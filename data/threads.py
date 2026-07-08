"""Thread persistence and pinning for Ved.

Owns the thread state machine: create / load / save / switch / rename /
delete, plus the per-thread pin/unpin metadata. Extracted from chatbot.py
during the Phase 3 refactor so thread logic lives next to the other
`data/` modules (plans, threads, memories).

Thread file format
------------------
Threads persist to `data/threads.json` as a dict keyed by thread id, with
each entry containing {id, title, created_at, messages}. Messages are
serialized to plain dicts via `_serialize_message` / `_deserialize_message`
so the JSON is human-inspectable and not tied to LangChain internals.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# Cap matches graph/state.py limit_messages. Kept here as the source of truth
# for on-disk trimming; the in-memory reducer in state.py enforces it for
# LLM-bound messages.
THREAD_MESSAGE_CAP = 40  # 1 system prompt + up to 39 other messages.

# Tool messages persisted to history get truncated to this many chars.
# The LLM usually only needs the first chunk to know what the tool returned.
_TOOL_HISTORY_TRUNCATE_CHARS = 800  # ~200 tokens

# Compress long AI content to head + tail summary for history persistence.
# Full content is saved to thread RAG separately; the summary stays inline.
_AI_SUMMARY_THRESHOLD_CHARS = 1200  # ~300 tokens
_AI_SUMMARY_HEAD_WORDS = 30
_AI_SUMMARY_TAIL_WORDS = 30


# ---- Serialization helpers ----

def _serialize_message(msg) -> dict:
    cls_name = type(msg).__name__
    if cls_name == "HumanMessage":
        role = "human"
    elif cls_name == "AIMessage":
        role = "ai"
    elif cls_name == "SystemMessage":
        role = "system"
    elif cls_name == "ToolMessage":
        role = "tool"
    else:
        role = cls_name.lower()
    content = msg.content
    # Compact long tool outputs in history. The LLM usually only needs the
    # first chunk to know what the tool returned; full output can be
    # recovered later via retrieve_rag if needed.
    if cls_name == "ToolMessage" and isinstance(content, str) and len(content) > _TOOL_HISTORY_TRUNCATE_CHARS:
        content = (
            content[:_TOOL_HISTORY_TRUNCATE_CHARS]
            + f"\n\n...[truncated; full output recoverable via retrieve_rag]"
        )
    out = {"role": role, "content": content}
    # ToolMessage carries tool_call_id which the LLM uses to associate the
    # result with the originating tool_call. Preserve it across save/load.
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    # Preserve additional_kwargs (e.g., "pinned": True so FIFO doesn't drop pinned turns).
    extra = getattr(msg, "additional_kwargs", None)
    if extra:
        out["additional_kwargs"] = dict(extra)
    return out


def _deserialize_message(data: dict) -> BaseMessage:
    role = data.get("role", "")
    content = data.get("content", "")
    extra = data.get("additional_kwargs") or {}
    tool_call_id = data.get("tool_call_id")
    if role == "human":
        msg = HumanMessage(content=content)
    elif role == "ai":
        msg = AIMessage(content=content)
    elif role == "system":
        msg = SystemMessage(content=content)
    elif role == "tool":
        # ToolMessage requires tool_call_id; use a placeholder if missing.
        # Older saves (pre-tool-persistence) may not have stored one.
        msg = ToolMessage(content=content, tool_call_id=tool_call_id or "legacy_unpaired")
    else:
        msg = HumanMessage(content=content)
    if extra:
        # Restore pinned flag (and any future additional_kwargs) on reload.
        try:
            msg.additional_kwargs.update(extra)
        except Exception:
            pass
    return msg


def _trim_thread_messages(messages: list) -> list:
    """Keep at most THREAD_MESSAGE_CAP messages: first SystemMessage (if any) + most recent (CAP-1) others."""
    if len(messages) <= THREAD_MESSAGE_CAP:
        return messages
    system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    others = [m for m in messages if not isinstance(m, SystemMessage)]
    if system is not None:
        return [system] + others[-(THREAD_MESSAGE_CAP - 1):]
    return others[-THREAD_MESSAGE_CAP:]


def _autotitle_from_message(text: str) -> str:
    """Use the first ~40 chars of the user's first message as the thread title."""
    stripped = (text or "").strip()
    if len(stripped) <= 40:
        return stripped
    return stripped[:40]


# ---- ThreadStore ----

class ThreadStore:
    """Owns the threads dict, active-thread pointer, and pin metadata.

    Persistence: writes the full `_threads` dict to `db_path` (typically
    `data/threads.json`) on every mutation. Reads happen once at
    construction time via `load()`.

    Thread file integration: `delete_thread` notifies the
    `thread_files` callback (if set) to drop the thread's chunks from
    the vector DB. Caller passes the callback in __init__ to avoid
    loading the embedding engine when threads.py is imported.
    """

    def __init__(self, db_path: Path, thread_files=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._threads: dict = {}
        self._active_thread_id: Optional[str] = None
        # Optional callback fired when a thread is deleted. Used to drop
        # the thread's chunks from the vector DB. Set by Chatbot.__init__
        # after the ThreadFileStore is constructed (avoids loading the
        # embedding engine at threads.py import time).
        self._on_thread_deleted = None
        if thread_files is not None:
            self._on_thread_deleted = lambda tid: thread_files.clear_thread(tid)
        self.load()

    # ---- Persistence ----

    def load(self) -> None:
        path = self.db_path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            entries = list(raw.values())
        else:
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tid = entry.get("id")
            if not tid or not isinstance(tid, str):
                continue
            msgs_raw = entry.get("messages", [])
            if not isinstance(msgs_raw, list):
                msgs_raw = []
            messages = [_deserialize_message(m) for m in msgs_raw if isinstance(m, dict)]
            messages = _trim_thread_messages(messages)
            self._threads[tid] = {
                "id": tid,
                "title": entry.get("title", "New Thread"),
                "created_at": entry.get("created_at", time.time()),
                "messages": messages,
            }
        if self._threads and (self._active_thread_id is None or self._active_thread_id not in self._threads):
            self._active_thread_id = next(
                iter(sorted(self._threads.keys(), key=lambda k: self._threads[k]["created_at"]))
            )

    def save(self) -> None:
        path = self.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        # Iterate threads in REVERSE chronological order so the JSON file
        # has newest threads first. Dict iteration order is preserved in
        # Python 3.7+, so this puts the most recent thread at the top of
        # the file for easy human inspection and predictable load order.
        for tid, thread in sorted(
            self._threads.items(),
            key=lambda kv: kv[1].get("created_at", 0.0),
            reverse=True,
        ):
            thread["messages"] = _trim_thread_messages(thread["messages"])
            payload[tid] = {
                "id": thread["id"],
                "title": thread["title"],
                "created_at": thread["created_at"],
                "messages": [_serialize_message(m) for m in thread["messages"]],
            }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- CRUD ----

    def create_starter(self) -> str:
        """Create and activate the very first thread. Used when no threads exist."""
        tid = f"thr_{secrets.token_hex(4)}"
        self._threads[tid] = {
            "id": tid,
            "title": "New Thread",
            "created_at": time.time(),
            "messages": [],
        }
        self._active_thread_id = tid
        self.save()
        return tid

    def create(self, title: Optional[str] = None) -> str:
        tid = f"thr_{secrets.token_hex(4)}"
        self._threads[tid] = {
            "id": tid,
            "title": title if title else "New Thread",
            "created_at": time.time(),
            "messages": [],
        }
        self._active_thread_id = tid
        self.save()
        return tid

    def list(self) -> list:
        """Return threads sorted newest-first by created_at."""
        return sorted(
            ({"id": t["id"], "title": t["title"], "created_at": t["created_at"]} for t in self._threads.values()),
            key=lambda d: d["created_at"],
            reverse=True,
        )

    def set_active(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._active_thread_id = thread_id
        return True

    def rename(self, thread_id: str, title: str) -> bool:
        if thread_id not in self._threads:
            return False
        self._threads[thread_id]["title"] = title
        self.save()
        return True

    def delete(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        if len(self._threads) <= 1:
            return False
        was_active = (self._active_thread_id == thread_id)
        del self._threads[thread_id]
        # Drop the thread's chunks and metadata from the vector DB.
        if self._on_thread_deleted is not None:
            try:
                self._on_thread_deleted(thread_id)
            except Exception:
                pass
        if was_active:
            if self._threads:
                oldest = min(self._threads.values(), key=lambda t: t["created_at"])
                self._active_thread_id = oldest["id"]
            else:
                # Edge case: deleting the only thread (already guarded above
                # by len(self._threads) <= 1), but defensive fallback.
                self.create_starter()
                return True
        self.save()
        return True

    def get_active(self) -> dict:
        if self._active_thread_id is None or self._active_thread_id not in self._threads:
            if not self._threads:
                self.create_starter()
            else:
                self._active_thread_id = next(iter(self._threads))
        return self._threads[self._active_thread_id]

    # ---- Property bridge (back-compat with code that read _conversation_history) ----

    @property
    def conversation_history(self) -> list:
        return self.get_active()["messages"]

    @conversation_history.setter
    def conversation_history(self, value):
        if self._active_thread_id and self._active_thread_id in self._threads:
            self._threads[self._active_thread_id]["messages"] = value
        else:
            self.get_active()["messages"] = value

    # ---- Pinning ----
    # Pinned messages are marked with `additional_kwargs["pinned"] = True`.
    # They live inside their own thread and never leak into other threads.
    # The state.limit_messages reducer preserves them from being trimmed.

    def get_pinned_messages_in_active_thread(self) -> list:
        """Return the pinned messages in the current thread, oldest first."""
        thread = self.get_active()
        out = []
        for m in thread.get("messages", []):
            if getattr(m, "additional_kwargs", {}).get("pinned", False):
                out.append(m)
        return out

    def pin_last_turn_in_active_thread(self) -> int:
        """Mark the last AI message (and its preceding Human) as pinned.

        Returns the number of messages newly pinned. 0 if nothing to pin.
        """
        thread = self.get_active()
        msgs = thread.get("messages", [])
        # Find the last AIMessage in the thread.
        last_ai_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage):
                last_ai_idx = i
                break
        if last_ai_idx < 0:
            return 0
        pinned_count = 0
        msgs[last_ai_idx].additional_kwargs["pinned"] = True
        pinned_count += 1
        # Also pin the immediately preceding HumanMessage if present.
        if last_ai_idx > 0 and isinstance(msgs[last_ai_idx - 1], HumanMessage):
            msgs[last_ai_idx - 1].additional_kwargs["pinned"] = True
            pinned_count += 1
        self.save()
        return pinned_count

    def unpin_in_active_thread(self, index_1based: int) -> int:
        """Unpin the Nth pinned message in the current thread (1-based)."""
        thread = self.get_active()
        pinned = self.get_pinned_messages_in_active_thread()
        if index_1based < 1 or index_1based > len(pinned):
            return 0
        target = pinned[index_1based - 1]
        target.additional_kwargs["pinned"] = False
        self.save()
        return 1

    def unpin_all_in_active_thread(self) -> int:
        """Clear the pinned flag on all messages in the current thread."""
        thread = self.get_active()
        cleared = 0
        for m in thread.get("messages", []):
            if getattr(m, "additional_kwargs", {}).get("pinned", False):
                m.additional_kwargs["pinned"] = False
                cleared += 1
        if cleared:
            self.save()
        return cleared
