"""Persistent plan file for the planner-executor pipeline.

A plan is a JSON file at `data/plans/<plan_id>.json` with the schema:

    {
      "plan_id": "<hex>",
      "task": "<original user request>",
      "created_at": <float>,
      "chunks": [
        {
          "id": 1,
          "instruction": "<what the executor should do>",
          "status": "pending" | "executing" | "done" | "failed",
          "output": "<executor output>" | null,
          "executed_at": <float> | null
        },
        ...
      ],
      "current_chunk": <int> | null,
      "final_summary": "<text>" | null,
      "status": "in_progress" | "waiting" | "complete" | "aborted",
      "waiting_reason": "<text>" | null,
      "planner_messages": [{"role": "system"|"human"|"ai"|"tool", "content": "<text>", "tool_call_id": "..."|null}, ...] | null
    }

Used by the planner node (writes/reads), executor node (writes outputs),
and the LangGraph loop (uses current_chunk to drive iteration).
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_ROOT = Path(__file__).resolve().parent
PLANS_ROOT = DATA_ROOT / "plans"

# Cap on persisted plans. When the cap is exceeded, the oldest plans
# (by mtime) are deleted. Keeps disk usage bounded for long-running
# sessions and prevents the plans/ directory from accumulating stale state.
MAX_PLANS = 5


def _ensure_root() -> Path:
    PLANS_ROOT.mkdir(parents=True, exist_ok=True)
    return PLANS_ROOT


def _plan_path(plan_id: str) -> Path:
    if not plan_id or not all(c in "0123456789abcdef" for c in plan_id):
        raise ValueError(f"Invalid plan_id: {plan_id!r}")
    return _ensure_root() / f"{plan_id}.json"


def new_plan_id() -> str:
    return secrets.token_hex(6)


# ---- File I/O ----

def save_plan(plan: Dict[str, Any]) -> None:
    """Persist a plan dict to disk. Atomic write via temp file."""
    plan_id = plan.get("plan_id")
    if not plan_id:
        raise ValueError("plan dict missing 'plan_id'")
    path = _plan_path(plan_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_plan(plan_id: str) -> Optional[Dict[str, Any]]:
    path = _plan_path(plan_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_plans() -> List[str]:
    _ensure_root()
    paths = sorted(PLANS_ROOT.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    return [p.stem for p in paths]


def cleanup_old_plans(max_count: int = MAX_PLANS) -> int:
    """Delete oldest plan files until at most `max_count` remain.

    Plans are sorted by mtime descending (newest first). Files beyond
    the cap are removed. Returns the number of plans deleted.

    Safe to call from make_blank_plan or as a periodic janitor.
    """
    _ensure_root()
    paths = sorted(PLANS_ROOT.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    if len(paths) <= max_count:
        return 0
    deleted = 0
    for stale in paths[max_count:]:
        try:
            stale.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


# ---- Mutators ----

def make_blank_plan(task: str, chunks: List[str]) -> Dict[str, Any]:
    # Enforce the storage cap *before* creating a new plan file. We leave
    # room for the new file, otherwise the cap would always overshoot by 1.
    cleanup_old_plans(max_count=MAX_PLANS - 1)
    plan_id = new_plan_id()
    now = time.time()
    return {
        "plan_id": plan_id,
        "task": task,
        "created_at": now,
        "chunks": [
            {
                "id": i + 1,
                "instruction": instr.strip(),
                "status": "pending",
                "output": None,
                "tool_calls": [],
                "executed_at": None,
                "context_blocks": [],
            }
            for i, instr in enumerate(chunks)
            if instr and instr.strip()
        ],
        "current_chunk": None,
        "final_summary": None,
        "status": "in_progress",
    }


def mark_executing(plan: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["status"] = "executing"
            plan["current_chunk"] = chunk_id
            return plan
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def mark_done(
    plan: Dict[str, Any],
    chunk_id: int,
    output: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mark a chunk as done.

    `tool_calls` is the structured list of tool invocations made by the
    executor: each entry is a dict with `name`, `args`, `result`, `ok`
    (bool), and optionally `error`. The planner reads this directly
    instead of trusting the LLM's prose summary.
    """
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["status"] = "done"
            c["output"] = output
            c["tool_calls"] = tool_calls or []
            c["executed_at"] = time.time()
            return plan
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def mark_failed(
    plan: Dict[str, Any],
    chunk_id: int,
    error: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mark a chunk as failed.

    `tool_calls` is the structured list of tool invocations made before
    the failure (partial work — useful for the planner to decide
    whether to retry, skip, or stop).
    """
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["status"] = "failed"
            c["output"] = f"FAILED: {error}"
            c["tool_calls"] = tool_calls or []
            c["executed_at"] = time.time()
            return plan
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def next_pending(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for c in plan.get("chunks", []):
        if c["status"] == "pending":
            return c
    return None


def has_staged_chunks(plan: Dict[str, Any]) -> bool:
    return any(c.get("status") == "staged" for c in plan.get("chunks", []))


def finalize(plan: Dict[str, Any], summary: str) -> Dict[str, Any]:
    plan["status"] = "complete"
    plan["final_summary"] = summary
    plan["current_chunk"] = None
    return plan


def abort(plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
    plan["status"] = "aborted"
    plan["final_summary"] = f"ABORTED: {reason}"
    plan["current_chunk"] = None
    return plan


def set_waiting(plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Mark the plan as waiting for an external event or human input.

    Unlike abort, the plan is not terminated — chunks stay in their
    current state. The planner can be resumed later via resume_waiting().
    Use cases:
      - Planner asks the user a clarifying question mid-plan
      - Plan depends on a scheduled job / external webhook
      - Human-approval gate on a destructive operation
    """
    plan["status"] = "waiting"
    plan["waiting_reason"] = reason
    plan["current_chunk"] = None
    return plan


def resume_waiting(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Resume a plan that was in 'waiting' state back to 'in_progress'.

    No-op if the plan is not currently waiting.
    """
    if plan.get("status") == "waiting":
        plan["status"] = "in_progress"
        plan["waiting_reason"] = None
    return plan


def is_plan_terminal(status: str) -> bool:
    """A plan is terminal if it won't resume on its own: complete or aborted.

    'waiting' is NOT terminal — it's a paused state that can be resumed
    via resume_waiting(). 'in_progress' is the active state.
    """
    return status in ("complete", "aborted")


# ---- Persistent planner context ----
# In coder mode the 7B planner should stay "alive" across executor chunks
# instead of rebuilding its context from scratch each iteration. We persist
# the planner's message stream (system prompt, RAG retrievals, AI responses,
# tool calls/results) to the plan file. Next iteration loads this as the
# starting point and re-injects dynamic content (RAG block, task pin).
#
# This is a capability stub — full wiring into planner_node requires
# refactoring _build_planner_prompt to accept a saved-context prefix.
# Until then, the field is populated by save_planner_messages() but not
# consumed on reload (planner still builds fresh each turn).

def save_planner_messages(plan: Dict[str, Any], messages: List[Any]) -> Dict[str, Any]:
    """Persist a planner message stream to the plan file.

    Accepts a list of LangChain messages (SystemMessage/HumanMessage/
    AIMessage/ToolMessage) or pre-serialized dicts. Serializes to a
    plain JSON-safe form.

    Best-effort: skips messages that fail to serialize rather than
    raising, so a partial save doesn't break the planner loop.
    """
    serialized: List[Dict[str, Any]] = []
    for msg in messages or []:
        try:
            if isinstance(msg, dict):
                serialized.append(msg)
                continue
            # LangChain message object
            role = getattr(msg, "type", None) or _msg_role(msg)
            entry: Dict[str, Any] = {
                "role": role,
                "content": getattr(msg, "content", "") or "",
            }
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            serialized.append(entry)
        except Exception:
            continue
    plan["planner_messages"] = serialized
    return plan


def load_planner_messages(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the persisted planner message stream, or [] if none."""
    return plan.get("planner_messages") or []


def clear_planner_messages(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the persisted planner context (call on plan finalization)."""
    plan["planner_messages"] = None
    return plan


def _msg_role(msg: Any) -> str:
    """Best-effort role extraction for a LangChain message object."""
    cls_name = type(msg).__name__.lower()
    if "system" in cls_name:
        return "system"
    if "human" in cls_name:
        return "human"
    if "ai" in cls_name:
        return "ai"
    if "tool" in cls_name:
        return "tool"
    return "unknown"


# ---- FIM-style plan edits ----
# The planner uses these to mutate an existing plan mid-execution in
# response to a partial failure: add a fix-up chunk, replace a chunk's
# instruction with a different approach, or drop a chunk entirely.

def add_chunk_after(plan: Dict[str, Any], anchor_id: int, instruction: str) -> Dict[str, Any]:
    """Insert a new pending chunk immediately after `anchor_id`.

    Returns the new chunk's id (a fresh int, max(existing)+1). Raises
    KeyError if the anchor chunk doesn't exist.
    """
    instr = (instruction or "").strip()
    if not instr:
        raise ValueError("instruction is empty")
    chunks = plan.get("chunks", [])
    new_id = (max((c["id"] for c in chunks), default=0)) + 1
    new_chunk = {
        "id": new_id,
        "instruction": instr,
        "status": "pending",
        "output": None,
        "executed_at": None,
        "context_blocks": [],
    }
    for i, c in enumerate(chunks):
        if c["id"] == anchor_id:
            chunks.insert(i + 1, new_chunk)
            return new_chunk
    raise KeyError(f"No chunk with id={anchor_id} in plan {plan.get('plan_id')}")


def replace_chunk(plan: Dict[str, Any], chunk_id: int, instruction: str) -> Dict[str, Any]:
    """Replace the instruction of an existing chunk.

    Resets status to pending and clears output. Idempotent on a chunk
    that's already pending. Raises KeyError if the chunk doesn't exist.
    """
    instr = (instruction or "").strip()
    if not instr:
        raise ValueError("instruction is empty")
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["instruction"] = instr
            c["status"] = "pending"
            c["output"] = None
            c["executed_at"] = None
            # Preserve any existing context_blocks (carried over from the
            # original chunk). Don't wipe — the new instruction still
            # inherits the project's RAG context the planner gathered.
            if "context_blocks" not in c:
                c["context_blocks"] = []
            return c
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def remove_chunk(plan: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
    """Drop a chunk from the plan entirely.

    Returns the removed chunk. Raises KeyError if it doesn't exist.
    """
    chunks = plan.get("chunks", [])
    for i, c in enumerate(chunks):
        if c["id"] == chunk_id:
            removed = chunks.pop(i)
            # If the plan was pointing at this chunk, clear current_chunk.
            if plan.get("current_chunk") == chunk_id:
                plan["current_chunk"] = None
            return removed
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def skip_chunk(plan: Dict[str, Any], chunk_id: int, reason: str = "") -> Dict[str, Any]:
    """Mark a chunk as skipped. Skipped chunks don't block progress.

    Use this when the planner decides the failure isn't critical — the
    executor's mistake or a benign tool error. The chunk's status is set
    to 'skipped' and its output records the reason. next_pending() skips
    skipped chunks automatically.
    """
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["status"] = "skipped"
            c["output"] = f"SKIPPED: {reason}" if reason else "SKIPPED"
            c["executed_at"] = time.time()
            return c
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def mark_staged(plan: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
    """Mark a chunk as staged_in_memory.

    Used by the batch executor when a chunk's edits have been staged in
    STAGING_REGISTRY but not yet committed to disk. Staged chunks are
    skipped on subsequent executor sweeps; the planner treats them as
    awaiting user approval. Once approved, staged chunks transition to
    'done'.
    """
    for c in plan.get("chunks", []):
        if c["id"] == chunk_id:
            c["status"] = "staged"
            c["output"] = c.get("output") or "STAGED: awaiting user approval"
            c["executed_at"] = c.get("executed_at") or time.time()
            return plan
    raise KeyError(f"No chunk with id={chunk_id} in plan {plan.get('plan_id')}")


def resume_staged_to_pending(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Reset all 'staged' chunks back to 'pending'.

    Called when the user rejects the staged batch so the plan can be
    retried or modified without recreating it from scratch.
    """
    for c in plan.get("chunks", []):
        if c["status"] == "staged":
            c["status"] = "pending"
            c["output"] = None
            c["executed_at"] = None
    return plan


def finalize_staged(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Mark all 'staged' chunks as 'done' after user approval.

    The physical writes are performed by the chatbot worker thread
    before this is called. This function updates the plan state so the
    planner can finalize the plan.
    """
    for c in plan.get("chunks", []):
        if c["status"] == "staged":
            c["status"] = "done"
            c["executed_at"] = time.time()
            if not c.get("output"):
                c["output"] = "Applied (user-approved)"
    return plan


def is_chunk_terminal(status: str) -> bool:
    """A chunk is terminal if it won't run again: done, failed, skipped, or staged.

    'staged' is terminal for the executor because the edit is in memory
    awaiting user approval; the planner decides what to do next.
    """
    return status in ("done", "failed", "skipped", "staged")
