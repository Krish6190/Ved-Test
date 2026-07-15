"""Planner node — same llama model as chat, but a focused role.

The planner has exactly one tool: `retrieve_rag`. It can look up past
content from the thread's RAG store (AI responses that got compressed,
uploaded files, etc.) but it cannot call read_file / execute_python /
edit_file / etc. — those are the executor's job.

The planner's outputs are text markers (CREATE_PLAN, DIRECT_ANSWER,
EXECUTE_NEXT, FINAL_SUMMARY) parsed by the graph loop.

Difference from the previous design:
  - Planner now HAS the retrieve_rag tool (was bind_tools([]) before).
  - Planner sees a capped message history (10 for llama, 40 for qwen)
    so it can plan in context. Previously the planner only saw the latest
    user message — that was too lean.
  - Executor no longer gets the full msg history — it only sees the
    chunk instruction + a brief plan-status line. The planner writes
    SELF-CONTAINED chunk instructions so the executor doesn't need
    prior conversation context.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from data import plans as plan_store
from graph.nodes._helpers import _build_rag_block
from graph.nodes._hints import _FRESH_QUESTION_HINT
from graph.nodes._stream_helpers import _stream_text, _clean_chunk
from graph.nodes.planner_diagnostics import escalate, EscalationAction
from graph.state import VedState
from graph.tools._common import coerce_retrieve_rag_args, normalize_tool_args
from graph.tools.rag_retrieve import retrieve_rag

# ---- Output marker parsing (unchanged) ----

_CREATE_PLAN_RE = re.compile(r"CREATE_PLAN:\s*(\[.*?\])", re.DOTALL)
_DIRECT_ANSWER_RE = re.compile(r"\bDIRECT_ANSWER:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)", re.DOTALL)
_EXECUTE_NEXT_RE = re.compile(r"\bEXECUTE_NEXT\b")
_FINAL_SUMMARY_RE = re.compile(r"\bFINAL_SUMMARY:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)", re.DOTALL)
_ADD_CHUNK_AFTER_RE = re.compile(
    r"ADD_CHUNK_AFTER\s+(\d+)\s*:\s*\n\s*INSTRUCTION\s*:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)",
    re.DOTALL,
)
_REPLACE_CHUNK_RE = re.compile(
    r"REPLACE_CHUNK\s+(\d+)\s*:\s*\n\s*INSTRUCTION\s*:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)",
    re.DOTALL,
)
_REMOVE_CHUNK_RE = re.compile(r"REMOVE_CHUNK\s+(\d+)\b")
_SKIP_CHUNK_RE = re.compile(
    r"SKIP_CHUNK\s+(\d+)(?:\s+REASON\s*:\s*(.*?))?(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK|SKIP_CHUNK|RECOMMEND_CODER):|\Z)",
    re.DOTALL,
)
_RECOMMEND_CODER_RE = re.compile(
    r"\bRECOMMEND_CODER_MODE(?:\s+REASON\s*:\s*(.*?))?(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK|SKIP_CHUNK|RECOMMEND_CODER):|\Z)",
    re.DOTALL,
)
def parse_planner_output(text: str) -> Tuple[str, Any]:
    """Parse planner text into (kind, payload).
    Kinds:
      - create_plan        -> list[str]   chunk instructions
      - direct_answer      -> str        answer text
      - execute_next       -> None
      - final_summary      -> str        summary text
      - add_chunk_after    -> (anchor_id: int, instruction: str)
      - replace_chunk      -> (chunk_id: int, instruction: str)
      - remove_chunk       -> int (chunk_id)
      - skip_chunk         -> (chunk_id: int, reason: str)
      - recommend_coder    -> str        reason (when std mode hits coding)
      - fallback           -> str        raw text
    """
    if not text:
        return "fallback", ""
    m = _CREATE_PLAN_RE.search(text)
    if m:
        raw = m.group(1).strip()
        try:
            chunks = json.loads(raw)
            if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
                return "create_plan", chunks
        except Exception:
            pass
    m = _RECOMMEND_CODER_RE.search(text)
    if m:
        reason = (m.group(1) or "").strip()
        return "recommend_coder", reason
    m = _DIRECT_ANSWER_RE.search(text)
    if m:
        answer = m.group(1).strip()
        if answer:
            return "direct_answer", answer
    m = _FINAL_SUMMARY_RE.search(text)
    if m:
        summary = m.group(1).strip()
        if summary:
            return "final_summary", summary
    m = _ADD_CHUNK_AFTER_RE.search(text)
    if m:
        anchor = int(m.group(1))
        instr = m.group(2).strip()
        if instr:
            return "add_chunk_after", (anchor, instr)
    m = _REPLACE_CHUNK_RE.search(text)
    if m:
        chunk_id = int(m.group(1))
        instr = m.group(2).strip()
        if instr:
            return "replace_chunk", (chunk_id, instr)
    m = _REMOVE_CHUNK_RE.search(text)
    if m:
        return "remove_chunk", int(m.group(1))
    m = _SKIP_CHUNK_RE.search(text)
    if m:
        chunk_id = int(m.group(1))
        reason = (m.group(2) or "").strip()
        return "skip_chunk", (chunk_id, reason)
    if _EXECUTE_NEXT_RE.search(text):
        return "execute_next", None
    return "fallback", text.strip()


# ---- System prompt ----

_PLANNER_SYSTEM_BASE = (
    "You are the PLANNER in a planner-executor pipeline.\n"
    "\n"
    "TOOL: retrieve_rag(query, scope, k) — pulls relevant text from the thread/project RAG store. "
    "Use it for past-chat recall, codebase exploration, or when the user points at a directory. "
    "You do NOT have file-editing or execution tools; those belong to the executor.\n"
    "\n"
    "OUTPUT EXACTLY ONE marker per turn:\n"
    "  CREATE_PLAN: [\"<chunk 1>\", ...] — first turn, 1-5 self-contained chunks\n"
    "  DIRECT_ANSWER: <text> — simple non-tool replies only\n"
    "  EXECUTE_NEXT — continue an in-progress plan\n"
    "  FINAL_SUMMARY: <text> — all chunks complete\n"
    "  ADD_CHUNK_AFTER <id>: INSTRUCTION: <text> — insert a fix-up step\n"
    "  REPLACE_CHUNK <id>: INSTRUCTION: <text> — rewrite and retry a failed chunk\n"
    "  REMOVE_CHUNK <id> — drop an irrelevant chunk\n"
    "  SKIP_CHUNK <id> REASON: <text> — benign failure, move on\n"
    "\n"
    "Rules:\n"
    "  - Default to CREATE_PLAN for any tool/file/code/search request. DIRECT_ANSWER only for chitchat or facts.\n"
    "  - Each chunk instruction must be self-contained (action, target, context); the executor sees no history.\n"
    "  - Failure triage: REPLACE_CHUNK (executor mistake), SKIP_CHUNK (benign/broken tool), or STOP with FINAL_SUMMARY.\n"
    "  - retrieve_rag(query, paths=['dir/']) — query is always required; use paths only as an optional directory filter. Empty RAG = use search_files.\n"
    "  - Never call web_search for files in this project; it is not in your tool registry.\n"
)

_PLANNER_RECOMMEND_CODER_BLOCK = (
    "\n"
    "RECOMMEND_CODER_MODE REASON: <why>\n"
    "  Use ONLY in standard/turbo mode when the user asks to write, edit, delete, or run code. "
    "Do NOT emit DIRECT_ANSWER for editing tasks; use RECOMMEND_CODER_MODE instead. "
    "Tell the user to switch to coder mode.\n"
)

_PLANNER_SYSTEM = SystemMessage(content=_PLANNER_SYSTEM_BASE + _PLANNER_RECOMMEND_CODER_BLOCK)


def _build_planner_system_prompt(mode: str) -> str:
    """Return the planner system prompt, omitting the coder recommendation in coder mode."""
    if mode == "coder":
        return _PLANNER_SYSTEM_BASE
    return _PLANNER_SYSTEM_BASE + _PLANNER_RECOMMEND_CODER_BLOCK


# ---- Model-aware message cap ----
# Smaller models (llama3.2:3b) choke on long histories; larger ones
# (qwen2.5-coder:7b) can handle more. Cap based on current mode.

_MSG_HISTORY_CAP = {
    "standard": 10,   # llama3.2:3b — small context window, keep tight
    "turbo":   10,   # same model family as standard
    "coder":   40,   # qwen2.5-coder:7b — bigger context, can fit more
    "hibernate": 0,  # no LLM running; cap irrelevant
}


def _recent_human_ai(messages, cap: int) -> List:
    """Return the last `cap` HumanMessage + AIMessage (skip System/Tool)."""
    if cap <= 0:
        return []
    recent: List = []
    for m in reversed(messages):
        if isinstance(m, (HumanMessage, AIMessage)):
            recent.append(m)
            if len(recent) >= cap:
                break
    return list(reversed(recent))


def _build_planner_prompt(state: VedState, plan: Optional[Dict[str, Any]], config: Optional[RunnableConfig] = None) -> list:
    """Compose the message stream sent to the planner LLM.

    Includes:
      - Planner system prompt (mentions retrieve_rag tool)
      - Per-turn fresh-question hint
      - A capped slice of recent HumanMessage + AIMessage history
      - The user's current request (or a "continue the plan" instruction
        if a plan is already in progress)
    """
    user_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
    last_user = user_msgs[-1].content if user_msgs else ""

    msgs = [SystemMessage(content=_build_planner_system_prompt(state.mode)), _FRESH_QUESTION_HINT]
    # RAG auto-injection: same as chat_node does. Without this, a user
    # who uploads a file via /files/thread and then asks about it gets
    # a DIRECT_ANSWER with no awareness of the uploaded content. With
    active_thread_id = None
    if config and isinstance(config.get("configurable"), dict):
        active_thread_id = config["configurable"].get("active_thread_id")
    try:
        rag_block = _build_rag_block(last_user, active_thread_id)
        if rag_block:
            msgs.append(SystemMessage(content=rag_block))
    except Exception:
        pass

    # Virtual sandbox injection: tell the Planner which files currently
    # have staged edits in STAGING_REGISTRY. This prevents the Planner
    # from re-reading the unchanged physical disk / raw RAG store and
    # concluding that a chunk failed. It also lets the Planner advance
    # to the next chunk instead of recreating the plan.
    if active_thread_id:
        try:
            from graph.tools.staging_registry import STAGING_REGISTRY
            staged_paths = sorted(
                STAGING_REGISTRY.get_tasks(active_thread_id).keys()
            )
            if staged_paths:
                msgs.append(SystemMessage(content=(
                    "STAGED EDITS (awaiting user approval) — these files "
                    "have been updated in memory but are NOT yet committed "
                    "to disk. Treat them as complete for planning purposes; "
                    "do not rewrite them unless the user explicitly asks:\n"
                    + "\n".join(f"  - {p}" for p in staged_paths)
                )))
        except Exception:
            pass

    # Task pin: when a plan is active, always include the original task
    # verbatim near the top. This survives history-clipping in long plans
    # so the planner never loses sight of what it's executing.
    if plan is not None and plan.get("task"):
        msgs.append(SystemMessage(content=(
            f"PLAN TASK (always visible across all turns): {plan['task']}"
        )))

    # Capped message history — the planner needs context to plan well,
    # but small models can't handle the full 40-message cap.
    cap = _MSG_HISTORY_CAP.get(getattr(state, "mode", "standard"), 10)
    history = _recent_human_ai(state.messages, cap)
    if history:
        msgs.append(SystemMessage(content=(
            "Recent conversation history (most recent last; the LATEST "
            "human message at the bottom is the current request):\n\n"
            + "\n".join(
                f"[{type(m).__name__.replace('Message','').upper()}] {m.content[:400]}"
                for m in history
            )
        )))

    if plan is None:
        msgs.append(HumanMessage(content=(
            f"Decide whether to PLAN or DIRECTLY ANSWER this request:\n\n"
            f"USER REQUEST: {last_user}\n\n"
            "If the request references past context, call retrieve_rag first. "
            "Then output one marker:\n"
            "  - CREATE_PLAN: [\"...\", \"...\"]  if 2+ steps are needed\n"
            "  - DIRECT_ANSWER: <text>          if the task is simple\n"
        )))
    else:
        # 'staged' chunks have had their edits stored in STAGING_REGISTRY
        # but not yet committed to disk. They are complete from the
        # executor's point of view and must NOT be re-executed. The
        # planner should advance past them just like 'done' chunks.
        chunks_done = [
            c for c in plan["chunks"]
            if c["status"] in ("done", "failed", "staged")
        ]
        pending = [c for c in plan["chunks"] if c["status"] == "pending"]
        staged = [c for c in plan["chunks"] if c["status"] == "staged"]
        progress_lines = []
        for c in plan["chunks"]:
            line = f"  [{c['id']}] {c['status'].upper():9} - {c['instruction'][:80]}"
            if c["status"] == "failed" and c.get("output"):
                # Surface the error so the planner can decide whether to retry
                # with a tweaked instruction, add a fix-up chunk, or fail out.
                line += f"   ERROR: {c['output'][:200]}"
            progress_lines.append(line)
        progress = "\n".join(progress_lines)
        last_done = chunks_done[-1] if chunks_done else None
        last_output_excerpt = ""
        if last_done and last_done.get("output"):
            last_output_excerpt = (
                f"\n\nLAST EXECUTOR OUTPUT (chunk {last_done['id']}):\n"
                f"{last_done['output'][:600]}"
                + ("..." if len(last_done.get("output", "")) > 600 else "")
            )
        if pending:
            instruction = (
                f"Continue the plan. {len(pending)} chunk(s) remain. "
                f"{len(staged)} chunk(s) are already staged in memory "
                f"awaiting user approval.\n\n"
                f"PLAN PROGRESS:\n{progress}{last_output_excerpt}\n\n"
                f"You may call retrieve_rag to look up more context before "
                f"deciding the next chunk's instruction. When ready, output "
                f"EXECUTE_NEXT or FINAL_SUMMARY: <text>."
            )
        elif staged:
            instruction = (
                f"All chunks have been executed. {len(staged)} edit(s) are "
                f"staged in memory awaiting user approval. Do NOT recreate "
                f"the plan. Write a FINAL_SUMMARY that briefly describes "
                f"what was done and notes that the edits are pending user "
                f"approval.\n\n"
                f"PLAN PROGRESS:\n{progress}{last_output_excerpt}\n\n"
                f"Output FINAL_SUMMARY: <one paragraph wrap-up for the user>"
            )
        else:
            instruction = (
                f"All chunks complete. Write a FINAL_SUMMARY.\n\n"
                f"PLAN PROGRESS:\n{progress}{last_output_excerpt}\n\n"
                f"Output FINAL_SUMMARY: <one paragraph wrap-up for the user>"
            )
        msgs.append(HumanMessage(content=instruction))
    return msgs


# ---- Tool-call loop ----

_MAX_PLANNER_TOOL_ROUNDS = 3
_MAX_FALLBACK_RETRIES = 1  # re-prompt once if the model forgets markers


# ---- Dual-role Planner (Thinker) prompt template ----

_PLANNER_THINKER_PROMPT_TEMPLATE = (
    "Find the logic error in this code snippet and write a single, clear "
    "instruction explaining exactly how to rewrite the line to fix it. "
    "Do not rewrite the full code. Code: {code}"
)


def _build_thinker_prompt(code_snippet: str) -> HumanMessage:
    """Return the exact Planner (Thinker) prompt for a code snippet."""
    return HumanMessage(content=_PLANNER_THINKER_PROMPT_TEMPLATE.format(code=code_snippet))


def _execute_planner_tool_call(
    tool_call: Dict[str, Any],
    config: RunnableConfig,
    call_cache: Optional[Dict[str, str]] = None,
) -> str:
    """Run a single planner tool call. Returns the content for a ToolMessage.

    `call_cache` is an optional per-invocation dict that memoizes retrieve_rag
    results by (query, scope, k) so the same query isn't re-embedded + re-searched
    twice within one planner turn (e.g. when the LLM retries the same call).
    """
    name = tool_call.get("name")
    if name == "retrieve_rag":
        raw_args = tool_call.get("args", {}) or {}
        args = normalize_tool_args(name, raw_args)
        query = args.get("query", "")
        scope = args.get("scope")
        k = args.get("k", 5)
        cache_key = f"{query}|{scope}|{k}"
        if call_cache is not None and cache_key in call_cache:
            return call_cache[cache_key]
        try:
            result = retrieve_rag.invoke(args, config=config)
        except Exception:
            retry_args = coerce_retrieve_rag_args(args)
            result = retrieve_rag.invoke(retry_args, config=config)
        if call_cache is not None:
            call_cache[cache_key] = result
        return result
    return f"ERROR: unknown tool '{name}'"


def _stream_with_tool_loop(llm, msgs, config, token_queue, max_rounds=_MAX_PLANNER_TOOL_ROUNDS):
    """Run an LLM stream with a small tool-call loop.

    Yields text tokens to token_queue as they arrive. Returns
    (final_ai_msg, tool_messages, last_rag_results) when done. Memoizes
    retrieve_rag results in `call_cache` so duplicate queries within one
    planner turn aren't re-embedded + re-searched. `last_rag_results` is a
    list of formatted result strings from every retrieve_rag call executed
    during this turn — the planner node attaches these to each new chunk
    as `context_blocks` so the executor can surface them as background.
    """
    import json as _json
    tool_messages: List[ToolMessage] = []
    call_cache: Dict[str, str] = {}
    last_rag_results: List[str] = []
    for round_idx in range(max_rounds + 1):
        full_content = ""
        tool_calls_acc: Dict[int, Dict] = {}
        for chunk in llm.stream(msgs):
            if hasattr(chunk, "content") and chunk.content:
                raw = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_content += raw
                c = _clean_chunk(raw)
                if c is None:
                    continue
                if token_queue:
                    try:
                        token_queue.put(c)
                    except Exception:
                        pass
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"name": "", "args": "", "id": ""}
                    if tc.get("id"):
                        tool_calls_acc[idx]["id"] = tc["id"]
                    if tc.get("name"):
                        tool_calls_acc[idx]["name"] = tc["name"]
                    if tc.get("args"):
                        tool_calls_acc[idx]["args"] += tc["args"]
        tool_calls_list: List[Dict] = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            try:
                args = _json.loads(tc["args"]) if tc["args"] else {}
            except Exception:
                args = {}
            tool_calls_list.append({"id": tc["id"], "name": tc["name"], "args": args})
        ai_msg = AIMessage(content=full_content, tool_calls=tool_calls_list)

        if not tool_calls_list:
            return ai_msg, tool_messages, last_rag_results

        if round_idx >= max_rounds:
            return ai_msg, tool_messages, last_rag_results
        for tc in tool_calls_list:
            result_content = _execute_planner_tool_call(tc, config, call_cache)
            tool_messages.append(ToolMessage(
                content=result_content,
                tool_call_id=tc["id"],
            ))
            if tc.get("name") == "retrieve_rag" and not result_content.startswith("ERROR"):
                last_rag_results.append(result_content)
        # Loop again with the tool messages appended.
        msgs = list(msgs) + [ai_msg] + tool_messages

    return ai_msg, tool_messages, last_rag_results  # unreachable but satisfies the type checker

# ---- Human-in-the-loop: plan approval gate ----

def _wait_for_plan_approval(
    token_queue,
    proposed_chunks: List[str],
    plan_approval_event,
    plan_approval_state,
) -> bool:
    """Block until the user approves or rejects the proposed plan.

    Emits a `("plan_approval_request", {"chunks": [...]})` event through
    the token_queue, waits on `plan_approval_event`, then reads the
    boolean decision out of `plan_approval_state["value"]`. The event is
    cleared and the state value is reset to None before returning so the
    next approval round starts from a clean slate.

    Mirrors the pattern used by `content_pipeline_node` for the
    content-generation approval gate. Returns True if the user approved,
    False if they rejected.
    """
    if token_queue is not None:
        try:
            token_queue.put(("plan_approval_request", {"chunks": list(proposed_chunks)}))
        except Exception:
            pass
    plan_approval_event.wait()
    approved = bool(plan_approval_state.get("value"))
    plan_approval_state["value"] = None
    plan_approval_event.clear()
    return approved


# ---- Node ----

def planner_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Run the planner LLM (with retrieve_rag tool), parse the marker, and decide routing.

    Side effects:
      - On CREATE_PLAN: writes a new plan file at data/plans/<id>.json
      - On FINAL_SUMMARY: updates the plan file with the summary
      - On tool calls (retrieve_rag): executes them and feeds results back
        to the planner for up to _MAX_PLANNER_TOOL_ROUNDS rounds
      - Surfaces plan progress to the UI via token_queue ("plan_update" events)
    """
    plan_id = state.active_plan_id if hasattr(state, "active_plan_id") else None
    plan: Optional[Dict[str, Any]] = None
    if plan_id:
        plan = plan_store.load_plan(plan_id)
    else:
        user_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
        if user_msgs:
            for pid in plan_store.list_plans():
                candidate = plan_store.load_plan(pid)
                if candidate and candidate.get("status") == "in_progress":
                    if candidate.get("task") == user_msgs[-1].content:
                        plan = candidate
                        plan_id = pid
                        break
    if getattr(state, "summary_emitted", False):
        return {
            "messages": [],
            "route_intent": "A",
            "mode": state.mode,
            "active_plan_id": None,
            "current_chunk_id": None,
            "current_step": None,
            "last_step_status": "",
        }
    current_step = getattr(state, "current_step", None)
    last_step_status = getattr(state, "last_step_status", "")
    if current_step is not None and last_step_status == "dispatched":
        return {
            "messages": [],
            "route_intent": "P",
            "mode": state.mode,
            "active_plan_id": getattr(state, "active_plan_id", None),
            "current_chunk_id": getattr(state, "current_chunk_id", None),
            "current_step": current_step,
            "last_step_status": last_step_status,
        }

    if plan is not None:
        for c in plan.get("chunks", []):
            if c.get("retry_count", 0) >= 4:
                decision = escalate(c["retry_count"], c.get("output", ""), plan)
                if decision.action == EscalationAction.HARD_HALT_USER_INTERVENTION:
                    plan_store.abort(plan, decision.halt_message)
                    plan_store.save_plan(plan)
                    try:
                        tq = config["configurable"]["token_queue"]
                        tq.put(("plan_update", {
                            "event": "plan_halted",
                            "reason": decision.reason,
                            "chunk_id": c["id"],
                        }))
                    except (KeyError, TypeError, AttributeError):
                        pass
                    return {
                        "messages": [AIMessage(content=decision.halt_message)],
                        "route_intent": "A",
                        "mode": state.mode,
                        "active_plan_id": None,
                        "chunk_retry_count": 0,
                    }
    factory = None
    if config and isinstance(config.get("configurable"), dict):
        factory = config["configurable"].get("planner_llm_factory")
    if factory is not None:
        llm = factory(state.mode)
    else:
        llm = get_llm()
    if llm is None:
        return {
            "messages": [AIMessage(content="No local model is available. Start Ollama.")],
            "route_intent": state.route_intent, "mode": state.mode,
        }
    # Planner has ONE tool: retrieve_rag. Everything else stays executor-side.
    llm_planner = llm.bind_tools([retrieve_rag])

    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    plan_approval_event = None
    plan_approval_state = None
    if config and isinstance(config.get("configurable"), dict):
        plan_approval_event = config["configurable"].get("plan_approval_event")
        plan_approval_state = config["configurable"].get("plan_approval_state")
    dual_phase = getattr(state, "dual_role_phase", "")
    if dual_phase == "analyze":
        code_snippet = getattr(state, "target_file_content", "") or ""
        target_path = getattr(state, "target_file_path", "") or ""
        if not code_snippet:
            # No code loaded yet — ask the executor to read the file first.
            updates = {
                "messages": [],
                "route_intent": "P",
                "mode": state.mode,
                "dual_role_phase": "read_target",
            }
            return updates
        thinker_msgs = [
            SystemMessage(content=_build_planner_system_prompt(state.mode)),
            _build_thinker_prompt(code_snippet),
        ]
        full_content = _stream_text(llm, thinker_msgs, token_queue)
        return {
            "messages": [],
            "route_intent": "P",
            "mode": state.mode,
            "dual_role_phase": "execute",
            "fix_instruction": full_content.strip(),
        }
    if dual_phase == "stage":
        pending = list(getattr(state, "pending_file_targets", []) or [])
        index = getattr(state, "current_file_target_index", 0) + 1
        completed = list(getattr(state, "completed_file_targets", []) or [])
        last_path = getattr(state, "target_file_path", "")
        if last_path:
            completed.append(last_path)
        if index < len(pending):
            return {
                "messages": [],
                "route_intent": "P",
                "mode": state.mode,
                "dual_role_phase": "analyze",
                "current_file_target_index": index,
                "target_file_path": pending[index],
                "target_file_content": "",
                "fix_instruction": "",
                "executor_generated_code": "",
                "completed_file_targets": completed,
            }
        return {
            "messages": [],
            "route_intent": "P",
            "mode": state.mode,
            "dual_role_phase": "",
            "target_file_path": "",
            "target_file_content": "",
            "fix_instruction": "",
            "executor_generated_code": "",
            "completed_file_targets": completed,
        }

    msgs_to_stream = _build_planner_prompt(state, plan, config)
    ai_msg = None
    tool_messages: List[ToolMessage] = []
    last_rag_results: List[str] = []
    for attempt in range(_MAX_FALLBACK_RETRIES + 1):
        ai_msg, tool_messages, last_rag_results = _stream_with_tool_loop(
            llm_planner, msgs_to_stream, config, token_queue
        )
        full_content = (
            ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        )
        kind, payload = parse_planner_output(full_content)
        if kind != "fallback" or attempt == _MAX_FALLBACK_RETRIES:
            break
        reminder = HumanMessage(content=(
            "Your previous response did not contain one of the required "
            "output markers. Output EXACTLY ONE of:\n"
            "  CREATE_PLAN: [\"...\", \"...\"]\n"
            "  DIRECT_ANSWER: <text>\n"
            "  EXECUTE_NEXT\n"
            "  FINAL_SUMMARY: <text>\n"
            "Do not add conversational filler."
        ))
        msgs_to_stream = list(msgs_to_stream) + [ai_msg] + tool_messages + [reminder]

    assert ai_msg is not None
    full_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    kind, payload = parse_planner_output(full_content)

    # Surface plan events to the UI.
    if token_queue:
        try:
            if kind == "create_plan":
                token_queue.put(("plan_update", {"event": "plan_created", "chunks": payload}))
            elif kind == "execute_next":
                token_queue.put(("plan_update", {"event": "execute_next"}))
            elif kind == "final_summary":
                token_queue.put(("plan_update", {"event": "finalized"}))
            elif kind == "direct_answer":
                token_queue.put(("plan_update", {"event": "direct_answer"}))
        except Exception:
            pass
    updates: Dict[str, Any] = {
        "messages": [],
        "route_intent": state.route_intent, "mode": state.mode,
    }
    if kind == "create_plan":
        user_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
        task = user_msgs[-1].content if user_msgs else ""
        if plan_approval_event is not None and plan_approval_state is not None:
            approved = _wait_for_plan_approval(
                token_queue, payload, plan_approval_event, plan_approval_state
            )
            if not approved:
                return {
                    "messages": [AIMessage(content=(
                        "Plan rejected. Please refine your request and I'll "
                        "draft a new plan that better fits what you want."
                    ))],
                    "route_intent": "A",
                    "mode": state.mode,
                    "active_plan_id": None,
                    "current_chunk_id": None,
                }

        new_plan = plan_store.make_blank_plan(task, payload)
        if last_rag_results:
            for chunk in new_plan["chunks"]:
                chunk["context_blocks"] = list(last_rag_results)
        first = new_plan["chunks"][0]
        plan_store.mark_executing(new_plan, first["id"])
        plan_store.save_plan(new_plan)
        updates["active_plan_id"] = new_plan["plan_id"]
        updates["current_chunk_id"] = first["id"]
        updates["current_step"] = first["id"]
        updates["last_step_status"] = "dispatched"
        updates["route_intent"] = "P"
    elif kind == "execute_next":
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        if plan.get("current_chunk") is not None:
            pass
        nxt = plan_store.next_pending(plan)
        if nxt is None:
            updates["route_intent"] = "P_FINALIZE"
            plan_store.save_plan(plan)
        else:
            plan_store.mark_executing(plan, nxt["id"])
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = nxt["id"]
            updates["current_step"] = nxt["id"]
            updates["last_step_status"] = "dispatched"
            updates["route_intent"] = "P"

    elif kind == "add_chunk_after":
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        anchor_id, instruction = payload
        try:
            new_chunk = plan_store.add_chunk_after(plan, anchor_id, instruction)
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = new_chunk["id"]
            updates["current_step"] = new_chunk["id"]
            updates["last_step_status"] = "dispatched"
            updates["route_intent"] = "P"  # executor will pick up the new chunk
            if token_queue:
                try:
                    token_queue.put(("plan_update", {
                        "event": "chunk_added",
                        "after": anchor_id,
                        "new_id": new_chunk["id"],
                    }))
                except Exception:
                    pass
        except (KeyError, ValueError):
            # Bad anchor id or empty instruction; loop back to planner.
            updates["route_intent"] = "P"

    elif kind == "replace_chunk":
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        chunk_id, instruction = payload
        try:
            replaced = plan_store.replace_chunk(plan, chunk_id, instruction)
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = replaced["id"]
            updates["current_step"] = replaced["id"]
            updates["last_step_status"] = "dispatched"
            updates["route_intent"] = "P"
            if token_queue:
                try:
                    token_queue.put(("plan_update", {
                        "event": "chunk_replaced",
                        "id": chunk_id,
                    }))
                except Exception:
                    pass
        except (KeyError, ValueError):
            updates["route_intent"] = "P"

    elif kind == "remove_chunk":
        # FIM-style: drop a chunk from the plan entirely.
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        chunk_id = payload
        try:
            plan_store.remove_chunk(plan, chunk_id)
            plan_store.save_plan(plan)
            if token_queue:
                try:
                    token_queue.put(("plan_update", {
                        "event": "chunk_removed",
                        "id": chunk_id,
                    }))
                except Exception:
                    pass
            nxt = plan_store.next_pending(plan)
            if nxt is None:
                updates["route_intent"] = "P_FINALIZE"
            else:
                plan_store.mark_executing(plan, nxt["id"])
                plan_store.save_plan(plan)
                updates["current_chunk_id"] = nxt["id"]
                updates["current_step"] = nxt["id"]
                updates["last_step_status"] = "dispatched"
                updates["route_intent"] = "P"
        except KeyError:
            updates["route_intent"] = "P"

    elif kind == "skip_chunk":
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        chunk_id, reason = payload
        try:
            plan_store.skip_chunk(plan, chunk_id, reason=reason)
            plan_store.save_plan(plan)
            if token_queue:
                try:
                    token_queue.put(("plan_update", {
                        "event": "chunk_skipped",
                        "id": chunk_id,
                        "reason": reason,
                    }))
                except Exception:
                    pass
            nxt = plan_store.next_pending(plan)
            if nxt is None:
                updates["route_intent"] = "P_FINALIZE"
            else:
                plan_store.mark_executing(plan, nxt["id"])
                plan_store.save_plan(plan)
                updates["current_chunk_id"] = nxt["id"]
                updates["current_step"] = nxt["id"]
                updates["last_step_status"] = "dispatched"
                updates["route_intent"] = "P"
        except KeyError:
            updates["route_intent"] = "P"

    elif kind == "final_summary":
        if plan is not None:
            plan_store.finalize(plan, payload)
            plan_store.save_plan(plan)
        # Store the user-facing final answer.
        updates["messages"] = [AIMessage(content=payload)]
        updates["final_summary"] = payload
        updates["summary_emitted"] = True
        updates["active_plan_id"] = None
        updates["current_chunk_id"] = None
        updates["current_step"] = None
        updates["last_step_status"] = ""
        updates["route_intent"] = "A"

    elif kind == "recommend_coder":
        updates["route_intent"] = "A"
        reason = payload or "This task needs file-editing tools."
        updates["messages"] = [AIMessage(content=(
            f"This request looks like a coding/file-editing task. "
            f"Please switch to `coder` mode to use the full tool set. "
            f"Reason: {reason}"
        ))]
        updates["active_plan_id"] = None
        updates["current_chunk_id"] = None
        updates["current_step"] = None
        updates["last_step_status"] = ""

    elif kind == "direct_answer":
        updates["route_intent"] = "A"
        updates["messages"] = [AIMessage(content=payload)]
        updates["active_plan_id"] = None
        updates["current_chunk_id"] = None
        updates["current_step"] = None
        updates["last_step_status"] = ""

    else:  # fallback
        updates["route_intent"] = "A"
        updates["messages"] = [AIMessage(content=full_content or payload or "")]
        updates["active_plan_id"] = None
        updates["current_chunk_id"] = None
        updates["current_step"] = None
        updates["last_step_status"] = ""

    return updates