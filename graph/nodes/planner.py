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
from graph.state import VedState
from graph.tools.rag_retrieve import retrieve_rag


# ---- Output marker parsing (unchanged) ----

_CREATE_PLAN_RE = re.compile(r"CREATE_PLAN:\s*(\[.*?\])", re.DOTALL)
_DIRECT_ANSWER_RE = re.compile(r"DIRECT_ANSWER:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)", re.DOTALL)
_EXECUTE_NEXT_RE = re.compile(r"\bEXECUTE_NEXT\b")
_FINAL_SUMMARY_RE = re.compile(r"FINAL_SUMMARY:\s*(.*?)(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK):|\Z)", re.DOTALL)
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
    r"RECOMMEND_CODER_MODE(?:\s+REASON\s*:\s*(.*?))?(?=\n(?:CREATE_PLAN|EXECUTE_NEXT|FINAL_SUMMARY|DIRECT_ANSWER|ADD_CHUNK_AFTER|REPLACE_CHUNK|REMOVE_CHUNK|SKIP_CHUNK|RECOMMEND_CODER):|\Z)",
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
    # Check RECOMMEND_CODER_MODE before DIRECT_ANSWER — if the planner
    # emits both, RECOMMEND_CODER wins because it's the more specific
    # 'ask the user to switch modes' response. The downstream DIRECT_ANSWER
    # would otherwise be picked up by its regex even though RECOMMEND_CODER
    # appears first in the text.
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

_PLANNER_SYSTEM = SystemMessage(content=(
    "You are the PLANNER role in a planner-executor pipeline.\n"
    "\n"
    "TOOLS AVAILABLE: you have ONE tool: `retrieve_rag(query, scope, k)`. "
    "It pulls relevant content from the thread's RAG store (past AI "
    "responses that got compressed, uploaded files, etc.). Use it when "
    "you need to look up what was discussed or decided in earlier turns. "
    "You do NOT have read_file, edit_file, execute_python, or any other "
    "destructive tools — those are the executor's job.\n"
    "\n"
    "OUTPUT FORMAT — output exactly ONE of these markers per turn:\n"
    "\n"
    "  CREATE_PLAN: [\"<chunk 1>\", \"<chunk 2>\", ...]\n"
    "    Use on the FIRST turn when the task needs multiple tool-using "
    "steps. Each chunk should be SELF-CONTAINED: include the action, the "
    "file/data to act on, and any context from prior chunks the executor "
    "needs. List 1-5 chunks.\n"
    "\n"
    "  DIRECT_ANSWER: <your answer>\n"
    "    Use when the task is SIMPLE and does not need any tools: factual "
    "questions, chitchat, explanations, opinions.\n"
    "\n"
    "  EXECUTE_NEXT\n"
    "    Use on subsequent turns when the plan is in progress and the "
    "executor just finished chunk N.\n"
    "\n"
    "  FINAL_SUMMARY: <one paragraph wrap-up>\n"
    "    Use when all chunks are complete.\n"
    "\n"
    "  ADD_CHUNK_AFTER <id>:\n"
    "    INSTRUCTION: <new chunk text>\n"
    "    Insert a new pending chunk right after chunk <id>. Use when the\n"
    "    executor failed and you want to insert a fix-up step BEFORE\n"
    "    retrying (e.g., 'first locate the file, then edit').\n"
    "\n"
    "  REPLACE_CHUNK <id>:\n"
    "    INSTRUCTION: <new instruction>\n"
    "    Overwrite an existing chunk's instruction. Resets it to pending.\n"
    "    Use when a chunk needs a different approach (e.g., wrong file path).\n"
    "\n"
    "  REMOVE_CHUNK <id>\n"
    "    Drop a chunk from the plan. Use when a chunk is no longer\n"
    "    relevant (e.g., task was already done by an earlier chunk).\n"
    "\n"
    "  SKIP_CHUNK <id> REASON: <text>\n"
    "    Mark a chunk as skipped (terminal). Use when the failure is not\n"
    "    critical — executor used the tool wrong, or the tool itself\n"
    "    is broken in a way downstream chunks don't depend on. The\n"
    "    executor moves on to the next pending chunk.\n"
    "\n"
    "  RECOMMEND_CODER_MODE REASON: <text>\n"
    "    Emit this when in standard mode and the task requires coding\n"
    "    (file edits, script execution, tool design). Standard mode's\n"
    "    executor only has read-only tools — emit this to ask the user\n"
    "    to switch to coder mode instead of attempting the task.\n"
    "\n"
    "WHEN TO EMIT RECOMMEND_CODER_MODE (important):\n"
    "  - The task asks to EDIT, MODIFY, UPDATE, FIX, REFACTOR, WRITE, or\n"
    "    CREATE a file.\n"
    "  - The task asks to RUN a script or execute code.\n"
    "  - The task asks to CREATE a new persistent tool.\n"
    "  In all these cases, standard mode's executor CANNOT do the work\n"
    "  (no edit_file, no execute_python, no propose_tool). Emit\n"
    "  RECOMMEND_CODER_MODE with a short reason. Do NOT emit\n"
    "  DIRECT_ANSWER with instructions for the user to follow manually —\n"
    "  that just tells the user to do the work themselves.\n"
    "\n"
    "FAILURE TRIAGE (important):\n"
    "  When a chunk fails, decide between:\n"
    "  - REPLACE_CHUNK: the executor made a mistake (wrong path, wrong args).\n"
    "    Rewrite the instruction to be clearer and retry.\n"
    "  - SKIP_CHUNK: the tool itself is broken / the failure is benign\n"
    "    AND downstream chunks don't need this chunk's output.\n"
    "  - SKIP_CHUNK + emit FINAL_SUMMARY next: the tool is broken AND the\n"
    "    failure blocks downstream chunks. Stop here with an honest error.\n"
    "  - ADD_CHUNK_AFTER: insert a prerequisite step first (e.g., 'first\n"
    "    locate the file'). Use sparingly — SKIP/REPLACE usually cover it.\n"
    "\n"
    "CHUNK INSTRUCTION QUALITY (important):\n"
    "  - Each chunk is executed IN ISOLATION by the executor. The executor "
    "    does NOT see the conversation history. So every chunk instruction "
    "    must be self-contained: state the action, the target, and any "
    "    relevant context the executor can't infer.\n"
    "  - BAD: 'Refactor that function.'\n"
    "  - GOOD: 'Open foo.py and refactor the `login(user, password)` "
    "    function on line 42 to use `auth_lib.login()` instead. The "
    "    function should return a Token object, not a dict.'\n"
    "\n"
    "GUIDELINES:\n"
       "  - DEFAULT to CREATE_PLAN when the task involves tools, files, code,\n"
       "search, or any concrete action. Most real tasks fall here.\n"
       "  - Use DIRECT_ANSWER ONLY for clearly non-tool queries: chitchat,\n"
       "greetings, factual lookups, opinions, explanations of general\n"
       "knowledge. If the user is asking 'how' or 'why' about a concept,\n"
       "that's DIRECT_ANSWER. If they're asking 'do X to Y', that's a plan.\n"
       "  - In standard mode, if the task needs file editing or code execution,\n"
       "emit RECOMMEND_CODER_MODE instead of CREATE_PLAN (the executor\n"
       "can't actually do the work in standard mode).\n"
       "  - When in doubt, prefer CREATE_PLAN over DIRECT_ANSWER. The plan\n"
       "can be one chunk if it's truly simple; DIRECT_ANSWER skips the\n"
       "executor entirely so the user gets no tool support at all.\n"
       "  - Before deciding, call retrieve_rag if the user's message\n"
       "references past context ('as we discussed', 'the bug from\n"
       "yesterday', etc.) so you don't miss it.\n"
))


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

    msgs = [_PLANNER_SYSTEM, _FRESH_QUESTION_HINT]

    # RAG auto-injection: same as chat_node does. Without this, a user
    # who uploads a file via /files/thread and then asks about it gets
    # a DIRECT_ANSWER with no awareness of the uploaded content. With
    # this, the planner sees top-k relevant chunks inline.
    # Source the thread id from config (chat_node does the same) — state
    # has no thread field, and reusing active_plan_id here meant uploads
    # were always queried in the wrong scope (None -> global fallback).
    active_thread_id = None
    if config and isinstance(config.get("configurable"), dict):
        active_thread_id = config["configurable"].get("active_thread_id")
    try:
        rag_block = _build_rag_block(last_user, active_thread_id)
        if rag_block:
            msgs.append(SystemMessage(content=rag_block))
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
        chunks_done = [c for c in plan["chunks"] if c["status"] in ("done", "failed")]
        pending = [c for c in plan["chunks"] if c["status"] == "pending"]
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
                f"Continue the plan. {len(pending)} chunk(s) remain.\n\n"
                f"PLAN PROGRESS:\n{progress}{last_output_excerpt}\n\n"
                f"You may call retrieve_rag to look up more context before "
                f"deciding the next chunk's instruction. When ready, output "
                f"EXECUTE_NEXT or FINAL_SUMMARY: <text>."
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
        args = tool_call.get("args", {}) or {}
        query = args.get("query", "")
        scope = args.get("scope")
        k = args.get("k", 5)
        cache_key = f"{query}|{scope}|{k}"
        if call_cache is not None and cache_key in call_cache:
            return call_cache[cache_key]
        # Config is injected by the framework; we pass it through so
        # retrieve_rag can use the active thread's scope.
        result = retrieve_rag.invoke(args, config=config)
        if call_cache is not None:
            call_cache[cache_key] = result
        return result
    return f"ERROR: unknown tool '{name}'"


def _stream_with_tool_loop(llm, msgs, config, token_queue, max_rounds=_MAX_PLANNER_TOOL_ROUNDS):
    """Run an LLM stream with a small tool-call loop.

    Yields text tokens to token_queue as they arrive. Returns
    (final_ai_msg, tool_messages) when done. Memoizes retrieve_rag results
    in `call_cache` so duplicate queries within one planner turn aren't
    re-embedded + re-searched.
    """
    import json as _json
    tool_messages: List[ToolMessage] = []
    call_cache: Dict[str, str] = {}
    for round_idx in range(max_rounds + 1):
        full_content = ""
        tool_calls_acc: Dict[int, Dict] = {}
        for chunk in llm.stream(msgs):
            if hasattr(chunk, "content") and chunk.content:
                c = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_content += c
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
            return ai_msg, tool_messages

        if round_idx >= max_rounds:
            # Hit tool-call budget. Return whatever the LLM said plus
            # any tool messages accumulated so far.
            return ai_msg, tool_messages

        # Execute each tool call, collect results, and re-invoke.
        # call_cache memoizes retrieve_rag results within this planner turn.
        for tc in tool_calls_list:
            result_content = _execute_planner_tool_call(tc, config, call_cache)
            tool_messages.append(ToolMessage(
                content=result_content,
                tool_call_id=tc["id"],
            ))
        # Loop again with the tool messages appended.
        msgs = list(msgs) + [ai_msg] + tool_messages

    return ai_msg, tool_messages  # unreachable but satisfies the type checker


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

    msgs_to_stream = _build_planner_prompt(state, plan, config)
    ai_msg, tool_messages = _stream_with_tool_loop(
        llm_planner, msgs_to_stream, config, token_queue
    )

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

    # New design: only user-facing responses (DIRECT_ANSWER and FINAL_SUMMARY)
    # get stored in state.messages. Intermediate markers (CREATE_PLAN,
    # EXECUTE_NEXT) return empty messages — the plan file holds the
    # context for those, and the planner reads it on the next turn.
    updates: Dict[str, Any] = {
        "messages": [],
        "route_intent": state.route_intent, "mode": state.mode,
    }

    if kind == "create_plan":
        user_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
        task = user_msgs[-1].content if user_msgs else ""
        new_plan = plan_store.make_blank_plan(task, payload)
        first = new_plan["chunks"][0]
        plan_store.mark_executing(new_plan, first["id"])
        plan_store.save_plan(new_plan)
        updates["active_plan_id"] = new_plan["plan_id"]
        updates["current_chunk_id"] = first["id"]
        updates["route_intent"] = "P"
        # messages already []

    elif kind == "execute_next":
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        # The executor_node already wrote its output to the plan file via
        # mark_done. We just need to confirm the plan state is consistent
        # and pick the next chunk.
        if plan.get("current_chunk") is not None:
            # If for some reason the executor didn't mark_done (e.g. it
            # returned early), the chunk is still 'executing'. Leave it
            # alone — the executor will re-run on the next iteration.
            pass
        nxt = plan_store.next_pending(plan)
        if nxt is None:
            # No more chunks — next planner call will emit FINAL_SUMMARY.
            updates["route_intent"] = "P_FINALIZE"
            plan_store.save_plan(plan)
        else:
            plan_store.mark_executing(plan, nxt["id"])
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = nxt["id"]
            updates["route_intent"] = "P"

    elif kind == "add_chunk_after":
        # FIM-style: insert a new pending chunk after `anchor_id`.
        # Used when the executor fails and the planner wants to add a
        # fix-up step before retrying the failed chunk.
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        anchor_id, instruction = payload
        try:
            new_chunk = plan_store.add_chunk_after(plan, anchor_id, instruction)
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = new_chunk["id"]
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
        # FIM-style: overwrite a chunk's instruction. Status resets to pending.
        if plan is None:
            updates["route_intent"] = state.route_intent
            return updates
        chunk_id, instruction = payload
        try:
            replaced = plan_store.replace_chunk(plan, chunk_id, instruction)
            plan_store.save_plan(plan)
            updates["current_chunk_id"] = replaced["id"]
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
            # If there are remaining pending chunks, route to executor;
            # otherwise the next planner turn will emit FINAL_SUMMARY.
            nxt = plan_store.next_pending(plan)
            if nxt is None:
                updates["route_intent"] = "P_FINALIZE"
            else:
                plan_store.mark_executing(plan, nxt["id"])
                plan_store.save_plan(plan)
                updates["current_chunk_id"] = nxt["id"]
                updates["route_intent"] = "P"
        except KeyError:
            updates["route_intent"] = "P"

    elif kind == "skip_chunk":
        # Mark the failed chunk as skipped. next_pending() automatically
        # skips status=skipped chunks, so the next executor turn picks up
        # the following pending chunk (or we route to FINALIZE if none).
        # Use when the failure is non-critical — executor error or a benign
        # tool bug that downstream chunks don't depend on.
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
                updates["route_intent"] = "P"
        except KeyError:
            updates["route_intent"] = "P"

    elif kind == "final_summary":
        if plan is not None:
            # The plan file already has all chunk outputs. Just record
            # the final summary.
            plan_store.finalize(plan, payload)
            plan_store.save_plan(plan)
        # Store the user-facing final answer.
        updates["messages"] = [AIMessage(content=payload)]
        updates["final_summary"] = payload
        updates["active_plan_id"] = None
        updates["current_chunk_id"] = None
        updates["route_intent"] = "A"

    elif kind == "direct_answer":
        updates["route_intent"] = "A"
        # Store the direct response. Tool messages from retrieve_rag are
        # not stored (planner can re-query on the next turn if needed).
        updates["messages"] = [AIMessage(content=payload)]
        updates["active_plan_id"] = None

    elif kind == "recommend_coder":
        # Planner detected a coding task in standard mode. Don't run the
        # executor — Path A's tools can't do real coding work. Emit a
        # user-facing message asking them to switch to coder mode.
        reason = payload or "this task needs code execution or file editing."
        msg_text = (
            f"This looks like a coding task ({reason}). "
            "Standard mode has read-only tools and can't run code or edit files. "
            "Switch to coder mode for full capabilities: `/mode coder`, "
            "then send your request again."
        )
        updates["messages"] = [AIMessage(content=msg_text)]
        updates["route_intent"] = "A"
        updates["active_plan_id"] = None
        if token_queue:
            try:
                token_queue.put(("plan_update", {
                    "event": "recommend_coder",
                    "reason": reason,
                }))
            except Exception:
                pass

    else:  # fallback
        updates["route_intent"] = "A"
        updates["messages"] = [AIMessage(content=full_content or payload or "")]
        updates["active_plan_id"] = None

    return updates

