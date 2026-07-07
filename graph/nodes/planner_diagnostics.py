"""Planner diagnostic escalation ladder.

When a chunk fails, the planner needs to decide how to react. This module
implements a 4-phase escalation ladder keyed off chunk_retry_count:

  Phase 1 (retry 1): REPLACE_CHUNK
      The executor likely made a mistake (wrong path, wrong args). Rewrite
      the chunk instruction with a clearer version and retry.

  Phase 2 (retry 2): TRIGGER_WORKSPACE_GREP
      The executor can't find what it needs. Inject a search_files step
      before retrying so it can locate the symbol/path it's missing.

  Phase 3 (retry 3): TRIGGER_RAG_REBUILD
      Stale chunks in the RAG store are likely poisoning retrieval (e.g.
      a file was edited and the old signature is still indexed). Drop the
      affected chunks from LocalVectorDB and force a fresh retrieve.

  Phase 4 (retry >= 4): HARD_HALT_USER_INTERVENTION
      Stop the plan. Emit a user-facing error with the full failure log
      and request human help. No automatic retry past this point.

The ladder is mode-agnostic — it reacts to errors, not to which model is
calling or which tools are bound. Path A's read-only executor and coder's
full-execute executor hit the same ladder; only the error patterns differ.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional


class EscalationAction(str, Enum):
    REPLACE_CHUNK = "replace_chunk"
    TRIGGER_WORKSPACE_GREP = "trigger_workspace_grep"
    TRIGGER_RAG_REBUILD = "trigger_rag_rebuild"
    HARD_HALT_USER_INTERVENTION = "hard_halt_user_intervention"


class EscalationDecision(NamedTuple):
    action: EscalationAction
    reason: str
    # Suggested grep target for Phase 2 (best-effort extraction from the error).
    # Empty string if nothing usable was found; planner can fall back to a
    # generic "search for the symbol mentioned in the task" instruction.
    grep_target: str = ""
    # Suggested halt message for Phase 4. The planner may override with a
    # richer message that includes plan/chunk context.
    halt_message: str = ""


# Match file-like paths: foo.py, bar/baz.js, src/config.json
_FILE_PATH_RE = re.compile(r'["\']?([\w./-]+\.[a-zA-Z]{1,5})["\']?')
# Match quoted identifiers (Python/JS symbols): `my_func`, 'MyClass'
_QUOTED_IDENT_RE = re.compile(r'[`\'"]([A-Za-z_][\w]*)[`\'"]')
# Match bare identifiers after common error keywords like "name 'X' is not defined"
_NAME_NOT_DEFINED_RE = re.compile(r"name ['\"]([A-Za-z_][\w]*) ['\"]")


def _extract_grep_target(error: str) -> str:
    """Best-effort extraction of a file path or symbol from an error message.

    Tries file paths first (more actionable), then quoted identifiers, then
    bare names from "name X is not defined" errors. Returns "" if nothing
    usable is found.
    """
    if not error:
        return ""
    m = _FILE_PATH_RE.search(error)
    if m:
        return m.group(1)
    m = _QUOTED_IDENT_RE.search(error)
    if m:
        return m.group(1)
    m = _NAME_NOT_DEFINED_RE.search(error)
    if m:
        return m.group(1)
    return ""


def escalate(
    retry_count: int,
    last_error: str,
    plan: Optional[Dict[str, Any]] = None,
) -> EscalationDecision:
    """Decide the next escalation action based on retry count and error.

    retry_count semantics (set by executor.py on failure):
      retry_count == 1: first failure → Phase 1
      retry_count == 2: second failure → Phase 2
      retry_count == 3: third failure → Phase 3
      retry_count >= 4: fourth+ failure → Phase 4 (halt)

    `plan` is passed for future use (e.g. to count total failures across
    the plan, or to look up chunk context). Currently unused.
    """
    error_excerpt = (last_error or "")[:200]

    if retry_count <= 0:
        # No failure yet — caller error. Return a no-op-safe default.
        return EscalationDecision(
            action=EscalationAction.REPLACE_CHUNK,
            reason="escalate() called with retry_count <= 0; defaulting to REPLACE_CHUNK",
        )

    if retry_count == 1:
        return EscalationDecision(
            action=EscalationAction.REPLACE_CHUNK,
            reason=f"Chunk failed once. Error: {error_excerpt}. Retry with a refined instruction.",
        )

    if retry_count == 2:
        grep_target = _extract_grep_target(last_error)
        if grep_target:
            reason = (
                f"Chunk failed twice. Injecting workspace grep for "
                f"{grep_target!r} before retrying."
            )
        else:
            reason = (
                "Chunk failed twice. Injecting a workspace grep step before "
                "retrying — the executor likely can't locate what it needs."
            )
        return EscalationDecision(
            action=EscalationAction.TRIGGER_WORKSPACE_GREP,
            reason=reason,
            grep_target=grep_target,
        )

    if retry_count == 3:
        return EscalationDecision(
            action=EscalationAction.TRIGGER_RAG_REBUILD,
            reason=(
                "Chunk failed 3x. Stale RAG chunks may be poisoning retrieval. "
                "Rebuilding the affected scope before the next attempt."
            ),
        )

    # retry_count >= 4: halt
    halt_msg = (
        f"Chunk has failed {retry_count} times in a row. Stopping the plan.\n\n"
        f"Last error: {(last_error or '')[:400]}\n\n"
        f"Please review the failure, fix the underlying issue, and try again. "
        f"You may also switch to /mode coder for full tool access if the "
        f"task requires code execution that Path A can't perform."
    )
    return EscalationDecision(
        action=EscalationAction.HARD_HALT_USER_INTERVENTION,
        reason=f"Chunk failed {retry_count} times — halting plan for human review.",
        halt_message=halt_msg,
    )


__all__ = [
    "EscalationAction",
    "EscalationDecision",
    "escalate",
]
