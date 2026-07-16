"""Read-only file access tool for Ved.

LangChain `@tool`-formatted. The agent binds this via `llm.bind_tools([...])`
and emits a structured `read_file(path=...)` call when it needs to inspect
a file. State (for the self_healing flag) is injected via `InjectedState`.

Two modes, toggled by `state.self_healing`:
  - DEFAULT (self_healing=False): user-driven reads. System paths and other
    users' profiles are blocked, otherwise the agent can read anywhere.
  - SELF-HEALING (self_healing=True): restricted to the project root.

The `path` argument is OPTIONAL. If omitted or empty, the tool scans the
last user message for a filename hint and globs the project for it. If
nothing is found, it returns a clear "couldn't find file" error so the LLM
can re-plan.

The actual read is delegated to `read_file_action` in graph/actions/.
"""
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.actions.filesystem import read_file_action
from graph.state import VedState
from graph.tools._common import (
    PROJECT_ROOT,
    is_safe_default,
    is_safe_self_healing,
    resolve_implicit_target,
    _resolve_fuzzy_path,
)
from graph.tools.staging_registry import STAGING_REGISTRY


@tool
def read_file(
    path: str = "",
    state: Annotated[VedState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Read the contents of a file and return them as a string.

    Args:
        path: Absolute path, or a path relative to the project root (self-healing
              mode) or the current working directory (default mode). If empty,
              the tool infers the file from the most recent user message.

    Returns:
        The file contents (truncated to 8000 chars), or `ERROR: ...` if
        the file cannot be located, read, or the path is blocked by safety
        policy.
    """
    self_healing = bool(getattr(state, "self_healing", False))

    # Fallback: if the LLM omitted `path`, try to infer it from the conversation.
    if not path:
        inferred = resolve_implicit_target(state)
        if not inferred:
            return (
                "ERROR: No file path provided and could not infer one from the "
                "conversation. Ask the user for the file name or pass an explicit path."
            )
        path = inferred

    candidate = Path(path)
    if not candidate.is_absolute():
        anchor = PROJECT_ROOT if self_healing else Path.cwd()
        candidate = anchor / candidate

    # Fuzzy path resolution: if the exact path doesn't exist, try to find
    # a close match by searching from the anchor directory.
    if not candidate.exists():
        anchor = PROJECT_ROOT if self_healing else Path.cwd()
        fuzzy = _resolve_fuzzy_path(str(candidate), base=anchor)
        if fuzzy:
            candidate = Path(fuzzy)

    safety = is_safe_self_healing if self_healing else is_safe_default
    if not safety(candidate):
        if self_healing:
            return (
                f"ERROR: Refused to read '{path}' - self-healing mode restricts "
                f"reads to the project root ({PROJECT_ROOT})."
            )
        return (
            f"ERROR: Refused to read '{path}' - system path, other user's "
            f"profile, or insufficient permissions."
        )

    raw = read_file_action(str(candidate), allowed_roots=(str(PROJECT_ROOT),))

    # Virtual read overlay: if this file has a pending staged edit in the
    # active response, present the post-edit content to the model so it
    # never operates against stale code.
    thread_id = getattr(state, "active_thread_id", "")
    if thread_id and STAGING_REGISTRY.has_session(thread_id):
        raw = STAGING_REGISTRY.get_overlay(thread_id, str(candidate.resolve()), raw)

    from graph.tools._common import ingest_path_to_thread_rag_sync
    ingest_path_to_thread_rag_sync(str(candidate.resolve()), thread_id, chunker="ast")
    return raw
