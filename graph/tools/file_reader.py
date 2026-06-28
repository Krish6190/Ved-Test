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

Output is truncated at 8000 chars to keep prompts bounded.
"""
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.state import VedState
from graph.tools._common import (
    ALWAYS_SKIP_DIRS,
    PROJECT_ROOT,
    is_safe_default,
    is_safe_self_healing,
    resolve_implicit_target,
)

_MAX_CHARS = 8000


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

    if not candidate.exists():
        return f"ERROR: File not found: `{candidate}`"

    try:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: Failed to read `{candidate}`: {exc}"

    total = len(content)
    truncated = total > _MAX_CHARS
    body = content[:_MAX_CHARS] if truncated else content
    header = (
        f"FILE: {candidate}\n"
        f"SIZE: {total} chars\n"
        + (f"NOTE: Truncated at {_MAX_CHARS} chars.\n" if truncated else "")
        + (f"MODE: SELF-HEALING (project root only)\n" if self_healing else "")
    )
    return f"{header}\n```\n{body}\n```"
