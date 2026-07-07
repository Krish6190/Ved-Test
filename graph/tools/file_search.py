"""Filesystem search tool for Ved.

LangChain `@tool`-formatted. The agent binds this via `llm.bind_tools([...])`
and emits a structured `search_files(pattern=..., directory=...)` call when
it needs to find files. State (for the self_healing flag) is injected via
`InjectedState`.

Two modes, toggled by `state.self_healing`:
  - DEFAULT (self_healing=False): search the entire user-accessible filesystem.
    System paths and other users' profiles are blocked; cache dirs are skipped.
  - SELF-HEALING (self_healing=True): restricted to the project root.

The `pattern` argument is OPTIONAL. If omitted, the tool extracts a likely
filename pattern from the last user message (e.g. "config.py" or the last
meaningful word) and uses that as the glob. The actual search delegates to
`search_files_action` in graph/actions/, which uses the same 3-strategy
matcher (exact glob, case-insensitive name match, full-path substring).
"""
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.actions.filesystem import search_files_action
from graph.state import VedState
from graph.tools._common import (
    ALWAYS_SKIP_DIRS,
    MAX_RESULTS,
    PROJECT_ROOT,
    extract_search_pattern,
    is_safe_default,
    is_safe_self_healing,
    last_user_message_text,
)


@tool
def search_files(
    pattern: str = "",
    directory: str = ".",
    state: Annotated[VedState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Search for files matching a glob pattern under a directory.

    Uses a 3-strategy matcher (exact glob, case-insensitive name match,
    full-path substring) so users can say "find the readme" without
    knowing the exact filename. Returns up to 100 paths. Cache dirs
    (`.venv`, `.git`, `__pycache__`, etc.) are always skipped.

    Args:
        pattern: Glob pattern, e.g. `*.py`, `requirements*.txt`, `voice*`. If
                 empty, the tool extracts a likely pattern from the last user
                 message.
        directory: Where to search. Relative paths anchor at project root
                   (self-healing mode) or cwd (default mode). Defaults to `.`.

    Returns:
        A newline-separated list of matching paths, or `ERROR: ...` if the
        search was blocked or produced zero matches.
    """
    self_healing = bool(getattr(state, "self_healing", False))

    # Fallback: if the LLM omitted `pattern`, infer from the conversation.
    if not pattern:
        text = last_user_message_text(state)
        inferred = extract_search_pattern(text) if text else None
        if not inferred:
            return (
                "ERROR: No search pattern provided and could not infer one "
                "from the conversation. Pass an explicit `pattern` argument."
            )
        pattern = inferred

    base = Path(directory)
    if not base.is_absolute():
        anchor = PROJECT_ROOT if self_healing else Path.cwd()
        base = anchor / base

    safety = is_safe_self_healing if self_healing else is_safe_default
    if not safety(base):
        if self_healing:
            return (
                f"ERROR: Refused to search in '{directory}' - self-healing "
                f"mode restricts searches to the project root ({PROJECT_ROOT})."
            )
        return (
            f"ERROR: Refused to search in '{directory}' - system path or "
            f"insufficient permissions."
        )

    skip_dirs = tuple(sorted(ALWAYS_SKIP_DIRS))
    raw = search_files_action(
        pattern,
        directory=str(base),
        skip_dirs=skip_dirs,
        max_results=MAX_RESULTS,
    )
    if raw.startswith("ERROR:"):
        return raw
    matches = [m for m in raw.split("\n") if m]

    if not matches:
        return f"ERROR: No files matched '{pattern}' in '{base}'. Try a broader pattern."

    mode_tag = " [SELF-HEALING MODE]" if self_healing else ""
    listing = "\n".join(matches)
    return (
        f"Found {len(matches)} match(es) for '{pattern}' in {base}{mode_tag}:\n"
        f"```\n{listing}\n```"
    )
