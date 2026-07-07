"""File editing tools for Ved.

LangChain `@tool`-formatted. Two tools here:
  - `edit_file(path, old_text, new_text)` - in-place replace with backup + popup.
  - `overwrite_file(path, content)` - full-file overwrite with backup + popup.

Both follow the project's dual-mode safety policy and accept OPTIONAL
`path` arguments - if the LLM omits `path`, the tool infers it from the
last user message via `resolve_implicit_target`.

Self-healing mode (`state.self_healing=True`) restricts edits to the
project root; default mode allows any user-accessible file but blocks
system paths and other users' profiles.

The actual write is delegated to `edit_file_action` / `overwrite_file_action`
in graph/actions/.
"""
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.actions.filesystem import edit_file_action, overwrite_file_action
from graph.state import VedState
from graph.tools._common import (
    PROJECT_ROOT,
    is_safe_default,
    is_safe_self_healing,
    resolve_implicit_target,
)

def _request_approval(path: Path, old: str, new: str, self_healing: bool, overwrite: bool) -> bool:
    # tkinter is imported lazily so this module can be imported (e.g. during
    # `pytest --collect-only`) in environments where tkinter is unavailable
    # (headless Linux CI, slim venvs, etc.). The popup is the only consumer
    # of tkinter, so deferring the import keeps the import surface clean.
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return False  # no GUI available -> secure fallback: deny
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        intent_label = "FULL FILE OVERWRITE" if overwrite else "IN-PLACE EDIT"
        mode_banner = (
            "\n[ SELF-HEALING MODE: scope restricted to project root ]\n"
            if self_healing
            else ""
        )
        choice = messagebox.askyesno(
            title="Ved File Edit Approval",
            message=(
                f"Ved requests permission for this file change ({intent_label}):{mode_banner}\n\n"
                f"FILE: {path}\n"
                f"----------------------------------------\n"
                f"OLD:\n{(old or '(empty - full overwrite)')[:300]}"
                f"{'...[Truncated]' if len(old) > 300 else ''}\n"
                f"----------------------------------------\n"
                f"NEW:\n{new[:300]}{'...[Truncated]' if len(new) > 300 else ''}\n"
                f"----------------------------------------\n\n"
                f"Authorize this edit?"
            ),
            parent=root,
        )
        root.destroy()
        return choice
    except Exception:
        return False  # secure fallback: deny on UI failure


def _resolve_and_check(path_str: str, self_healing: bool) -> tuple[Path | None, str | None]:
    try:
        candidate = Path(path_str)
        if not candidate.is_absolute():
            anchor = PROJECT_ROOT if self_healing else Path.cwd()
            candidate = anchor / candidate
        resolved = candidate.resolve()
    except Exception as exc:
        return None, f"ERROR: Could not resolve path '{path_str}': {exc}"

    safety = is_safe_self_healing if self_healing else is_safe_default
    if not safety(resolved):
        if self_healing:
            return None, (
                f"ERROR: Refused to edit '{resolved}' - self-healing mode "
                f"restricts edits to the project root ({PROJECT_ROOT})."
            )
        return None, (
            f"ERROR: Refused to edit '{resolved}' - system path, other user's "
            f"profile, or insufficient permissions."
        )
    return resolved, None


@tool
def edit_file(
    path: str = "",
    old_text: str = "",
    new_text: str = "",
    state: Annotated[VedState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Replace `old_text` with `new_text` inside the file at `path`.

    Shows a tkinter approval popup before writing. The first occurrence of
    `old_text` is replaced. Both `path` and `old_text` are optional - if
    omitted, the tool infers the file from the last user message.

    Args:
        path: Absolute path, or path relative to project root (self-healing)
              or cwd (default mode). Empty triggers implicit discovery.
        old_text: Exact text to find. Must match the file byte-for-byte.
                  Empty string returns an error (use `overwrite_file` instead).
        new_text: Replacement text.

    Returns:
        Status string describing the result, or `ERROR: ...` on failure.
    """
    self_healing = bool(getattr(state, "self_healing", False))

    if not path:
        inferred = resolve_implicit_target(state)
        if not inferred:
            return "ERROR: No file path provided and could not infer one from the conversation."
        path = inferred

    if not old_text:
        return (
            "ERROR: `old_text` is empty. For a full file replacement, call "
            "`overwrite_file(path=..., content=...)` instead."
        )

    resolved, err = _resolve_and_check(path, self_healing)
    if err:
        return err

    if not _request_approval(resolved, old_text, new_text, self_healing, overwrite=False):
        return "ERROR: User denied edit authorization."

    return edit_file_action(
        str(resolved),
        old_text,
        new_text,
        allowed_roots=(str(PROJECT_ROOT),),
        backup_dir=None,
    )


@tool
def overwrite_file(
    path: str = "",
    content: str = "",
    state: Annotated[VedState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Replace the entire contents of the file at `path` with `content`.

    Shows a tkinter approval popup before writing. `path` is optional and
    is inferred from the last user message when omitted.

    Args:
        path: Absolute path, or path relative to project root (self-healing)
              or cwd (default mode). Empty triggers implicit discovery.
        content: The full new file contents.

    Returns:
        Status string describing the result, or `ERROR: ...` on failure.
    """
    self_healing = bool(getattr(state, "self_healing", False))

    if not path:
        inferred = resolve_implicit_target(state)
        if not inferred:
            return "ERROR: No file path provided and could not infer one from the conversation."
        path = inferred

    resolved, err = _resolve_and_check(path, self_healing)
    if err:
        return err

    if not _request_approval(resolved, "", content, self_healing, overwrite=True):
        return "ERROR: User denied overwrite authorization."

    return overwrite_file_action(
        str(resolved),
        content,
        allowed_roots=(str(PROJECT_ROOT),),
        backup_dir=None,
    )
