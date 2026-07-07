"""Action layer - stateless primitives for every local side effect.

These functions are the swap surface for the future client-server
migration. Today they are in-process Python calls; tomorrow they can be
replaced with WebSocket / HTTP round-trips without touching the agent
loop or the LangChain @tool layer above them.

Rules (enforced by tests + grep):
  - Primitive argument types only (str, int, dict, tuple).
  - No upward imports into the tool or state layers, or into the langchain or data modules.
  - No upward dependency on the tool layer.
  - Allowed roots / skip dirs are passed by the caller.
"""
from graph.actions.filesystem import (
    edit_file_action,
    overwrite_file_action,
    read_file_action,
    search_files_action,
)
from graph.actions.process import execute_python_action
from graph.actions.apps import open_app_action
from graph.actions.tool_creator_actions import propose_tool_action

__all__ = [
    "read_file_action",
    "edit_file_action",
    "overwrite_file_action",
    "search_files_action",
    "execute_python_action",
    "open_app_action",
    "propose_tool_action",
]
