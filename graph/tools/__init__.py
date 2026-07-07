"""Ved's tool registry.

The agent binds these to its LLM via `llm.bind_tools(VED_TOOLS)` so the model
can emit structured tool calls during conversation. ToolNode in
`graph/__init__.py` executes them.

Tool list, with one-line summaries:
  - read_file      — read any text file (system paths blocked; project-only in self-healing mode)
  - edit_file      — replace old_text with new_text in a file (approval popup) [CODER ONLY]
  - overwrite_file — replace a file's full contents (approval popup) [CODER ONLY]
  - search_files   — recursive glob search for files
  - execute_python — run a Python code block in a subprocess (approval popup, 10s timeout)
  - propose_tool   — design a new tool at runtime, ask human, save + register (approval modal) [CODER ONLY]
  - open_app       — launch any application by name (always requires human approval)
  - retrieve_rag   — query thread/global RAG (Path A); in coder mode also falls back to project indexer scope
"""
from graph.tools.file_reader import read_file
from graph.tools.file_editor import edit_file, overwrite_file
from graph.tools.file_search import search_files
from graph.tools.python_runner import execute_python
from graph.tools.tool_creator import propose_tool
from graph.tools.app_launcher import open_app
from graph.tools.rag_retrieve import retrieve_rag

VED_TOOLS = [
    read_file,
    edit_file,
    overwrite_file,
    search_files,
    execute_python,
    propose_tool,
    open_app,
    retrieve_rag,
]

# Tools available to Path A's executor (standard/turbo mode). Path A is a
# smart chatbot that can read files, search the project, run standalone
# Python scripts (math, scripts, calculations), launch apps, and query
# thread-scoped past-chat memory — but it CANNOT edit/overwrite workspace
# files or create new persistent tools. Those are coder-only.
# In coder mode the executor gets the full VED_TOOLS set instead.
PATH_A_EXECUTOR_TOOLS = [
    read_file,
    search_files,
    execute_python,
    open_app,
    retrieve_rag,
]

# Tools that are coder-only — Path A's executor must never see them.
# If a future refactor accidentally binds VED_TOOLS in standard/turbo
# mode, this assertion catches it at executor entry instead of letting
# the model silently edit files or create persistent tools.
_CODER_ONLY_TOOLS = frozenset({"edit_file", "overwrite_file", "propose_tool"})


def _assert_tool_isolation(mode, tools):
    """Raise AssertionError if a restricted tool leaks into a non-coder mode.

    Called by executor_node before binding tools to the LLM. In production
    the check is cheap (one set intersection); in tests / refactors it
    provides a loud, fast failure if PATH_A_EXECUTOR_TOOLS gets out of sync
    with VED_TOOLS. Note: execute_python is intentionally NOT in the
    coder-only set because Path A's chatbot can run standalone scripts.
    """
    if mode == "coder":
        return
    tool_names = {t.name for t in tools}
    leaked = tool_names & _CODER_ONLY_TOOLS
    if leaked:
        raise AssertionError(
            f"Tool isolation violation: mode={mode!r} must not bind "
            f"coder-only tools {sorted(leaked)}. Use PATH_A_EXECUTOR_TOOLS "
            f"for standard/turbo mode or switch to coder mode."
        )


from graph.tools import user_tools 

__all__ = [
    "VED_TOOLS", "PATH_A_EXECUTOR_TOOLS", "_assert_tool_isolation",
    "read_file", "edit_file", "overwrite_file",
    "search_files", "execute_python", "propose_tool", "open_app",
    "retrieve_rag",
]
