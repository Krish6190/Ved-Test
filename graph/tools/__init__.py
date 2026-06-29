"""Ved's tool registry.

The agent binds these to its LLM via `llm.bind_tools(VED_TOOLS)` so the model
can emit structured tool calls during conversation. ToolNode in
`graph/__init__.py` executes them.

Tool list, with one-line summaries:
  - read_file      — read any text file (system paths blocked; project-only in self-healing mode)
  - edit_file      — replace old_text with new_text in a file (approval popup)
  - overwrite_file — replace a file's full contents (approval popup)
  - search_files   — recursive glob search for files
  - execute_python — run a Python code block in a subprocess (approval popup, 10s timeout)
  - propose_tool   — design a new tool at runtime, ask human, save + register (approval modal)
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

# Tools available to Path A's executor (standard/turbo mode). Excludes the
# coding-only tools (edit_file, overwrite_file, execute_python,
# propose_tool) so llama can't accidentally write code or run scripts.
# In coder mode the executor gets full VED_TOOLS instead.
PATH_A_EXECUTOR_TOOLS = [
    read_file,
    search_files,
    retrieve_rag,
    open_app,
]
from graph.tools import user_tools 

__all__ = [
    "VED_TOOLS", "PATH_A_EXECUTOR_TOOLS",
    "read_file", "edit_file", "overwrite_file",
    "search_files", "execute_python", "propose_tool", "open_app",
    "retrieve_rag",
]
