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
"""
from graph.tools.file_reader import read_file
from graph.tools.file_editor import edit_file, overwrite_file
from graph.tools.file_search import search_files
from graph.tools.python_runner import execute_python

VED_TOOLS = [
    read_file,
    edit_file,
    overwrite_file,
    search_files,
    execute_python,
]

__all__ = ["VED_TOOLS", "read_file", "edit_file", "overwrite_file", "search_files", "execute_python"]
