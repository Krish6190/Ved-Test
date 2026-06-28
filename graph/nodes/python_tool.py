"""Legacy `/run ...` slash-command entrypoint.

With LangChain tool calling wired into `chat_node`/`coder_chat_node`, the
LLM normally invokes `execute_python` as a LangChain tool. This node
remains for the explicit `/run ...` slash command: it extracts the Python
code from the last message and runs it via the `execute_python` @tool,
which still enforces the approval gate + 10s timeout + cleanup.
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from graph.state import VedState

def python_tool_node(state: VedState, config: RunnableConfig) -> dict:
    """Extract Python code from the last message and run it via execute_python."""
    last_text = ""
    for msg in reversed(state.messages):
        if hasattr(msg, "content") and msg.content:
            last_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    fence = re.search(r"```python\s*([\s\S]*?)```", last_text)
    raw_code = fence.group(1).strip() if fence else last_text.strip()
    from graph.tools.python_runner import execute_python
    result = execute_python.invoke({"code": raw_code, "state": state})
    return {"messages": [HumanMessage(content=result)], "route_intent": "", "mode": state.mode}
