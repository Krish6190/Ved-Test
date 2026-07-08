"""Tests for the planner's context-window helpers and executor's lean prompt.

Pure-Python tests; no LLM required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph.nodes.planner_prompt import (
    _MSG_HISTORY_CAP,
    _recent_human_ai,
)


# ---- Model-aware msg cap ----

def test_msg_history_cap_per_mode():
    assert _MSG_HISTORY_CAP["standard"] == 10
    assert _MSG_HISTORY_CAP["turbo"] == 10
    assert _MSG_HISTORY_CAP["coder"] == 40
    assert _MSG_HISTORY_CAP["hibernate"] == 0


# ---- _recent_human_ai ----

def test_recent_human_ai_returns_last_n_in_order():
    msgs = [
        HumanMessage(content="h1"),
        AIMessage(content="a1"),
        HumanMessage(content="h2"),
        AIMessage(content="a2"),
        HumanMessage(content="h3"),
    ]
    out = _recent_human_ai(msgs, cap=3)
    assert [m.content for m in out] == ["h2", "a2", "h3"]  # last 3 human+ai, in order


def test_recent_human_ai_skips_system_and_tool():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="h1"),
        ToolMessage(content="tool out", tool_call_id="c1"),
        AIMessage(content="a1"),
        HumanMessage(content="h2"),
    ]
    out = _recent_human_ai(msgs, cap=10)
    # Should only include the 3 HumanMessage + AIMessage (h1, a1, h2).
    assert [m.content for m in out] == ["h1", "a1", "h2"]


def test_recent_human_ai_zero_cap_returns_empty():
    msgs = [HumanMessage(content="h1"), AIMessage(content="a1")]
    assert _recent_human_ai(msgs, cap=0) == []


def test_recent_human_ai_cap_larger_than_history():
    msgs = [HumanMessage(content="h1")]
    out = _recent_human_ai(msgs, cap=100)
    assert len(out) == 1


