"""Tests for the RECOMMEND_CODER_MODE marker (planner asks user to switch modes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.nodes.planner_markers import parse_planner_output


def test_parse_recommend_coder_with_reason():
    text = (
        "This task requires editing files and running tests.\n"
        "RECOMMEND_CODER_MODE REASON: needs edit_file + execute_python"
    )
    kind, payload = parse_planner_output(text)
    assert kind == "recommend_coder"
    assert payload == "needs edit_file + execute_python"


def test_parse_recommend_coder_without_reason():
    text = "RECOMMEND_CODER_MODE"
    kind, payload = parse_planner_output(text)
    assert kind == "recommend_coder"
    assert payload == ""


def test_recommend_coder_takes_priority_over_execute_next():
    """If both markers appear, RECOMMEND_CODER_MODE wins (parse order is
    RECOMMEND_CODER before EXECUTE_NEXT in the parser chain)."""
    text = (
        "This needs code work.\n"
        "RECOMMEND_CODER_MODE REASON: edit_file needed\n"
        "EXECUTE_NEXT"
    )
    kind, _ = parse_planner_output(text)
    assert kind == "recommend_coder"


def test_other_markers_still_parse():
    """Sanity: existing markers aren't broken by the new pattern."""
    assert parse_planner_output('CREATE_PLAN: ["a"]')[0] == "create_plan"
    assert parse_planner_output("EXECUTE_NEXT")[0] == "execute_next"
    assert parse_planner_output("SKIP_CHUNK 2 REASON: x")[0] == "skip_chunk"
    assert parse_planner_output("REPLACE_CHUNK 3:\n  INSTRUCTION: x")[0] == "replace_chunk"
