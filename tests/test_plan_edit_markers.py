"""Tests for FIM-style plan-edit markers (ADD_CHUNK_AFTER, REPLACE_CHUNK, REMOVE_CHUNK).

Covers:
  - Parser recognizes the new markers and extracts anchor_id / instruction
  - data.plans mutators add/replace/remove chunks correctly
  - Planner_node routes the new kinds correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.plans as plan_store
from graph.nodes.planner_markers import parse_planner_output


# ---- Parser tests ----

def test_parse_add_chunk_after():
    text = (
        "Chunk 2 needs a prerequisite step.\n"
        "ADD_CHUNK_AFTER 2:\n"
        "  INSTRUCTION: Run `find . -name 'foo.py'` to locate the file first."
    )
    kind, payload = parse_planner_output(text)
    assert kind == "add_chunk_after"
    assert payload == (2, "Run `find . -name 'foo.py'` to locate the file first.")


def test_parse_replace_chunk():
    text = (
        "Wrong approach. Use a different file path.\n"
        "REPLACE_CHUNK 3:\n"
        "  INSTRUCTION: Read src/auth.py (NOT test_auth.py) and find the login function."
    )
    kind, payload = parse_planner_output(text)
    assert kind == "replace_chunk"
    assert payload == (3, "Read src/auth.py (NOT test_auth.py) and find the login function.")


def test_parse_remove_chunk():
    text = "Chunk 5 is redundant.\nREMOVE_CHUNK 5"
    kind, payload = parse_planner_output(text)
    assert kind == "remove_chunk"
    assert payload == 5


def test_parse_add_chunk_with_other_markers():
    """Verify the regex doesn't get confused by multiple markers in one output."""
    text = (
        "Add a fix-up, then proceed.\n"
        "ADD_CHUNK_AFTER 1:\n"
        "  INSTRUCTION: Locate foo.py first.\n"
        "Then run the new chunk.\n"
        "EXECUTE_NEXT"
    )
    # ADD_CHUNK_AFTER takes priority over EXECUTE_NEXT in the parser.
    kind, payload = parse_planner_output(text)
    assert kind == "add_chunk_after"
    assert payload[0] == 1


# ---- plan_store mutator tests ----

def test_add_chunk_after_inserts_at_correct_position(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    new_chunk = plan_store.add_chunk_after(plan, 2, "fix-up step")
    # New chunk inserted between 2 and 3.
    assert [c["id"] for c in plan["chunks"]] == [1, 2, 4, 3]
    assert new_chunk["id"] == 4
    assert new_chunk["instruction"] == "fix-up step"
    assert new_chunk["status"] == "pending"


def test_add_chunk_after_appends_when_anchor_is_last(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b"])
    new_chunk = plan_store.add_chunk_after(plan, 2, "after last")
    assert [c["id"] for c in plan["chunks"]] == [1, 2, 3]


def test_replace_chunk_resets_status_and_clears_output(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b"])
    plan_store.mark_done(plan, 2, "previous result")
    plan_store.replace_chunk(plan, 2, "new approach")
    chunk2 = next(c for c in plan["chunks"] if c["id"] == 2)
    assert chunk2["instruction"] == "new approach"
    assert chunk2["status"] == "pending"
    assert chunk2["output"] is None


def test_remove_chunk_drops_from_list(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    plan_store.remove_chunk(plan, 2)
    assert [c["id"] for c in plan["chunks"]] == [1, 3]


def test_remove_chunk_clears_current_chunk_if_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b"])
    plan["current_chunk"] = 2
    plan_store.remove_chunk(plan, 2)
    assert plan["current_chunk"] is None


def test_add_chunk_after_raises_for_unknown_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a"])
    try:
        plan_store.add_chunk_after(plan, 99, "x")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_replace_chunk_raises_for_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a"])
    try:
        plan_store.replace_chunk(plan, 99, "x")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_remove_chunk_raises_for_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a"])
    try:
        plan_store.remove_chunk(plan, 99)
        assert False, "expected KeyError"
    except KeyError:
        pass


# ---- SKIP_CHUNK marker + planner triage ----

def test_parse_skip_chunk_with_reason():
    text = (
        "Chunk 2 hit a transient error but downstream doesn't need it.\n"
        "SKIP_CHUNK 2 REASON: tool was misbehaving, output already includes what we need"
    )
    kind, payload = parse_planner_output(text)
    assert kind == "skip_chunk"
    assert payload == (2, "tool was misbehaving, output already includes what we need")


def test_parse_skip_chunk_without_reason():
    text = "Just skip it.\nSKIP_CHUNK 3"
    kind, payload = parse_planner_output(text)
    assert kind == "skip_chunk"
    assert payload == (3, "")


def test_skip_chunk_marks_terminal_and_next_pending_advances(tmp_path, monkeypatch):
    """Skipped chunks are terminal — next_pending skips them."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    plan_store.skip_chunk(plan, 2, reason="benign tool error")
    # Chunk 2 is skipped, chunks 1 and 3 untouched.
    chunk2 = next(c for c in plan["chunks"] if c["id"] == 2)
    assert chunk2["status"] == "skipped"
    assert "benign tool error" in chunk2["output"]
    # next_pending skips skipped chunks.
    nxt = plan_store.next_pending(plan)
    assert nxt["id"] == 1


def test_skip_chunk_in_middle_does_not_block_later_chunks(tmp_path, monkeypatch):
    """Skipping chunk 2 should let chunk 3 (later) still be picked up."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    plan_store.mark_done(plan, 1, "ok")
    plan_store.skip_chunk(plan, 2, reason="irrelevant")
    nxt = plan_store.next_pending(plan)
    assert nxt["id"] == 3


def test_is_chunk_terminal():
    assert plan_store.is_chunk_terminal("done")
    assert plan_store.is_chunk_terminal("failed")
    assert plan_store.is_chunk_terminal("skipped")
    assert not plan_store.is_chunk_terminal("pending")
    assert not plan_store.is_chunk_terminal("executing")
