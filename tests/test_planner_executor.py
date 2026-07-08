"""Tests for the planner-executor pipeline.

Pure-Python tests; no LLM required.
  - data.plans: file I/O + mutators
  - graph.nodes.planner.parse_planner_output: marker parsing
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.plans as plan_store
import graph.nodes.planner as planner_mod


# ---- Plan file ----

def test_make_blank_plan_assigns_incremental_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("refactor foo.py", [
        "Read foo.py", "Rewrite it", "Test it"
    ])
    assert len(plan["chunks"]) == 3
    assert [c["id"] for c in plan["chunks"]] == [1, 2, 3]
    assert all(c["status"] == "pending" for c in plan["chunks"])
    assert all(c["output"] is None for c in plan["chunks"])
    assert plan["status"] == "in_progress"
    assert plan["task"] == "refactor foo.py"


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    plan = plan_store.make_blank_plan("task", ["a", "b"])
    plan_store.save_plan(plan)
    loaded = plan_store.load_plan(plan["plan_id"])
    assert loaded is not None
    assert loaded["plan_id"] == plan["plan_id"]
    assert loaded["chunks"][0]["instruction"] == "a"


def test_load_plan_returns_none_for_missing(monkeypatch):
    # Use a tmp root that doesn't have the file.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        monkeypatch.setattr(plan_store, "PLANS_ROOT", Path(td))
        assert plan_store.load_plan("aabbccdd") is None


def test_invalid_plan_id_rejected(tmp_path, monkeypatch):
    """Non-hex ids and empty strings are rejected."""
    monkeypatch.setattr(plan_store, "PLANS_ROOT", tmp_path)
    for bad in ("", "UPPERCASE", "with space", "../etc/passwd", "xyz123", "deadbeef!", "a b c"):
        try:
            plan_store._plan_path(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass
    # Hex ids of any length are accepted (path safety comes from the
    # restricted character set, not a length cap).
    for good in ("abc", "deadbeef", "a" * 64):
        plan_store._plan_path(good)  # should not raise


def test_mark_done_updates_chunk(monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", __import__("pathlib").Path(__file__).parent / "_tmp_plans")
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    plan_store.mark_executing(plan, 1)
    assert plan["chunks"][0]["status"] == "executing"
    plan_store.mark_done(plan, 1, "the output here")
    assert plan["chunks"][0]["status"] == "done"
    assert plan["chunks"][0]["output"] == "the output here"
    assert plan["chunks"][0]["executed_at"] is not None


def test_next_pending_returns_first_pending(monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", __import__("pathlib").Path(__file__).parent / "_tmp_plans")
    plan = plan_store.make_blank_plan("task", ["a", "b", "c"])
    plan_store.mark_done(plan, 1, "done")
    nxt = plan_store.next_pending(plan)
    assert nxt["id"] == 2
    plan_store.mark_done(plan, 2, "done")
    nxt = plan_store.next_pending(plan)
    assert nxt["id"] == 3
    plan_store.mark_done(plan, 3, "done")
    assert plan_store.next_pending(plan) is None


def test_finalize_marks_complete(monkeypatch):
    monkeypatch.setattr(plan_store, "PLANS_ROOT", __import__("pathlib").Path(__file__).parent / "_tmp_plans")
    plan = plan_store.make_blank_plan("task", ["a"])
    plan_store.finalize(plan, "all done")
    assert plan["status"] == "complete"
    assert plan["final_summary"] == "all done"


# ---- Planner output parser ----

def test_parse_create_plan_valid_json():
    text = 'CREATE_PLAN: ["Read foo.py", "Rewrite it", "Test it"]'
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "create_plan"
    assert payload == ["Read foo.py", "Rewrite it", "Test it"]


def test_parse_create_plan_with_reasoning():
    text = (
        "This needs 3 steps. Let me lay them out.\n"
        'CREATE_PLAN: ["Read foo.py", "Rewrite", "Test"]'
    )
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "create_plan"
    assert payload == ["Read foo.py", "Rewrite", "Test"]


def test_parse_direct_answer():
    text = "DIRECT_ANSWER: Paris is the capital of France."
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "direct_answer"
    assert payload == "Paris is the capital of France."


def test_parse_direct_answer_with_reasoning():
    text = (
        "This is a simple factual question.\n"
        "DIRECT_ANSWER: 42 is the answer to everything."
    )
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "direct_answer"
    assert "42" in payload


def test_parse_execute_next():
    text = "EXECUTE_NEXT"
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "execute_next"
    assert payload is None


def test_parse_execute_next_with_context():
    text = "Chunk 1 finished. Moving on.\nEXECUTE_NEXT"
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "execute_next"


def test_parse_final_summary():
    text = "FINAL_SUMMARY: The refactor is complete. All tests pass."
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "final_summary"
    assert "refactor is complete" in payload


def test_parse_fallback_when_no_marker():
    """If the LLM forgets to emit a marker, treat the whole text as a direct answer."""
    text = "I think the best approach is to refactor incrementally."
    kind, payload = planner_mod.parse_planner_output(text)
    assert kind == "fallback"
    assert payload == text


def test_parse_empty_text():
    kind, payload = planner_mod.parse_planner_output("")
    assert kind == "fallback"
    assert payload == ""


def test_parse_create_plan_with_invalid_json_falls_through():
    """Malformed CREATE_PLAN JSON falls through to other markers / fallback."""
    text = "CREATE_PLAN: [not json"
    kind, payload = planner_mod.parse_planner_output(text)
    # Should NOT match CREATE_PLAN; check what it matches instead.
    assert kind in ("direct_answer", "fallback", "execute_next", "final_summary")
