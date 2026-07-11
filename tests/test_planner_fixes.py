"""Tests for the planner fixes:

Issue 1: Editing/coding tasks in standard mode should trigger
         RECOMMEND_CODER_MODE (not DIRECT_ANSWER with instructions).

Issue 2: RAG context is auto-injected into the planner's message stream
         so the LLM sees uploaded file content without an explicit
         retrieve_rag call.
"""
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from graph.nodes import planner as planner_mod
from graph.state import VedState
from graph.nodes.planner_prompt import _PLANNER_SYSTEM, _build_planner_prompt
from graph.nodes.planner_markers import parse_planner_output


# ---- Issue 1: system prompt guides toward RECOMMEND_CODER ----

def test_planner_system_prompt_instructs_to_emit_recommend_coder_for_edits():
    """The planner's system prompt must explicitly say: editing tasks in
    standard mode -> RECOMMEND_CODER_MODE, NOT DIRECT_ANSWER with
    instructions."""
    content = _PLANNER_SYSTEM.content
    # Must mention RECOMMEND_CODER_MODE marker explicitly
    assert "RECOMMEND_CODER_MODE" in content
    # Must mention the editing/file-modification triggers
    assert "EDIT" in content or "edit" in content
    assert "WRITE" in content or "write" in content
    # Must explicitly tell the planner NOT to emit DIRECT_ANSWER for these
    content_lower = content.lower()
    assert ("do not emit" in content_lower and "direct_answer" in content_lower), \
        "prompt must explicitly instruct against DIRECT_ANSWER for editing tasks"


def test_planner_parser_recognizes_recommend_coder_marker():
    """Sanity: the RECOMMEND_CODER marker is still parsed correctly."""
    text = (
        "This needs code work.\n"
        "RECOMMEND_CODER_MODE REASON: refactor requires edit_file"
    )
    kind, payload = parse_planner_output(text)
    assert kind == "recommend_coder"
    assert "refactor requires edit_file" in payload


# ---- Issue 2: RAG auto-injection in planner prompt ----

def test_planner_prompt_includes_rag_block_when_rag_returns_results():
    """When the RAG mixer has content for the user's query, the planner's
    message stream includes a SystemMessage with [RAG Context]..."""
    user_prompt = "summarize the project structure"
    state = VedState(messages=[HumanMessage(content=user_prompt)], route_intent="A")

    fake_rag_block = (
        "[RAG Context]\n\n"
        "(1) [thread] README.md\nThis is a sample README describing the project.\n\n"
        "(2) [thread] setup.py\nPackage configuration goes here."
    )

    with patch.object(planner_mod, "_build_rag_block", return_value=fake_rag_block):
        msgs = _build_planner_prompt(state, plan=None)

    # At least one SystemMessage in the stream contains the RAG block.
    rag_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[RAG Context]" in m.content]
    assert rag_msgs, "RAG block was not injected into the planner prompt"
    # The injected block contains the uploaded file content.
    assert "README.md" in rag_msgs[0].content
    assert "sample README" in rag_msgs[0].content


def test_planner_prompt_handles_missing_rag_block_gracefully():
    """When _build_rag_block returns '' (no relevant content), the planner
    prompt still builds successfully without crashing."""
    state = VedState(messages=[HumanMessage(content="hello")], route_intent="A")

    with patch.object(planner_mod, "_build_rag_block", return_value=""):
        msgs = _build_planner_prompt(state, plan=None)

    # No SystemMessage with [RAG Context] marker.
    rag_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[RAG Context]" in m.content]
    assert not rag_msgs
    # But the prompt is still well-formed.
    assert any(isinstance(m, SystemMessage) for m in msgs)


def test_planner_prompt_handles_rag_exception_gracefully():
    """If _build_rag_block raises (e.g., embeddings unavailable), the
    planner prompt still builds without crashing."""
    state = VedState(messages=[HumanMessage(content="hello")], route_intent="A")

    with patch.object(planner_mod, "_build_rag_block", side_effect=RuntimeError("rag down")):
        msgs = _build_planner_prompt(state, plan=None)

    # Prompt still builds.
    assert any(isinstance(m, SystemMessage) for m in msgs)
    # No RAG block was injected (because the call raised).
    rag_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[RAG Context]" in m.content]
    assert not rag_msgs


# ---- End-to-end: simulator confirms the right marker gets emitted ----

def test_planner_recommend_coder_marker_priority_over_direct_answer():
    """If the LLM outputs both, RECOMMEND_CODER_MODE wins (parse order
    is RECOMMEND_CODER before DIRECT_ANSWER in the parser)."""
    text = (
        "Open foo.py and refactor the login function.\n"
        "RECOMMEND_CODER_MODE REASON: needs edit_file\n"
        "DIRECT_ANSWER: Open foo.py and refactor the login function..."
    )
    kind, _ = parse_planner_output(text)
    assert kind == "recommend_coder"
