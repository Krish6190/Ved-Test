"""Tests for graph/nodes/intent_router.py routing rules."""
from langchain_core.messages import HumanMessage

from graph.nodes.intent_router import intent_router_node
from graph.state import VedState


def _state(text: str) -> VedState:
    # Minimal state with just the messages field populated.
    return VedState(
        messages=[HumanMessage(content=text)],
        mode="standard",
        route_intent="A",
    )


def _route(text: str) -> str:
    s = _state(text)
    # get_llm is unused by the router but the signature requires it.
    return intent_router_node(s, get_llm=lambda: None)["route_intent"]


# ----- Should route to A (default / tool-needing) -----

def test_routes_read_to_A():
    assert _route("read this file") == "A"


def test_routes_create_to_A():
    assert _route("create a new function") == "A"


def test_routes_make_to_A():
    assert _route("make a config file") == "A"


def test_routes_build_to_A():
    assert _route("build a CLI tool") == "A"


def test_routes_summarize_to_A():
    assert _route("summarize foo.py") == "A"


def test_routes_summary_of_to_A():
    assert _route("summary of the bug") == "A"


def test_routes_generate_to_A():
    assert _route("generate a test for this") == "A"


def test_routes_produce_to_A():
    assert _route("produce a config") == "A"


def test_routes_craft_to_A():
    assert _route("craft a function") == "A"


def test_routes_fix_to_A():
    assert _route("fix the bug in bar.py") == "A"


def test_routes_explain_to_A():
    assert _route("explain what this code does") == "A"


def test_routes_chitchat_to_A():
    assert _route("hi") == "A"


def test_routes_slash_command_to_A():
    assert _route("/threads") == "A"


def test_routes_explicit_A():
    assert _route("use path A") == "A"


# ----- Should route to B (clear content-gen) -----

def test_routes_write_essay_to_B():
    assert _route("write me an essay") == "B"


def test_routes_write_poem_to_B():
    assert _route("write a poem about love") == "B"


def test_routes_draft_blog_to_B():
    assert _route("draft a blog post about AI") == "B"


def test_routes_compose_letter_to_B():
    assert _route("compose a letter to my landlord") == "B"


def test_routes_essay_phrase_to_B():
    assert _route("essay on climate change") == "B"


def test_routes_length_with_prose_verb_to_B():
    assert _route("write 5 paragraphs about cats") == "B"


def test_routes_length_with_content_phrase_to_B():
    # Length spec ("200 words") + content phrase ("essay about") together
    # trigger Path B. "essay topic" alone is not in _GENERATION_PHRASES.
    assert _route("give me 200 words on an essay about topic") == "B"


def test_routes_blog_post_to_B():
    assert _route("blog post about productivity") == "B"


def test_routes_story_to_B():
    assert _route("story about a dragon") == "B"


def test_routes_explicit_B():
    assert _route("use path B") == "B"