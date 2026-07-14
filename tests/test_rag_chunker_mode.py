"""Tests for the mode-aware RAG chunking in Chunk 5.

These tests cover three things:
  1. ingest_local_file with chunker="text" tags every registry record with
     chunker="text" and layer="body".
  2. ingest_local_file with chunker="ast" on a .py file produces at least
     one record tagged chunker="ast" with layer in {"sig", "body"}.
  3. Chatbot._rag_chunker() returns "ast" when mode="coder" and "text"
     otherwise.

The embeddings engine is replaced with a FakeEmbeddings object that
returns 768-dim zero vectors, so no Ollama/network calls happen. The DB
path is redirected to a per-test tmp directory via monkeypatch so we
never write to the real project `data/vectordb/index.bin`.

Chatbot construction is sandboxed using the same technique as
tests/test_threads.py: monkeypatch `chatbot.Path` so that
`Path(__file__).resolve().parent` lands inside tmp_path.
"""
from pathlib import Path as RealPath

import pytest

import chatbot
from chatbot import Chatbot
from graph.rag.vector_engine import LocalVectorDB
from graph.nodes.planner import planner_node
from graph.nodes.executor import executor_node
from graph.state import VedState
from langchain_core.messages import SystemMessage
from model_adapter import ModelAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    """Stand-in for langchain_ollama.OllamaEmbeddings.

    Returns a 768-dim zero vector per input text. Dimension is arbitrary
    but fixed, mirroring a real embedding model's behaviour.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


@pytest.fixture
def make_vectordb(tmp_path, monkeypatch):
    """Return a factory that builds a LocalVectorDB with a FakeEmbeddings.

    The DB file is redirected to tmp_path / "index.bin" so the real
    project database is never touched.
    """

    def _factory():
        monkeypatch.setenv("DB_PATH", str(tmp_path / "index.bin"))
        db = LocalVectorDB()
        db.embeddings_engine = FakeEmbeddings()
        return db

    return _factory


@pytest.fixture
def make_chatbot(tmp_path, monkeypatch):
    """Return a factory that builds Chatbot instances with file I/O sandboxed to tmp_path."""

    def _patched_path(*args, **kwargs):
        # When chatbot.py's __init__ does `Path(__file__).resolve().parent`,
        # return a fake Path inside tmp_path so .parent == tmp_path.
        if args and isinstance(args[0], str) and args[0].endswith("chatbot.py"):
            return RealPath(tmp_path) / "chatbot.py"
        return RealPath(*args, **kwargs)

    monkeypatch.setattr(chatbot, "Path", _patched_path)

    def _factory(mode="hibernate"):
        return Chatbot(mode=mode)

    return _factory


# ---------------------------------------------------------------------------
# Test 1: text chunker tags every record with chunker="text", layer="body"
# ---------------------------------------------------------------------------
def test_text_chunker_tagged_in_registry(make_vectordb, tmp_path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text(
        "This is the first paragraph of the document. It contains some prose.\n\n"
        "Here is a second paragraph. It elaborates on the first point with "
        "additional detail so it is long enough to be its own chunk.\n\n"
        "Finally a third paragraph that closes the document with a summary "
        "and a concluding remark.",
        encoding="utf-8",
    )

    db = make_vectordb()
    initial_count = len(db.registry)
    db.ingest_local_file(str(txt_path), scope="test", chunker="text", source="doc.txt")

    new_records = db.registry[initial_count:]
    assert len(new_records) >= 1, "Text chunker should produce at least one record"
    for record in new_records:
        assert record["chunker"] == "text"
        assert record["layer"] == "body"
        assert record["scope"] == "test"
        assert record["source"] == "doc.txt"


# ---------------------------------------------------------------------------
# Test 2: AST chunker tags records with chunker="ast" and layer in {sig, body}
# ---------------------------------------------------------------------------
def test_ast_chunker_tagged_for_python(make_vectordb, tmp_path):
    py_path = tmp_path / "sample.py"
    py_path.write_text(
        '"""Sample module for AST chunker tests."""\n'
        "\n"
        "import math\n"
        "\n"
        "CONSTANT = 42\n"
        "\n"
        "\n"
        "def greet(name):\n"
        '    """Return a friendly greeting."""\n'
        '    return f"hello, {name}"\n'
        "\n"
        "\n"
        "class Calculator:\n"
        '    """A simple calculator."""\n'
        "\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def multiply(self, a, b):\n"
        "        return a * b\n",
        encoding="utf-8",
    )

    db = make_vectordb()
    initial_count = len(db.registry)
    db.ingest_local_file(str(py_path), scope="test", chunker="ast", source="sample.py")

    new_records = db.registry[initial_count:]
    assert len(new_records) >= 1, "AST chunker should produce at least one record"
    assert len(new_records) >= 2, (
        "Expected multiple AST records (sig + body) for a file with "
        "a function and a class"
    )

    # Every new record must be tagged as AST with a valid layer.
    for record in new_records:
        assert record["chunker"] == "ast"
        assert record["layer"] in {"sig", "body"}
        assert record["scope"] == "test"
        assert record["source"] == "sample.py"

    # At least one record must be a sig and at least one a body, since
    # we have at least one function (sig+body) and one class (sig+body).
    layers = {record["layer"] for record in new_records}
    assert "sig" in layers, "Expected at least one sig-layer AST chunk"
    assert "body" in layers, "Expected at least one body-layer AST chunk"


# ---------------------------------------------------------------------------
# Test 3: Chatbot._rag_chunker() respects the mode
# ---------------------------------------------------------------------------
def test_chatbot_rag_chunker_by_mode(make_chatbot):
    coder_bot = make_chatbot(mode="coder")
    assert coder_bot.mode == "coder"
    assert coder_bot._rag_chunker() == "ast"

    standard_bot = make_chatbot(mode="standard")
    assert standard_bot.mode == "standard"
    assert standard_bot._rag_chunker() == "text"


def test_coder_mode_sanitizes_recommendation_payload(make_chatbot):
    bot = make_chatbot(mode="coder")
    bot.adapters["coder"] = ModelAdapter(
        model_name="dummy",
        device="cpu",
        params={},
        system_prompt=(
            "You are Ved in coder mode.\n"
            "RECOMMEND_CODER_MODE REASON: use coder mode for file edits.\n"
            "Execute coder tasks directly."
        ),
    )

    prompt = bot._sanitize_system_prompt(bot.adapters["coder"].system_prompt, bot.mode)
    assert "RECOMMEND_CODER_MODE" not in prompt
    assert "Execute coder tasks directly." in prompt


def test_set_mode_coder_clears_legacy_recommendation_system_messages(make_chatbot):
    bot = make_chatbot(mode="standard")
    active_thread = bot.get_active_thread()
    active_thread["messages"].append(SystemMessage(content="RECOMMEND_CODER_MODE REASON: should switch to coder"))

    bot.set_mode("coder")
    assert bot.mode == "coder"
    assert not any(
        isinstance(msg, SystemMessage)
        and "RECOMMEND_CODER_MODE" in msg.content
        for msg in active_thread["messages"]
    )


def test_compact_system_prompt_removes_duplicate_lines(make_chatbot):
    bot = make_chatbot(mode="standard")
    prompt = (
        "You are Ved.\n\n"
        "You are Ved.\n"
        "Use tools only when needed.\n"
        "Use tools only when needed.\n\n"
        "Be concise."
    )
    compacted = bot._compact_system_prompt(prompt)
    assert compacted.count("You are Ved.") == 1
    assert compacted.count("Use tools only when needed.") == 1
    assert "\n\n\n" not in compacted


def test_planner_node_respects_summary_emitted():
    state = VedState(
        messages=[],
        route_intent="P",
        mode="standard",
        active_thread_id="thr_test",
        summary_emitted=True,
    )
    result = planner_node(state, lambda: None, None)
    assert result["route_intent"] == "A"
    assert result["messages"] == []


def test_executor_node_respects_summary_emitted():
    state = VedState(
        messages=[],
        route_intent="P",
        mode="standard",
        active_thread_id="thr_test",
        summary_emitted=True,
    )
    result = executor_node(state, lambda: None, None)
    assert result["route_intent"] == "A"
    assert result["active_plan_id"] is None
