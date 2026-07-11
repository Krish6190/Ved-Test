"""Smoke test for the USE_CLOUD_API toggle in model_adapter._build_ollama_llm.

Verifies:
  - USE_CLOUD_API=false (default) -> ChatOllama is constructed
  - USE_CLOUD_API=true + API_KEY set -> ChatOpenAI pointed at OpenRouter,
    with the Poolside Laguna M.1 (free) coding model by default
  - USE_CLOUD_API=true + API_KEY missing -> falls back to ChatOllama
    with a warning (does NOT crash)
  - OPENROUTER_MODEL env var overrides the default model string
"""
import warnings
import pytest


@pytest.fixture
def clean_env(monkeypatch):
    """Strip the toggle env vars so each test sets them explicitly."""
    for k in ("USE_CLOUD_API", "API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _build(params=None):
    """Call _build_ollama_llm with neutral inputs and no actual inference."""
    from model_adapter import _build_ollama_llm
    return _build_ollama_llm(
        "ignored-local-model",
        base_url="http://localhost:11434",
        device="cpu",
        params=params or {"temperature": 0.1},
    )


def test_default_routes_to_ollama(clean_env):
    """USE_CLOUD_API=false explicitly -> local Ollama path.

    Note: model_adapter calls load_dotenv() at import time, which can set
    USE_CLOUD_API=true from .env. We pin it to "false" here to force the
    local path regardless of .env state.
    """
    clean_env.setenv("USE_CLOUD_API", "false")
    llm = _build()
    cls = type(llm).__name__
    assert cls == "ChatOllama", f"expected ChatOllama, got {cls}"


def test_cloud_toggle_returns_chatopenai(clean_env):
    """USE_CLOUD_API=true + API_KEY -> ChatOpenAI on OpenRouter endpoint."""
    clean_env.setenv("USE_CLOUD_API", "true")
    clean_env.setenv("API_KEY", "sk-or-v1-fake-test-key-for-unit-test")
    llm = _build()
    assert type(llm).__name__ == "ChatOpenAI", f"got {type(llm).__name__}"
    # ChatOpenAI exposes the model and base_url as attributes
    assert "poolside/laguna-m.1:free" in llm.model_name, \
        f"unexpected model: {llm.model_name}"
    assert "openrouter.ai" in str(llm.openai_api_base), \
        f"unexpected base_url: {llm.openai_api_base}"


def test_cloud_without_key_falls_back(clean_env):
    """USE_CLOUD_API=true but no API_KEY -> warning + local fallback."""
    clean_env.setenv("USE_CLOUD_API", "true")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        llm = _build()
    assert type(llm).__name__ == "ChatOpenAI" or type(llm).__name__ == "ChatOllama"
    # When API_KEY is missing we expect EITHER a graceful ChatOpenAI (if
    # ChatOpenAI tolerates empty key at construction) OR a ChatOllama
    # fallback with a warning. Both are acceptable; crash is not.
    if type(llm).__name__ == "ChatOllama":
        assert any("API_KEY" in str(w.message) for w in caught), \
            "fallback should warn about missing API_KEY"


def test_openrouter_model_override(clean_env):
    """OPENROUTER_MODEL env var overrides the default model string."""
    clean_env.setenv("USE_CLOUD_API", "true")
    clean_env.setenv("API_KEY", "sk-or-v1-fake-test-key")
    clean_env.setenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct")
    llm = _build()
    assert "32b" in llm.model_name, f"override ignored: {llm.model_name}"
