"""Phase 1.1 — model router split.

Coder mode -> planner = qwen2.5-coder:7b, executor = llama3.2:3b.
Standard/turbo -> both default to llama3.2:3b (or whatever `params["model"]` says).
"""
import pytest
from model_adapter import get_executor_llm, get_planner_llm


@pytest.fixture(autouse=True)
def _force_local_models(monkeypatch):
    """Model-router tests exercise the local Ollama path and must not be
    redirected to the cloud API even if .env has USE_CLOUD_API=true.
    """
    monkeypatch.setenv("USE_CLOUD_API", "false")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


def test_coder_mode_splits_planner_and_executor_models():
    """Coder mode: planner gets the 7B coder, executor keeps the 3B."""
    planner = get_planner_llm("coder", base_url="http://x", device="cpu", params={})
    executor = get_executor_llm("coder", base_url="http://x", device="cpu", params={})
    assert planner.model == "qwen2.5-coder:7b"
    assert executor.model == "llama3.2:3b"


def test_standard_mode_uses_default_small_model():
    """Standard mode: both planner and executor default to llama3.2:3b."""
    planner = get_planner_llm("standard", base_url="http://x", device="cpu", params={})
    executor = get_executor_llm("standard", base_url="http://x", device="cpu", params={})
    assert planner.model == "llama3.2:3b"
    assert executor.model == "llama3.2:3b"


def test_turbo_mode_uses_default_small_model():
    """Turbo mode: same default as standard (no regression)."""
    planner = get_planner_llm("turbo", base_url="http://x", device="cpu", params={})
    executor = get_executor_llm("turbo", base_url="http://x", device="cpu", params={})
    assert planner.model == "llama3.2:3b"
    assert executor.model == "llama3.2:3b"


def test_standard_mode_respects_params_model_override():
    """Standard mode honors params['model'] when caller wants to override."""
    planner = get_planner_llm(
        "standard", base_url="http://x", device="cpu",
        params={"model": "custom:1b"},
    )
    executor = get_executor_llm(
        "standard", base_url="http://x", device="cpu",
        params={"model": "custom:1b"},
    )
    assert planner.model == "custom:1b"
    assert executor.model == "custom:1b"


def test_coder_mode_ignores_params_model_for_planner():
    """Coder planner is locked to the 7B coder model regardless of params."""
    planner = get_planner_llm(
        "coder", base_url="http://x", device="cpu",
        params={"model": "should-be-ignored:9b"},
    )
    assert planner.model == "qwen2.5-coder:7b"
