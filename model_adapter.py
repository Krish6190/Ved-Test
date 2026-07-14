import os
import re
from pathlib import Path

def _load_env_fallback() -> None:
    """Minimal .env loader used only if python-dotenv is unavailable.

    Sets env vars from KEY=VALUE lines if not already in the process
    environment. Shell-set vars win over .env values. Handles blank
    lines and # comments. No variable expansion.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.getenv(key):
                os.environ[key] = value
    except Exception:
        pass
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    _load_env_fallback()

class ModelAdapter:
    def __init__(self, model_name: str = "local-stub", device: str = "cpu", params=None, system_prompt: str = ""):
        self.model_name = model_name
        self.device = device
        self.params = params or {}
        self.system_prompt = system_prompt

    def create_llm(self, base_url: str):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("langchain-ollama is required for the graph flow.") from exc

        kwargs = {
            "model": self.model_name,
            "base_url": base_url,
            "temperature": float(self.params.get("temperature", 0.1)),
            "keep_alive": "20m",
        }
        for key, value in self.params.items():
            kwargs[key] = value
        kwargs.setdefault("temperature", 0.1)
        kwargs["num_gpu"] = 1 if self.device == "gpu" else 0
        return ChatOllama(**kwargs)

# ---- Mode-aware factories (Phase 1.1) ----
#
# Model names are NEVER hardcoded — they come from the Modelfile.{mode}
# files in the project root. This way changing models (e.g. swapping
# qwen2.5-coder:7b for a different quant, or using a different 3B for
# the executor) requires zero code changes — just edit the Modelfile.
#
# Routing:
#   - get_planner_llm(mode)  -> reads Modelfile.{mode}
#   - get_executor_llm(mode) -> reads Modelfile.executor (3B) for both
#                               turbo and coder modes.
# Mode model mapping:
#   - standard: 3B CPU (casual chat, no tools)
#   - turbo:    8B GPU (planner) + 3B executor
#   - coder:    7B GPU (planner) + 3B executor
#   - hibernate: no model


def _resolve_model_name(mode: str, *, modelfile_dir: str = ".") -> str:
    """Read the `FROM <model>` line from Modelfile.{mode}.

    Falls back to a sensible default if the Modelfile is missing or
    malformed, so a broken config doesn't crash the graph — it just
    logs a warning and uses the fallback.
    """
    path = Path(modelfile_dir) / f"Modelfile.{mode}"
    info = parse_modelfile(path)
    if info.get("from"):
        return info["from"]
    # Fallback: only hit this if the Modelfile is missing entirely.
    # Should never happen in a properly-installed project.
    import warnings
    warnings.warn(
        f"Modelfile.{mode} not found or has no FROM line; "
        f"falling back to a generic small model.",
        stacklevel=2,
    )
    return "llama3.2:3b"


def _local_ollama_available(base_url: str) -> bool:
    """Return True if a local Ollama server is reachable at `base_url`."""
    try:
        import requests
        endpoint = base_url.rstrip("/") + "/api/ls"
        resp = requests.get(endpoint, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _build_ollama_llm(model_name: str, *, base_url: str, device: str, params: dict):
    """Construct a ChatOllama with the same kwargs layout as
    ModelAdapter.create_llm — keeps temperature, keep_alive, and
    num_gpu handling consistent across the codebase."""
    if os.getenv("USE_CLOUD_API", "").lower() in ("1", "true", "yes"):
        local_available = _local_ollama_available(base_url)
        api_key = os.getenv("API_KEY")
        if local_available:
            pass
        elif not api_key:
            import warnings
            warnings.warn(
                "USE_CLOUD_API=true but no API_KEY set in environment; "
                "falling back to local Ollama.",
                stacklevel=2,
            )
        else:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "USE_CLOUD_API=true requires `pip install langchain-openai`."
                ) from exc
            openrouter_model = os.getenv(
                "OPENROUTER_MODEL", "poolside/laguna-m.1:free",
            )
            temperature = float(params.get("temperature", 0.1))
            return ChatOpenAI(
                model=openrouter_model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
            )
 
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain-ollama is required for the graph flow.") from exc

    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "temperature": float(params.get("temperature", 0.1)),
        "keep_alive": "20m",
    }
    for key, value in params.items():
        if key == "model":
            continue  # model is set explicitly above
        kwargs[key] = value
    kwargs.setdefault("temperature", 0.1)
    kwargs["num_gpu"] = 1 if device == "gpu" else 0
    return ChatOllama(**kwargs)


def get_planner_llm(mode: str, *, base_url: str, device: str, params: dict, modelfile_dir: str = "."):
    """Return the LLM the planner node should use for the given mode.

    Model name is read from Modelfile.{mode}. For coder mode that's the
    7B reasoning model; for turbo/standard it's whatever the Modelfile
    specifies (typically the same 3B model used by chat).

    `params` can override the model via `params["model"]` — useful for
    testing or env-based overrides.
    """
    p = dict(params) if params else {}
    if "model" in p:
        model = p.pop("model")
    else:
        model = _resolve_model_name(mode, modelfile_dir=modelfile_dir)
    return _build_ollama_llm(model, base_url=base_url, device=device, params=p)


def get_executor_llm(mode: str, *, base_url: str, device: str, params: dict, modelfile_dir: str = "."):
    """Return the LLM the executor node should use for the given mode.

    The executor always reads Modelfile.executor — a dedicated small/fast
    model for running tool loops. This keeps it separate from the planner
    model so the planner (7B in coder mode, 8B in non-coder modes) can
    focus on reasoning while the executor (3B) focuses on fast tool calls.

    `params` can override the model via `params["model"]`.
    """
    p = dict(params) if params else {}
    if "model" in p:
        model = p.pop("model")
    else:
        model = _resolve_model_name("executor", modelfile_dir=modelfile_dir)
    return _build_ollama_llm(model, base_url=base_url, device=device, params=p)


def parse_modelfile(path: Path) -> dict:
    data = {"from": None, "params": {}, "system": ""}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8")
    m = re.search(r"FROM\s+(.+)", text)
    if m:
        data["from"] = m.group(1).strip()
    for pm in re.finditer(r"PARAMETER\s+(\S+)\s+(.+)", text):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except Exception:
                pass
        data["params"][key] = val
    sys_m = re.search(r"SYSTEM\s+\"\"\"([\s\S]*?)\"\"\"", text)
    if sys_m:
        data["system"] = sys_m.group(1).strip()
    return data
