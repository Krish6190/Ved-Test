"""Project-root conftest for pytest.

Responsibilities:
- Make sure the project root is on ``sys.path`` so ``import graph.*``,
  ``import voice.*``, ``import ui.*`` etc. resolve without requiring an
  installed package.
- Provide an opt-in ``pytest_plugins`` marker for any future plugin needs.
- Stub out optional platform-specific modules (tkinter, sounddevice, etc.)
  on Linux CI/test venvs so ``pytest --collect-only`` never crashes with
  ModuleNotFoundError. Tests that genuinely exercise GUI/voice should use
  ``pytest.importorskip(...)``.
"""
import sys
import types
from pathlib import Path

# Project root = the directory containing this conftest.py.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Some modules are heavy or platform-specific (tkinter on Linux/WSL,
# sounddevice requiring PortAudio, etc.). Provide minimal stubs so test
# collection succeeds without them. Runtime code still uses real imports
# when the deps are installed.
_MISSING_MODULE_STUBS = [
    "tkinter",
    "tkinter.messagebox",
    "tkinter.simpledialog",
    "tkinter.filedialog",
    "sounddevice",
    "soundfile",
    "pywebview",
    "faster_whisper",
    "openwakeword",
    "openwakeword.model",
    "piper",
    "tensorflow",
    "winsound",
]


for _mod_name in _MISSING_MODULE_STUBS:
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except Exception:
            _parts = _mod_name.split(".")
            for _i in range(1, len(_parts) + 1):
                _parent = ".".join(_parts[:_i])
                if _parent not in sys.modules:
                    sys.modules[_parent] = types.ModuleType(_parent)

# Keep the marker symbol importable for any test that wants to assert
# pytest plugin registration behaviour. Empty marker is acceptable too.
pytest_plugins = []
