"""Project-root conftest for pytest.

Responsibilities:
- Make sure the project root is on ``sys.path`` so ``import graph.*``,
  ``import voice.*``, ``import ui.*`` etc. resolve without requiring an
  installed package.
- Provide an opt-in ``pytest_plugins`` marker for any future plugin needs.
- Deliberately import nothing heavy at module load (no tkinter, pywebview,
  faster_whisper, etc.) so ``pytest --collect-only`` stays fast and never
  crashes with a ModuleNotFoundError on a slim Linux venv.

Individual test files that genuinely need GUI/voice stacks should import
them inside the test function or behind ``pytest.importorskip(...)``.
"""
import sys
from pathlib import Path

# Project root = the directory containing this conftest.py.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep the marker symbol importable for any test that wants to assert
# pytest plugin registration behaviour. Empty marker is acceptable too.
pytest_plugins = []
