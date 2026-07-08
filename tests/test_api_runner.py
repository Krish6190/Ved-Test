"""Unit tests for api.runner. Uses tiny temp scripts — no real workloads."""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path when pytest runs from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import runner


def test_run_python_script_basic(tmp_path: Path):
    p = tmp_path / "hello.py"
    p.write_text("print('hello from script')")
    result = asyncio.run(runner.run_python_script(str(p)))
    assert result["exit_code"] == 0
    assert "hello from script" in result["stdout"]
    assert result["stderr"] == ""
    assert result["timed_out"] is False
    assert result["duration_seconds"] >= 0


def test_run_python_script_nonzero_exit(tmp_path: Path):
    p = tmp_path / "fail.py"
    p.write_text("import sys; sys.stderr.write('oops'); sys.exit(2)")
    result = asyncio.run(runner.run_python_script(str(p)))
    assert result["exit_code"] == 2
    assert "oops" in result["stderr"]
    assert result["timed_out"] is False


def test_run_python_script_timeout(tmp_path: Path):
    p = tmp_path / "slow.py"
    p.write_text("import time; time.sleep(5)")
    result = asyncio.run(runner.run_python_script(str(p), timeout_seconds=1))
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_run_python_script_passes_args(tmp_path: Path):
    p = tmp_path / "echo_args.py"
    p.write_text("import sys; print(' '.join(sys.argv[1:]))")
    result = asyncio.run(runner.run_python_script(str(p), args=["a", "b", "c"]))
    assert result["exit_code"] == 0
    assert "a b c" in result["stdout"]


def test_run_python_script_rejects_missing_file():
    try:
        asyncio.run(runner.run_python_script("/does/not/exist.py"))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_run_python_script_rejects_non_py(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("hello")
    try:
        asyncio.run(runner.run_python_script(str(p)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_format_run_output_success():
    out = runner.format_run_output({
        "exit_code": 0, "stdout": "ok", "stderr": "",
        "timed_out": False, "duration_seconds": 0.1,
        "truncated_stdout": False, "truncated_stderr": False,
    }, "x.py")
    assert "x.py" in out
    assert "ok" in out


def test_format_run_output_timeout():
    out = runner.format_run_output({
        "exit_code": -1, "stdout": "", "stderr": "",
        "timed_out": True, "duration_seconds": 30.0,
        "truncated_stdout": False, "truncated_stderr": False,
    }, "slow.py")
    assert "timed out" in out.lower()
    assert "slow.py" in out
