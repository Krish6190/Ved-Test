"""Unit tests for the filesystem action layer.

Pure-Python tests against `tmp_path`. No LLM, no LangChain imports. The
goal is to lock in the action-layer contract (primitive args in, primitive
result out) so the future client-server split can replace the action
implementations without breaking tool-layer behavior.

Run with: `cd /mnt/c/Users/krish/OneDrive/Desktop/ved && .venv/bin/python -m pytest tests/test_actions_filesystem.py -v`
"""
from pathlib import Path

from graph.actions.filesystem import (
    edit_file_action,
    overwrite_file_action,
    read_file_action,
    search_files_action,
)


# ---------------------------------------------------------------------------
# read_file_action
# ---------------------------------------------------------------------------

def test_read_existing_file(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("hello from the test", encoding="utf-8")
    result = read_file_action(str(target), allowed_roots=(str(tmp_path),))
    assert "hello from the test" in result
    assert "FILE:" in result
    assert "ERROR" not in result


def test_read_missing_file(tmp_path: Path):
    """A missing file must return an error string, never raise."""
    target = tmp_path / "does_not_exist.txt"
    result = read_file_action(str(target), allowed_roots=(str(tmp_path),))
    assert isinstance(result, str)
    assert result.startswith("ERROR:")
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# edit_file_action
# ---------------------------------------------------------------------------

def test_edit_and_verify(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    result = edit_file_action(
        str(target),
        "world",
        "VED",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "hello VED"
    # Chunk 4: physical .bak files must never be produced by edit_file_action.
    backup = target.with_suffix(target.suffix + ".bak")
    assert not backup.exists()
    assert not (tmp_path / "notes.txt.bak").exists()


def test_edit_old_text_not_found(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    result = edit_file_action(
        str(target),
        "missing phrase",
        "x",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert isinstance(result, str)
    assert result.startswith("ERROR:")
    assert "could not locate" in result.lower() or "not found" in result.lower()
    # File must be untouched.
    assert target.read_text(encoding="utf-8") == "hello world"


# ---------------------------------------------------------------------------
# overwrite_file_action
# ---------------------------------------------------------------------------

def test_overwrite_and_verify(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("original content", encoding="utf-8")
    result = overwrite_file_action(
        str(target),
        "brand new content",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )
    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "brand new content"
    # Chunk 4: physical .bak files must never be produced by overwrite_file_action.
    backup = target.with_suffix(target.suffix + ".bak")
    assert not backup.exists()
    assert not (tmp_path / "data.txt.bak").exists()


# ---------------------------------------------------------------------------
# search_files_action
# ---------------------------------------------------------------------------

def test_search_with_match(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("a", encoding="utf-8")
    (tmp_path / "beta.py").write_text("b", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("c", encoding="utf-8")

    result = search_files_action(
        "*.py",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert "alpha.py" in result
    assert "beta.py" in result
    assert "gamma.txt" not in result
    assert "ERROR" not in result


def test_search_no_match(tmp_path: Path):
    """An unmatched pattern must return an empty string (no error, no raise).

    The tool layer formats the empty case as an ERROR for the LLM; the
    action's contract is just 'no matches -> empty string'.
    """
    (tmp_path / "alpha.py").write_text("a", encoding="utf-8")
    result = search_files_action(
        "*.zzz",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert result == ""


# ---------------------------------------------------------------------------
# Backup-artifact masking (*.bak, *.tmp) -- Chunk 1 acceptance criteria
# ---------------------------------------------------------------------------

def test_search_files_action_skips_backup_artifacts(tmp_path: Path):
    """Chunk 1 AC #2: backup artifacts must never appear in search results.

    Even when ``*.bak`` (or ``*.tmp``) explicitly matches files, the action
    filters them out in all three match strategies so the agent can never
    read or edit stale backup copies.
    """
    (tmp_path / "real.py").write_text("r", encoding="utf-8")
    (tmp_path / "notes.bak").write_text("old", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_text("tmp", encoding="utf-8")

    # Strategy 1 (exact glob *.bak) must return "" -- no results, no error.
    result_bak = search_files_action(
        "*.bak",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert result_bak == "", (
        f"*.bak glob must return '' even when .bak files exist; got: {result_bak!r}"
    )

    # Same contract for *.tmp
    result_tmp = search_files_action(
        "*.tmp",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert result_tmp == "", (
        f"*.tmp glob must return '' even when .tmp files exist; got: {result_tmp!r}"
    )

    # Sanity check: a non-backup pattern still returns real files.
    result_py = search_files_action(
        "*.py",
        directory=str(tmp_path),
        skip_dirs=(),
        max_results=50,
    )
    assert "real.py" in result_py
    assert "notes.bak" not in result_py
    assert "scratch.tmp" not in result_py


def test_project_indexer_skips_backup_artifacts(tmp_path: Path):
    """Chunk 1 AC #3: ``index_workspace`` must not ingest ``.bak``/``.tmp``.

    Uses a recording fake DB so we can assert the *exact* set of files
    the indexer handed off to RAG. The DB never touches disk and never
    calls embeddings.
    """
    from graph.rag.project_indexer import index_workspace

    class _RecordingDB:
        def __init__(self):
            self.ingested: list[tuple[str, str, str]] = []  # (path, scope, source)

        def ingest_local_file(self, file_path, scope, chunker, source):
            self.ingested.append((file_path, scope, source))
            return True

    # Lay out a workspace with a mix of real, backup, and tmp files.
    (tmp_path / "keep.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "keep.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "notes.bak").write_text("old backup\n", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_text("leftover\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.bak").write_text("nested backup\n", encoding="utf-8")
    (sub / "nested.tmp").write_text("nested tmp\n", encoding="utf-8")
    (sub / "good.py").write_text("x = 1\n", encoding="utf-8")

    db = _RecordingDB()
    stats = index_workspace(tmp_path, db)

    ingested_paths = [Path(p).name for (p, _, _) in db.ingested]

    # Real files must be indexed.
    assert "keep.py" in ingested_paths
    assert "keep.md" in ingested_paths
    assert "good.py" in ingested_paths

    # Backup artifacts must never reach the DB.
    for forbidden in ("notes.bak", "scratch.tmp", "nested.bak", "nested.tmp"):
        assert forbidden not in ingested_paths, (
            f"index_workspace ingested forbidden backup artifact: {forbidden}"
        )

    # And stats should reflect the skips.
    assert stats["files_indexed"] >= 3
    assert stats["files_skipped"] >= 4


# ---------------------------------------------------------------------------
# Chunk 4 -- in-memory diff capture replaces physical .bak writes
# ---------------------------------------------------------------------------

class _FakeRAGForDiff:
    """Minimal in-memory RAG stub mirroring DiffHistoryStore's surface.

    Records every (text, scope, source) tuple handed to ingest_text so a
    test can assert that a diff chunk was stored under DIFF_HISTORY_SCOPE
    without touching Ollama or the real LocalVectorDB.
    """

    def __init__(self) -> None:
        self.ingests: list[tuple[str, str, str]] = []  # (text, scope, source)
        self.deleted: list[tuple[str, str]] = []  # (scope, source)

    def ingest_text(
        self,
        text: str,
        scope: str = "__GLOBAL__",
        source: str = "raw_text",
        chunker: str = "text",
    ) -> None:
        self.ingests.append((text, scope, source))

    def delete_by_source(self, scope: str, source: str) -> int:
        self.deleted.append((scope, source))
        before = len(self.ingests)
        self.ingests = [
            t for t in self.ingests
            if not (t[1] == scope and t[2] == source)
        ]
        return before - len(self.ingests)

    def query_similarity(self, query_text: str, k: int = 2, **kwargs):
        return []


def test_edit_file_action_creates_no_bak(tmp_path: Path):
    """Chunk 4 AC #1: edit_file_action never writes a .bak file.

    Run the action with the legacy backup_dir=None argument to confirm
    that even when callers pass it (the historical default), no .bak
    file appears on disk after a successful edit.
    """
    target = tmp_path / "no_bak.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = edit_file_action(
        str(target),
        "beta",
        "BETA",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )

    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    # The on-disk reality: no .bak anywhere in or around the file.
    assert not (tmp_path / "no_bak.txt.bak").exists()
    assert not target.with_suffix(target.suffix + ".bak").exists()
    leftover = list(tmp_path.glob("*.bak"))
    assert leftover == [], f"Unexpected .bak files after edit: {leftover}"


def test_overwrite_file_action_creates_no_bak(tmp_path: Path):
    """Chunk 4 AC #2: overwrite_file_action never writes a .bak file."""
    target = tmp_path / "no_bak_overwrite.txt"
    target.write_text("original\n", encoding="utf-8")

    result = overwrite_file_action(
        str(target),
        "replacement\n",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
    )

    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "replacement\n"

    assert not (tmp_path / "no_bak_overwrite.txt.bak").exists()
    assert not target.with_suffix(target.suffix + ".bak").exists()
    leftover = list(tmp_path.glob("*.bak"))
    assert leftover == [], f"Unexpected .bak files after overwrite: {leftover}"


def test_edit_file_action_stores_diff_when_store_provided(tmp_path: Path):
    """Chunk 4 AC #3: when a DiffHistoryStore is supplied, the unified
    diff of (old, new) is ingested into RAG under DIFF_HISTORY_SCOPE.
    """
    from graph.rag.diff_history import DIFF_HISTORY_SCOPE, DiffHistoryStore

    target = tmp_path / "edited.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    fake_rag = _FakeRAGForDiff()
    store = DiffHistoryStore(rag_db=fake_rag, meta_path=tmp_path / "diff_history.json")

    result = edit_file_action(
        str(target),
        "b = 2",
        "b = 22",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
        diff_history_store=store,
    )

    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "a = 1\nb = 22\n"

    # Exactly one chunk should have been stored, under DIFF_HISTORY_SCOPE.
    assert len(fake_rag.ingests) == 1, fake_rag.ingests
    text, scope, source = fake_rag.ingests[0]
    assert scope == DIFF_HISTORY_SCOPE
    assert source.startswith(str(target.resolve()))
    assert "diff::" in source
    # Unified-diff markers must be present in the stored text.
    assert "---" in text
    assert "+++" in text
    assert "-b = 2" in text
    assert "+b = 22" in text

    # No .bak should have been created as a side effect.
    assert not (tmp_path / "edited.py.bak").exists()


def test_overwrite_file_action_stores_diff_when_store_provided(tmp_path: Path):
    """Chunk 4 AC #4: same as edit but for overwrite_file_action."""
    from graph.rag.diff_history import DIFF_HISTORY_SCOPE, DiffHistoryStore

    target = tmp_path / "whole.py"
    target.write_text("first version\n", encoding="utf-8")

    fake_rag = _FakeRAGForDiff()
    store = DiffHistoryStore(rag_db=fake_rag, meta_path=tmp_path / "diff_history.json")

    result = overwrite_file_action(
        str(target),
        "second version\nthird line\n",
        allowed_roots=(str(tmp_path),),
        backup_dir=None,
        diff_history_store=store,
    )

    assert result.startswith("OK:")
    assert target.read_text(encoding="utf-8") == "second version\nthird line\n"

    assert len(fake_rag.ingests) == 1, fake_rag.ingests
    text, scope, source = fake_rag.ingests[0]
    assert scope == DIFF_HISTORY_SCOPE
    assert source.startswith(str(target.resolve()))
    assert "diff::" in source
    # Unified-diff markers must be present.
    assert "---" in text
    assert "+++" in text

    # No .bak should have been created as a side effect.
    assert not (tmp_path / "whole.py.bak").exists()
