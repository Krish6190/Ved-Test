"""Tests for the thread management backend in chatbot.py and command_processor.py.

These tests isolate all file I/O to a per-test tmp_path by monkeypatching
`Path` inside the `chatbot` module so that `Path(__file__).resolve().parent`
returns the tmp directory. This prevents any writes to the real
`data/threads.json`.
"""
from pathlib import Path as RealPath
import json
import pytest
import chatbot
from chatbot import Chatbot
from langchain_core.messages import HumanMessage, SystemMessage


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
# Test 1: create_thread assigns unique ids
# ---------------------------------------------------------------------------
def test_create_thread_assigns_unique_ids(make_chatbot):
    c = make_chatbot()
    id_a = c.create_thread()
    id_b = c.create_thread()
    assert isinstance(id_a, str) and id_a
    assert isinstance(id_b, str) and id_b
    assert id_a != id_b


# ---------------------------------------------------------------------------
# Test 2: switch_thread changes the active thread
# ---------------------------------------------------------------------------
def test_switch_thread_changes_active(make_chatbot):
    c = make_chatbot()
    a = c.create_thread()
    b = c.create_thread()
    # After create_thread(b), b is active. Switch back to a explicitly.
    assert c.switch_thread(a) is True
    assert c.get_active_thread()["id"] == a
    assert c.switch_thread(b) is True
    assert c.get_active_thread()["id"] == b


# ---------------------------------------------------------------------------
# Test 3: deleting the active thread falls back to the oldest remaining
# ---------------------------------------------------------------------------
def test_delete_active_falls_back_to_oldest(make_chatbot):
    c = make_chatbot()
    # The starter thread is the oldest; create A explicitly with a forced older ts.
    a = c.create_thread()
    # Force A's created_at to be the oldest.
    c._threads[a]["created_at"] = 1.0
    b = c.create_thread()
    c._threads[b]["created_at"] = 2.0
    # b is the currently active thread (create_thread sets active).
    assert c.get_active_thread()["id"] == b
    # Delete the active thread; should fall back to A (oldest).
    assert c.delete_thread(b) is True
    assert c.get_active_thread()["id"] == a


# ---------------------------------------------------------------------------
# Test 4: deleting the only remaining thread is refused
# ---------------------------------------------------------------------------
def test_delete_last_thread_refused(make_chatbot):
    c = make_chatbot()
    # A starter thread exists from __init__; capture its id and ensure it's the only one.
    only_id = c.get_active_thread()["id"]
    assert len(c._threads) == 1
    assert c.delete_thread(only_id) is False
    # Thread still exists.
    assert only_id in c._threads
    assert c.get_active_thread()["id"] == only_id


# ---------------------------------------------------------------------------
# Test 5: rename_thread updates the title and persists across reloads
# ---------------------------------------------------------------------------
def test_rename_updates_title_and_persists(make_chatbot, tmp_path):
    c = make_chatbot()
    tid = c.get_active_thread()["id"]
    assert c.rename_thread(tid, "my-renamed-thread") is True
    assert c._threads[tid]["title"] == "my-renamed-thread"
    c._save_threads()

    # Re-instantiate over the same tmp_path; the new instance must load the title back.
    c2 = make_chatbot()
    # Find the thread by title.
    titles = [t["title"] for t in c2.list_threads()]
    assert "my-renamed-thread" in titles


# ---------------------------------------------------------------------------
# Test 6: persistence round-trip — manually appended messages survive reload
# ---------------------------------------------------------------------------
def test_persistence_round_trip(make_chatbot):
    c = make_chatbot()
    active = c.get_active_thread()
    from langchain_core.messages import HumanMessage, AIMessage

    active["messages"].append(HumanMessage(content="hello there"))
    active["messages"].append(AIMessage(content="hi!"))
    c._save_threads()

    # Fresh instance over the same tmp_path.
    c2 = make_chatbot()
    new_active = c2.get_active_thread()
    assert len(new_active["messages"]) == 2
    types = [type(m).__name__ for m in new_active["messages"]]
    assert "HumanMessage" in types
    assert "AIMessage" in types


# ---------------------------------------------------------------------------
# Test 7: _autotitle_from_message trims long content and keeps short content
# ---------------------------------------------------------------------------
def test_first_message_autotitles(make_chatbot):
    c = make_chatbot()
    long_title = c._autotitle_from_message("a" * 100)
    assert len(long_title) == 40
    assert long_title == "a" * 40

    short_title = c._autotitle_from_message("short")
    assert short_title == "short"


# ---------------------------------------------------------------------------
# Test 8: command router produces expected strings for thread commands
# ---------------------------------------------------------------------------
def test_commands_router(make_chatbot):
    c = make_chatbot()
    # Create a second thread so /switch and /delete have something to target.
    c.create_thread()

    # /new
    r_new = c.handle_command("/new")
    assert isinstance(r_new, str) and r_new.startswith("Created thread")

    # /threads
    r_threads = c.handle_command("/threads")
    assert isinstance(r_threads, str) and r_threads.startswith("Threads:")

    # /switch 1 — references the first thread (oldest)
    r_switch = c.handle_command("/switch 1")
    assert isinstance(r_switch, str) and r_switch.startswith("Switched")

    # /rename
    r_rename = c.handle_command("/rename my-renamed-thread")
    assert isinstance(r_rename, str) and "Renamed" in r_rename

    # /delete 1 — delete the first (oldest) thread; there are 3+ threads so this is allowed
    r_delete = c.handle_command("/delete 1")
    assert isinstance(r_delete, str) and "Deleted" in r_delete

    # /clear
    r_clear = c.handle_command("/clear")
    assert isinstance(r_clear, str) and "Cleared" in r_clear


# ---------------------------------------------------------------------------
# Test 9: save format is a dict keyed by thread id
# ---------------------------------------------------------------------------
def test_save_format_is_dict_keyed_by_id(make_chatbot):
    c = make_chatbot()
    starter_id = c.get_active_thread()["id"]
    id_a = c.create_thread("alpha")
    id_b = c.create_thread("beta")
    c._save_threads()

    raw = RealPath(c.threads_db_path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == {starter_id, id_a, id_b}


# ---------------------------------------------------------------------------
# Test 10: save output is pretty-printed (multi-line)
# ---------------------------------------------------------------------------
def test_save_output_is_pretty_printed(make_chatbot):
    c = make_chatbot()
    tid = c.create_thread("pretty")
    c._threads[tid]["messages"].append(HumanMessage(content="hi there"))
    c._save_threads()

    raw_bytes = RealPath(c.threads_db_path).read_bytes()
    assert b"\n" in raw_bytes
    assert raw_bytes.count(b"\n") >= 1


# ---------------------------------------------------------------------------
# Test 11: _load_threads tolerates the OLD list-format JSON
# ---------------------------------------------------------------------------
def test_load_tolerates_old_list_format(make_chatbot, tmp_path):
    data_dir = RealPath(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "threads.json").write_text(
        json.dumps([
            {"id": "thr_aaa", "title": "old-a", "created_at": 1.0,
             "messages": [{"role": "human", "content": "hi"}]},
            {"id": "thr_bbb", "title": "old-b", "created_at": 2.0,
             "messages": [{"role": "human", "content": "yo"}, {"role": "ai", "content": "hello"}]},
        ]),
        encoding="utf-8",
    )

    c = make_chatbot()
    assert "thr_aaa" in c._threads
    assert "thr_bbb" in c._threads
    msgs_a = c._threads["thr_aaa"]["messages"]
    msgs_b = c._threads["thr_bbb"]["messages"]
    assert len(msgs_a) == 1
    assert msgs_a[0].content == "hi"
    assert len(msgs_b) == 2
    assert msgs_b[0].content == "yo"
    assert msgs_b[1].content == "hello"


# ---------------------------------------------------------------------------
# Test 12: message cap is enforced on save (100 -> 40)
# ---------------------------------------------------------------------------
def test_message_cap_enforced_on_save(make_chatbot, tmp_path):
    c = make_chatbot()
    tid = c.create_thread("cap-test")
    c._threads[tid]["messages"] = [HumanMessage(content=f"msg-{i}") for i in range(100)]
    c._save_threads()

    c2 = make_chatbot()
    # The thread id may have changed (it persists), so find the thread by title.
    target = next(t for tid_, t in c2._threads.items() if t["title"] == "cap-test")
    assert len(target["messages"]) == 40


# ---------------------------------------------------------------------------
# Test 13: message cap keeps the system prompt first
# ---------------------------------------------------------------------------
def test_message_cap_keeps_system_prompt(make_chatbot, tmp_path):
    c = make_chatbot()
    tid = c.create_thread("sys-cap")
    c._threads[tid]["messages"] = [SystemMessage(content="you are ved")] + [
        HumanMessage(content=f"h-{i}") for i in range(100)
    ]
    c._save_threads()

    c2 = make_chatbot()
    target = next(t for tid_, t in c2._threads.items() if t["title"] == "sys-cap")
    msgs = target["messages"]
    assert len(msgs) == 40
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "you are ved"


# ---------------------------------------------------------------------------
# Test 14: oversized existing data on disk is trimmed on load
# ---------------------------------------------------------------------------
def test_message_cap_trims_oversized_existing_data_on_load(make_chatbot, tmp_path):
    data_dir = RealPath(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    big_msgs = [{"role": "human", "content": f"m{i}"} for i in range(100)]
    (data_dir / "threads.json").write_text(
        json.dumps({
            "thr_big": {
                "id": "thr_big",
                "title": "huge",
                "created_at": 1.0,
                "messages": big_msgs,
            }
        }),
        encoding="utf-8",
    )

    c = make_chatbot()
    assert "thr_big" in c._threads
    msgs = c._threads["thr_big"]["messages"]
    assert len(msgs) == 40
    # Most recent messages should be preserved.
    assert msgs[-1].content == "m99"
