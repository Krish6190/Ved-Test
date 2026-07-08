"""Unit tests for the wake-word hysteresis + cooldown helper (Chunk A).

The helper `evaluate_wake_hit` in `voice/audio_loop.py` is pure (no I/O,
no audio device, no clock), so we can exhaustively exercise the state
machine with table-driven pytest cases.

Run with: `cd /mnt/c/Users/krish/OneDrive/Desktop/ved && python -m pytest tests/test_wake_hysteresis.py -v`
"""
import pytest

from voice.audio_loop import (
    WAKE_THRESHOLD,
    WAKE_REQUIRED_HITS,
    WAKE_COOLDOWN_SECONDS,
    evaluate_wake_hit,
)


# ---------------------------------------------------------------------------
# Constants sanity (locks in the values the plan explicitly chose)
# ---------------------------------------------------------------------------

def test_constants_match_plan():
    assert WAKE_THRESHOLD == 0.45
    assert WAKE_REQUIRED_HITS == 3
    assert WAKE_COOLDOWN_SECONDS == 2.0


# ---------------------------------------------------------------------------
# Acceptance criterion 1: a single spike above threshold does NOT trigger.
# ---------------------------------------------------------------------------

def test_single_spike_above_threshold_does_not_trigger():
    # Frame 1: above threshold -> hit counter becomes 1, no trigger.
    hits, triggered = evaluate_wake_hit(
        score=0.9,
        current_hits=0,
        now=1000.0,
        cooldown_until=0.0,
    )
    assert triggered is False
    assert hits == 1

    # Frame 2: drops below threshold -> counter resets to 0.
    hits, triggered = evaluate_wake_hit(
        score=0.1,
        current_hits=1,
        now=1000.08,
        cooldown_until=0.0,
    )
    assert triggered is False
    assert hits == 0


# ---------------------------------------------------------------------------
# Acceptance criterion 2: three consecutive above-threshold frames trigger
# exactly on the 3rd frame.
# ---------------------------------------------------------------------------

def test_three_consecutive_frames_trigger_on_third():
    now = 1000.0
    # Frame 1
    hits, triggered = evaluate_wake_hit(0.9, 0, now, 0.0)
    assert (hits, triggered) == (1, False)
    # Frame 2
    hits, triggered = evaluate_wake_hit(0.9, 1, now + 0.08, 0.0)
    assert (hits, triggered) == (2, False)
    # Frame 3 -> triggers
    hits, triggered = evaluate_wake_hit(0.9, 2, now + 0.16, 0.0)
    assert triggered is True
    assert hits == 3  # helper returns the hit count at the moment of trigger


# ---------------------------------------------------------------------------
# Acceptance criterion 3: alternating above/below never accumulates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", [
    # All patterns below: no run of 3+ consecutive frames above threshold,
    # AND they end on a low frame so the trailing hits==0 invariant holds.
    [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],     # alternating high/low, resets every other frame
    [0.5, 0.6, 0.1, 0.7, 0.8, 0.1],    # mid-stream reset caps any streak at 2
    [0.46, 0.46, 0.1, 0.46, 0.46, 0.1],  # just-above-threshold scores behave identically
])
def test_alternating_patterns_never_trigger(pattern):
    hits = 0
    triggered_any = False
    for i, score in enumerate(pattern):
        hits, triggered = evaluate_wake_hit(
            score=score,
            current_hits=hits,
            now=1000.0 + i * 0.08,
            cooldown_until=0.0,
        )
        if triggered:
            triggered_any = True
    assert triggered_any is False
    assert hits == 0  # always ends reset after the last low frame


# ---------------------------------------------------------------------------
# Acceptance criterion 4: once triggered, cooldown suppresses further hits.
# Mirrors the real loop: caller resets _wake_hits=0 and arms cooldown.
# ---------------------------------------------------------------------------

def test_cooldown_suppresses_subsequent_high_scores():
    cooldown_until = 1000.0 + WAKE_COOLDOWN_SECONDS  # 1002.0

    # Inside the cooldown window (1002.0 not yet reached).
    for offset in (0.1, 0.5, 1.0, 1.5, 1.99):
        hits, triggered = evaluate_wake_hit(
            score=0.95,
            current_hits=0,
            now=1000.0 + offset,
            cooldown_until=cooldown_until,
        )
        assert triggered is False
        assert hits == 0  # counter not advanced while in cooldown


def test_cooldown_suppresses_high_scores_even_if_streak_was_building():
    # Simulates the real-world case where the model's own TTS echo keeps
    # scoring high for several frames. The cooldown must short-circuit
    # BEFORE the streak accumulates.
    cooldown_until = 1000.0 + WAKE_COOLDOWN_SECONDS
    hits = 2  # pretend we already had 2 hits before the cooldown armed
    for offset in (0.05, 0.10, 0.15):
        hits, triggered = evaluate_wake_hit(
            score=0.95,
            current_hits=hits,
            now=1000.0 + offset,
            cooldown_until=cooldown_until,
        )
        assert triggered is False
        # Cooldown path does not increment or reset the streak; the caller
        # is expected to keep `hits` as-is until cooldown elapses.
        assert hits == 2


# ---------------------------------------------------------------------------
# Acceptance criterion 5: after cooldown elapses, normal detection resumes.
# Caller has already reset _wake_hits=0 (as the production loop does).
# ---------------------------------------------------------------------------

def test_detection_resumes_after_cooldown():
    cooldown_until = 1000.0 + WAKE_COOLDOWN_SECONDS  # 1002.0

    # Just inside the cooldown window: still suppressed by strict `<` check
    # (must use a timestamp STRICTLY less than cooldown_until, otherwise the
    # helper falls through to the score check and starts the streak at 1).
    hits, triggered = evaluate_wake_hit(0.9, 0, 1001.999, cooldown_until)
    assert (hits, triggered) == (0, False)

    # Exactly at the boundary: cooldown has elapsed (strict `<` fails), so
    # this is the first post-cooldown frame and the streak begins at 1.
    hits, triggered = evaluate_wake_hit(0.9, 0, 1002.0, cooldown_until)
    assert (hits, triggered) == (1, False)

    # Second post-cooldown frame: streak accumulates.
    hits, triggered = evaluate_wake_hit(0.9, 1, 1002.081, cooldown_until)
    assert (hits, triggered) == (2, False)

    # Third post-cooldown frame: streak reaches 3 and triggers.
    hits, triggered = evaluate_wake_hit(0.9, 2, 1002.161, cooldown_until)
    assert (hits, triggered) == (3, True)


# ---------------------------------------------------------------------------
# Boundary: score exactly AT threshold does NOT count as a hit (strict `>`).
# ---------------------------------------------------------------------------

def test_score_at_threshold_does_not_count():
    hits, triggered = evaluate_wake_hit(
        score=WAKE_THRESHOLD,  # exactly equal
        current_hits=2,
        now=1000.0,
        cooldown_until=0.0,
    )
    assert triggered is False
    assert hits == 0  # reset because the strict `>` check failed


# ---------------------------------------------------------------------------
# Table-driven: full enumeration of small input windows.
# This is the regression net for the state machine.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scores, expected_trigger_frame", [
    ([0.1, 0.1, 0.1], None),
    ([0.9, 0.1, 0.9], None),
    ([0.9, 0.9, 0.1], None),
    ([0.9, 0.9, 0.9], 2),  # 0-indexed: third frame
    ([0.1, 0.9, 0.9, 0.9], 3),
    ([0.9, 0.9, 0.9, 0.9], 2),  # triggers on the earliest possible frame
])
def test_trigger_timing_table(scores, expected_trigger_frame):
    hits = 0
    trigger_frame = None
    for i, s in enumerate(scores):
        hits, triggered = evaluate_wake_hit(s, hits, 1000.0 + i * 0.08, 0.0)
        if triggered and trigger_frame is None:
            trigger_frame = i
    assert trigger_frame == expected_trigger_frame
