"""Unit tests for the ambient-noise calibration helper (Chunk B).

The helper `_compute_silence_threshold` in `voice/voice_module.py` is pure
(no I/O, no audio device, no clock), so we can exhaustively exercise the
math with table-driven pytest cases.

`VoiceSystem.calibrate_ambient_noise` is also exercised via its optional
`rms_samples` test seam to confirm it actually updates `self.silence_threshold`.

Run with: `python -m pytest tests/test_ambient_calibration.py -v`
"""
import numpy as np
import pytest

from voice.voice_module import _compute_silence_threshold


# ---------------------------------------------------------------------------
# Acceptance criterion 1: empty samples -> floor (80).
# ---------------------------------------------------------------------------

def test_empty_samples_returns_floor():
    """No samples means we cannot measure noise; fall back to the floor."""
    assert _compute_silence_threshold([]) == 80


def test_empty_iterator_returns_floor():
    """A consumed-empty iterator must also return the floor (not raise)."""
    assert _compute_silence_threshold(iter([])) == 80


# ---------------------------------------------------------------------------
# Acceptance criterion 2: constant RMS=50 -> max(int(50 * 1.8), 80) = 90.
# ---------------------------------------------------------------------------

def test_constant_rms_fifty_returns_ninety_single_sample():
    # Single sample -> np.percentile([50], 75) == 50 -> max(90, 80) == 90.
    assert _compute_silence_threshold([50.0]) == 90


def test_constant_rms_fifty_returns_ninety_long_run():
    # 12 samples all equal to 50 -> same result, regardless of length.
    assert _compute_silence_threshold([50.0] * 12) == 90


# ---------------------------------------------------------------------------
# Acceptance criterion 3: a single transient pop is dampened by the 75th
# percentile, so the resulting threshold stays near the room noise level
# rather than shooting up toward the spike.
#
# numpy's default linear interpolation on [10, 10, 200, 10] gives:
#   sorted = [10, 10, 10, 200]; position = 0.75 * 3 = 2.25
#   value = 10 + 0.25 * (200 - 10) = 57.5
#   threshold = max(int(57.5 * 1.8), 80) = max(103, 80) = 103
#
# Note: the original plan text claimed the percentile would be 50 and the
# threshold 90; that arithmetic was off. The behaviour asserted here is what
# `np.percentile(samples, 75)` actually produces, which is what the plan
# explicitly specifies we should call.
# ---------------------------------------------------------------------------

def test_transient_spike_dampened_by_75th_percentile():
    threshold = _compute_silence_threshold([10.0, 10.0, 200.0, 10.0])
    assert threshold == 103
    # Sanity: the spike (200) would have produced ~360 if we naively used
    # the max. Assert the 75th-percentile path keeps the threshold well below
    # that, so a single pop cannot dominate the energy gate.
    naive_max_threshold = max(int(max([10.0, 10.0, 200.0, 10.0]) * 1.8), 80)
    assert threshold < naive_max_threshold
    assert naive_max_threshold == 360


# ---------------------------------------------------------------------------
# Acceptance criterion 4: loud room RMS=300 -> int(300 * 1.8) = 540.
# ---------------------------------------------------------------------------

def test_loud_room_rms_three_hundred_returns_540():
    assert _compute_silence_threshold([300.0]) == 540
    assert _compute_silence_threshold([300.0] * 12) == 540


# ---------------------------------------------------------------------------
# Monotonicity: a louder room should never produce a lower threshold than
# a quieter one (the function is non-decreasing in the percentile).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quiet, loud", [
    ([10.0] * 12, [50.0] * 12),
    ([50.0] * 12, [300.0] * 12),
    ([10.0, 10.0, 200.0, 10.0], [300.0, 300.0, 300.0, 300.0]),
])
def test_threshold_is_non_decreasing(quiet, loud):
    assert _compute_silence_threshold(quiet) <= _compute_silence_threshold(loud)


# ---------------------------------------------------------------------------
# Floor protection: even at a very loud room, the threshold cannot drop
# below 80 (the helper raises with RMS, never lowers toward the floor for
# loud input).
# ---------------------------------------------------------------------------

def test_floor_holds_for_low_rms():
    # RMS well below 80/1.8 ~= 44.44: the floor takes over.
    assert _compute_silence_threshold([10.0]) == 80
    assert _compute_silence_threshold([30.0]) == 80  # int(30*1.8)=54 < 80
    assert _compute_silence_threshold([44.0]) == 80  # int(44*1.8)=79 < 80


def test_floor_releases_just_above_threshold():
    # At RMS ~= 44.5 the multiplier finally exceeds 80 and the floor releases.
    # int(44.5 * 1.8) = 80 -> max(80, 80) = 80 (still on the floor).
    assert _compute_silence_threshold([44.5]) == 80
    # int(45 * 1.8) = 81 -> max(81, 80) = 81 (just above the floor).
    assert _compute_silence_threshold([45.0]) == 81


# ---------------------------------------------------------------------------
# Integration: VoiceSystem.calibrate_ambient_noise(rms_samples=...) writes
# the computed threshold to the instance, without requiring a real audio
# device. We invoke the unbound function against a plain object to avoid
# the heavy __init__ (which loads Whisper/Piper/OWW models).
# ---------------------------------------------------------------------------

class _CalibrationProbe:
    """Minimal stand-in: only the attributes the method touches."""

    def __init__(self):
        self.silence_threshold = None


def test_method_sets_silence_threshold_from_rms_samples():
    from voice import voice_module
    probe = _CalibrationProbe()
    # Bind the unbound method to the probe so `self` resolves correctly.
    bound = voice_module.VoiceSystem.calibrate_ambient_noise.__get__(probe)
    bound(rms_samples=[50.0] * 12)
    assert probe.silence_threshold == 90


def test_method_uses_floor_for_empty_samples():
    from voice import voice_module
    probe = _CalibrationProbe()
    bound = voice_module.VoiceSystem.calibrate_ambient_noise.__get__(probe)
    bound(rms_samples=[])
    assert probe.silence_threshold == 80


def test_method_returns_assigned_threshold():
    from voice import voice_module
    probe = _CalibrationProbe()
    bound = voice_module.VoiceSystem.calibrate_ambient_noise.__get__(probe)
    returned = bound(rms_samples=[300.0])
    assert returned == 540
    assert probe.silence_threshold == returned
