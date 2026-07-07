"""Voice package.

The ``VoiceSystem`` re-export is wrapped in a try/except so the package can
be imported (e.g. during ``pytest --collect-only``) on a slim Linux venv
that lacks the heavy voice / audio deps (``sounddevice``, ``faster_whisper``,
``openwakeword``, ``piper``, etc.). Callers that actually need ``VoiceSystem``
should import it directly from :mod:`voice.voice_module` and handle the
``ImportError`` themselves - ``ui.gui`` already does that.
"""
try:
    from .voice_module import VoiceSystem  # noqa: F401
except Exception:
    # Heavy deps (sounddevice / faster_whisper / openwakeword / piper) are
    # absent. Leave ``VoiceSystem`` undefined; the GUI entrypoint imports it
    # directly and will raise a clear error if the user actually tries to
    # launch the voice stack.
    VoiceSystem = None  # type: ignore[assignment]
