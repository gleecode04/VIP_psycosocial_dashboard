"""Shared text-to-speech helpers for posture_live_replay.py.

Intentionally duplicated from ../biomarkers/tts_utils.py (not imported across
the posture/biomarkers folder boundary) so each pathway's folder stays
self-contained and runnable on its own without sys.path tricks. It's a small,
dependency-free leaf module; if it ever needs real changes, change both
copies.

Uses Kokoro-82M (Apache 2.0) -- a free, local, offline neural TTS model with
far more natural prosody than pyttsx3's system voices. No API key and no
network needed after the one-time model download (~330 MB from Hugging Face).
Audio is synthesized to a temp wav and played with the platform's native
player, so call sites don't need to know anything changed:
init_engine() + speak(engine, text) still work the same.

Requires the espeak-ng system package (macOS: `brew install espeak-ng`).
"""
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from kokoro import KPipeline

_SAMPLE_RATE = 24000
_LANG_AMERICAN = "a"
_DEFAULT_VOICE = "af_bella"
_VOICE_NAME_HINTS = {
    "bella": "af_bella",
    "heart": "af_heart",
    "nicole": "af_nicole",
    "sarah": "af_sarah",
    "sky": "af_sky",
    "michael": "am_michael",
    "adam": "am_adam",
}
_DEFAULT_MALE_VOICE = "am_michael"
_BASELINE_RATE = 175  # words/min the old pyttsx3 call sites passed in


class _KokoroEngine:
    """Drop-in replacement for the pyttsx3 engine object."""

    def __init__(self, voice, rate, volume):
        self.voice = voice
        self.speed = max(0.5, min(rate / _BASELINE_RATE, 2.0))
        self.volume = volume
        self._pipeline = KPipeline(lang_code=_LANG_AMERICAN, repo_id="hexgrad/Kokoro-82M")
        self._pending_text = ""

    def _player_command(self, path):
        system = platform.system()
        if system == "Darwin":
            return ["afplay", path]
        if system == "Linux":
            return ["aplay", "-q", path]
        if system == "Windows":
            return [
                "powershell", "-c",
                f"(New-Object Media.SoundPlayer '{path}').PlaySync();",
            ]
        raise RuntimeError(f"No audio player configured for platform: {system}")

    def _synthesize(self, text, path):
        chunks = [
            np.asarray(audio)
            for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed)
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for text: {text!r}")
        sf.write(path, np.concatenate(chunks), _SAMPLE_RATE)

    def say(self, text):
        self._pending_text = text

    def runAndWait(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            self._synthesize(self._pending_text, path)
            if self.volume > 0:
                subprocess.run(self._player_command(path), check=True)
        finally:
            os.unlink(path)


def init_engine(rate=175, volume=None, prefer_female=True, name_hint=None):
    """Build a Kokoro engine. Default voice is American female "Bella".

    volume defaults to the TTS_VOLUME env var (or 1.0) so automated tests can
    turn audio down (0 = synthesize but don't play) without touching call
    sites. name_hint accepts a short name ("bella", "michael") or any full
    Kokoro voice id (e.g. "af_sarah").
    """
    if volume is None:
        volume = float(os.environ.get("TTS_VOLUME", "1.0"))

    voice = _DEFAULT_VOICE
    if name_hint:
        voice = _VOICE_NAME_HINTS.get(name_hint.lower(), name_hint)
    elif not prefer_female:
        voice = _DEFAULT_MALE_VOICE

    engine = _KokoroEngine(voice, rate, volume)
    print(f"[tts] Using voice: {voice}", file=sys.stderr)
    return engine


def speak(engine, text):
    print(f"[tts] {text}")
    engine.say(text)
    engine.runAndWait()
