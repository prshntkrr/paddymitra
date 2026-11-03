"""Odia text-to-speech, spoken by Sarvam AI.

WHY THE FILE IS STILL CALLED narakeet_tts
-----------------------------------------
Narakeet was the first provider and the name stuck to the module. Two callers
import `synthesize` from here by that path (`pipeline.tts_synthesize` and its
older copy), so the module keeps its name and its one-positional-argument
contract; only what happens inside changed. Renaming it would be a rename of
call sites in files this change has no business touching.

THE KEY IS NOT IN THIS FILE
---------------------------
It used to be. A subscription key sat in the source as a literal, which means
it was readable by anyone with the repository and travelled into every clone
and branch. It now comes from SARVAM_API_KEY — the same setting the Sarvam
speech-to-text path already reads, so one key configures both directions — via
`settings` first (python-decouple reads `.env` and the real environment) and the
process environment second, so the module also works outside Django.

FAILURES ARE LOUD
-----------------
A farmer who gets no audio has still got the written answer; a farmer who gets
a half-second of silence believes the system answered and heard nothing. So
nothing here returns empty or placeholder audio: a missing key, an API error, or
a response with no audio all raise, and the pipeline records the failure against
the query (see `tts_synthesize`'s caller, which already catches and traces it).
"""
from __future__ import annotations

import base64
import binascii
import logging
import os

log = logging.getLogger("irri")

# The SDK's own vocabulary, and the values this deployment speaks with. All four
# are overridable through settings, so a change of voice does not need a code
# change — see irri_ai/settings/base.py.
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "shubh"
DEFAULT_LANGUAGE_CODE = "od-IN"
# Slightly under natural speed: the answers carry doses and stage names, and a
# farmer hears them once. This is what the previous implementation used.
DEFAULT_PACE = 0.9


class SarvamNotConfigured(RuntimeError):
    """No SARVAM_API_KEY, so no speech can be made."""


class SarvamTTSError(RuntimeError):
    """Sarvam was called and did not return usable audio."""


def _setting(name: str, default):
    try:
        from django.conf import settings
        value = getattr(settings, name, None)
    except Exception:                                       # noqa: BLE001
        value = None                                        # used outside Django
    return default if value in (None, "") else value


def _api_key() -> str:
    """The subscription key, from settings or the environment. Never logged."""
    key = (_setting("SARVAM_API_KEY", "") or os.environ.get("SARVAM_API_KEY", "")).strip()
    if not key:
        raise SarvamNotConfigured(
            "SARVAM_API_KEY is not set. Add it to the environment (or .env) to "
            "enable Odia speech; see .env.example.")
    return key


def _redact(message: str, secret: str) -> str:
    """A message safe to log or re-raise.

    The key is a header, so it should never come back in an error — "should
    never" is not a guarantee worth taking with a credential, and an exception
    string ends up in logs, in a pipeline trace, and on a page.
    """
    text = str(message)
    return text.replace(secret, "***") if secret else text


def _new_client(api_key: str):
    """A Sarvam client. Imported lazily so the module loads without the SDK."""
    try:
        from sarvamai import SarvamAI
    except ImportError as exc:                              # pragma: no cover
        raise SarvamNotConfigured(
            "The sarvamai package is not installed; run "
            "`pip install sarvamai` to enable Sarvam speech.") from exc
    return SarvamAI(api_subscription_key=api_key)


# One client per key. Building it is cheap but not free, and the key can change
# between calls in tests and after a settings reload, so the key itself is the
# cache key — the client is reused only while the key it was built with is the
# one in force.
_client_cache: dict = {}


def _client(api_key: str):
    client = _client_cache.get(api_key)
    if client is None:
        client = _new_client(api_key)
        _client_cache[api_key] = client
    return client


def reset_client_cache() -> None:
    """Forget the cached client — after a key change, or between tests."""
    _client_cache.clear()


def synthesize(
    text,
    voice: str = DEFAULT_SPEAKER,
    target_language_code: str = DEFAULT_LANGUAGE_CODE,
    *,
    model: str | None = None,
    pace: float | None = None,
) -> bytes:
    """Speak `text` and return the audio bytes.

    The signature is the one the callers already use: `synthesize(text)` with
    the voice and language defaulted, and `voice` still naming what the SDK
    calls a speaker.

    Raises `ValueError` for empty text, `SarvamNotConfigured` when there is no
    key, and `SarvamTTSError` when Sarvam returns no usable audio. It never
    returns silence.
    """
    spoken = (text or "").strip() if isinstance(text, str) else ""
    if not spoken:
        raise ValueError("Nothing to synthesize: the text is empty.")

    api_key = _api_key()
    speaker = voice or _setting("SARVAM_TTS_SPEAKER", DEFAULT_SPEAKER)
    language = target_language_code or _setting("SARVAM_TTS_LANGUAGE_CODE",
                                                DEFAULT_LANGUAGE_CODE)
    chosen_model = model or _setting("SARVAM_TTS_MODEL", DEFAULT_MODEL)
    chosen_pace = pace if pace is not None else _setting("SARVAM_TTS_PACE", DEFAULT_PACE)

    # Only parameters the installed SDK declares (sarvamai 0.1.28,
    # TextToSpeechClient.convert): text, target_language_code, speaker, model,
    # pace. Nothing is passed that the SDK would reject.
    try:
        response = _client(api_key).text_to_speech.convert(
            text=spoken,
            language_code=language,
            speaker=speaker,
            model=chosen_model,
            pace=float(chosen_pace),
            output_audio_codec="mp3",
        )
    except Exception as exc:                                # noqa: BLE001
        log.warning("sarvam tts: request failed (%s, %s): %s",
                    chosen_model, language, _redact(exc, api_key))
        raise SarvamTTSError(
            f"Sarvam text-to-speech failed: {_redact(exc, api_key)}") from None

    audios = getattr(response, "audios", None) or []
    if not audios:
        raise SarvamTTSError("Sarvam returned no audio for this text.")

    try:
        audio_bytes = base64.b64decode(audios[0])
    except (binascii.Error, TypeError, ValueError) as exc:
        raise SarvamTTSError(f"Sarvam returned audio that is not base64: {exc}") from None
    if not audio_bytes:
        raise SarvamTTSError("Sarvam returned an empty audio track.")

    log.info("sarvam tts: %d char(s) -> %d byte(s) (%s, %s, speaker=%s)",
             len(spoken), len(audio_bytes), chosen_model, language, speaker)
    return audio_bytes
