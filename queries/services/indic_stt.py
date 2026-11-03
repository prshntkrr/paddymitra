"""
Odia speech-to-text.

Primary path: **Sarvam AI** hosted ASR (Indic-tuned). Used whenever
`SARVAM_API_KEY` is set in `.env`.

Fallback path: **local Wav2Vec2** (`Harveenchadha/odia_large_wav2vec2`).
Offline, but slow on CPU (~5–20 s per query). Engages automatically if the
Sarvam call fails (network error, 4xx/5xx, empty transcript, or no key set).

Pipeline:
    audio (any container) ─ffmpeg→ WAV 16 kHz mono ─→ Sarvam | wav2vec2 ─→ Odia text

Called from `queries.services.pipeline.stt_transcribe` only when the user's
selected language is Odia. All other languages still use OpenAI Whisper.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import requests
from django.conf import settings

log = logging.getLogger("irri")

def _bool_setting(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class NotConfigured(RuntimeError):
    """Raised when neither path is available."""


# ---------- Sarvam (primary) ----------

def _sarvam_transcribe(wav_path: str) -> str:
    api_key = getattr(settings, "SARVAM_API_KEY", "") or ""
    if not api_key:
        raise NotConfigured("SARVAM_API_KEY not set; cannot use Sarvam STT.")
    url = getattr(settings, "SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
    model = getattr(settings, "SARVAM_STT_MODEL", "saaras:v3")
    language_code = getattr(settings, "SARVAM_STT_LANGUAGE_CODE", "od-IN")
    mode = getattr(settings, "SARVAM_STT_MODE", "transcribe")

    with open(wav_path, "rb") as fh:
        files = {"file": ("audio.wav", fh, "audio/wav")}
        data = {"model": model, "language_code": language_code}
        if model.startswith("saaras:") and mode:
            data["mode"] = mode
        headers = {"api-subscription-key": api_key}
        log.info("Sarvam STT (%s): POST %s, model=%s", language_code, url, model)
        r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Sarvam returned {r.status_code}: {r.text[:400]}")
    payload = r.json() if r.content else {}
    return _extract_sarvam_transcript(payload)


def _extract_sarvam_transcript(payload: dict) -> str:
    """Return transcript text across Sarvam REST response variants."""
    for key in ("transcript", "text"):
        text = payload.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()

    results = payload.get("results") or payload.get("transcripts")
    if isinstance(results, list):
        parts = []
        for item in results:
            if not isinstance(item, dict):
                continue
            text = item.get("transcript") or item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return " ".join(parts)

    diarized = payload.get("diarized_transcript")
    if isinstance(diarized, dict) and isinstance(diarized.get("entries"), list):
        parts = []
        for item in diarized["entries"]:
            if not isinstance(item, dict):
                continue
            text = item.get("transcript") or item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return " ".join(parts)

    return ""


# ---------- Local Wav2Vec2 (fallback) ----------

_LOCAL_MODEL_ID = "Harveenchadha/odia_large_wav2vec2"
_pipe = None


def _get_local_pipe():
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline  # lazy import
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False
        except ImportError as exc:
            raise NotConfigured(
                "transformers is not installed and Sarvam is unavailable. "
                "Run `pip install transformers torch torchaudio`."
            ) from exc
        log.info("Loading local Odia Wav2Vec2 fallback %s …", _LOCAL_MODEL_ID)
        # Use GPU if available (device=0), otherwise CPU (device=-1)
        device = 0 if cuda_available else -1
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=_LOCAL_MODEL_ID,
            device=device,
        )
        print(f"[STT] Local Wav2Vec2 loaded: {_LOCAL_MODEL_ID} (device={'cuda' if cuda_available else 'cpu'})")
    return _pipe


def _local_transcribe(wav_path: str) -> str:
    pipe = _get_local_pipe()
    log.info("Local Wav2Vec2 transcribing %s", wav_path)
    print(f"[STT] Transcribing locally with {_LOCAL_MODEL_ID} — processing {wav_path}")
    # Let the HF pipeline chunk long audio; chunk_length_s improves memory usage
    try:
        result = pipe(wav_path, chunk_length_s=30)
    except TypeError:
        # Older pipeline versions may not support chunking; fall back
        result = pipe(wav_path)

    # pipeline may return a dict, a list of dicts, or a simple dict-like result
    text = ""
    if isinstance(result, dict):
        # common case: {'text': '...'}
        text = result.get("text") or ""
    elif isinstance(result, list):
        # chunked results — join per-chunk text
        texts = [r.get("text") if isinstance(r, dict) else str(r) for r in result]
        text = " ".join(t for t in texts if t)
    else:
        try:
            text = str(result)
        except Exception:
            text = ""

    return text.strip()


# ---------- Common ffmpeg helper ----------

def _to_wav_16k_mono(src_path: str) -> str:
    """Convert any audio to 16 kHz mono WAV via ffmpeg. Returns a temp path."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000",
             "-f", "wav", out],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed. Install it "
            "(Windows: choco install ffmpeg, Linux: apt install ffmpeg)."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed: {exc.stderr.decode('utf-8', errors='ignore')[:300]}"
        )
    return out


# ---------- Public entry point ----------

def transcribe_odia(audio_path: str) -> str:
    """Transcribe an Odia audio file. Returns plain text in Odia script.

    Tries Sarvam first (fast); falls back to local Wav2Vec2 on any failure.
    Set FORCE_LOCAL_ODIA_STT=1 in `.env` to skip Sarvam entirely and test only
    the local Wav2Vec2 model.
    """
    force_local_only = _bool_setting(getattr(settings, "FORCE_LOCAL_ODIA_STT", False))
    wav_path = _to_wav_16k_mono(audio_path)
    try:
        if not force_local_only:
            # 1) Sarvam
            try:
                text = _sarvam_transcribe(wav_path)
                if text:
                    print("[STT] Transcribed via Sarvam API")
                    return text
                log.warning("Sarvam returned empty transcript; falling back to local model.")
                print("[STT] Sarvam returned empty transcript; falling back to local model")
            except NotConfigured as exc:
                log.info("%s — using local Wav2Vec2 fallback.", exc)
                print(f"[STT] {exc} — using local Wav2Vec2 fallback")
            except Exception as exc:
                log.warning("Sarvam STT failed (%s); falling back to local model.", exc)
                print(f"[STT] Sarvam STT failed ({exc}); falling back to local model")
        else:
            print("[STT] FORCE_LOCAL_ODIA_STT enabled — skipping Sarvam and using local Wav2Vec2")

        # 2) Local Wav2Vec2 fallback
        print("[STT] Using local Wav2Vec2 fallback model")
        return _local_transcribe(wav_path)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
