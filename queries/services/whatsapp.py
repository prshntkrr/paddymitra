"""
Picky Assist outbound sender.
Docs: https://www.pickyassist.com/docs/

WHAT CHANGED AND WHY
--------------------
Three failures, all of which lost part of the answer the pipeline had already
produced:

  * A LONG ANSWER WAS DROPPED AT THE GATEWAY. The body was cut to 4000
    characters with a slice, and WhatsApp itself refuses anything over 4096, so a
    long answer either arrived truncated mid-sentence or not at all. It is now
    split on sentence boundaries and sent as consecutive messages.

  * SENDING AUDIO REPLACED THE TEXT. `type` was switched to "audio" on the same
    payload, so the text field went unread by the gateway and a farmer who sent
    a voice note got audio with no written answer to read back. Text and audio
    are now separate messages.

  * ONE ATTEMPT ONLY. A 429 or a 502 from the gateway lost the answer outright,
    after every expensive part of the work had already been paid for. Transient
    failures are retried.

The voice file the pipeline generates for every hook message was never sent at
all: the send site passed `audio_url=None`. send_answer() below is what puts it
on the wire, and it needs PUBLIC_BASE_URL to build a URL the gateway can fetch.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests
from django.conf import settings

log = logging.getLogger("irri")

_RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)
_ATTEMPTS = 3


def _chunks(text: str, limit: int) -> list:
    """`text` in pieces of at most `limit` characters, split where a reader would.

    Sentence boundaries first, then words. A dose split across two messages
    ("apply 20-" / "25 kg/ha") is worse than a slightly shorter message, so a
    piece is never cut mid-word.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?।])\s+", text):
        while len(sentence) > limit:            # one sentence longer than a message
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def _payload(number: str, message: str, media_url: str = "") -> dict:
    """The gateway's bulk-send body.

    This is the shape the account actually accepts:

        {"token": …, "priority": "1", "application": "121",
         "globalmedia": "<url or empty>",
         "data": [{"number": "+9198…", "message": "…"}]}

    `globalmedia` carries the voice note, so the text and its audio go in ONE
    call rather than two. It is always present, empty when there is no audio,
    because that is how the receiving script is written and a missing key is one
    more thing that can quietly not work.

    `application` is the account's app id and is NOT optional — a send without it
    is accepted and delivers nothing, which is the worst possible outcome, so it
    fails loudly here instead.
    """
    application = str(getattr(settings, "PICKY_ASSIST_APPLICATION", "") or "").strip()
    if not application:
        raise RuntimeError(
            "PICKY_ASSIST_APPLICATION is not configured — the gateway needs the "
            "application id (e.g. 121) or it accepts the request and sends nothing.")
    return {
        "token": settings.PICKY_ASSIST_TOKEN,
        "priority": str(getattr(settings, "PICKY_ASSIST_PRIORITY", "1") or "1"),
        "application": application,
        "globalmedia": media_url or "",
        "data": [{"number": number, "message": message}],
    }


def _post(payload: dict) -> dict:
    """One gateway call, retried while the failure still looks transient."""
    last = {}
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            r = requests.post(settings.PICKY_ASSIST_SEND_URL, json=payload, timeout=30)
        except requests.RequestException as exc:
            last = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            log.warning("PickyAssist send failed (attempt %s/%s): %s",
                        attempt, _ATTEMPTS, exc)
        else:
            try:
                data = r.json()
            except ValueError:
                data = {"status_code": r.status_code, "text": r.text[:500]}
            if r.status_code < 400:
                return {"ok": True, "response": data}
            last = {"ok": False, "status_code": r.status_code, "response": data}
            log.warning("PickyAssist error %s (attempt %s/%s): %s",
                        r.status_code, attempt, _ATTEMPTS, data)
            if r.status_code not in _RETRY_STATUS:
                return last                    # a 401 will not fix itself
        if attempt < _ATTEMPTS:
            time.sleep(2 ** (attempt - 1))     # 1s, 2s
    return last


def send_via_picky_assist(hook, to_number: str, text: str,
                          audio_url: Optional[str] = None) -> dict:
    """Send `text` (and optionally a voice note) to one number.

    Kept at this name and signature because the pipeline calls it; the behaviour
    behind it is what changed.
    """
    if not settings.PICKY_ASSIST_TOKEN:
        raise RuntimeError("PICKY_ASSIST_TOKEN not configured.")
    if not to_number:
        raise ValueError("no recipient number")

    limit = max(500, int(getattr(settings, "WHATSAPP_MAX_CHARS", 3800)))
    results = []
    parts = _chunks(text, limit)

    if not parts:
        # Audio only. Reached on the hook path, where the pipeline has already
        # sent the text and this call exists purely to carry the voice note.
        if audio_url:
            results.append(_post(_payload(to_number, "", audio_url)))
    else:
        for i, part in enumerate(parts, 1):
            # "(1/3)" so a split answer reads as one answer rather than three.
            body = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
            # The voice note rides with the text when there is only one message.
            # With several it goes on its own call below instead: `globalmedia`
            # applies to every entry in the batch, so attaching it here would send
            # the farmer the same recording two or three times.
            media = audio_url if len(parts) == 1 else ""
            results.append(_post(_payload(to_number, body, media)))
        if audio_url and len(parts) > 1:
            results.append(_post(_payload(to_number, "", audio_url)))

    ok = all(r.get("ok") for r in results) and bool(results)
    if not ok:
        log.warning("PickyAssist delivery incomplete for %s: %s", to_number, results)
    return {"ok": ok, "messages": len(results), "results": results}


def voice_url_for(query) -> str:
    """A URL the gateway can fetch for this answer's audio, or "".

    Empty unless PUBLIC_BASE_URL is set: a relative path or a localhost URL is
    one the gateway cannot resolve, and sending it produces a message the farmer
    sees as a broken attachment. Text-only is the honest fallback.
    """
    if not getattr(settings, "WHATSAPP_SEND_VOICE", True):
        return ""
    base = (getattr(settings, "PUBLIC_BASE_URL", "") or "").rstrip("/")
    if not base:
        return ""
    audio = getattr(query, "ai_response_voice_file", None)
    if not audio:
        return ""
    try:
        url = audio.url
    except ValueError:                          # no file associated
        return ""
    return url if url.startswith("http") else f"{base}{url}"


def send_answer(hook, to_number: str, query, text_already_sent: bool = False) -> dict:
    """Deliver a finished Query to the farmer: the text, then the voice note.

    `text_already_sent` exists because the pipeline sends the text itself as soon
    as it has it — the farmer should not wait for TTS to read their answer. This
    then adds only the audio, so nothing is sent twice.
    """
    answer = (getattr(query, "ai_response_text_local", "")
              or getattr(query, "ai_response_text", ""))
    audio_url = voice_url_for(query)
    if text_already_sent:
        if not audio_url:
            return {"ok": True, "messages": 0, "results": [], "skipped": "no audio url"}
        return send_via_picky_assist(hook, to_number, "", audio_url=audio_url)
    if not answer:
        return {"ok": False, "error": "no answer text to send"}
    return send_via_picky_assist(hook, to_number, answer, audio_url=audio_url or None)
