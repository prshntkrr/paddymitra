"""
hooks/api.py
============
The provider-independent WhatsApp API: POST /api/v1/whatsapp/message/

WHAT IT IS FOR
--------------
Anything that can make an HTTP request can put a farmer's question through this
system: Meta's Cloud API, Twilio, Picky Assist, a Node middleware, another
internal service. No hook, no trigger keyword, no gateway-specific field names,
no per-provider routing. One JSON contract, one API key.

WHAT IT IS NOT
--------------
It is not a second pipeline, and it does not build a hook to borrow the first
one. Validation and authentication happen here; everything after that is
hooks/service.py, which the Picky Assist webhook goes through as well.

WHY IT ANSWERS 202 AND NOT THE ANSWER
-------------------------------------
Retrieval, generation, translation and text-to-speech take tens of seconds. A
caller — and every WhatsApp gateway there is — treats a request that slow as a
failure and sends it again, which is how one question becomes three paid answers.
So the request is recorded, queued, and acknowledged; the answer goes to the
farmer over WhatsApp.
"""
from __future__ import annotations

import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import service
from .models import InboundMessage, WhatsAppApiKey
# Only the exception is imported by name. Everything else is reached through the
# `service` module on purpose: `from .service import dispatch` would bind a copy
# of the function at import time, so a test that replaces the dispatcher would be
# quietly ignored and the real one would run — which is exactly what happened, and
# put pipeline jobs onto live threads during the test run.
from .service import NotConfigured

log = logging.getLogger("irri")

PROVIDER = InboundMessage.PROVIDER_GENERIC

# Where a key may arrive. X-API-Key is the documented one; the others exist
# because callers and gateways each have their own habit, and a key that works
# only when placed in one exact header is a support ticket.
_KEY_HEADERS = ("HTTP_X_API_KEY", "HTTP_X_APIKEY", "HTTP_API_KEY")
_KEY_PARAMS = ("api_key", "apikey", "key")


def _presented_key(request, payload: dict) -> str:
    for header in _KEY_HEADERS:
        value = (request.META.get(header) or "").strip()
        if value:
            return value
    auth = (request.META.get("HTTP_AUTHORIZATION") or "").strip()
    if auth:
        # "Bearer <key>", "Token <key>", or the bare value.
        return auth.split(None, 1)[1].strip() if " " in auth else auth
    for param in _KEY_PARAMS:
        value = (request.GET.get(param) or "").strip()
        if value:
            return value
    value = payload.get("api_key")
    return value.strip() if isinstance(value, str) else ""


def _error(message: str, status: int, **extra):
    return JsonResponse({"error": message, **extra}, status=status)


def _flag(payload: dict, name: str, default: bool) -> bool:
    """A boolean from JSON true/false or from the strings a form post sends."""
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _answer_body(inbound, query, delivered: bool) -> dict:
    """The finished answer, for a caller that waited for it."""
    from queries.services.whatsapp import voice_url_for

    return {
        "status": inbound.status,
        "message_id": inbound.message_id,
        "query_id": query.pk,
        # The farmer's language first: that is the message to send them.
        "answer": query.ai_response_text_local or query.ai_response_text,
        "answer_en": query.ai_response_text,
        "question_en": query.translated_query_text,
        "language": query.original_query_language,
        "audio_url": voice_url_for(query) or None,
        "delivered": delivered,
        "response_time_seconds": query.response_time_seconds,
    }


@csrf_exempt
@require_POST
def whatsapp_message(request):
    """Accept one WhatsApp message and answer it out of band.

    Request (application/json):

        {
          "message_id": "msg_123",        # optional — a UUID is generated
          "phone":      "+919876543210",  # REQUIRED
          "message":    "How much urea per acre?",   # REQUIRED for text
          "message_type": "text",         # optional — "text" (default) or "audio"
          "audio_url":  "https://…/voice.ogg",       # REQUIRED for audio
          "language":   "or",             # optional — ISO code or name
          "catalogue":  4,                # optional — only if several exist
          "district":   "Bargarh",        # optional
          "wait":       false,            # optional — hold the connection for the answer
          "deliver":    true              # optional — false: return it, send nothing
        }

    Responses:

        202  {"status": "accepted",  "message_id": "…"}   queued (default)
        200  {"status": "answered",  "answer": "…", …}    wait=true
        200  {"status": "duplicate", "message_id": "…"}   already seen; not re-run
        400  {"error": "phone is required"}               malformed request
        401  {"error": "…"}                               bad or missing key
        500  {"error": "…", "message_id": "…"}            wait=true and it failed
        503  {"error": "…"}                               deployment not configured
    """
    # ── the body ──────────────────────────────────────────────────────────
    content_type = (request.content_type or "").split(";")[0].strip()
    if content_type == "application/json" or not request.POST:
        try:
            payload = json.loads((request.body or b"{}").decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error("body is not valid JSON", 400)
    else:
        payload = dict(request.POST.items())
    if not isinstance(payload, dict):
        return _error("body must be a JSON object", 400)

    # ── authentication ────────────────────────────────────────────────────
    presented = _presented_key(request, payload)
    if not presented:
        return _error("missing API key — send it as 'X-API-Key: <key>' or "
                      "'Authorization: Bearer <key>'", 401)
    key = WhatsAppApiKey.authenticate_generic(presented)
    if key is None:
        # Deliberately one message for every way a key can be unusable: telling a
        # caller whether a key is unknown, revoked or merely expired tells anyone
        # holding a stolen key how far they got.
        log.warning("generic whatsapp api: rejected key %s…", presented[:12])
        return _error("invalid API key", 401)

    # ── validation ────────────────────────────────────────────────────────
    phone = str(payload.get("phone") or payload.get("from") or "").strip()
    if not phone:
        return _error("phone is required", 400)

    message_type = str(payload.get("message_type") or "text").strip().lower()
    if message_type not in ("text", "audio"):
        return _error("message_type must be 'text' or 'audio'", 400)

    text = str(payload.get("message") or payload.get("text") or "").strip()
    audio_url = str(payload.get("audio_url") or payload.get("media_url") or "").strip()

    if message_type == "audio":
        if not audio_url:
            return _error("audio_url is required when message_type is 'audio'", 400)
    elif not text:
        return _error("message is required", 400)

    # Fail here rather than on the worker: a deployment that cannot answer
    # anything should say so while someone is still looking at the response,
    # instead of accepting the message and dropping it silently.
    try:
        service.default_catalogue(payload.get("catalogue"))
    except NotConfigured as exc:
        log.error("generic whatsapp api is not answerable: %s", exc)
        return _error(str(exc), 503)

    language = service.language_name(payload.get("language"))

    # ── record, de-duplicate, queue ───────────────────────────────────────
    inbound, created = service.record(
        provider=PROVIDER,
        message_id=str(payload.get("message_id") or "").strip(),
        phone=phone,
        message_type=message_type,
        text=text,
        media_url=audio_url,
        language=language,
        raw_payload=payload,
        hook=None,                       # this API has no hook, by design
    )
    wait = _flag(payload, "wait", False)
    deliver = _flag(payload, "deliver", True)

    if not created:
        log.info("generic whatsapp api: duplicate %s ignored", inbound.message_id)
        body = {"status": "duplicate", "message_id": inbound.message_id}
        # A caller that waits gets the answer it already paid for, rather than a
        # bare "duplicate" it can do nothing with.
        if wait and inbound.query_id:
            body = {**_answer_body(inbound, inbound.query, delivered=False),
                    "status": "duplicate"}
        return JsonResponse(body, status=200)

    if wait:
        # SYNCHRONOUS. The caller has asked to hold the connection open for the
        # whole pipeline — retrieval, generation, translation and TTS, so 15-30
        # seconds. Two things to know before using it:
        #   * the server's request timeout must exceed that (gunicorn's default
        #     is 30s, which will cut this off mid-answer)
        #   * a caller that retries on timeout gets a `duplicate` on the second
        #     try rather than a second paid answer
        log.info("generic whatsapp api: %s running inline (deliver=%s)",
                 inbound.message_id, deliver)
        status = service.answer_message(inbound.pk, deliver=deliver)
        inbound.refresh_from_db()
        if inbound.query_id is None:
            # 500 rather than 200: the caller is waiting for an answer and there
            # is none, so it must be able to tell that apart from success. The
            # message stays recorded, so whatsapp_retry can still finish it.
            return JsonResponse(
                {"error": inbound.error or "the answer could not be produced",
                 "status": status, "message_id": inbound.message_id}, status=500)
        return JsonResponse(
            _answer_body(inbound, inbound.query,
                         delivered=deliver and status == InboundMessage.STATUS_ANSWERED),
            status=200)

    queued_on = service.dispatch(inbound.pk)
    log.info("generic whatsapp api: accepted %s from %s (queued on %s)",
             inbound.message_id, phone, queued_on)
    return JsonResponse({"status": "accepted", "message_id": inbound.message_id},
                        status=202)


def health_fields() -> dict:
    """What the health endpoint reports about the GENERIC API.

    Lives here next to the endpoint it describes, and is merged into the single
    /api/v1/whatsapp/health/ response rather than served from a second URL of its
    own — one place to look is the entire value of a health check.

    Every value is a BOOLEAN or a name. No key, token or secret is read, so
    nothing here can leak one.
    """
    try:
        catalogue = service.default_catalogue()
        rag_configured = bool(catalogue and settings.OPENAI_API_KEY)
        catalogue_name = catalogue.name if catalogue else None
    except NotConfigured as exc:
        rag_configured, catalogue_name = False, str(exc)

    usable = [k for k in WhatsAppApiKey.objects.filter(hook__isnull=True,
                                                       revoked_at__isnull=True)
              if k.is_usable]
    return {
        "status": "ok",
        "service": "whatsapp",
        "generic_endpoint": "/api/v1/whatsapp/message/",
        # A usable credential for the generic API exists.
        "api_configured": bool(usable),
        "generic_api_keys": len(usable),
        "celery_configured": service.celery_available(),
        "rag_configured": rag_configured,
        "catalogue": catalogue_name,
        "stt_configured": bool(getattr(settings, "SARVAM_API_KEY", "")
                               or settings.OPENAI_API_KEY),
        "tts_configured": bool(getattr(settings, "NARAKEET_KEY", "")
                               or settings.OPENAI_API_KEY),
    }
