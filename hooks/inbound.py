"""
hooks/inbound.py
================
Turning a WhatsApp gateway request into an answered farmer message.

WHY THIS IS NOT IN views.py
---------------------------
Four separate things have to be right before the RAG pipeline is allowed to run,
and every one of them is a decision that must be testable without an HTTP client
and without the gateway:

    1. IS IT REALLY THE GATEWAY?   a shared secret, checked in constant time
    2. WHAT DID THE FARMER SAY?    payload shapes differ per account and per
                                   gateway version, so field names are aliases
    3. HAVE WE SEEN IT BEFORE?     gateway retries must not answer twice
    4. WHICH HOOK ANSWERS IT?      keyword, then receiving number, then the only
                                   active hook — in that order

The view is left as the thin thing it should be: parse JSON, call `receive()`,
return its status code.

THE 200 IS NOT THE ANSWER
-------------------------
The pipeline takes tens of seconds (retrieval, generation, translation, TTS).
Gateways retry when a webhook is slow, so answering inside the request produced
duplicate replies to the farmer, each one a fresh set of paid API calls. The
webhook now acks as soon as the message is safely recorded, and the answer is
produced on a worker thread and pushed back through the gateway's send API.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re

from django.conf import settings
from django.db import connection
from django.utils import timezone

from . import service
from .models import Hook, InboundMessage, WhatsAppApiKey

log = logging.getLogger("irri")


# ── 1. Is it really the gateway? ──────────────────────────────────────────
#
# Picky Assist can attach a custom header or a query parameter to its outbound
# webhook call; which one is easiest differs by account, so all the plausible
# carriers are accepted and compared against the one configured secret.
_SECRET_HEADERS = ("HTTP_X_WEBHOOK_SECRET", "HTTP_X_PICKY_SECRET",
                   "HTTP_X_API_KEY", "HTTP_X_HOOK_SECRET")
_SECRET_FIELDS = ("secret", "webhook_secret", "token", "api_key")
_KEY_PARAMS = ("api_key", "apikey", "key", "secret")


def _presented_secrets(request, payload: dict) -> list:
    found = [request.META.get(h, "") for h in _SECRET_HEADERS]
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    if auth:
        # "Bearer <s>", "Token <s>", or the bare value.
        found.append(auth.split(None, 1)[1].strip() if " " in auth else auth.strip())
    for param in _KEY_PARAMS:
        found.append(request.GET.get(param, ""))
    for field in _SECRET_FIELDS:
        value = payload.get(field)
        if isinstance(value, str):
            found.append(value)
    return [f for f in found if f]


def authenticate(request, payload: dict) -> tuple:
    """(ok, reason, key). Refuses anything that cannot prove it is the gateway.

    Two credentials are accepted, and an API KEY is tried first because it says
    more: it identifies which integration is calling and which hook answers, so
    it can be rotated or revoked on its own. The shared env secret remains valid
    for gateways configured before keys existed.

    A request that presents something SHAPED like a key is judged as a key and
    never falls through to the secret check — a mistyped or revoked key must be
    reported as such, not as a missing secret.
    """
    presented_keys = [p for p in _presented_secrets(request, payload)
                      if WhatsAppApiKey.looks_like_key(p)]
    for candidate in presented_keys:
        key = WhatsAppApiKey.authenticate(candidate)
        if key:
            if key.is_generic:
                # A generic key belongs to /api/v1/whatsapp/message/, which routes
                # by nothing at all. Accepting it here would leave this endpoint
                # with no hook to answer with; it is a wrong credential for this
                # door, not a broken one.
                return False, ("this is a generic API key — use "
                               "/api/v1/whatsapp/message/, or a hook key here"), None
            if key.hook.status != Hook.STATUS_ACTIVE:
                return False, (f"the key is valid but hook "
                               f"'{key.hook.trigger_keyword}' is "
                               f"{key.hook.status}"), None
            return True, "", key
    if presented_keys:
        return False, "invalid, revoked or expired API key", None

    expected = (settings.PICKY_ASSIST_WEBHOOK_SECRET or "").strip()
    if not expected:
        if settings.WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK:
            log.warning("whatsapp webhook is running UNAUTHENTICATED "
                        "(WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK=1)")
            return True, "", None
        return False, ("no API key was presented and "
                       "PICKY_ASSIST_WEBHOOK_SECRET is not configured, so this "
                       "endpoint cannot tell the gateway from anyone else"), None
    for presented in _presented_secrets(request, payload):
        # compare_digest, not ==: a plain comparison leaks the secret one
        # character at a time to anything that can measure the response.
        if hmac.compare_digest(presented, expected):
            return True, "", None
    return False, "missing or incorrect webhook secret", None


# ── 2. What did the farmer say? ───────────────────────────────────────────
#
# Field names are ALIASES, not a guess at one true shape. The original view
# assumed exactly {"from", "trigger_keyword", "message_type", "text",
# "audio_url"}; a gateway that calls the body "message" or the sender "mobile"
# produced an empty question and a 200, which looks from the outside like the RAG
# system having nothing to say.
_ALIASES = {
    "message_id": ("message_id", "messageId", "msg_id", "id", "uuid",
                   "message_uuid", "wamid"),
    "from_number": ("from", "from_number", "sender", "mobile", "phone",
                    "phone_number", "number", "msisdn", "contact", "wa_id"),
    "to_number": ("to", "to_number", "receiver", "sender_id", "business_number",
                  "whatsapp_number", "destination"),
    "text": ("text", "message", "body", "message_text", "content", "caption",
             "question", "query"),
    "media_url": ("audio_url", "media_url", "attachment", "attachment_url",
                  "file_url", "url", "voice_url", "media"),
    "message_type": ("message_type", "type", "msg_type", "media_type"),
    "district": ("district", "location", "region", "block"),
    "keyword": ("trigger_keyword", "keyword", "trigger"),
}

# Gateways wrap the message in a container as often as not.
_ENVELOPES = ("data", "payload", "message", "body", "result", "webhook_data")

_AUDIO_HINTS = ("audio", "voice", "ptt", "ogg", "opus", "amr", "mp3", "m4a", "wav")


def _walk(payload: dict, depth: int = 0) -> dict:
    """One flat mapping of the payload, outermost keys winning.

    Nested containers are merged rather than searched for by name, because the
    same field can arrive at the top level from one account and one level down
    from another.
    """
    flat = {}
    if not isinstance(payload, dict) or depth > 3:
        return flat
    for key, value in payload.items():
        if isinstance(value, dict) and (key.lower() in _ENVELOPES or depth < 2):
            for k, v in _walk(value, depth + 1).items():
                flat.setdefault(k, v)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # "messages": [ { ... } ] — the first message in the batch.
            for k, v in _walk(value[0], depth + 1).items():
                flat.setdefault(k, v)
        elif isinstance(value, (str, int, float)) and value != "":
            flat.setdefault(key, value)
    return flat


def _pick(flat: dict, field: str) -> str:
    lowered = {str(k).lower(): v for k, v in flat.items()}
    for alias in _ALIASES[field]:
        value = lowered.get(alias)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _synthetic_id(from_number: str, text: str, media_url: str) -> str:
    """A stable id for a gateway that sends none.

    Content-based and time-bucketed to the minute: a retry of the same message
    lands on the same id and is deduped, while the same farmer legitimately
    asking the same question again later is a new message.
    """
    bucket = timezone.now().strftime("%Y%m%d%H%M")
    digest = hashlib.sha256(
        f"{from_number}|{text}|{media_url}|{bucket}".encode("utf-8")).hexdigest()
    return f"syn-{digest[:24]}"


def normalise(payload: dict) -> dict:
    """The one shape the rest of this module works with."""
    flat = _walk(payload or {})
    text = _pick(flat, "text")
    media_url = _pick(flat, "media_url")
    declared = _pick(flat, "message_type").lower()

    # A voice note is what the URL and the declared type say it is, together: a
    # gateway that only sets "media_url" and a gateway that only sets
    # "type": "ptt" both have to be read as audio, or the farmer's voice message
    # is answered as if it were empty text.
    is_audio = any(h in declared for h in _AUDIO_HINTS) or (
        not text and bool(media_url))
    if media_url and not is_audio:
        is_audio = any(h in media_url.lower() for h in _AUDIO_HINTS)

    from_number = _pick(flat, "from_number")
    message_id = _pick(flat, "message_id") or _synthetic_id(from_number, text, media_url)
    return {
        "message_id": message_id,
        "from_number": from_number,
        "to_number": _pick(flat, "to_number"),
        "message_type": "audio" if is_audio else "text",
        "text": text,
        "media_url": media_url if is_audio else "",
        "district": _pick(flat, "district"),
        "keyword": _pick(flat, "keyword").upper(),
    }


# ── 4. Which hook answers it? ─────────────────────────────────────────────

def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def resolve_hook(message: dict):
    """(hook, question_text). The keyword decides; the number is the fallback.

    Returns (None, text) when nothing claims the message — an unroutable message
    is recorded and ignored, never answered by a hook the farmer did not address.
    """
    text = (message.get("text") or "").strip()
    active = Hook.objects.filter(status=Hook.STATUS_ACTIVE)

    # a) The keyword the gateway extracted for us.
    keyword = (message.get("keyword") or "").upper().strip()
    if keyword:
        hook = active.filter(trigger_keyword=keyword).first()
        if hook:
            return hook, _strip_keyword(text, hook.trigger_keyword)

    # b) The keyword the farmer typed. Gateways that forward the raw body leave
    #    the trigger word at the front of the message, and it was going into the
    #    RAG query — "DHAN how much urea per acre?" retrieves for "DHAN" too.
    first = re.split(r"[\s,:;.\-]+", text, maxsplit=1)[0].upper() if text else ""
    if first:
        hook = active.filter(trigger_keyword=first).first()
        if hook:
            return hook, _strip_keyword(text, hook.trigger_keyword)

    # c) The number the farmer wrote TO — one hook per business number.
    to_digits = _digits(message.get("to_number"))
    if to_digits:
        for hook in active:
            hook_digits = _digits(hook.whatsapp_number)
            # Suffix match: numbers are stored with and without country code.
            if hook_digits and (hook_digits.endswith(to_digits[-10:])
                                or to_digits.endswith(hook_digits[-10:])):
                return hook, text

    # d) Exactly one active hook — then there is nothing to be ambiguous about.
    if active.count() == 1:
        return active.first(), text
    return None, text


def _strip_keyword(text: str, keyword: str) -> str:
    """The question without the trigger word the farmer had to type to reach us.

    Returns "" when the keyword was the whole message — that is a greeting, not a
    question, and the caller answers it with the hook's welcome message. Falling
    back to the raw text here (as this did) sent "DHAN" into retrieval as though
    it were the farmer's question.
    """
    if not (text and keyword):
        return text
    stripped = re.sub(rf"^\s*{re.escape(keyword)}\b[\s,:;.\-]*", "", text,
                      count=1, flags=re.IGNORECASE)
    return stripped.strip()


# ── 3. Have we seen it before? ────────────────────────────────────────────

def _record(provider: str, message: dict, payload: dict, hook):
    """(inbound, created). `created` False means the gateway is retrying.

    The recording and de-duplication themselves are provider-independent, so they
    live in service.record(); this only maps the normalised Picky Assist message
    onto its arguments.
    """
    return service.record(
        provider=provider,
        message_id=message["message_id"],
        phone=message["from_number"],
        message_type=message["message_type"],
        text=message["text"],
        media_url=message["media_url"],
        raw_payload=payload,
        hook=hook,
    )


def process(inbound_id: int) -> None:
    """Run the pipeline for one recorded Picky Assist message and reply.

    Delegates to service.answer_message(), which is the one implementation both
    gateways share — recording, running the pipeline once, delivering, and
    writing down what happened. The hook path inside it is unchanged: the hook is
    still handed to process_incoming_message(), which still sends the text answer
    itself, and the voice note still follows.

    Kept as a function of this name because it is what the thread pool, the retry
    command and the existing tests call.
    """
    from .service import answer_message
    answer_message(inbound_id)


_DEFAULT_WELCOME = ("Namaskar! Send me your question about paddy — you can type it "
                    "or send a voice message.")


def send_welcome(inbound_id: int) -> None:
    """Answer a farmer who sent the trigger word and nothing else.

    "DHAN" on its own is how a farmer starts a conversation. It is not a
    question, and running retrieval on the keyword produced a confident answer to
    a question nobody asked. The hook's own welcome_message is what belongs here.
    """
    from queries.services.whatsapp import send_via_picky_assist

    inbound = InboundMessage.objects.filter(pk=inbound_id).first()
    if inbound is None or inbound.hook is None:
        return
    try:
        send_via_picky_assist(inbound.hook, inbound.from_number,
                              inbound.hook.welcome_message.strip() or _DEFAULT_WELCOME)
        inbound.status = InboundMessage.STATUS_ANSWERED
        inbound.error = ""
    except Exception as exc:                     # noqa: BLE001
        log.warning("welcome message to %s failed: %s", inbound.from_number, exc)
        inbound.status = InboundMessage.STATUS_FAILED
        inbound.error = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        inbound.processed_at = timezone.now()
        inbound.save(update_fields=["status", "error", "processed_at"])
        if not connection.in_atomic_block:
            connection.close()


# The dashboard counters are updated by the shared service, for both gateways.
_update_hook_stats = service.update_hook_stats


# ── The entry point the view calls ────────────────────────────────────────

def receive(request, provider: str = Hook.GATEWAY_PICKY,
            run_inline: bool = False) -> tuple:
    """(response_dict, http_status) for one gateway webhook call.

    Every outcome that is not our fault answers 200: a gateway that gets an error
    retries, and there is nothing to retry when the message was a duplicate, had
    no text, or named no hook. Only a genuine server fault gets a 5xx.
    """
    try:
        payload = json.loads((request.body or b"{}").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Some gateways post form-encoded instead of JSON.
        payload = {k: v for k, v in request.POST.items()} if request.POST else None
        if payload is None:
            return {"ok": False, "error": "invalid json"}, 400
    if not isinstance(payload, dict):
        return {"ok": False, "error": "expected a json object"}, 400

    ok, reason, key = authenticate(request, payload)
    if not ok:
        log.warning("whatsapp webhook rejected: %s", reason)
        return {"ok": False, "error": reason}, 401

    message = normalise(payload)
    if not (message["text"] or message["media_url"]):
        return {"ok": True, "status": "ignored",
                "reason": "no text or audio in the message"}, 200

    if key is not None:
        # THE KEY DECIDES. It was issued for one hook, so there is nothing to
        # infer from a keyword and no way for the message to go unrouted. The
        # keyword is still stripped when the farmer typed it, because it is not
        # part of their question either way.
        hook = key.hook
        question = _strip_keyword(message["text"], hook.trigger_keyword)
    else:
        hook, question = resolve_hook(message)
    if hook is None:
        log.info("whatsapp message from %s matched no active hook",
                 message["from_number"] or "?")
        InboundMessage.objects.update_or_create(
            provider=provider, message_id=message["message_id"],
            defaults={"from_number": message["from_number"][:30],
                      "message_type": message["message_type"],
                      "text": message["text"],
                      "media_url": message["media_url"][:1000],
                      "raw_payload": payload,
                      "status": InboundMessage.STATUS_IGNORED,
                      "error": "no active hook matched this message"},
        )
        return {"ok": True, "status": "ignored",
                "reason": "no active hook matched (keyword, number, or single hook)"}, 200

    message["text"] = question
    inbound, created = _record(provider, message, payload, hook)
    if not created:
        log.info("whatsapp duplicate ignored: %s", message["message_id"])
        return {"ok": True, "status": "duplicate",
                "message_id": message["message_id"],
                "inbound_id": getattr(inbound, "pk", None)}, 200

    # The trigger word on its own — a greeting. Answered from the hook, not from
    # the documents, and it never reaches the pipeline.
    if not message["text"] and not message["media_url"]:
        if run_inline:
            send_welcome(inbound.pk)
        else:
            service._pool().submit(send_welcome, inbound.pk)
        return {"ok": True, "status": "welcome", "inbound_id": inbound.pk,
                "hook": hook.trigger_keyword}, 200

    if run_inline:
        process(inbound.pk)
        inbound.refresh_from_db()
        return {"ok": True, "status": inbound.status,
                "inbound_id": inbound.pk,
                "query_id": inbound.query_id}, 200

    # Queued through the shared dispatcher, so this gateway and the generic API
    # get the same behaviour — Celery when a broker is configured, the thread pool
    # otherwise. Never inline: the gateway is holding this request open.
    # Logged, not returned: the response shape this gateway already relies on
    # stays exactly as it was, and where the work runs is our business.
    log.info("whatsapp: queued %s on %s", message["message_id"],
             service.dispatch(inbound.pk))
    # Accepted, not answered. The farmer's reply arrives over WhatsApp.
    return {"ok": True, "status": "accepted", "inbound_id": inbound.pk,
            "hook": hook.trigger_keyword, "message_id": message["message_id"]}, 200
