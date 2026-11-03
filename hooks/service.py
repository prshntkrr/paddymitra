"""
hooks/service.py
================
Answering one WhatsApp message, whatever brought it in.

WHERE THIS SITS
---------------
    Picky Assist webhook ──► hooks/inbound.py  (adapter: payload aliases,
                                                shared secret, keyword routing)
                                     │
    Generic API ─────────────► hooks/api.py    (adapter: JSON contract,
                                                generic API key)
                                     │
                                     ▼
                             hooks/service.py  ◄── you are here
                                     │
                     queries/services/pipeline.py   (RAG, STT, TTS — untouched)
                     queries/services/whatsapp.py   (delivery — untouched)

The adapters own everything provider-shaped. This module owns everything that is
the same either way: recording the message, refusing a duplicate, choosing where
the work runs, running it once, and delivering the result. Nothing here knows a
field name from any gateway, and nothing here contains a line of retrieval.

TWO RULES THAT SHAPE THE CODE
-----------------------------
1. THE ANSWER IS THE EXPENSIVE PART. A message that has already produced a Query
   is never put through the pipeline again — not by a retry, not by a duplicate,
   not by this module's own error handling. `answer()` re-sends instead.

2. A FAILED DELIVERY IS NOT A FAILED ANSWER. They get different statuses because
   they need opposite treatment, and conflating them either loses an answer that
   was already generated or pays for it twice.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from django.conf import settings
from django.db import IntegrityError, connection, transaction
from django.utils import timezone

from .models import Hook, InboundMessage

log = logging.getLogger("irri")


# ── Language ──────────────────────────────────────────────────────────────
#
# Callers send ISO codes ("or", "en"); this codebase names languages in full
# ("Odia", "English") and routes STT and TTS on those names — Sarvam is chosen by
# `language_hint == "Odia"`, Narakeet by `language == "Odia"`. Translating here
# keeps that untouched and keeps the ISO codes out of everything below.
_LANGUAGES = {
    "en": "English", "eng": "English", "english": "English",
    "or": "Odia", "od": "Odia", "ori": "Odia", "ory": "Odia", "odia": "Odia",
    "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
    "bn": "Bengali", "ben": "Bengali", "bengali": "Bengali",
    "te": "Telugu", "tel": "Telugu", "telugu": "Telugu",
    "ta": "Tamil", "tam": "Tamil", "tamil": "Tamil",
}
DEFAULT_LANGUAGE = "Odia"


def language_name(value: str, default: str = "") -> str:
    """"or" -> "Odia". An unknown value is returned unchanged, not discarded."""
    raw = (value or "").strip()
    if not raw:
        return default
    return _LANGUAGES.get(raw.lower(), raw)


# ── Catalogue ─────────────────────────────────────────────────────────────

class NotConfigured(RuntimeError):
    """The deployment cannot answer anything yet, and says which part is missing."""


def default_catalogue(catalogue_ref=None):
    """Which documents a generic message is answered from.

    A hook names its own catalogue. The generic API deliberately does not require
    one — so it comes from, in order:

        1. an explicit `catalogue` in the request (id or slug), for deployments
           serving several
        2. WHATSAPP_DEFAULT_CATALOGUE_ID
        3. the only Published catalogue, or failing that the only catalogue at all
           — there is nothing to choose between when there is one

    Raises NotConfigured rather than guessing between several: answering from the
    wrong corpus is worse than saying the deployment is not set up.
    """
    from catalogues.models import Catalogue

    if catalogue_ref not in (None, ""):
        catalogue = (Catalogue.objects.filter(pk=catalogue_ref).first()
                     if str(catalogue_ref).isdigit()
                     else Catalogue.objects.filter(slug=str(catalogue_ref)).first())
        if catalogue is None:
            raise NotConfigured(f"no catalogue '{catalogue_ref}'")
        return catalogue

    configured = getattr(settings, "WHATSAPP_DEFAULT_CATALOGUE_ID", None)
    if configured:
        catalogue = Catalogue.objects.filter(pk=configured).first()
        if catalogue is None:
            raise NotConfigured(
                f"WHATSAPP_DEFAULT_CATALOGUE_ID={configured} does not exist")
        return catalogue

    published = list(Catalogue.objects.filter(status=Catalogue.STATUS_PUBLISHED)[:2])
    if len(published) == 1:
        return published[0]
    every = list(Catalogue.objects.all()[:2])
    if len(every) == 1:
        return every[0]
    if not every:
        raise NotConfigured("there are no catalogues to answer from")
    raise NotConfigured(
        "several catalogues exist and none is chosen — set "
        "WHATSAPP_DEFAULT_CATALOGUE_ID, or send \"catalogue\": <id> in the request")


# ── Recording, and refusing a duplicate ───────────────────────────────────

def record(provider: str, message_id: str = "", phone: str = "",
           message_type: str = "text", text: str = "", media_url: str = "",
           language: str = "", raw_payload: dict = None, hook=None) -> tuple:
    """(inbound, created). `created` False means this message is already known.

    The unique constraint on (provider, message_id) is what makes a duplicate
    cheap: the second insert fails, and that failure IS the answer. Two requests
    racing therefore cannot both start the pipeline.
    """
    message_id = (message_id or "").strip() or f"gen-{uuid.uuid4().hex[:20]}"
    window = timezone.now() - timedelta(
        hours=getattr(settings, "WHATSAPP_DEDUPE_WINDOW_HOURS", 24))

    existing = InboundMessage.objects.filter(
        provider=provider, message_id=message_id).first()
    if existing:
        if existing.created_at >= window:
            return existing, False
        # Outside the window the id counts as new, but the row must stay unique.
        existing.message_id = (f"{existing.message_id}:"
                               f"{int(existing.created_at.timestamp())}")
        existing.save(update_fields=["message_id"])
    try:
        with transaction.atomic():
            return InboundMessage.objects.create(
                provider=provider,
                message_id=message_id,
                hook=hook,
                from_number=(phone or "")[:30],
                message_type=message_type or "text",
                text=text or "",
                media_url=(media_url or "")[:1000],
                language=language or "",
                raw_payload=raw_payload if isinstance(raw_payload, dict) else {},
            ), True
    except IntegrityError:
        # Another request won the insert a moment ago. That is the dedupe working.
        return InboundMessage.objects.filter(
            provider=provider, message_id=message_id).first(), False


# ── Where the work runs ───────────────────────────────────────────────────

_executor = None
_executor_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, getattr(settings, "WHATSAPP_WORKERS", 4)),
                thread_name_prefix="whatsapp")
        return _executor


def celery_available() -> bool:
    """True when there is a Celery app with a broker to hand work to.

    Celery is NOT currently installed in this project, so the default path is the
    thread pool. This is a runtime check rather than an assumption so that adding
    celery + a broker later needs no code change here: the moment a broker is
    configured, dispatch starts using it.
    """
    try:
        from .tasks import answer_message_task
    except Exception:                                # pragma: no cover
        return False
    if answer_message_task is None:
        return False
    return bool(getattr(settings, "CELERY_BROKER_URL", "")
                or getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False))


def dispatch(inbound_id: int) -> str:
    """Queue the answering of one message. Returns how it was queued.

    Never runs the pipeline inline: the caller is holding an HTTP request open,
    and the gateway on the other end retries slow webhooks.
    """
    if celery_available():
        from .tasks import answer_message_task
        answer_message_task.delay(inbound_id)
        return "celery"
    _pool().submit(answer_message, inbound_id)
    return "thread"


# ── Answering ─────────────────────────────────────────────────────────────

def _run_pipeline(inbound) -> object:
    """The RAG pipeline, for a message that has no hook behind it.

    Calls the SAME functions the Picky Assist path calls — transcription, then
    run_pipeline_for_text — so retrieval, rewriting, reranking, the glossary, the
    quantity guard, Odia handling and TTS are all whatever they already are. The
    only difference is where the catalogue and the language come from, because
    there is no hook to read them off.

    `hook=None` is passed deliberately: run_pipeline_for_text only sends a
    WhatsApp message when it has a hook AND a number, so with no hook it produces
    the Query and sends nothing. Delivery is then this module's job, in one place
    for both paths.
    """
    from queries.services.pipeline import (fetch_audio, run_pipeline_for_text,
                                            transcribe_audio_bytes)

    language = language_name(inbound.language, DEFAULT_LANGUAGE)
    asked_for = (inbound.raw_payload or {}).get("catalogue")
    catalogue = default_catalogue(asked_for)

    query_text, trace, source_audio = inbound.text, None, None
    if inbound.message_type == "audio":
        # Downloaded once and kept: the recording is stored on the Query, so a
        # question asked by voice can be heard again when its transcript looks
        # wrong. The gateway's media URL expires, so it is not a copy.
        data, suffix = fetch_audio(inbound.media_url)
        source_audio = (data, suffix)
        query_text, backend = transcribe_audio_bytes(data, suffix,
                                                     language_hint=language)
        trace = {"stt_backend": backend}
        if not (query_text or "").strip():
            raise ValueError("the voice note could not be transcribed")

    # WHICH CORPUS ANSWERS THIS. Only when the caller did not name one: an
    # explicit "catalogue" in the request is an instruction, not a suggestion.
    #
    # Deliberately AFTER transcription. A voice note has no text to read until
    # here, and routing it on the empty string would send every spoken question
    # to the default — which is most of them.
    category, catalogue_ids = "", []
    if not asked_for:
        from .routing import plan as plan_catalogues
        decision = plan_catalogues(query_text, default=catalogue)
        category = decision["primary"]
        catalogue_ids = decision["catalogue_ids"]
        if decision["catalogues"]:
            # The FIRST is the primary: it owns the Query record, and it is the
            # one the FAQ layer is scoped to. Any others widen retrieval only.
            catalogue = decision["catalogues"][0]
        if decision["secondary"]:
            log.info("whatsapp generic: question spans %s + %s",
                     decision["primary"], ", ".join(decision["secondary"]))

    # LOGGED, not written into the trace: `initial_trace` is the pipeline's own
    # input and other things assert on its exact shape. The log line carries
    # everything needed to explain which corpus answered and why.
    log.info("whatsapp generic: routing category=%s -> catalogue id=%s %s",
             category or "none (default)", getattr(catalogue, "pk", None),
             getattr(catalogue, "name", ""))

    return run_pipeline_for_text(
        catalogue=catalogue,
        farmer_language=language,
        query_text=query_text,
        district=(inbound.raw_payload or {}).get("district", "") or "",
        generate_tts=True,
        hook=None,
        # Recorded on the Query so a conversation can be followed by phone
        # number, and so the answer is attributable to a person.
        whatsapp_number=inbound.from_number,
        initial_trace=trace,
        source_audio=source_audio,
        catalogue_ids=catalogue_ids,
    )


def _deliver(inbound, query, text_already_sent: bool) -> tuple:
    """(ok, error). Sends the answer; never raises."""
    from queries.services.whatsapp import send_answer

    if not inbound.from_number:
        return False, "no recipient number on the message"
    try:
        result = send_answer(inbound.hook, inbound.from_number, query,
                             text_already_sent=text_already_sent)
    except Exception as exc:                         # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if isinstance(result, dict) and result.get("ok") is False:
        return False, str(result.get("error") or result)[:500]
    return True, ""


def answer_message(inbound_id: int, deliver: bool = True) -> str:
    """Produce and deliver the answer for one recorded message. Returns its status.

    Runs on a worker (thread or Celery), so it owns its errors: an exception
    escaping here would surface only in a log nobody reads, with the farmer
    waiting. Everything is written to the InboundMessage row instead.

    `deliver=False` produces the answer and sends nothing. It exists for a caller
    that waits for the response and puts the message on WhatsApp itself — without
    it, such a caller and this code would both send, and the farmer would get the
    same advice twice.
    """
    inbound = InboundMessage.objects.filter(pk=inbound_id).select_related(
        "hook", "query").first()
    if inbound is None:
        log.warning("whatsapp: no inbound message %s to answer", inbound_id)
        return "missing"
    if inbound.status == InboundMessage.STATUS_ANSWERED:
        # Already done. Re-running would send a second answer for one question.
        return inbound.status

    InboundMessage.objects.filter(pk=inbound_id).update(
        status=InboundMessage.STATUS_PROCESSING, attempts=inbound.attempts + 1)

    try:
        query = inbound.query
        text_already_sent = False
        if query is None:
            if inbound.hook is not None:
                # THE HOOK PATH, unchanged: the pipeline is given the hook, and
                # sends the text answer itself as it always has.
                from queries.services.pipeline import process_incoming_message
                query = process_incoming_message(inbound.hook, {
                    "from": inbound.from_number,
                    "message_type": inbound.message_type,
                    "text": inbound.text,
                    "audio_url": inbound.media_url,
                    "district": (inbound.raw_payload or {}).get("district", ""),
                })
                text_already_sent = True
            else:
                query = _run_pipeline(inbound)
            inbound.query = query
            inbound.save(update_fields=["query"])
        else:
            # A retry of a message whose answer already exists: deliver it again
            # rather than paying for it again.
            log.info("whatsapp: re-delivering the existing answer for %s",
                     inbound.message_id)

        if not deliver:
            # The caller is waiting for this answer and will send it. Marked
            # answered because the work IS done; nothing was put on the wire.
            log.info("whatsapp: answered %s, delivery left to the caller",
                     inbound.message_id)
            inbound.status = InboundMessage.STATUS_ANSWERED
            inbound.error = ""
        else:
            ok, error = _deliver(inbound, query, text_already_sent)
            inbound.status = (InboundMessage.STATUS_ANSWERED if ok
                              else InboundMessage.STATUS_DELIVERY_FAILED)
            inbound.error = "" if ok else f"delivery failed: {error}"
            if not ok:
                log.error("whatsapp: answered %s but could not deliver it: %s",
                          inbound.message_id, error)
    except Exception as exc:                         # noqa: BLE001 — see docstring
        log.exception("whatsapp: answering %s failed", inbound.message_id)
        inbound.status = InboundMessage.STATUS_FAILED
        inbound.error = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        inbound.processed_at = timezone.now()
        inbound.save(update_fields=["query", "status", "error", "processed_at"])
        update_hook_stats(inbound.hook)
        # A pooled thread outlives the request that created it, and a connection
        # left open here is one the database holds until it times out. Not inside
        # a transaction, though — under TestCase that would tear down the
        # enclosing atomic block and fail every query after it.
        if not connection.in_atomic_block:
            connection.close()
    return inbound.status


def update_hook_stats(hook) -> None:
    """The counters the hooks dashboard shows.

    Recomputed from the rows rather than incremented, so a retry or a manual
    reprocess cannot drift them. Generic messages have no hook and no dashboard,
    so there is nothing to update for them.
    """
    if hook is None:
        return
    from django.db.models import Avg, Count, Q as Qs
    try:
        stats = InboundMessage.objects.filter(hook=hook).aggregate(
            total=Count("pk"),
            answered=Count("pk", filter=Qs(status=InboundMessage.STATUS_ANSWERED)),
        )
        avg = hook.queries.filter(status="Responded").aggregate(
            ms=Avg("response_time_seconds"))["ms"] or 0.0
        total = stats["total"] or 0
        Hook.objects.filter(pk=hook.pk).update(
            messages_processed_count=total,
            avg_response_time=round(avg, 2),
            success_rate=round((stats["answered"] / total) * 100, 1) if total else 0.0,
        )
    except Exception:                                # noqa: BLE001
        log.warning("could not update stats for hook %s", hook.pk, exc_info=True)
