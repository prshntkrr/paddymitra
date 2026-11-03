"""Run a question through the real pipeline and keep nothing.

WHY A ROLLBACK AND NOT A `persist=False` FLAG
---------------------------------------------
The obvious change is a flag threaded through `run_pipeline_for_text`. Counting
the writes on that path shows why it is the riskier of the two:

    Query.objects.create(...)                  the row itself
    q.save() x3                                after generation, after m2m, on failure
    q.source_documents.set(doc_ids)            the m2m
    q.original_query_voice_file.save(...)      inbound audio
    q.ai_response_voice_file.save(...)         the spoken answer
    _answer_from_faq: q.save(), .set(), and
        faq_record_hit(faq)                    FAQ.hit_count and last_used_at

Nine sites in the function that answers every farmer's question, plus one in the
FAQ path that increments a counter on a production row. A flag means editing all
of them and being right about all of them — and being right again every time
someone adds a write. Getting it wrong means either a test question saved as
real, or production silently not saving.

A transaction that is rolled back needs NO edit to the pipeline at all. It
cannot miss a write, including writes added later, and production keeps running
byte-for-byte the code it ran before. That is the whole reason for this module.

WHAT IT DOES NOT COVER, AND HOW THAT IS HANDLED
-----------------------------------------------
Files are not transactional: an audio file written to disk would survive the
rollback. So TTS is off and no inbound audio is passed — the two writes that
produce a file. Neither is needed to debug an answer.

ONE COST, STATED PLAINLY: a rolled-back INSERT still consumes the table's
auto-increment counter, so heavy use of this page leaves gaps in Query ids. The
alternative was nine edits to the production pipeline.
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.db import transaction

log = logging.getLogger("irri")

# Settings whose values must never reach the browser, whatever an exception
# says. A provider error can quote the request it was given.
_SECRET_SETTINGS = ("OPENAI_API_KEY", "SARVAM_API_KEY", "GEMINI_API_KEY",
                    "GOOGLE_TRANSLATE_API_KEY", "NARAKEET_KEY", "SECRET_KEY",
                    "PICKY_ASSIST_TOKEN", "PICKY_ASSIST_WEBHOOK_SECRET",
                    "DB_PASSWORD")


def safe_error(exc: Exception) -> str:
    """A message the tester can act on, with no secret and no traceback."""
    text = f"{type(exc).__name__}: {exc}"
    for name in _SECRET_SETTINGS:
        value = str(getattr(settings, name, "") or "")
        if len(value) >= 8 and value in text:
            text = text.replace(value, "***")
    return text[:600]


def _sources(trace: dict) -> list:
    """The retrieved chunks, as the trace already recorded them."""
    sources = []
    for chunk in (trace.get("retrieved_chunks") or []):
        sources.append({
            "rank": chunk.get("rank"),
            "document_title": chunk.get("document_title") or "Untitled document",
            "document_id": chunk.get("document_id"),
            "chunk_index": chunk.get("chunk_index"),
            "score": chunk.get("score"),
            "distance": chunk.get("distance"),
            "document_url": chunk.get("document_url") or "",
            "file_url": chunk.get("file_url") or "",
            "preview": (chunk.get("chunk") or chunk.get("preview") or "")[:600],
        })
    return sources


def _snapshot(query) -> dict:
    """Everything the test page shows, read while the row still exists.

    Read INSIDE the transaction on purpose: after the rollback the row and its
    m2m are gone, and touching the instance would query for rows that no longer
    exist. Nothing but plain data leaves this function.
    """
    trace = query.pipeline_trace if isinstance(query.pipeline_trace, dict) else {}
    return {
        "question": query.original_query_text,
        "question_en": query.translated_query_text,
        "answer": query.ai_response_text,
        "answer_local": query.ai_response_text_local,
        "language": query.original_query_language,
        "status": query.status,
        "response_time_seconds": query.response_time_seconds,
        "classification": {
            "category": query.category,
            "task_type": query.task_type,
            "problem_entity": query.problem_entity,
            "season": query.season,
            "crop_stage": query.crop_stage,
            "missing_context": query.missing_context,
        },
        "sources": _sources(trace),
        # The trace the production path stores on the Query — the intent, the
        # accepted and rejected candidates, the steps. Returned from memory
        # rather than read back from a row, because there will be no row.
        "intent": trace.get("intent") or {},
        "candidates": trace.get("candidates") or {},
        "trace": trace,
    }


def run_test_query(*, catalogue, question: str, language: str = "English",
                   user_type: str = "farmer", district: str = "") -> dict:
    """Answer `question` with the production pipeline and persist nothing.

    Returns {"ok": True, ...snapshot} or {"ok": False, "error": "..."}.
    """
    from queries.services.pipeline import run_pipeline_for_text

    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "Enter a question to test."}
    if catalogue is None:
        return {"ok": False, "error": "Choose a catalogue to test against."}

    payload = None
    try:
        with transaction.atomic():
            try:
                query = run_pipeline_for_text(
                    catalogue=catalogue,
                    farmer_language=language,
                    query_text=question,
                    district=district or "",
                    # The two writes that would leave a FILE behind, which no
                    # rollback can undo. Neither is needed to debug an answer.
                    generate_tts=False,
                    source_audio=None,
                    # No hook and no number: the pipeline only sends WhatsApp
                    # when it has both, so this cannot message anyone.
                    hook=None,
                    whatsapp_number="",
                    user_type=user_type,
                )
                payload = _snapshot(query)
            finally:
                # ALWAYS, including when the pipeline raised: the failure path
                # saves the Query with status Failed, and that must not survive
                # either.
                transaction.set_rollback(True)
    except Exception as exc:                                # noqa: BLE001
        log.exception("test query failed: %r", question[:120])
        return {"ok": False, "error": safe_error(exc)}

    log.info("test query answered and rolled back: %r", question[:120])
    return {"ok": True, **payload}
