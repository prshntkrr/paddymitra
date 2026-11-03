"""
glossary_learn.py
=================
Auto-learning for the Odia glossary.

When the TTS prep step (pipeline.prepare_for_tts) finds English / Latin-script
words that odia_glossary.json does not yet cover, this module translates them to
Odia and persists them back into the glossary. The next time the same word
appears it is replaced instantly by apply_glossary() - no LLM call needed - so
the glossary keeps improving on its own and Narakeet stops reading English.

It also learns in bulk from document text (learn_from_texts / *_async) so the
glossary auto-updates whenever a new document is added or a catalogue's vector
DB is rebuilt.

Translation provider order:
    1. Gemini  (gemini-3.1-flash-lite) if GEMINI_API_KEY is set - same model the
       project already uses for glossary verification.
    2. OpenAI  (gpt-4.1-mini) as a fallback.

Both are asked to TRANSLATE when a natural Odia word exists, and TRANSLITERATE
into Odia script otherwise. Values that still contain Latin letters are dropped.
"""
from __future__ import annotations

import json
import logging
import re
import threading

from django.conf import settings

from .odia_glossary import add_terms, get_glossary, is_auto_learn_blocked

log = logging.getLogger("irri")

# Tokens we never want to "learn": handled elsewhere (TTS_SPOKEN_LATIN), or not
# real words worth a glossary entry.
_SKIP = {
    "dap", "mop", "npk", "ssp", "n", "p", "k",
    "kg", "g", "mg", "ha", "ml", "l", "ph", "ec", "co2",
}

_TRANSLATE_PROMPT = (
    "You are an expert Odia agricultural linguist. You will receive a JSON array "
    "of English agricultural words or short phrases.\n"
    "Return ONLY a JSON object mapping EACH input term to its Odia text:\n"
    "  - Use a natural Odia TRANSLATION when a common Odia word exists.\n"
    "  - Use Odia TRANSLITERATION (Odia script) when no common Odia word exists.\n"
    "  - The values must contain NO Latin/English letters.\n"
    "  - Keep the keys exactly as given.\n"
    "No markdown, no explanation, no comments."
)


def _candidate_terms(words: list[str]) -> list[str]:
    """De-duplicate and drop tokens not worth learning.

    The words handed to this function are whatever was still in Latin script in
    an Odia answer, which is NOT the same thing as "agricultural terms the
    glossary is missing". When a translation leaves an ordinary word behind —
    "strong", "usually", "you" — this used to learn it as a term, and
    apply_glossary() then rewrote that word mid-sentence forever after.

    `is_auto_learn_blocked` is the LEARNER'S test, and it is deliberately wider
    than the one a human editing the glossary has to pass. A person adding
    "strong" on the glossary page has decided it is worth an entry; this function
    has decided nothing — it is looking at whatever a translation happened to
    leave in Latin script. Provenance is the difference, not the word.
    """
    seen: set[str] = set()
    out: list[str] = []
    skipped: list[str] = []
    for w in words:
        term = w.strip()
        key = term.lower()
        if not key or key in _SKIP or key in seen or len(key) < 2:
            continue
        seen.add(key)
        if is_auto_learn_blocked(key):
            skipped.append(key)
            continue
        out.append(term)
    if skipped:
        log.info("glossary_learn: not learning %d ordinary word(s): %s",
                 len(skipped), sorted(skipped))
    return out


def _translate_with_gemini(terms: list[str]) -> dict[str, str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[_TRANSLATE_PROMPT, json.dumps(terms, ensure_ascii=False)],
            # temperature=0, for the same reason _chat and the pipeline's own
            # Gemini calls pin it: Gemini defaults to about 1.0, and what this
            # returns is WRITTEN INTO THE GLOSSARY. An unpinned call gives one
            # Odia spelling of a term today and another tomorrow, so the same
            # question comes back worded differently — and the difference
            # outlives the request, because the glossary keeps it.
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = (resp.text or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {str(k): str(v) for k, v in data.items()}
    finally:
        client.close()


def _translate_with_openai(terms: list[str]) -> dict[str, str]:
    # Imported lazily to avoid a circular import (pipeline imports this module).
    from .pipeline import _chat

    raw = _chat(
        [
            {"role": "system", "content": _TRANSLATE_PROMPT},
            {"role": "user", "content": json.dumps(terms, ensure_ascii=False)},
        ],
        json_mode=True,
    )
    data = json.loads(raw)
    return {str(k): str(v) for k, v in data.items()}


def learn_terms(words: list[str]) -> dict[str, str]:
    """
    Translate unknown Latin words to Odia (Gemini preferred, OpenAI fallback) and
    persist them into odia_glossary.json.

    Returns the mapping that was newly added (english_lower -> odia), or {} if
    there was nothing to learn or translation failed.
    """
    terms = _candidate_terms(words)
    if not terms:
        return {}

    mapping: dict[str, str] = {}
    if settings.GEMINI_API_KEY:
        try:
            mapping = _translate_with_gemini(terms)
        except Exception as exc:
            log.warning("glossary_learn: Gemini failed (%s); falling back to OpenAI", exc)

    if not mapping:
        try:
            mapping = _translate_with_openai(terms)
        except Exception as exc:
            log.warning("glossary_learn: OpenAI translation failed (%s)", exc)
            return {}

    # add_terms() itself rejects values with Latin letters and skips existing keys.
    added = add_terms(mapping)
    if added:
        log.info("glossary_learn: persisted %d new term(s): %s", len(added), added)
    return added


# -- Bulk learning from document text (auto-update on new documents) --------

_EXTRACTION_PROMPT = (
    "You are an expert agricultural linguist. Given excerpts from rice-farming "
    "documents, extract every distinct English agricultural term, crop name, "
    "disease name, technique, or agri-chemical name.\n"
    "Rules:\n"
    "- Return ONLY a JSON array of strings. No markdown, no other text.\n"
    "- Each entry is a lowercase term or short phrase (1-4 words).\n"
    "- Do NOT include generic words, numbers, or units (kg, ha).\n"
    "- DO include fertilizers, pesticides, diseases, crop stages, techniques, "
    "soil types, crop varieties, irrigation methods, organisms, nutrients."
)


def _extract_terms_from_texts(texts: list[str], batch_chars: int = 12000) -> list[str]:
    """Extract candidate English agri terms from raw document/chunk text via GPT."""
    from .pipeline import _chat   # lazy import to avoid a circular import

    # Group pieces into batches that stay comfortably under the context limit.
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for piece in texts:
        piece = (piece or "").strip()
        if not piece:
            continue
        if cur and cur_len + len(piece) > batch_chars:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        batches.append(cur)

    found: set[str] = set()
    for batch in batches:
        combined = "\n\n---\n\n".join(batch)
        try:
            raw = _chat(
                [
                    {"role": "system", "content": _EXTRACTION_PROMPT},
                    {"role": "user", "content": combined},
                ],
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            terms = json.loads(raw)
            for t in terms:
                if isinstance(t, str) and t.strip():
                    found.add(t.strip().lower())
        except Exception as exc:
            log.warning("glossary_learn: term extraction batch failed (%s)", exc)
            continue
    return sorted(found)


def learn_from_texts(texts: list[str], label: str = "") -> dict[str, str]:
    """
    Extract agri terms from document/chunk text, keep only the ones NOT already in
    odia_glossary.json (so we never pay to re-translate known terms), translate the
    new ones to Odia, and persist them.

    Returns the dict of newly added terms.
    """
    terms = _extract_terms_from_texts(texts)
    if not terms:
        return {}

    known = set(get_glossary().keys())   # keys are already lower-cased
    new_terms = [t for t in terms if t not in known and t not in _SKIP]
    if not new_terms:
        log.info("glossary_learn[%s]: no new terms (%d already known)", label, len(terms))
        return {}

    log.info("glossary_learn[%s]: %d new term(s) to translate", label, len(new_terms))
    return learn_terms(new_terms)


def learn_from_texts_async(texts: list[str], label: str = "") -> threading.Thread:
    """
    Fire-and-forget version of learn_from_texts(): runs in a daemon thread so the
    caller (a web request / vector-DB build) returns immediately. Failures are
    logged, never raised.
    """
    def _run() -> None:
        try:
            added = learn_from_texts(texts, label=label)
            if added:
                log.info("glossary auto-learn[%s]: added %d term(s)", label, len(added))
        except Exception as exc:
            log.warning("glossary auto-learn[%s] failed: %s", label, exc)

    thread = threading.Thread(target=_run, name="glossary-learn", daemon=True)
    thread.start()
    return thread
