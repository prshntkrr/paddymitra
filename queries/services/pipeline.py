"""
End-to-end real AI pipeline for a farmer query:

    STT (voice only) -> Translate to English -> Retrieve from Chroma -> Classify
    -> Generate answer -> Translate to farmer's language -> TTS (optional)
    -> Send via WhatsApp (Picky Assist)

All OpenAI calls are real (model names configurable via settings).
The pipeline is resilient: on failure it records the error into Query.pipeline_trace.error
and attempts to send a graceful fallback text reply to the farmer.\n"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, unquote_plus

import requests
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from .odia_glossary import (apply_glossary, build_prompt_glossary, freezable_sub_terms,
                            glossary_action_stems, glossary_term_class,
                            relevant_glossary, relevant_reverse_glossary)
from .odia_translit import apply_translit_fixes, apply_tts_spoken
from .technical_terms import (compose_odia, is_allowed_latin, normalize_odia,
                              pin_list_for_prompt, protect_input_terms,
                              reassert_tokens, tts_spoken_forms)
from .odia_numbers import conversions_for_prompt, convert_odia_numbers
from . import entity_intent
from .synonyms import canonicalise
from .odia_register import ODIA_INFLECTION, to_spoken, to_imperative
from .odia_semantics import validate_odia_to_english
# Imported under clearer names: `lookup` and `record_hit` say nothing at the call
# site three hundred lines away.
from .faq import (_durations as faq_durations,
                  duration_for_variety as faq_duration_for_variety,
                  lookup as faq_lookup, nearest as faq_nearest,
                  record_hit as faq_record_hit)
from .chat_history import (format_history_for_prompt, load_chat_history,
                           save_chat_turn)

try:
    import torch
except ImportError:
    torch = None

from catalogues.services.vector_store import (search_catalogue, search_catalogue_batch,
                                              keyword_search)
from queries.models import Query

log = logging.getLogger("irri")

# ASCII digits -> Odia numerals, so the Odia TTS voice actually speaks numbers
# (Narakeet skips/garbles Latin "45", "902"). Applied to the final Odia TTS text.
_ODIA_DIGITS = str.maketrans("0123456789", "୦୧୨୩୪୫୬୭୮୯")

# Odia numerals -> ASCII (for parsing numbers back out of the display text).
_ASCII_DIGITS = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")

# The Odia block, used as a "letter" class. Odia script has no \b the regex
# engine understands, so word boundaries are written by hand.
_ODIA_RANGE = r"଀-୿"

# Spoken Odia number-words. Narakeet does NOT vocalize digit glyphs (Odia OR
# Latin), so for the AUDIO path we must spell numbers out as words. The ON-SCREEN
# text keeps Odia numerals (handled separately in translate()).
# NOTE: 1-99 are irregular in Odia — this table is the single place to correct a
# spelling if a number is mispronounced.
_ODIA_ONES = [
    "ଶୂନ୍ୟ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ",
    "ଦଶ", "ଏଗାର", "ବାର", "ତେର", "ଚଉଦ", "ପନ୍ଦର", "ଷୋହଳ", "ସତର", "ଅଠର", "ଅଣେଇଶ",
    "କୋଡ଼ିଏ", "ଏକୋଇଶି", "ବାଇଶି", "ତେଇଶି", "ଚବିଶି", "ପଚିଶି", "ଛବିଶି", "ସତାଇଶି", "ଅଠାଇଶି", "ଅଣତିରିଶି",
    "ତିରିଶି", "ଏକତିରିଶି", "ବତିଶି", "ତେତିଶି", "ଚଉତିରିଶି", "ପଞ୍ଚତିରିଶି", "ଛତିଶି", "ସତତିରିଶି", "ଅଠତିରିଶି", "ଅଣଚାଳିଶି",
    "ଚାଳିଶି", "ଏକଚାଳିଶି", "ବୟାଳିଶି", "ତେୟାଳିଶି", "ଚଉରାଳିଶି", "ପଞ୍ଚଚାଳିଶି", "ଛୟାଳିଶି", "ସତଚାଳିଶି", "ଅଠଚାଳିଶି", "ଅଣଚାଶ",
    "ପଚାଶ", "ଏକାବନ", "ବାଉନ", "ତେପନ", "ଚଉବନ", "ପଞ୍ଚାବନ", "ଛପନ", "ସତାବନ", "ଅଠାବନ", "ଅଣଷଠି",
    "ଷାଠିଏ", "ଏକଷଠି", "ବାଷଠି", "ତେଷଠି", "ଚଉଷଠି", "ପଞ୍ଚଷଠି", "ଛଅଷଠି", "ସତଷଠି", "ଅଠଷଠି", "ଅଣସ୍ତରୀ",
    "ସତୁରୀ", "ଏକସ୍ତରୀ", "ବାସ୍ତରୀ", "ତେସ୍ତରୀ", "ଚଉସ୍ତରୀ", "ପଞ୍ଚସ୍ତରୀ", "ଛଅସ୍ତରୀ", "ସତସ୍ତରୀ", "ଅଠସ୍ତରୀ", "ଅଣାଅଶୀ",
    "ଅଶୀ", "ଏକାଅଶୀ", "ବୟାଅଶୀ", "ତେୟାଅଶୀ", "ଚଉରାଅଶୀ", "ପଞ୍ଚାଅଶୀ", "ଛୟାଅଶୀ", "ସତାଅଶୀ", "ଅଠାଅଶୀ", "ଅଣାନବେ",
    "ନବେ", "ଏକାନବେ", "ବୟାନବେ", "ତେୟାନବେ", "ଚଉରାନବେ", "ପଞ୍ଚାନବେ", "ଛୟାନବେ", "ସତାନବେ", "ଅଠାନବେ", "ନିରାନବେ",
]


def _int_to_odia_words(n: int) -> str:
    """Whole number (0-9,99,999) -> Odia words."""
    if n < 0:
        return "ଋଣାତ୍ମକ " + _int_to_odia_words(-n)
    if n < 100:
        return _ODIA_ONES[n]
    if n < 1000:
        h, r = divmod(n, 100)
        out = _ODIA_ONES[h] + " ଶହ"
        return out + (" " + _int_to_odia_words(r) if r else "")
    if n < 100000:
        t, r = divmod(n, 1000)
        out = _int_to_odia_words(t) + " ହଜାର"
        return out + (" " + _int_to_odia_words(r) if r else "")
    l, r = divmod(n, 100000)
    out = _int_to_odia_words(l) + " ଲକ୍ଷ"
    return out + (" " + _int_to_odia_words(r) if r else "")


def _number_to_odia_words(token: str) -> str:
    """A numeric token (integer or decimal, Odia or ASCII digits) -> Odia words."""
    token = token.translate(_ASCII_DIGITS)
    if "." in token:
        whole, frac = token.split(".", 1)
        words = _int_to_odia_words(int(whole or "0")) + " ଦଶମିକ"
        for d in frac:
            if d.isdigit():
                words += " " + _ODIA_ONES[int(d)]
        return words
    return _int_to_odia_words(int(token))


_NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")

# Words that, right BEFORE a number, mark it as a rice VARIETY CODE (read it
# digit-by-digit: 315 -> "three one five") rather than a quantity.
#
# ଧାନ USED TO BE IN THIS LIST AND MUST NOT BE. It is the everyday Odia word for
# rice/paddy and appears throughout any rice-farming answer, so it matched
# ordinary quantities and made the voice read them out digit-by-digit:
#     ଧାନ ୬୦୨ ଅମଳ  ->  "six zero two"   (wanted: "six hundred two")
# In a real variety name ଧାନ is always preceded by an institute/brand prefix
# (ସିଆର ଧାନ ୩୧୦), so it is handled by _VARIETY_BRANDS below instead of here.
_VARIETY_PREFIXES = (
    "ଏମଟିୟୁ", "ଏମ.ଟି.ୟୁ", "MTU", "ସ୍ୱର୍ଣ୍ଣ", "ସ୍ୱର୍ଣ", "ନବୀନ", "ନଭୀନ",
    "ସବ୍", "ସବ", "ଆଇଆର", "ଆଇ.ଆର୍", "IR", "ବିନା", "ପୂଜା", "ପୁଜା", "ଲଳାଟ",
)

# Institute / brand prefixes. A ଧାନ directly after one of these is part of a
# VARIETY NAME rather than the ordinary noun, so the number that follows is a
# code. Latin spellings are listed too: the English->Odia guard now keeps CR and
# MTU in Latin script, so TTS text can read "CR ଧାନ ୩୧୦".
_VARIETY_BRANDS = (
    "ସିଆର", "ସି ଆର", "ସି.ଆର୍", "ସିଆର୍", "CR",
    "ଏମଟିୟୁ", "MTU", "ଏନଆରଆରଆଇ", "NRRI", "ଡିଆରଆର", "DRR", "CRRI",
)

# ଧାନ written either way (the input-side guard rewrites it to Latin "Dhan").
_DHAN_FORMS = ("ଧାନ", "Dhan", "dhan")


def _is_variety_context(before: str) -> bool:
    """True if the text ending at `before` names a rice VARIETY.

    Either an unambiguous marker (ଏମଟିୟୁ, ସ୍ୱର୍ଣ୍ଣ, ଆଇଆର …), or ଧାନ preceded by an
    institute/brand prefix. A bare ଧାନ is NOT a variety context — see the note on
    _VARIETY_PREFIXES.
    """
    b = (before or "").rstrip()
    if any(b.endswith(p) for p in _VARIETY_PREFIXES):
        return True
    for dhan in _DHAN_FORMS:
        if b.endswith(dhan):
            stem = b[: -len(dhan)].rstrip()
            return any(stem.endswith(brand) for brand in _VARIETY_BRANDS)
    return False
# Unit words that, right AFTER a number, mark it as a QUANTITY (read full number
# even if a variety word precedes it, e.g. "ଧାନ ୨୦ କିଲୋ" = 20 kg of paddy seed).
_UNIT_WORDS = (
    "କିଲୋ", "ଗ୍ରାମ", "ମିଲିଗ୍ରାମ", "ଲିଟର", "ମିଲିଲିଟର", "ହେକ୍ଟର", "ଦିନ",
    "ଶତକଡ଼ା", "ଟନ", "ବ୍ୟାଗ", "ସେଣ୍ଟି", "ମିଟର", "ଏକର", "ରୁ", "ପ୍ରତି",
)


def _spell_digits_odia(token: str) -> str:
    """Read a number digit-by-digit (for variety codes). 315 -> ତିନି ଏକ ପାଞ୍ଚ."""
    token = token.translate(_ASCII_DIGITS)
    return " ".join(_ODIA_ONES[int(d)] for d in token if d.isdigit())


def _spell_numbers_odia(text: str) -> str:
    """Replace every number in the text with spoken Odia words. Used ONLY on the
    TTS text (Narakeet skips digit glyphs). Variety codes (a number right after
    'ଧାନ'/MTU/Swarna and NOT followed by a unit) are read digit-by-digit; all
    other numbers are read as full number-words."""
    def repl(m):
        num = m.group(0)
        before = text[:m.start()].rstrip()
        after = text[m.end():].lstrip()
        # Count only the digits (ignore a decimal point) to decide length.
        digit_count = sum(c.isdigit() for c in num)
        is_after_variety = _is_variety_context(before)
        is_before_unit = any(after.startswith(u) for u in _UNIT_WORDS)
        # Read digit-by-digit ONLY for a real variety CODE: 3+ digits, right after
        # a variety word, and not a quantity. A short number like 45 is always a
        # normal number, so it is spoken "forty five", never "four five".
        if is_after_variety and not is_before_unit and digit_count >= 3:
            return _spell_digits_odia(num)
        return _number_to_odia_words(num)
    return _NUM_TOKEN_RE.sub(repl, text)



# ---------- OpenAI client helpers ----------

def _client():
    from openai import OpenAI
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; pipeline cannot run. "
            "Add it to .env or set USE_LOCAL_EMBEDDINGS and skip TTS/STT."
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _chat(messages: List[dict], model: Optional[str] = None, json_mode: bool = False,
          temperature: float = 0.0, seed: int = 1) -> str:
    # temperature=0 + fixed seed so the SAME question yields the SAME answer
    # every time. Field feedback: farmers/extension officers rerun the same
    # question and get slightly different advice, which erodes trust. Both
    # knobs are needed — seed alone with temperature > 0 still varies; temp=0
    # alone is nearly deterministic but the seed makes it reproducible across
    # server restarts and worker replicas too.
    kwargs = {
        "model": model or settings.OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _gemini_text(system_prompt: str, user_text: str, model: Optional[str] = None) -> str:
    """Generate plain text with Gemini. Used as a cheap Odia translation path."""
    from google import genai
    from google.genai import types

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model_names = [model or getattr(settings, "GEMINI_TRANSLATE_MODEL", "gemini-3.5-flash-lite")]
    for fallback in ("gemini-3.8-flash", "gemini-3.5-flash-lite"):
        if fallback not in model_names:
            model_names.append(fallback)

    # Force deterministic output. Gemini's default temperature is ~1.0 which
    # was the biggest source of "why does the same question give different
    # answers?" — cleanup/translation passes rewrote the same input
    # differently on each call. temperature=0 pins that.
    gen_config = types.GenerateContentConfig(temperature=0.0)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        last_exc: Exception | None = None
        for model_name in model_names:
            for attempt in range(1, 4):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=[system_prompt, user_text],
                        config=gen_config,
                    )
                    text = (resp.text or "").strip()
                    if text:
                        return text
                    raise RuntimeError(f"{model_name} returned empty text")
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    transient = (
                        "503" in msg
                        or "UNAVAILABLE" in msg
                    )
                    if transient and attempt < 3:
                        wait = 1.5 * attempt
                        log.info(
                            "Gemini Odia translation retrying %s after %ss (%s)",
                            model_name,
                            wait,
                            msg[:160],
                        )
                        time.sleep(wait)
                        continue
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        log.warning("Gemini quota/rate limit on %s; trying next model: %s", model_name, exc)
                        break
                    log.warning("Gemini Odia translation failed on %s: %s", model_name, exc)
                    break
        raise last_exc or RuntimeError("Gemini translation failed")
    finally:
        client.close()


# ---------- Google Cloud Translation API (answer translation) ----------
# Human-language name -> Google Translate ISO code.
_GOOGLE_LANG = {
    "English": "en", "Odia": "or", "Hindi": "hi",
    "Bengali": "bn", "Telugu": "te", "Tamil": "ta",
}


def _google_translate_key() -> str:
    """Read the Google Translate API key from settings, env, or .env (decouple),
    so no settings.py change is required."""
    key = getattr(settings, "GOOGLE_TRANSLATE_API_KEY", "") or os.environ.get(
        "GOOGLE_TRANSLATE_API_KEY", ""
    )
    if not key:
        try:
            from decouple import config
            key = config("GOOGLE_TRANSLATE_API_KEY", default="")
        except Exception:
            key = ""
    return key


def _log_baseline_translation(text: str, source_lang: str, target_lang: str,
                              direction: str) -> None:
    """Log what Google returns for the UNTOUCHED text — no glossary applied.

    The glossary goes in BEFORE Google sees the text (English spliced into the
    question, markers standing in for terms in the answer), so the normal logs
    can only ever show the already-influenced translation. This is the
    comparison: the same sentence translated with nothing forced, so the
    glossary's effect on a given answer can be read off the log directly.

    COSTS A SECOND API CALL, so it is off unless asked for — set
    LOG_RAW_TRANSLATION=True in settings or the environment. Never raises: a
    diagnostic must not be able to break the translation it is diagnosing.
    """
    enabled = getattr(settings, "LOG_RAW_TRANSLATION", None)
    if enabled is None:
        enabled = os.environ.get("LOG_RAW_TRANSLATION", "").lower() in (
            "1", "true", "yes", "on")
    if not enabled:
        return
    try:
        raw = _google_translate(text, source_lang, target_lang)
        log.info("%s | BASELINE google, no glossary = %r", direction, raw)
    except Exception as exc:                                # noqa: BLE001
        log.info("%s | baseline translation unavailable (%s)", direction, exc)


def _google_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate `text` with the Google Cloud Translation API (simple API key).

    Raises on missing key / unsupported language / HTTP error so the caller can
    fall back. Returns HTML-unescaped plain text."""
    import html

    api_key = _google_translate_key()
    if not api_key:
        raise RuntimeError("GOOGLE_TRANSLATE_API_KEY is not set")
    tgt = _GOOGLE_LANG.get(target_lang)
    if not tgt:
        raise RuntimeError(f"No Google language code for target '{target_lang}'")
    body = {"q": text, "target": tgt, "format": "text"}
    src = _GOOGLE_LANG.get(source_lang)
    if src:
        body["source"] = src
    resp = requests.post(
        "https://translation.googleapis.com/language/translate/v2",
        params={"key": api_key},
        data=body,
        headers={"Referer": "https://paddymitra-dev.indevconsultancy.in/"},
        timeout=20,
    )
    resp.raise_for_status()
    out = resp.json()["data"]["translations"][0]["translatedText"]
    return html.unescape(out or "").strip()


# Units whose slash means "per" (kg/ha = kilograms PER hectare), as opposed to a
# choice (DAP/SSP = DAP OR SSP).
_SPEECH_UNITS = {
    "kg", "g", "mg", "l", "ml", "t", "ton", "tons", "q", "ac", "acre",
    "ha", "km", "m", "cm", "mm", "day", "days", "week", "weeks",
}

# The SAME units once the translator has put them into Odia script. Without
# these the "per" test could only ever see the Latin spelling, so
# "୨୫ କିଲୋଗ୍ରାମ/ହେକ୍ଟର" fell through to the choice rule and the voice said
# "twenty five kilogram OR hectare" — the farmer was told to pick one, when the
# rate is per hectare. Matched as a PREFIX, because Odia glues its case endings
# on (ହେକ୍ଟରକୁ, ଏକରରେ).
_SPEECH_UNITS_ODIA = (
    "କିଲୋଗ୍ରାମ", "କିଲୋ", "ଗ୍ରାମ", "ମିଲିଗ୍ରାମ", "କ୍ୱିଣ୍ଟାଲ", "ଟନ",
    "ଲିଟର", "ମିଲିଲିଟର", "ହେକ୍ଟର", "ଏକର", "ଡେସିମାଲ", "ଗୁଣ୍ଠ",
    "ମିଟର", "ସେଣ୍ଟିମିଟର", "କିଲୋମିଟର", "ବର୍ଗମିଟର",
    "ଦିନ", "ସପ୍ତାହ", "ମାସ", "ବର୍ଷ", "ଥର", "ଗଛ", "ବ୍ୟାଗ",
)


def _is_speech_unit(word: str) -> bool:
    """True if `word` is a unit of measure, in either script."""
    w = (word or "").strip().strip("।.,")
    if not w:
        return False
    if w.lower() in _SPEECH_UNITS:
        return True
    bare = compose_odia(w).rstrip("୍")        # drop a trailing halant
    return any(bare.startswith(u) for u in _SPEECH_UNITS_ODIA)

# Spoken connector words per language for "/" and number ranges.
_SPEECH_CONNECTORS = {
    "Odia":    {"per": "ପ୍ରତି", "or": "କିମ୍ବା", "to": "ରୁ"},
    "Hindi":   {"per": "प्रति", "or": "या", "to": "से"},
    "Bengali": {"per": "প্রতি", "or": "বা", "to": "থেকে"},
    "English": {"per": "per", "or": "or", "to": "to"},
}


def _clean_for_speech(text: str, language: str) -> str:
    """Deterministically fix punctuation that a TTS voice mispronounces or that
    changes meaning when read aloud — brackets, slashes ('/'), and numeric
    ranges — using language-aware connector words. Example:
        "Apply Urea (25-30 kg/ha), DAP/SSP, and MOP."
      → "Apply Urea, 25 to 30 kg per ha, DAP or SSP, and MOP."
    """
    if not text or not text.strip():
        return text
    c = _SPEECH_CONNECTORS.get(language, _SPEECH_CONNECTORS["English"])

    # URLs go FIRST. They are dropped again further down the Odia path, but the
    # colon and slash rules below would have already chewed "https://" into
    # "https, ..." by then and the leftovers would be read out loud.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Fertilizer grades (10-26-26) -> separate numbers FIRST, so the range step
    # below doesn't read them as "10 to 26 to 26".
    text = re.sub(r"\b\d+(?:-\d+){2,}\b",
                  lambda m: " ".join(m.group(0).split("-")), text)

    # Ratios written with colons — the fertiliser grade (10:26:26) and the
    # nutrient triple (N:P:K). Narakeet reads a colon as a pause at best and as
    # the word "colon" at worst, and the farmer heard neither the grade nor the
    # nutrients as a list. Digits become separate numbers; words become a comma
    # so the voice pauses between them the way a person listing three things does.
    text = re.sub(r"(?<=\d)\s*:\s*(?=\d)", " ", text)
    text = re.sub(r"(?<=[^\s:])\s*:\s*(?=[^\s:\d])", ", ", text)

    # Slash: unit/unit -> "per" (kg/ha), anything else -> a choice, "or"
    # (DAP/SSP). ONE rule for both scripts. There used to be two — a Latin-only
    # one that decided per-vs-or, and a catch-all that always said "or" — so the
    # moment the translator wrote the units in Odia the decision was skipped and
    # "୨୫ କିଲୋଗ୍ରାମ/ହେକ୍ଟର" was read as "25 kilogram OR hectare". `_is_speech_unit`
    # knows both spellings, so the same test now applies whatever the script.
    #
    # A slash with a DIGIT on both sides is left exactly as it is: "1/2" is half,
    # not "one or two", and turning a dose into the wrong quantity is the one
    # mistake this pipeline must never make.
    _SIDE = r"[^\s/,;।.!?()\[\]{}]+"

    def _slash(m):
        a, b = m.group(1), m.group(2)
        if a.isdigit() and b.isdigit():
            return m.group(0)                       # a fraction — leave it alone
        joiner = c["per"] if _is_speech_unit(a) and _is_speech_unit(b) else c["or"]
        return f"{a} {joiner} {b}"

    text = re.sub(rf"({_SIDE})\s*/\s*({_SIDE})", _slash, text)
    text = re.sub(r"(?<!\d)/|/(?!\d)", " ", text)   # a stray slash is never spoken

    # Numeric ranges: 25-30 / 25–30 / 25 — 30 -> "25 to 30".
    text = re.sub(r"(\d+)\s*[–—-]\s*(\d+)", rf"\1 {c['to']} \2", text)

    # Brackets: KEEP the content (it carries the dose) but drop the symbols,
    # using a comma so the aside is spoken as a natural pause.
    text = re.sub(r"\s*[\(\[\{]\s*", ", ", text)
    text = re.sub(r"\s*[\)\]\}]\s*", ", ", text)

    # Tidy the commas/spaces produced above. The Odia full stop (danda, ।) is
    # listed alongside the Latin one everywhere below: it is the ONLY sentence
    # terminator in an Odia answer, so leaving it out of these rules is what
    # produced ", ।" and "।ପ୍ରୟୋଗ" — no pause between sentences, which is the
    # run-together delivery farmers reported.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:,\s*){2,}", ", ", text)       # ", , " -> ", "
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*([.!?;:।])", r"\1", text)     # ", ।" -> "।"
    text = re.sub(r"\s+([.!?;:,।])", r"\1", text)
    # A full stop and a comma must each be followed by a space, or the voice runs
    # the sentences together — "... କରନ୍ତୁ।ପ୍ରତି ହେକ୍ଟରକୁ ..." was delivered as one
    # breathless line with no pause where the farmer needed one.
    #
    # The digit guards are not optional. Without them "2.5 kg" becomes "2. 5 kg"
    # and the number-speller then reads it as TWO numbers — a 2.5-acre dose
    # spoken as "two, five". "1,000" breaks the same way.
    text = re.sub(r"(?<!\d)([.!?।])(?!\d)(?=[^\s.!?।])", r"\1 ", text)
    text = re.sub(r"(?<!\d)(,)(?!\d)(?=\S)", r"\1 ", text)
    text = re.sub(r"[।.]{2,}", "।" if language == "Odia" else ".", text)
    return text.strip()


def prepare_for_tts(text: str, language: str) -> str:
    if not text.strip():
        return text

    # Deterministic punctuation cleanup FIRST (brackets / slashes / ranges), so
    # neither the glossary pass nor the TTS engine ever sees them.
    text = _clean_for_speech(text, language)

    def _apply_tts_glossary(value: str) -> str:
        if language != "Odia":
            return value
        # Normalize common Latin misspellings before glossary replacement.
        value = re.sub(r"\btil+er+ing\b", "tillering", value, flags=re.IGNORECASE)
        value = apply_glossary(value)
        # Some words get TRANSLITERATED into Odia script upstream (e.g. tillering
        # -> ଟିଲେରିଂ). Narakeet then pronounces them like the English word, so we
        # replace the Odia transliteration with the real Odia term. The regex is
        # tolerant of spelling variants (ଟିଲେରିଂ / ଟିଲରିଂ / ଟିଲେରିଙ୍ଗ / ଟିଲ୍ଲେରିଂ ...).
        value = _fix_transliterations(value)
        value = _expand_element_symbols(value)
        return re.sub(r"\burea\b", "ୟୁରିଆ ସାର", value, flags=re.IGNORECASE)

    text = _apply_tts_glossary(text)

    # TTS-only SPOKEN forms are no longer hardcoded here. Word translations
    # (tillering, foliar spray) come from the glossary (apply_glossary) and
    # transliteration/halant fixes from the transliteration map — both applied in
    # _apply_tts_glossary above. The remaining SPOKEN overrides (how "DAP" or
    # nitrogen should sound, which differs from the on-screen form) are managed in
    # odia_tts_spoken.json and applied here.
    text = apply_tts_spoken(text)

    # Convert ranges like 40-50 or 40–50 to speech-friendly format
    text = re.sub(
        r'(\d+)\s*[–-]\s*(\d+)',
        r'\1 ରୁ \2',
        text
    )
    
    system_prompt = f"""
    You are preparing text for text-to-speech.

    Language: {language}

    Rules:
    - Keep the meaning exactly the same.
    - Do NOT modify numbers.
    - Do NOT change dosage quantities.
    - Do NOT change percentages.
    - Do NOT change product names.
    - Keep agricultural terminology unchanged.
    - Remove markdown symbols.
    - Remove URLs.
    - Make the text natural for speech.
    - Return plain text only.
    """

    # ===== PROTECT FERTILIZER GRADES =====
    # Convert fertilizer grades to speech-friendly format
    text = re.sub(
        r"\b\d+(?:-\d+)+\b",
        lambda m: " ".join(m.group(0).split("-")),
        text,
    )

    if language == "Odia":
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[*_`#>\[\]]", "", text)
        text = _apply_tts_glossary(text)
        # Re-apply the TTS spoken map here too. `_apply_tts_glossary` above
        # re-runs `_expand_element_symbols` (K -> ପୋଟାସିୟମ), so if any bare `K`
        # or Latin `potassium` survived the earlier pass we'd have ପୋଟାସିୟମ
        # in the text again — which Narakeet's Baisali voice mispronounces
        # as ପେଟାସିୟମ. Running the TTS map a second time collapses it back
        # to ପୋଟାସ୍ before synthesis. Idempotent on text that has none.
        text = apply_tts_spoken(text)

        # Narakeet's Odia voice pronounces ANY Latin-script token in English.
        # For DISPLAY we keep abbreviations/units in Latin, but for SPEECH every
        # remaining Latin token must become Odia script (phonetic), otherwise the
        # farmer hears stray English words. Convert the known ones directly...
        # Every token the display guard is allowed to KEEP in Latin script needs
        # a spoken Odia form here, or the voice reads it as English. The generated
        # entries cover the corpus acronyms and scientific names protected in
        # technical_terms.py (BGA, DSR, SSNM, Azolla, …); the explicit entries
        # below win where a hand-tuned pronunciation is better.
        # The ACRONYMS are no longer overridden here. They used to be, and the
        # hand-typed spellings quietly beat the generated ones — which is how
        # "MOP" ended up as ଏମ ଓ ପି and was heard as "MO OP".
        #
        # Odia PRONOUNCES the word-final inherent vowel (ରାମ is "ramo"), so a
        # letter name ending in a consonant needs its halant to stop it: ଏମ୍ is
        # "em", plain ଏମ is "emo" — an extra o right before the real ଓ.
        # tts_spoken_forms() builds every acronym from _LETTER_NAMES, which
        # carries the halant on exactly the right letters, so it is the single
        # source of truth and nothing may shadow it. Only UNITS are listed here;
        # they are words, not spelled-out letters, and _LETTER_NAMES has no
        # opinion about them.
        TTS_SPOKEN_LATIN = {
            **tts_spoken_forms(),
            # Element symbols, as a backstop: _expand_element_symbols has already
            # turned a bare N/P/K into the full nutrient name by this point.
            "N": "ଏନ୍", "P": "ପି", "K": "କେ",
            # Spaced, because the voice mispronounces the joined କିଲୋଗ୍ରାମ.
            "kg": "କିଲୋ ଗ୍ରାମ", "g": "ଗ୍ରାମ", "mg": "ମିଲି ଗ୍ରାମ",
            "ha": "ହେକ୍ଟର", "ml": "ମିଲି ଲିଟର", "L": "ଲିଟର", "pH": "ପିଏଚ୍",
        }
        # Longest first so MTU is not partially eaten by a shorter token.
        for abbr, spoken in sorted(TTS_SPOKEN_LATIN.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{re.escape(abbr)}\b", spoken, text)

        # ...and anything still in Latin: LEARN it. Translate the unknown words to
        # Odia (Gemini/OpenAI) and persist them into odia_glossary.json so the same
        # word is replaced instantly next time, then re-apply the glossary to fix
        # the CURRENT response too.
        leftover = re.findall(r"\b[a-zA-Z][a-zA-Z/-]*\b", text)
        if leftover:
            log.info("Odia TTS: leftover Latin words (auto-learn disabled): %s", leftover)

            # Safety net: transliterate anything STILL left in Latin (e.g. learning
            # was skipped/failed) so Narakeet never receives English characters.
            still = re.findall(r"\b[a-zA-Z][a-zA-Z/-]*\b", text)
            if still:
                try:
                    fix_prompt = (
                        "Rewrite the following Odia text for text-to-speech. Every word "
                        "still written in Latin/English script must be transliterated into "
                        "Odia script so it can be pronounced aloud. Keep numbers as digits. "
                        "Do NOT translate the meaning and do NOT add anything. "
                        "Return ONLY the corrected text.\n\n" + text
                    )
                    text = _chat(
                        [{"role": "user", "content": fix_prompt}],
                        model="gpt-4.1-mini",
                    )
                except Exception as exc:
                    log.warning("Odia TTS transliteration fix failed: %s", exc)

        # Make numbers speak in Odia: Narakeet skips digit glyphs, so spell every
        # number out as Odia WORDS for the audio (% -> Odia word). The on-screen
        # text keeps Odia numerals (set separately in translate()).
        text = text.replace("%", " ଶତକଡ଼ା ")
        text = _spell_numbers_odia(text)

        # Punctuation again, LAST. Everything above inserts words and spaces
        # (letter names, number words, ", " for brackets), so the run that
        # happened at the top of this function no longer describes the string
        # being sent. Narakeet phrases on the punctuation it receives, so the
        # final text is what has to be tidy: one space after every ।, one after
        # every comma, no doubled stops.
        text = _clean_for_speech(text, language)

        # Compose LAST of all: this is the exact string Narakeet receives, and a
        # decomposed ୋ is read as େ (କିଲୋଗ୍ରାମ heard as କିଲେଗ୍ରାମ). Any pass above
        # can have reintroduced NFD, so it is normalised here rather than trusted.
        return compose_odia(text).strip()

    cleaned = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model="gpt-4.1-mini"
    )

    return _apply_tts_glossary(cleaned)

# ---------- Pipeline stages ----------

def _stt_transcribe_raw(audio_path: str, language_hint: str = "") -> tuple[str, str]:
    """Transcribe audio to text in the source language.

    Routing:
      - Odia -> Sarvam STT first, then local Wav2Vec2, then Whisper.
      - All other languages -> OpenAI Whisper with the ISO-639-1 hint.

    Fallbacks:
      - If Sarvam/local Odia STT fails, Odia falls back to Whisper.
      - If Whisper rejects the language hint with `unsupported_language`,
        we retry without the hint (auto-detect).
    """
    language_hint = (language_hint or "").strip()
    if language_hint == "Odia":
        try:
            from .indic_stt import NotConfigured, transcribe_odia
            text = transcribe_odia(audio_path)
            if text:
                return text, "sarvam_odia"
            log.warning("Sarvam/local Odia STT returned empty transcript; falling back to Whisper.")
        except NotConfigured as exc:
            log.info("%s - falling back to Whisper for Odia.", exc)
        except Exception as exc:
            log.warning("Sarvam/local Odia STT failed (%s); falling back to Whisper.", exc)

    lang_map = {"Odia": "or", "Hindi": "hi", "Bengali": "bn", "English": "en",
                "Telugu": "te", "Tamil": "ta"}
    lang = lang_map.get(language_hint, None)

    def _call(with_lang: bool) -> str:
        with open(audio_path, "rb") as fh:
            kwargs = {"model": settings.OPENAI_STT_MODEL, "file": fh}
            if with_lang and lang:
                kwargs["language"] = lang
            return _client().audio.transcriptions.create(**kwargs).text

    try:
        return _call(with_lang=True), "openai_whisper"
    except Exception as exc:
        msg = str(exc)
        if "unsupported_language" in msg or "is not supported" in msg.lower():
            log.warning("STT model rejected language '%s'; retrying without hint", lang)
            return _call(with_lang=False), "openai_whisper"
        raise


# ── Speech-to-text transcript correction ─────────────────────────────────
#
# A farmer speaks and an STT model GUESSES. What reaches the pipeline is the
# guess, and a misheard word is not a translation problem or a retrieval
# problem — it is a different question, answered confidently. So the transcript
# is repaired before anything reads it.
#
# APPLIED HERE because stt_transcribe() is the one funnel every audio path goes
# through — the web demo (queries/views.py), the API (api_endpoints/views.py),
# the catalogues endpoint and WhatsApp (via transcribe_audio_bytes). No caller
# can miss it and no caller has to change.
#
# WHAT IT MAY NOT DO. It recovers the QUESTION; it never answers it, never
# adds information the farmer did not imply, and never drops any. A NUMBER may
# be corrected when the transcription clearly misheard one (rule 8) — but a
# number may not APPEAR or DISAPPEAR, which would be new information or lost
# information, so the COUNT is checked below and any value change is logged.
#
# RULE 13 — A CLARIFICATION IS NOT A QUESTION TO RETRIEVE ON. When the meaning
# cannot be recovered, the model is asked for a short clarification question
# instead of a guess. That is the right output for a human and the wrong string
# to search a knowledge base with: retrieval must never run on "which crop do
# you mean?". So a clarification is logged and the ORIGINAL transcript is kept.
# Putting the question to the farmer needs the delivery path to carry it, which
# is a change to WhatsApp/answer handling and not to this pass.
_ODIA_CHAR_RE = re.compile(r"[଀-୿]")

# A correction is a repair, not a rewrite. Far shorter means information was
# dropped (rules 5, 7); far longer means commentary or an answer was added
# (rules 6, 7, 12).
_TRANSCRIPT_MIN_RATIO, _TRANSCRIPT_MAX_RATIO = 0.5, 2.5

# How the model flags rule 13. Ours, not the operator's wording: the two
# possible outputs — a recovered question and a clarification question — are
# both plain Odia, and the code has to tell them apart.
_CLARIFY_PREFIX = "CLARIFY:"

_TRANSCRIPT_PROMPT = (
    "You are reviewing a farmer's question that was generated using "
    "speech-to-text transcription.\n"
    "Some audio recordings may be unclear, and the transcription may contain "
    "mistakes that change the original meaning of the farmer's question. Your "
    "task is to recover the most likely intended question before it is sent "
    "for knowledge-base retrieval.\n"
    "Instructions:\n"
    "1. Carefully review the transcribed question.\n"
    "2. Assume the farmer's spoken question is more likely to be correct than "
    "the transcription.\n"
    "3. Identify words, phrases, or sentences that appear unnatural, "
    "incomplete, out of context, or likely caused by speech-to-text errors.\n"
    "4. Correct transcription mistakes only when the intended meaning is "
    "reasonably clear from the surrounding context.\n"
    "5. Preserve the farmer's original intent.\n"
    "6. Do not answer the question.\n"
    "7. Do not add new information that was not implied by the original "
    "question.\n"
    "8. Do not change numbers, quantities, dates, crop stages, or measurements "
    "unless they are clearly transcription errors.\n"
    "9. If the transcription appears to have changed the meaning of the "
    "farmer's question, restore the most likely intended meaning.\n"
    "10. If a term appears unusual, mistranscribed, or contextually incorrect, "
    "infer the most likely intended agricultural term from the surrounding "
    "words.\n"
    "11. Avoid unnecessary rewriting when the transcription is already "
    "correct.\n"
    "12. Produce a clean, natural Odia question suitable for knowledge-base "
    "retrieval.\n"
    "13. If the intended meaning cannot be determined with reasonable "
    "confidence, return a short clarification question instead of guessing.\n"
    "14. Pay special attention to cases where a single mistranscribed word "
    "changes the entire meaning of the question.\n"
    "15. Before finalizing, compare the corrected question with the original "
    "transcription and verify that the correction improves clarity and "
    "preserves the farmer's likely intent.\n"
    "OUTPUT FORMAT: return ONLY the corrected Odia question, with no "
    "explanation, note or comment. If rule 13 applies, return ONLY a short "
    "Odia clarification question prefixed with " + _CLARIFY_PREFIX + "\n"
    "The next message is the transcribed question."
)


def _transcript_correction_enabled() -> bool:
    """The off switch for this extra model call: settings first, then the
    environment. On unless explicitly turned off."""
    value = getattr(settings, "STT_TRANSCRIPT_CORRECTION", None)
    if value is None:
        value = os.environ.get("STT_TRANSCRIPT_CORRECTION", "true")
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _correct_odia_transcript(transcript: str) -> str:
    """Recover the farmer's intended question from an STT transcript.

    Returns the transcript UNCHANGED whenever the recovery cannot be used:
    nothing to correct, no Odia script (an English or romanised transcript is
    not this pass's job), the model failed, it answered instead of correcting,
    it added or dropped a number, or it asked for clarification (rule 13) —
    which is the right reply for a person but not a question to retrieve on.
    """
    text = transcript or ""
    if not text.strip() or not _ODIA_CHAR_RE.search(text):
        return transcript
    if not _transcript_correction_enabled():
        return transcript

    reply = ""
    if settings.GEMINI_API_KEY:
        try:
            reply = _gemini_text(_TRANSCRIPT_PROMPT, text)
        except Exception as exc:                            # noqa: BLE001
            log.warning("Gemini transcript correction failed (%s); falling "
                        "back to OpenAI", exc)
    if not (reply or "").strip():
        try:
            reply = _chat(
                [{"role": "system", "content": _TRANSCRIPT_PROMPT},
                 {"role": "user", "content": text}],
                model="gpt-4.1-mini",
            )
        except Exception as exc:                            # noqa: BLE001
            log.warning("transcript correction failed (%s); keeping the "
                        "original transcript", exc)
            return transcript

    reply = _strip_emphasis(_dedupe_exact_repeat(reply or ""))
    if not reply.strip() or not _ODIA_CHAR_RE.search(reply):
        log.warning("transcript correction returned nothing usable; keeping "
                    "the original transcript")
        return transcript
    if reply.strip().upper().startswith(_CLARIFY_PREFIX):
        log.warning("transcript correction could not recover the question and "
                    "asked for clarification (%r); keeping the original "
                    "transcript", reply.strip())
        return transcript
    ratio = len(reply) / max(len(text.strip()), 1)
    if not _TRANSCRIPT_MIN_RATIO <= ratio <= _TRANSCRIPT_MAX_RATIO:
        log.warning("transcript correction changed the length %.2fx — it "
                    "answered or dropped something; keeping the original "
                    "transcript", ratio)
        return transcript
    before, after = _numbers_in(text), _numbers_in(reply)
    if len(before) != len(after):
        log.warning("transcript correction added or dropped a number (%s -> "
                    "%s); keeping the original transcript", before, after)
        return transcript
    if before != after:
        # Allowed by rule 8: a misheard digit is a transcription error like any
        # other. Never silently, though — this is the single change that can
        # alter what the farmer is told, so it is always in the log.
        log.warning("transcript correction changed a number (%s -> %s); "
                    "accepted as a mis-hearing", before, after)
    if reply.strip() != text.strip():
        log.info("stt correct | %r -> %r", text, reply)
    return reply


def stt_transcribe(audio_path: str, language_hint: str = "") -> tuple[str, str]:
    """Transcribe audio, then repair likely mis-hearings in the transcript.

    The contract is unchanged — (text, backend) — and so is every caller. See
    _correct_odia_transcript for what the correction refuses to do.
    """
    text, backend = _stt_transcribe_raw(audio_path, language_hint)
    return _correct_odia_transcript(text), backend


# Odia spelling corrections for common LLM errors.
#
# ONLY genuine misspellings belong here — a word that is simply spelled wrong and
# means the same thing once fixed. This table used to contain four entries that
# were not misspellings at all but DIFFERENT WORDS, and it destroyed every one of
# them, because -ୀ is not a typo in Odia: it is the productive suffix that turns
# a thing into the person who does it.
#
#     ଚାଷୀ  -> ଚାଷ    farmer   -> farming        (reported: "ଅନେକ ଚାଷ ..." for
#                                                  "many farmers ...", and
#                                                  "ଚାଷଙ୍କଠାରୁ" for "from farmers")
#     ରୋଗୀ  -> ରୋଗ    the sick one -> the disease
#     ଦ୍ରବ୍ୟ -> ଦ୍ରବ   substance -> liquid
#     ସାରି  -> ସାର    NOT a misspelling of fertiliser: ସାରି is "having finished",
#                     so "ରୋଇ ସାରିବା ପରେ" ("after finishing transplanting") — a
#                     real farmer question from this project — became ରୋଇ ସାରବା.
#
# The last one shows the second defect: the replacements ran as a bare
# str.replace(), so they also fired INSIDE longer words. Both are fixed below.
ODIA_CORRECTIONS = {
    "ଜିବନ": "ଜୀବନ",   # life — short i for long ī, a real misspelling
    # "ବୀଜ": "ବିହନ" removed — it CONTRADICTED the glossary, which gives
    # "seed rate" -> ବୀଜ ହାର. This rule silently rewrote the forced glossary term
    # to ବିହନ ହାର. If ବିହନ is the better word, change it in the glossary so both
    # layers agree; two layers disagreeing means whichever runs last wins.
}


@lru_cache(maxsize=1)
def _compiled_corrections() -> list:
    """Whole-word patterns for ODIA_CORRECTIONS, inflections allowed.

    Left boundary anchored and the right boundary limited to the inflectional
    endings, so a correction fixes ଜିବନରେ as well as ଜିବନ but can never fire
    inside an unrelated word — which is what turned ସାରିବା into ସାରବା.
    """
    out = []
    for wrong, right in ODIA_CORRECTIONS.items():
        if wrong == right:
            continue                      # a no-op entry helps nobody
        rx = re.compile(
            rf"(?<![{_ODIA_RANGE}]){re.escape(normalize_odia(wrong))}"
            rf"(?={normalize_odia(ODIA_INFLECTION)}(?![{_ODIA_RANGE}]))"
        )
        out.append((rx, right))
    return out


def _correct_odia(text: str) -> str:
    """Apply common Odia spelling corrections."""
    out = normalize_odia(text or "")
    for rx, right in _compiled_corrections():
        out = rx.sub(right, out)
    return _fix_variety_dhan(compose_odia(out))


# "CR Dhan 310" is translated to "ସିଆର ଧନ ୩୧୦" — but ଧନ means WEALTH; the paddy
# word is ଧାନ (with the ା matra). The farmer hears "CR wealth 310", and the TTS
# variety detector does not recognise ଧନ either, so the code is then read as a
# quantity. Only corrected directly after a variety brand, because ଧନ on its own
# is a perfectly good word that must not be rewritten.
_VARIETY_DHANA_RE = re.compile(
    r"((?:ସିଆର୍?|ସି\.?\s*ଆର୍?|CR|ଏମଟିୟୁ|MTU|NRRI|CRRI|DRR)\s*)ଧନ"
)


def _fix_variety_dhan(text: str) -> str:
    return _VARIETY_DHANA_RE.sub(r"\1ଧାନ", text)


# Some English words get TRANSLITERATED into Odia script by the translator
# (e.g. fertilizer -> ଫର୍ଟିଲାଇଜର, tillering -> ଟିଲେରିଂ). These then read/sound
# like English. The deterministic transliteration -> canonical-Odia map lives in
# odia_translit.py (a fuzzy regex layer + a JSON literal map, grown offline). This
# thin wrapper keeps the existing call sites (translator output AND TTS text).
def _fix_transliterations(text: str) -> str:
    """Replace Odia-script transliterations of English words with the real Odia term."""
    return apply_translit_fixes(text)

# Nutrient element symbols mean nothing to a farmer ('P', 'S'). Expand them to the
# full Odia nutrient name. Word boundaries keep fertilizer PRODUCT names (DAP, MOP,
# NPK, SSP) and numeric grades (10-26-26) intact.
_ELEMENT_NAMES = {
    "Zn": "ଜିଙ୍କ", "Fe": "ଲୌହ", "Ca": "କ୍ୟାଲସିୟମ", "Mg": "ମ୍ୟାଗ୍ନେସିୟମ",
    "Mn": "ମାଙ୍ଗାନିଜ", "Mo": "ମଲିବଡେନମ", "Si": "ସିଲିକନ", "Cu": "କପର", "Al": "ଆଲୁମିନିୟମ",
    "N": "ନାଇଟ୍ରୋଜେନ", "P": "ଫସଫରସ", "K": "ପୋଟାସିୟମ", "S": "ଗନ୍ଧକ", "B": "ବୋରନ",
}

def _expand_element_symbols(text: str) -> str:
    """Replace standalone nutrient element symbols with their full Odia name."""
    # First drop a redundant symbol in parentheses right after the name, e.g.
    # "ଫସଫରସ (P)" -> "ଫସଫରସ".
    for sym in _ELEMENT_NAMES:
        text = re.sub(rf"\s*\(\s*{sym}\s*\)", "", text)
    # Then expand any remaining bare symbol (2-letter first to avoid partials).
    for sym in sorted(_ELEMENT_NAMES, key=len, reverse=True):
        text = re.sub(rf"\b{sym}\b", _ELEMENT_NAMES[sym], text)
    return text

def _dedupe_exact_repeat(value: str) -> str:
    """An LLM cleanup pass occasionally echoes the whole text twice; keep one copy."""
    value = value.strip()
    if len(value) % 2:
        return value
    mid = len(value) // 2
    left = value[:mid].strip()
    right = value[mid:].strip()
    return left if left and left == right else value


def _strip_emphasis(value: str) -> str:
    """Remove markdown emphasis a model added: **x** / __x__ -> x.

    Gemini bolds words it thinks matter, and an asterisk in a farmer's text is
    printed on the screen and sent over WhatsApp. TWO passes need this — the
    fluency rewrite and the STT transcript correction — so it lives here
    instead of being written twice.
    """
    out = re.sub(r"(\*\*|__)(.+?)\1", r"\2", value or "")
    return out.replace("**", "")


def _detransliterate_odia(odia_text: str, source_en: str = "") -> str:
    """
    General cleanup pass: replace any English word written phonetically in Odia
    script (a transliteration) with its proper Odia translation, while leaving
    genuine Odia, numbers, and chemical / fertilizer / variety names that have no
    Odia word untouched. Uses a cheap model; returns the input unchanged on error.
    """
    if not odia_text.strip():
        return odia_text
    ref = f"\n\nFor reference, the English meaning was:\n{source_en}" if source_en else ""
    prompt = (
        "You are cleaning up an Odia agricultural text written for farmers.\n"
        "Some English words may have been written phonetically in Odia script "
        "(transliterated) instead of being translated. Replace ONLY those with their "
        "natural, common Odia translation.\n"
        "Examples: ଫର୍ଟିଲାଇଜର -> ସାର (fertilizer); "
        "ଟିଲେରିଂ -> ପିଲ ବାହାରିବା (tillering); "
        "ୱାଟର -> ପାଣି (water); "
        "କର୍ଣ୍ଣେଲ -> ମଜ୍ଜା (kernel).\n"
        # The open-ended instruction alone missed କର୍ଣ୍ଣେଲ inside "ନିମ ବିହନ
        # କର୍ଣ୍ଣେଲ ନିଷ୍କାସନ", so tell it HOW to find them, not only what to do
        # once found.
        "HOW TO FIND THEM: read every word aloud in your head. If it sounds like an "
        "ENGLISH word wearing Odia letters, and an ordinary Odia speaker would not use it "
        "in conversation, it is a transliteration — replace it with the real Odia word. "
        "Check EVERY word, including ones in the middle of a product name, and do not "
        "assume a long or technical-looking word must already be correct Odia.\n"
        "STRICT RULES:\n"
        "- Keep all real Odia words, numbers, percentages and punctuation EXACTLY as they are.\n"
        "- Do NOT change chemical names, fertilizer product names, or rice variety names that "
        "have no common Odia word (e.g. ଅମୋନିୟମ ସଲଫେଟ, ଡରି ଏ ପି) — leave them as they are.\n"
        "- Do NOT add, remove, or reorder information.\n"
        "- If nothing needs changing, return the text unchanged.\n"
        "- Return ONLY the corrected Odia text." + ref
    )

    if settings.GEMINI_API_KEY:
        try:
            out = _gemini_text(prompt, odia_text)
            return _dedupe_exact_repeat(out) or odia_text
        except Exception as exc:
            log.warning("Gemini Odia de-transliteration cleanup failed (%s); falling back to OpenAI", exc)

    try:
        out = _chat(
            [{"role": "system", "content": prompt}, {"role": "user", "content": odia_text}],
            model="gpt-4.1-mini",
        )
        return _dedupe_exact_repeat(out) or odia_text
    except Exception as exc:
        log.warning("Odia de-transliteration cleanup failed: %s", exc)
        return odia_text


# Google Translate is the base for English->Odia answers, and on longer
# sentences it keeps ENGLISH word order: the verb lands mid-sentence and a
# forced glossary noun sits where the English verb was. Real outputs heard by
# farmers:
#   "... ସାହାଯ୍ୟ କରିବା ପାଇଁ ନିୟନ୍ତ୍ରଣ ଏହି ଘାସ ପାଇଁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ"
#       ("control these weeds for important" — the glossary marker for
#        'control' was restored in the English verb slot)
#   "ଏହି ବୟସରେ ସାହାଯ୍ୟ କରେ ଚାରାଗୁଡ଼ିକ ... ମୂଳ ଧରିଥାଏ"
#       ("this age helps" glued to a dangling clause)
#   "ଆପଣ ନିୟନ୍ତ୍ରଣକୁ ରୋଗକୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ..."
#       (marker even picked up a case ending, ନିୟନ୍ତ୍ରଣ-କୁ)
# _detransliterate_odia cannot repair any of this: its prompt forbids
# reordering. This pass rebuilds each sentence in natural spoken Odia with the
# English source as the meaning reference, and changes nothing else.

_NUM_EXTRACT_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(text: str) -> list:
    """Every numeric token (Odia or ASCII digits), normalised, sorted."""
    return sorted(_NUM_EXTRACT_RE.findall((text or "").translate(_ASCII_DIGITS)))


# ── Naturalisation faults: the things a farmer hears as broken Odia ───────
#
# Each of these was heard in a real answer. They are CHECKED rather than only
# forbidden in the prompt, because a prompt rule is a request and this is a
# quality gate: an answer still carrying one of them goes back to the model
# once, with the fault named.
_UNNATURAL_PHRASES = (
    "ପ୍ରୟୋଗ କରିବାରେ",           # nominalised where an imperative belongs
    "ପ୍ରୟୋଗ କରିବା କରନ୍ତୁ",       # two verbs stacked
    "ପ୍ରୟୋଗ କରିବା ପିଲ ସାର",      # verb stranded before its object
    "ଦେଖୁପାରିବା",                 # not a word
    "ଦେଖୁବାକୁ",                   # not a word
    "କରିବା ଉଚିତ",                # textbook hedge, not advice
    "କରିବା ଆବଶ୍ୟକ",
    "କରିବା ଦରକାର",
)

# Broken constructions with something in the middle, so a substring test cannot
# see them.
#
# NARROW ON PURPOSE. "X କରିବା ପାଇଁ Y କରନ୍ତୁ" was reported as unacceptable, but
# as a general shape it is correct, idiomatic Odia and the construction this
# module's own glossary rule recommends:
#
#     ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ ମାନକୋଜେବ ସ୍ପ୍ରେ କରନ୍ତୁ।     <- perfectly good
#     ସାର ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରୟୋଗ କରନ୍ତୁ।                      <- broken
#
# What is broken is the SAME verb on both sides of the purpose clause, so that
# is what is matched. Flagging the general shape would send correct sentences
# back to be "fixed", which is how a repair pass starts causing the damage it
# exists to undo.
_UNNATURAL_PATTERNS = (
    (re.compile(r"([଀-୿]{2,})\s+କରିବା\s+ପାଇଁ\s+[^।]{0,40}?\1\s+କର"),
     "the same verb on both sides of 'କରିବା ପାଇଁ ... କରନ୍ତୁ'"),
    (re.compile(r"କରିବାରେ\s+[^।]{0,30}?କରନ୍ତୁ"),
     "'କରିବାରେ ... କରନ୍ତୁ'"),
    (re.compile(r"ହେବା\s+ପାଇଁ\s+ହୁଏ"), "'ହେବା ପାଇଁ ହୁଏ'"),
)

# "ଆରମ୍ଭ ଆରମ୍ଭ ହୁଏ" — the same word twice in a row. Odia does use deliberate
# reduplication (ଧୀରେ ଧୀରେ, ଟିକେ ଟିକେ), so those are left alone.
_DELIBERATE_REDUPLICATION = {"ଧୀରେ", "ଟିକେ", "ଅଳ୍ପ", "ବାରମ୍ବାର", "ପ୍ରତି"}
_DUPLICATE_WORD_RE = re.compile(r"\b([଀-୿]{2,})\s+\1\b")

# Symbols that belong in a dose, not in a sentence. A farmer hears "plus".
_SYMBOL_UNITS = {"kg", "g", "mg", "l", "ml", "ha", "acre", "ac", "t", "q",
                 "କିଲୋ", "ଗ୍ରାମ", "ଲିଟର", "ହେକ୍ଟର", "ଏକର"}
_SYMBOL_RE = re.compile(r"(\S*)\s*([+&/])\s*(\S*)")


def _bare_symbols(text: str) -> list:
    """`+`, `&` and `/` used as words rather than as notation.

    Exempt, per the spec: dosage notation (kg/ha), a ratio or sum between
    numbers (20+10, 3/4) and anything inside a single token that is a product
    name or chemical formula (K2O, N:P:K). Everything else is read aloud to a
    farmer as a symbol and has to become ଓ / ଏବଂ / ସହିତ.
    """
    faults = []
    for left, symbol, right in _SYMBOL_RE.findall(text or ""):
        near = (left.strip("()।,.").lower(), right.strip("()।,.").lower())
        if any(w in _SYMBOL_UNITS for w in near):
            continue                                  # kg/ha — dosage notation
        if all(w.replace(".", "").isdigit() for w in near if w):
            continue                                  # 20+10, 3/4 — arithmetic
        faults.append(f"{left}{symbol}{right}".strip())
    return faults


def _naturalisation_faults(text: str) -> list:
    """Every mechanically detectable reason this is not natural Odia yet."""
    found = [phrase for phrase in _UNNATURAL_PHRASES if phrase in (text or "")]
    found += [label for pattern, label in _UNNATURAL_PATTERNS
              if pattern.search(text or "")]
    found += [f"duplicated word: {m.group(1)}"
              for m in _DUPLICATE_WORD_RE.finditer(text or "")
              if m.group(1) not in _DELIBERATE_REDUPLICATION]
    found += [f"symbol: {s}" for s in _bare_symbols(text)]
    return found


def _fix_odia_fluency(odia_text: str, source_en: str = "",
                      glossary_terms=(), glossary_stems=()) -> str:
    """Re-say a machine-translated answer the way an Odia speaker would say it.

    Google gives a word-by-word draft. This pass is not a proof-read of that
    draft — it re-expresses the whole meaning in the everyday Odia a farmer
    speaks, then the guards below hold it to the facts: numbers, doses and
    product names are checked, and the original is kept if any number moved.

    `glossary_terms` are the approved Odia terms the glossary already forced
    into this text. THE GLOSSARY OUTRANKS THIS PASS: a term the operator
    approved is not a wording choice, so it is stated as mandatory in the prompt
    AND verified afterwards — a rewrite that dropped one is discarded, exactly
    as a rewrite that changed a number is. Without that, "say it more naturally"
    silently overrules the glossary, which is how ପିଲ (the approved term for
    tiller) came back as କୁଶି.
    """
    if not odia_text.strip():
        return odia_text
    # Only the terms actually present can be required to survive.
    required = [t for t in dict.fromkeys(glossary_terms or ())
                if t and normalize_odia(t) in normalize_odia(odia_text)]
    # Action-verb entries are NOT frozen (their verb must inflect), but the word
    # they carry is still the glossary's choice — ପ୍ରୟୋଗ for "apply", not
    # ବ୍ୟବହାର. Required at STEM level: present, with any ending. Unlike
    # `required` these are demanded even when the draft lost them, because the
    # draft is exactly where Google substituted its own verb.
    stems = [s for s in dict.fromkeys(glossary_stems or ()) if s]
    ref = (
        "\n\nThe English source. This is the AUTHORITY on what the answer means — "
        "say all of this, and nothing beyond it:\n" + source_en
    ) if source_en else ""
    # HIGHEST PRIORITY, so it is stated before anything it overrides.
    enforcement = (
        "GLOSSARY ENFORCEMENT (HIGHEST PRIORITY)\n"
        "An approved bilingual glossary governs this text. The glossary "
        "outranks translation, fluency, localisation, style, paraphrasing and "
        "synonym choice — every rule below included.\n"
        "These approved Odia terms are ALREADY in the text and MUST appear in "
        "your answer, character for character:\n"
        + "".join(f"  - {term}\n" for term in required)
        + "Do NOT replace one with a synonym, a more common word, or a more "
        "natural expression. Do NOT infer a different meaning for one. Improve "
        "only the sentence structure AROUND them. If a glossary term makes a "
        "sentence awkward, rebuild the rest of the sentence — never the term.\n"
        "Before you answer, check each term above appears exactly as given.\n"
    ) if required else ""
    if stems:
        enforcement = (enforcement or "GLOSSARY ENFORCEMENT (HIGHEST PRIORITY)\n") + (
            "These approved words carry an ACTION VERB. Use each word itself — no "
            "synonym — but INFLECT its verb to fit the sentence: in an instruction "
            "write 'ପ୍ରୟୋଗ କରନ୍ତୁ', never the infinitive 'ପ୍ରୟୋଗ କରିବା':\n"
            + "".join(f"  - {stem}\n" for stem in stems))
    prompt = (
        enforcement
        + "You are an expert Odia agricultural content writer and translator.\n"
        "Your task is NOT to translate the text below word-for-word. Rewrite it "
        "in natural, fluent, contextually correct Odia that an average "
        "Odia-speaking farmer can easily understand.\n"
        "REQUIREMENTS:\n"
        "1. Use simple, conversational, farmer-friendly Odia.\n"
        "2. Avoid literal translations of English phrases, technical jargon and "
        "English sentence structures.\n"
        "3. Replace awkward machine-translated wording with the agricultural "
        "terms actually used in Odisha.\n"
        "4. The result must sound as if it was originally written in Odia, not "
        "translated from English.\n"
        "5. Preserve the original meaning and the agricultural recommendations "
        "EXACTLY.\n"
        "6. Use proper Odia terminology for crops, irrigation, pests, diseases, "
        "fertilizers, tillers, seedlings, transplanting and harvesting.\n"
        "7. Remove unnatural phrases, redundant words and grammatical errors.\n"
        "8. Make pronoun usage, verb forms and sentence flow natural and "
        "culturally appropriate.\n"
        "9. If a direct translation sounds unnatural, rewrite the sentence "
        "completely while keeping the same meaning.\n"
        "10. NEVER copy English sentence structure into Odia. The verb belongs "
        "at the end of its clause, and case endings (କୁ, ର, ରେ, ପାଇଁ) attach to "
        "the word they belong to: 'ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ', never "
        "'ନିୟନ୍ତ୍ରଣକୁ ରୋଗକୁ'.\n"
        "11. Write practical advice the way an agricultural extension officer "
        "would speak to a farmer — a direct instruction (...ନ୍ତୁ), addressed to "
        "ଆପଣ. Never end advice with 'କରିବା ଉଚିତ', 'କରିବା ଆବଶ୍ୟକ' or "
        "'କରିବା ଦରକାର'; say 'କରନ୍ତୁ', 'ପ୍ରୟୋଗ କରନ୍ତୁ', 'ଦିଅନ୍ତୁ'. (A warning "
        "may keep the softer form.)\n"
        "12. For stage-wise fertilizer answers, use clear Odia stage headings: "
        "ମୂଳ ସାର ପ୍ରୟୋଗ ସମୟରେ, ପିଲ ବାହାରିବା ସମୟରେ, କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ "
        "ସମୟରେ. Do not transliterate English stage names.\n"
        "13. Keep it concise, clear and actionable.\n"
        "14. Every sentence must be complete, meaningful and grammatically "
        "correct — a finite verb before every '।'.\n"
        "15. Do not use uncommon Sanskritised words where a simpler Odia word "
        "exists: ପାଣି not ଜଳ, ଜମି not କ୍ଷେତ୍ର, ମାଟି not ମୃତ୍ତିକା, ସାର not "
        "ଉର୍ବରକ. THIS RULE STOPS AT A GLOSSARY TERM: if one of the approved "
        "terms above contains such a word, it stays — ଜଳସେଚନ is the approved "
        "term for irrigation and must not become ପାଣି ଦେବା.\n"
        "16. Before giving the final answer, check: does this sound natural to a "
        "native Odia speaker? Would a farmer immediately understand it? Is any "
        "phrase a literal translation from English? If yes, rewrite it.\n"
        # The examples now show the ONE thing that is easy to get wrong: the
        # sentence gets rebuilt, the approved term does not move. The earlier
        # pairs demonstrated the opposite (ପିଲ -> କୁଶି, ଜଳସେଚନ -> ପାଣି ଦିଅନ୍ତୁ)
        # and were teaching the model to overrule the glossary.
        "EXAMPLES — rebuild the sentence, never the approved term:\n"
        "  BAD  : ପିଲ ବିକଶିତ ହେବା ପରେ, ମାଟିର ଉପର ସ୍ତରକୁ ପାଖାପାଖି ସଂତୃପ୍ତ ରଖନ୍ତୁ।\n"
        "  GOOD : ପିଲ ବାହାରିବା ସମୟରେ ମାଟି ଭିଜା ରହିବାକୁ ଧ୍ୟାନ ଦିଅନ୍ତୁ।\n"
        "  WRONG: କୁଶି ବିକାଶ ସମୟରେ ମାଟି ଭିଜା ରହିବାକୁ ଧ୍ୟାନ ଦିଅନ୍ତୁ।  "
        "(ପିଲ is the approved term for tiller; କୁଶି and ଶାଖା are not.)\n"
        "  BAD  : ମାଟି ପୃଷ୍ଠରୁ ୧୫ ସେମି ତଳକୁ ଖସିଗଲେ ପୁନଃ ଜଳସେଚନ କରନ୍ତୁ।\n"
        "  GOOD : ପାଣି ସ୍ତର ମାଟି ପୃଷ୍ଠରୁ ୧୫ ସେ.ମି. ତଳକୁ ଯାଇଲେ ପୁଣି ଥରେ "
        "ଜଳସେଚନ କରନ୍ତୁ।\n"
        "  WRONG: ... ପୁଣି ଥରେ ପାଣି ଦିଅନ୍ତୁ।  (the approved term ଜଳସେଚନ was "
        "dropped for a more common word.)\n"
        "THE FACTS ARE FIXED (this is requirement 5, stated precisely):\n"
        "- Add no advice, drop no advice, invent no detail. Fluent Odia carrying "
        "wrong facts is far worse than stiff Odia.\n"
        "- Keep every number, percentage, dose, unit and time period EXACTLY as "
        "written. Never round, merge or recalculate one.\n"
        "- Keep every product name (DAP, MOP, NPK), chemical name and rice "
        "variety name exactly as written — these are not jargon to be "
        "simplified.\n"
        # LAST, so it is the instruction the model reads immediately before it
        # writes. A final gate catches what the rules above only discouraged.
        "NEVER PRODUCE THESE — they were all heard in real answers and a farmer "
        "hears them as broken Odia:\n"
        "  ପ୍ରୟୋଗ କରିବାରେ / ପ୍ରୟୋଗ କରିବା କରନ୍ତୁ / ପ୍ରୟୋଗ କରିବା ପିଲ ସାର / "
        "ଆରମ୍ଭ ଆରମ୍ଭ ହୁଏ / ଦେଖୁପାରିବା / ଜିଙ୍କ ଦେଖୁବାକୁ / ହେବା ପାଇଁ ହୁଏ / "
        "କରିବାରେ ... କରନ୍ତୁ — and any other incomplete clause, stacked verb or "
        "duplicated word.\n"
        "Never repeat the SAME verb on both sides of a purpose clause: "
        "'ସାର ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରୟୋଗ କରନ୍ତୁ' is broken. Two DIFFERENT verbs "
        "there are correct and welcome — 'ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ "
        "ମାନକୋଜେବ ସ୍ପ୍ରେ କରନ୍ତୁ' is exactly right.\n"
        "Keep natural Odia reduplication: ଧୀରେ ଧୀରେ, ଟିକେ ଟିକେ, ବାରମ୍ବାର. Only "
        "repetition created BY the translation is to be removed.\n"
        "USE THE VERB A FARMER EXPECTS: ପ୍ରୟୋଗ କରନ୍ତୁ, ଦିଅନ୍ତୁ, "
        "ବ୍ୟବହାର କରନ୍ତୁ, ଧ୍ୟାନ ଦିଅନ୍ତୁ, ଯାଞ୍ଚ କରନ୍ତୁ, ଅପେକ୍ଷା କରନ୍ତୁ, ରଖନ୍ତୁ.\n"
        "SYMBOLS ARE NOT WORDS. In an ordinary sentence write '+' as ଓ / ଏବଂ / "
        "ସହିତ, '&' as ଏବଂ, and '/' as ଏବଂ or କିମ୍ବା as the sense requires:\n"
        "  ୟୁରିଆ + ଏମଓପି -> ୟୁରିଆ ଓ ଏମଓପି\n"
        "  DAP/MOP -> DAP ଏବଂ MOP\n"
        "  Zinc + Urea -> ଜିଙ୍କ ସହିତ ୟୁରିଆ\n"
        "Keep the symbol ONLY inside dosage notation (20 kg/ha), scientific "
        "notation, a product name, a chemical formula or a variety name.\n"
        "FLUENCY REVIEW — read each sentence back and ask: \"Would a native "
        "Odia speaker naturally say this?\" If the answer is no, rewrite that "
        "sentence, preserving its meaning.\n"
        "FINAL CHECK — Before returning the Odia response, verify that every "
        "sentence sounds natural to a native Odia-speaking farmer. Rewrite any "
        "literal translation, unnatural phrase, English word, or grammatically "
        "awkward construction. The final answer should read as if written "
        "originally in Odia. (The product, chemical and variety names named "
        "above are the ONE exception: they stay exactly as written.)\n"
        "Output ONLY the final Odia response. Do not explain your translation "
        "choices, and add no preamble, notes or English." + ref
    )
    def ask(instructions: str) -> str:
        """One rewrite, Gemini first and OpenAI as the fallback."""
        out = ""
        if settings.GEMINI_API_KEY:
            try:
                out = _gemini_text(instructions, odia_text)
            except Exception as exc:
                log.warning("Gemini Odia fluency repair failed (%s); "
                            "falling back to OpenAI", exc)
        if not (out or "").strip():
            try:
                out = _chat(
                    [{"role": "system", "content": instructions},
                     {"role": "user", "content": odia_text}],
                    model="gpt-4.1-mini",
                )
            except Exception as exc:
                log.warning("Odia fluency repair failed: %s", exc)
                return ""
        # MARKDOWN IS NOT ODIA. Gemini bolds words it thinks matter — the real
        # API returned "… ୫ ଦିନ ଭିତରେ **ସାର ପ୍ରୟୋଗ** କରନ୍ତୁ।" for a farmer's
        # answer. It only vanished because the NEXT LLM pass
        # (_detransliterate_odia) happened to drop it, and that pass returns its
        # input unchanged on failure — so on a bad day the asterisks reach the
        # screen and WhatsApp. (The voice was already safe: prepare_for_tts
        # strips them.) Removed here, where they are introduced, and before the
        # number and glossary checks, so a bolded rewrite is judged on its words.
        return _strip_emphasis(_dedupe_exact_repeat(out or ""))

    def rejects(candidate: str) -> str:
        """The reason this rewrite cannot be used at all, or ""."""
        # A repair that changed a number is worse than broken word order — a
        # dose must never be silently rewritten.
        if _numbers_in(candidate) != _numbers_in(odia_text):
            return "it altered a number"
        # THE GLOSSARY IS NOT A SUGGESTION. Asking the model nicely is not
        # enforcement: a term the operator approved has to be checked. A
        # rewrite that swapped one for a "more natural" word is discarded
        # whole, because the wording it replaced it with is precisely what the
        # glossary exists to override.
        dropped = [t for t in required
                   if normalize_odia(t) not in normalize_odia(candidate)]
        if dropped:
            return f"it dropped approved glossary term(s) {dropped}"
        return ""

    fixed = ask(prompt)
    if not fixed.strip():
        return odia_text
    reason = rejects(fixed)
    if reason:
        log.warning("Odia fluency repair rejected (%s); keeping original text", reason)
        return odia_text

    # FINAL VALIDATION, done here rather than trusted to the model: a rewrite
    # still carrying a broken verb phrase, a duplicated word or a bare symbol
    # has not finished the job, so it goes back ONCE with the faults named.
    def stem_faults(candidate: str) -> list:
        return [f"approved word missing — use it, with the verb inflected: {s}"
                for s in stems if normalize_odia(s) not in normalize_odia(candidate)]

    def all_faults(candidate: str) -> list:
        return _naturalisation_faults(candidate) + stem_faults(candidate)

    faults = all_faults(fixed)
    if faults:
        log.info("Odia naturalisation | %d fault(s) left: %s — asking again",
                 len(faults), faults)
        retry = ask(
            prompt
            + "\n\nYOUR PREVIOUS ANSWER STILL CONTAINED THESE FAULTS:\n"
            + "".join(f"  - {fault}\n" for fault in faults)
            + "Rewrite it so none of them remain. Change nothing else: every "
              "fact, number, dose, unit, timing, product name and approved "
              "glossary term stays exactly as it is."
        )
        # The retry is used only if it is BOTH acceptable and actually cleaner.
        if retry.strip() and not rejects(retry):
            if len(all_faults(retry)) < len(faults):
                log.info("Odia naturalisation | retry cleared %d fault(s)",
                         len(faults) - len(all_faults(retry)))
                return retry
        log.warning("Odia naturalisation | faults remain after one retry: %s",
                    all_faults(fixed))
    return fixed


# English inflections the TRANSLATOR produces vs the form the DOCUMENTS use.
# Retrieval is lexical as well as semantic, so an inflection the corpus never
# contains is invisible to the keyword half and weak for the embedding.
# "lodged" appears ZERO times in the corpus; "lodging" appears 9 times, so a
# farmer reporting a fallen crop could not reach the chunk that answers them.
# Each entry must be a form the documents actually use — verify before adding.
_ENGLISH_CANONICAL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:has|have|had|is|are|was|were)\s+lodged\b", re.I), "has lodging"),
    (re.compile(r"\blodged\b", re.I), "lodging"),
    # "Swarna Sabana" / "Swarna Sabna" / "Swarna Savana" is how speech-to-text
    # mis-hears Swarna Sub-1 (ସ୍ଵର୍ଣ୍ଣ ସବ-୧); map it to the real variety so
    # retrieval and the answer use it. Bare "Swarna" (MTU 7029) is a DIFFERENT
    # variety and is left untouched.
    (re.compile(r"\bswarna\s+sa[bv]\w*", re.I), "Swarna Sub-1"),
]


# Glossary keys that are ordinary English words. Forcing these would wreck the
# sentence — 'the' is in the glossary as ଟି, so every "the" would become a
# classifier. Anything here is never protected.
_GLOSSARY_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "are", "is", "be", "of", "to", "in", "on",
    "at", "it", "as", "by", "with", "this", "that", "from", "not", "can", "do", "use",
    "used", "using", "good", "better", "best", "more", "less", "high", "low", "well",
    # Common VERBS must never be force-marked either: the marker freezes them as
    # a bare noun in the ENGLISH verb slot ("... ପାଇଁ ନିୟନ୍ତ୍ରଣ ଏହି ଘାସ ପାଇଁ ..."),
    # while Google inflects them correctly when left inline (ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ).
    # Multi-word glossary keys ("weed control") are separate entries and are
    # still forced as usual.
    "control", "controls", "controlling", "help", "helps", "grow", "grows",
    "growing", "keep", "keeps", "avoid", "start", "starts",
})

# Google preserves an opaque ASCII token through translation (verified), so a
# term can be lifted out, translated around, and put back in the glossary's own
# words. Without this the glossary has NO effect on the Google path at all: it is
# only ever passed to the LLM fallback, and apply_glossary() afterwards matches
# ENGLISH keys against text that is already Odia, so it is a no-op. That is why
# "panicle initiation" came back transliterated as ପ୍ୟାନିକଲ୍ ଆରମ୍ଭ instead of the
# glossary's କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ.
_MARK_RE = re.compile(r"ZQX(\d+)XQZ")
_MAX_FORCED_TERMS = 8


def _force_glossary_terms(text: str, exclude=()) -> tuple[str, dict]:
    """Replace glossary terms with opaque markers; return (text, {marker: odia}).

    Longest term first, so "panicle initiation" is taken before "panicle".
    Only PROTECTED terminology is frozen (see glossary_term_class). `exclude`
    holds Odia values to leave inline this time — used when Google dropped their
    marker and the sentence is being translated again.
    """
    rel = relevant_glossary(text or "")
    # A VERB MUST NEVER BECOME A MARKER. An opaque token cannot be conjugated,
    # so Google demotes it to a noun and invents a verb of its own. Live:
    #
    #   in    "For Veena rice, apply 70 kg of urea per acre."
    #   sent  "For ZQX4XQZ ZQX6XQZ, ZQX5XQZ 70 kg of ZQX7XQZ per acre."
    #   out   "ବୀଣା ଧାନ, ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରତି ଏକର ପାଇଁ 70 କିଲୋଗ୍ରାମ ୟୁରିଆ
    #          ବ୍ୟବହାର କରନ୍ତୁ।"    <- "for applying ... use": two verbs, one stranded
    #
    # _GLOSSARY_STOPWORDS was meant to stop exactly this, but it is a short
    # hand-kept list and "apply" was never added to it. Checking the English key
    # against the glossary module's complete verb list was the first repair, and
    # it was still not enough: the hazard is in the ODIA VALUE. "application" is
    # a noun, passes any English verb check, and maps to the same ପ୍ରୟୋଗ କରିବା;
    # "puddling" -> କାଦୁଅ କରିବା freezes into "କାଦୁଅ କରିବା କରନ୍ତୁ". So selection
    # now asks the ONE authoritative classifier, is_action_verb_entry(), which
    # reads both the key and the value (a light verb ending) — the same call the
    # LLM-fallback prompt makes. Stage nouns ending in a content verb
    # (tillering -> ପିଲ ଧରିବା) and multi-word terms ("weed control") still freeze.
    #
    # ONLY PROTECTED TERMINOLOGY FREEZES. glossary_term_class() splits entries
    # three ways — action (verb must inflect), contextual (a relational/temporal
    # phrase whose parts must reorder) and protected (noun-headed terminology).
    # Live, before this:
    #
    #   "within 5 days after transplanting" -> "within 5 ZQX0" -> "5 ରୁଆ ପରର ଦିନ"
    #   the number could not bind to ଦିନ sealed inside a placeholder, and the
    #   fluency pass, told to keep that exact string, produced
    #   "୫ ଦିନ ରୁଆ ପରର ଦିନ".
    #
    # A contextual phrase is translated compositionally, but the terminology
    # INSIDE it (transplanting -> ଧାନ ରୁଆ) is still frozen: freezable_sub_terms()
    # recovers it, since the longest-match scan had consumed it into the phrase.
    excluded = {normalize_odia(v) for v in (exclude or ())}
    candidates = dict(rel)
    for key, odia in rel.items():
        if glossary_term_class(key, odia) == "contextual":
            for sub_key, sub_odia in freezable_sub_terms(key).items():
                candidates.setdefault(sub_key, sub_odia)
    terms = [t for t, od in candidates.items()
             if len(t) >= 4 and t.lower() not in _GLOSSARY_STOPWORDS
             and glossary_term_class(t, od) == "protected"
             and normalize_odia(od) not in excluded]
    terms.sort(key=len, reverse=True)

    marks: dict = {}
    out = text or ""
    for term in terms[:_MAX_FORCED_TERMS]:
        rx = re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", re.IGNORECASE)
        if not rx.search(out):
            continue
        marker = f"ZQX{len(marks)}XQZ"
        out = rx.sub(marker, out)
        marks[marker] = candidates[term]
    if marks:
        log.info("forcing %d glossary term(s): %s", len(marks), list(marks.values()))
    return out, marks


# Odia case endings glue onto the noun. Because the marker was a separate token,
# Google leaves them standing alone ("... ଆରମ୍ଭ ରେ"), which reads as two words
# instead of one. Re-attach them after substitution.
_LOOSE_POSTPOSITION_RE = re.compile(r"(?<=[଀-୿])\s+(ରେ|କୁ|ରୁ|ର|ଙ୍କୁ|ଙ୍କର|ଙ୍କ|ଠାରୁ|ଠାରେ)(?![଀-୿])")


# ── Odia -> English: making the glossary bind on the Google path ──────────
#
# Google does not know this project's glossary, and asked in plain Odia it
# translates around it: ରୋଇବା came back as "planting" (the glossary says
# transplanting) and ହଳ କରିବା as "plowed" (the glossary says ploughing). The
# forward direction solves the same problem with opaque markers, but that does
# not work in this direction — replacing every matched term with ZQX0XQZ leaves
# a sentence with no content words, and Google returns
# "How many days in advance should ZQX0XQZ ZQX1XQZ ZQX2XQZ be?".
#
# What DOES work is substituting the glossary's ENGLISH straight into the Odia.
# Google keeps a Latin word it cannot translate — the same property
# protect_input_terms() already relies on for acronyms — and builds the sentence
# around it:
#
#   ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?
#     -> How many days before planting should the rice be plowed?   (plain)
#   rice transplanting କେତେ ଦିନ ଆଗରୁ ploughing ଉଚିତ?
#     -> How many days before rice transplanting should ploughing be done?
#
# FUNCTION WORDS ARE LEFT IN ODIA. Substituting ଆଗରୁ -> "before" puts an English
# preposition in a postposition's slot, and Google then mis-parses which
# operation comes first: "How many days before PLOUGHING should rice be
# TRANSPLANTED" — the reverse of what the farmer asked, and advice they would
# act on in the wrong order. Those entries still reach the OpenAI fallback's
# prompt, where they help; they are only kept out of the substitution.
_INLINE_SKIP_MEANINGS = frozenset({
    "before", "after", "during", "per", "with", "without", "between",
})


# A MONEY QUESTION THE FARMER NEVER ASKED. ଦେବି is "shall I give", and for
# "ସହଭାଗୀ ଧାନ ପାଇଁ କେତେ ୟୁରିଆ ଦେବି?" Google returned "How much urea should I PAY
# for Sahbhagi Dhan?" — so the pipeline answered the question it was handed and
# replied "The cost of urea is 10.03 INR per kg", a figure lifted from a price
# table. The farmer asked for a dose.
#
# Detected by comparing the two sides: English about paying, Odia with no word
# for money in it. Rejecting Google here sends the sentence to the LLM fallback,
# whose prompt carries the glossary and the farming context.
_MONEY_EN_RE = re.compile(
    r"\b(pay|pays|paid|paying|cost|costs|price|prices|priced|rupees?|inr|"
    r"expensive|cheap|afford)\b", re.I)
_MONEY_ODIA_WORDS = ("ଦାମ", "ମୂଲ୍ୟ", "ମୁଲ୍ୟ", "ଖର୍ଚ୍ଚ", "ଟଙ୍କା", "ପଇସା", "ଦର")


def _money_intrusion(odia: str, english: str) -> str:
    """The money word the translation invented, or "" if the farmer used one."""
    found = _MONEY_EN_RE.search(english or "")
    if not found:
        return ""
    haystack = normalize_odia(odia or "")
    if any(normalize_odia(word) in haystack for word in _MONEY_ODIA_WORDS):
        return ""                      # the farmer really did ask about price
    return found.group(0)


def _semantic_repair_enabled() -> bool:
    """The off switch for the repair call: settings first, then the
    environment. On unless explicitly turned off. Detection always runs —
    it costs nothing and the warnings are worth having either way."""
    value = getattr(settings, "ODIA_SEMANTIC_REPAIR", None)
    if value is None:
        value = os.environ.get("ODIA_SEMANTIC_REPAIR", "true")
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _repair_translation(odia: str, english: str, faults: list) -> str:
    """Re-translate a question whose English lost the farmer's meaning.

    Returns the repaired English, or "" when the repair cannot be trusted —
    the caller then keeps the original, exactly as the answer direction does.
    Called ONLY when validate_odia_to_english found something, so a good
    translation never pays for this.
    """
    named = "\n".join(f"  - {f}" for f in faults)
    system = (
        "You are correcting an English translation of an Odia question asked by "
        "a rice farmer. The translation lost or changed part of what the farmer "
        "said. Return ONLY the corrected English question — no explanation, no "
        "commentary, no answer.\n\n"
        "WHAT IS WRONG WITH IT:\n" + named + "\n\n"
        "RULES:\n"
        "1. NEVER answer the question. You are translating it, not advising.\n"
        "2. Preserve the farmer's intent and the kind of question asked.\n"
        "3. Keep every crop name and variety name exactly as written. Never "
        "invent a variety, and never translate the literal words inside a name.\n"
        "4. Keep every number, quantity, unit and duration exactly as written. "
        "Never recompute, round, merge or drop one, and never add one.\n"
        "5. Keep the agricultural stage the farmer named. Transplanting is NOT "
        "harvesting. Ploughing is NOT cutting. Sowing is NOT harvesting.\n"
        "6. A crop that ପଡ଼ିଗଲା has LODGED — fallen over. It is NOT ready, "
        "mature or harvested.\n"
        "7. Applying fertiliser or pesticide is NOT paying for it. Never turn a "
        "question about applying something into a question about cost.\n"
        "8. Keep negative instructions negative ('do not', 'should not').\n"
        "9. Never add information the Odia does not contain.\n"
        "10. Use simple, plain English suitable for searching a document "
        "collection.\n"
    )
    try:
        repaired = _chat([
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"ODIA (what the farmer said):\n{odia}\n\n"
                f"CURRENT ENGLISH (wrong):\n{english}"},
        ])
    except Exception as exc:                                # noqa: BLE001
        log.warning("translate_in | semantic repair call failed (%s); keeping "
                    "the original translation", exc)
        return ""
    # A reply that is not text is not a translation. Guarding here rather than
    # letting _strip_emphasis raise on it: this runs inside the question path,
    # and an exception would cost the farmer the whole answer over a repair
    # that was only ever optional.
    if not isinstance(repaired, str):
        log.warning("translate_in | semantic repair returned %s, not text; "
                    "keeping the original", type(repaired).__name__)
        return ""
    repaired = _strip_emphasis(repaired.strip())
    if not repaired:
        return ""
    # THE REPAIR MUST NOT BE AN ANSWER. Rule 1 is the one a model breaks, and a
    # statement in the question slot searches for the wrong thing entirely.
    # Cheap structural check: the Odia asked something, so the English must too.
    if len(repaired) > max(400, len(english) * 3):
        log.warning("translate_in | semantic repair returned something far too "
                    "long (%d chars); keeping the original", len(repaired))
        return ""
    return repaired


def _force_reverse_glossary_terms(text: str) -> tuple[str, dict]:
    """Substitute the glossary's English for each matched Odia term.

    Returns (text_for_google, {odia: english}) covering only the terms actually
    substituted — which is what the caller verifies survived the round trip.
    """
    rev = relevant_reverse_glossary(text or "")
    if not rev:
        return text or "", {}

    # The scanner works on normalised text and returns normalised keys, so the
    # haystack has to be normalised too or the substitution silently misses.
    out = normalize_odia(text or "")
    applied: dict = {}
    # Longest first: ହଳ କରିବା must be replaced before a shorter overlapping ହଳ
    # could be considered.
    for odia, english in sorted(rev.items(), key=lambda kv: len(kv[0]), reverse=True):
        if english.lower() in _INLINE_SKIP_MEANINGS:
            continue
        # One optional Odia case ending may be glued to the term; it goes with
        # the term, because the English word does not inflect.
        rx = re.compile(rf"(?<![{_ODIA_RANGE}]){re.escape(odia)}"
                        rf"{normalize_odia(ODIA_INFLECTION)}(?![{_ODIA_RANGE}])")
        if not rx.search(out):
            continue
        out = rx.sub(english, out)
        applied[odia] = english
    return out, applied


def _missing_glossary_terms(translated: str, applied: dict) -> list:
    """The substituted English terms that did NOT survive the translation.

    A MULTI-WORD TERM IS CHECKED WORD BY WORD. "transplanting rice" is a
    perfectly good translation when it comes back as "rice transplanting" — the
    meaning is intact and only the English word order moved, which is the
    translator's job. Requiring the phrase verbatim reported it missing and sent
    a good translation to the fallback for no reason. Every word must still be
    present; it is the term that must survive, not its word order.
    """
    low = (translated or "").lower()
    missing = []
    for english in dict.fromkeys(applied.values()):
        words = [w for w in re.split(r"[^a-z0-9]+", english.lower()) if w]
        if not all(re.search(_surviving_word_pattern(w), low) for w in words):
            missing.append(english)
    return missing


def _surviving_word_pattern(word: str) -> str:
    """Match `word` however English inflected it.

    A PLURAL IS NOT A DROPPED TERM. The glossary forced "rice variety" into the
    sentence and Google returned "Name the 105-day rice varieties." — a correct
    translation — but the check looked for the literal "variety", which is not a
    substring of "varieties", called the term dropped, and sent a good
    translation to the OpenAI fallback. The fallback rephrased it as "the rice
    varieties that take 105 days", and THAT wording made entity_intent read
    "take" and "days" as the conditions being asked about. One plural cost the
    question its meaning.

    Only ordinary English endings are allowed: -s, -es, and y -> ies. Anything
    further would start accepting a different word.
    """
    if word.endswith("y") and len(word) > 2:
        return rf"(?<![a-z]){re.escape(word[:-1])}(?:y|ies)(?![a-z])"
    return rf"(?<![a-z]){re.escape(word)}(?:e?s)?(?![a-z])"


def _restore_glossary_terms(translated: str, marks: dict) -> str:
    """Put the glossary's Odia back where the markers are."""
    out = translated or ""
    for marker, odia in marks.items():
        out = out.replace(marker, odia)
    out = _LOOSE_POSTPOSITION_RE.sub(r"\1", out)
    # A marker Google mangled or dropped must never reach the farmer.
    leftover = _MARK_RE.findall(out)
    if leftover:
        log.warning("glossary markers survived translation mangled: %s", leftover)
        out = _MARK_RE.sub("", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _canonicalize_english(text: str) -> str:
    """Rewrite the translated question into the corpus's own word forms."""
    out = text or ""
    for rx, repl in _ENGLISH_CANONICAL:
        out = rx.sub(repl, out)
    if out != text:
        log.info("canonicalised English for retrieval: %r -> %r", text, out)
    return out


# ── Mis-hearings that change the question ────────────────────────────────
#
# Odia has three sibilants a speech model routinely confuses, and one wrong
# letter turns a number into a verb. Live: ଶହେ (one hundred) came through as
# ସହେ. odia_numbers knows ଶହେ, so nothing converted; Google read "ସହେ ଦିନିଆ" as
# "Same day"; and the OpenAI fallback turned it into a variety that does not
# exist — "the Sahia Diniya short variety". The farmer had asked about the
# HUNDRED-DAY short variety, and retrieval went looking for a fictional seed.
#
# REPAIRED ONLY IN CONTEXT. ସହେ is also an ordinary word (a form of ସହିବା, to
# bear), so it is corrected only where the following word makes it a count.
# Everywhere else it is left exactly as the farmer said it.
_ODIA_TIME_WORDS = r"(?:ଦିନିଆ|ଦିନ|ଦିନର|ଦିନରେ|ମାସ|ମାସିଆ|ବର୍ଷ|ବର୍ଷିଆ|ସପ୍ତାହ)"
_ODIA_MISHEARINGS = (
    (re.compile(r"ସହେ(?=\s+" + _ODIA_TIME_WORDS + r"\b)"), "ଶହେ", "ଶହେ/hundred"),
)


def _repair_odia_mishearings(text: str) -> str:
    """Fix speech-recognition letter confusions that change the meaning."""
    out = text or ""
    for pattern, correct, why in _ODIA_MISHEARINGS:
        fixed = pattern.sub(correct, out)
        if fixed != out:
            log.info("translate_in | mis-hearing repaired (%s): %r -> %r",
                     why, out, fixed)
            out = fixed
    return out


# A NAME THE FALLBACK INVENTED. Google renders unknown Odia as ordinary words;
# the LLM sometimes transliterates the same words into a Title Case proper noun.
# A fabricated variety name is worse than a vague sentence, because retrieval
# then hunts for a variety that does not exist. Flagged ONLY when Google did not
# produce the name too — a genuine variety the farmer named is transliterated by
# both engines, so it is never touched.
_TITLE_NAME_RE = re.compile(r"[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+")


def _invented_name(fallback: str, google: str) -> str:
    """A Title Case name present only in the fallback translation, or ""."""
    if not (fallback or "").strip() or not (google or "").strip():
        return ""
    try:
        from .entity_intent import KNOWN_CULTIVARS
        known = {k.lower() for k in KNOWN_CULTIVARS}
    except Exception:                                       # noqa: BLE001
        known = set()
    google_low = google.lower()
    for match in _TITLE_NAME_RE.finditer(fallback):
        name = match.group(0)
        before = fallback[:match.start()].rstrip()
        if not before or before.endswith((".", "?", "!")):
            continue                        # sentence-initial capital, not a name
        low = name.lower()
        if low in google_low:
            continue                        # both engines produced it
        if any(word in known for word in low.split()):
            continue                        # a variety this project knows
        return name
    return ""


# ── Days -> duration class -> the answers written for that class ─────────
#
# "How many kg of urea for the 100-day short variety?" names a DAY COUNT. The
# answer is written for a DURATION CLASS ("short duration rice varieties like
# Sahabhagi Dhan ..."). Measured against the operator's own entries, the
# day-count phrasing scores 0.508 — far under the 0.60 context floor — while the
# class phrasing scores 0.625 and is offered as context. So the bridge is built
# here: the day count is classified, and the class phrasing is offered to the FAQ
# matcher as an ALTERNATIVE probe. The farmer's question itself is unchanged, and
# the probe is only tried when the question's own wording found nothing.
#
# The boundaries mirror FAQ-81 (short 100-110, medium 120-135, long 140-160),
# with the gaps between those bands going to the nearer class.
_DURATION_BANDS = ((115, "short"), (139, "medium"))         # anything else: long
# Carried into the probe so it looks for the entry about THAT input. Ordered:
# the first one the question names wins.
_NUTRIENTS = ("urea", "dap", "mop", "potash", "npk", "ssp", "zinc",
              "nitrogen", "phosphorus", "potassium", "fym", "compost")
# IS THE FARMER ASKING WHAT TO PUT ON THE CROP AT ALL?
#
# The probe rewrites a day count into the class phrasing the approved answers
# are written for — and it used to do that for ANY question carrying a day
# count. "ଶହେ ଦିନିଆ ଧାନ କିସମ କଣ କଣ ଅଛି?" ("what are the 100-day varieties?")
# became "For short duration rice varieties, how much fertilizer to be applied
# and when?", matched FAQ-137 at 0.891, and the farmer was sent a fertiliser
# schedule. The document that answers them — "Yes, Sahabhagidhan and CR Dhan 207
# are good 100-day varieties" — was never read, because a served FAQ skips
# retrieval entirely.
#
# entity_intent cannot tell these apart: "what to apply for a 100-day variety?"
# and "what are the 100-day varieties?" are BOTH ('variety', cultivar, list).
# What separates them is whether anything is being applied, so that is what is
# tested. Nutrient names count, because naming one IS the intent.
#
# "give" and "put" are deliberately absent: "which varieties GIVE good yield?"
# is a listing question, and including them put it straight back.
_APPLICATION_INTENT_RE = re.compile(
    r"\b(appl(?:y|ied|ies|ication)|dose|dosage|top\s*dress\w*|basal|spray\w*|"
    r"fertili[sz]\w*|manure|nutrient\w*|" + "|".join(_NUTRIENTS) + r")\b", re.I)
_DAY_COUNT_RE = re.compile(r"(\d{2,3})\s*[-– ]?\s*day(?:s)?\b", re.I)


def duration_class(question_en: str) -> str:
    """"short" / "medium" / "long", from a day count OR from the words used.

    READING THE WORDS MATTERS AS MUCH AS READING THE DAYS. "ଅଳ୍ପ କାଳୀନ କିସମ
    ପାଇଁ କେତେ ୟୁରିଆ ଦେବି?" arrives as "How much urea should I give for an
    early-maturing variety?" — no digits anywhere, so a day-count rule found
    nothing, no probe was built, and the generic "how much urea for my rice
    crop" entry (0.684) was handed over. Its 70 kg then went out under an
    "early-maturing variety" heading: the same mislabelled dose the day-count
    path was fixed for.

    Only ONE class counts. An entry or question naming several (a
    classification) is generic and gets no probe.
    """
    match = _DAY_COUNT_RE.search(question_en or "")
    if match:
        days = int(match.group(1))
        if not 60 <= days <= 250:
            return ""
        return next((name for limit, name in _DURATION_BANDS if days <= limit),
                    "long")
    named = faq_durations(question_en or "")
    return next(iter(named)) if len(named) == 1 else ""


def duration_probe(question_en: str, label: str = "") -> str:
    """A "<class> duration" phrasing of a question that names a day count.

    Returns "" when the question names no day count, names one that is not a
    crop duration (a 5-day interval after transplanting is not a variety class),
    or is not asking what to APPLY — see _APPLICATION_INTENT_RE.
    """
    label = label or duration_class(question_en)
    if not label:
        return ""
    # A DURATION IS NOT A REASON TO ASK ABOUT FERTILISER. The class phrasing
    # below always asks "how much ... to be applied", so building it for a
    # question that asked which varieties EXIST substitutes a question the
    # farmer never asked — and, because a probe match used to be served
    # verbatim, answered it with authority. See query 1030.
    if not _APPLICATION_INTENT_RE.search(question_en or ""):
        log.info("faq | %r names a %s duration but asks nothing about applying "
                 "anything; no duration probe", question_en, label)
        return ""
    # A TEMPLATE, NOT A SUBSTITUTION. Swapping "100-day" for "short duration"
    # inside the farmer's own sentence was measured at 0.568 against the entry
    # that answers it — still under the 0.60 floor, and only 0.001 ahead of the
    # LONG-duration entry, which is no separation at all. This phrasing scores
    # 0.787 against the right entry and 0.655 against the long one: a 0.132
    # margin, and below the 0.80 serve threshold, so it can add CONTEXT but can
    # never serve an approved answer off a sentence the farmer never said.
    nutrient = next((w for w in _NUTRIENTS
                     if re.search(rf"\b{w}\b", question_en, re.I)), "fertilizer")
    return (f"For {label} duration rice varieties, how much {nutrient} "
            f"to be applied and when?")


def translate(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    if source_lang == target_lang:
        return text

    # Technical tokens pinned out of the farmer's question (see the Odia->English
    # branch below). Empty for every other direction.
    pinned_tokens: list[str] = []
    # Canonical English terms the reverse glossary resolved for this question.
    glossary_terms: list[str] = []
    # Google's own attempt, kept so the LLM fallback can be judged against it.
    google_candidate: str = ""
    # True only when Google's output passed every existing guard and was used
    # as-is — the one path where nothing has yet looked at what it MEANS.
    accepted_from_google: bool = False

    # Use specialized prompt for Odia
    if target_lang == "Odia":
        # Give the model the EXACT Odia for every glossary term present in THIS text.
        # This is what stops it from transliterating a translatable word (e.g.
        # tillering -> ଟିଲେରିଂ, which then sounds like English in TTS).
        rel = relevant_glossary(text)
        # The SAME classifier as _force_glossary_terms. "Output EXACTLY" is a
        # freeze by prompt: listing 'apply → ପ୍ରୟୋଗ କରିବା' there told the model to
        # copy the infinitive into the verb slot. Action entries are listed by
        # their approved word instead, to be inflected; function-word entries
        # with nothing to keep are left out.
        #
        # "Output EXACTLY" is also wrong for a CONTEXTUAL phrase: it tells the
        # model to paste 'ରୁଆ ପରର ଦିନ' in whole. Those are offered as guidance,
        # to be fitted around their numbers and word order; the terminology
        # inside them stays in the exact list.
        exact = {en: od for en, od in rel.items()
                 if glossary_term_class(en, od) == "protected"}
        guidance = {en: od for en, od in rel.items()
                    if glossary_term_class(en, od) == "contextual"}
        for en in guidance:
            exact.update(freezable_sub_terms(en))
        inflect = glossary_action_stems(text)
        if rel:
            glossary_intro = (
                "12. The text contains these terms. Output EXACTLY the Odia shown for "
                "each — do NOT transliterate them and do NOT keep them in English:\n"
            )
            glossary_block = "\n".join(f"    - {en} → {od}" for en, od in sorted(exact.items()))
            if inflect:
                glossary_block += (
                    "\n    These carry an ACTION VERB. Use the Odia word shown, but "
                    "INFLECT the verb to fit the sentence — never copy an infinitive "
                    "into a verb slot:\n"
                    + "\n".join(f"    - {en}: use {stem}, with the verb inflected"
                                for en, stem in sorted(inflect.items())))
            if guidance:
                glossary_block += (
                    "\n    These are PHRASES, not fixed terms. Translate them "
                    "naturally, fitting any number and the word order around them; "
                    "the wording shown is guidance only:\n"
                    + "\n".join(f"    - {en}: e.g. {od}"
                                for en, od in sorted(guidance.items())))
        else:
            glossary_intro = "12. Common agricultural terms and their Odia equivalents:\n"
            glossary_block = build_prompt_glossary()   # loads from odia_glossary.json

        sys = (
            "You are an expert agricultural translator specializing in Odia (ଓଡ଼ିଆ). "
            "Translate the following agricultural text into natural, farmer-friendly Odia.\n\n"

            "CRITICAL RULES:\n"
            "1. Use simple, everyday SPOKEN Odia that a village farmer easily understands. "
            "Keep sentences short. Avoid bookish or heavily Sanskritized words; prefer the "
            "common spoken word. Say ପାଣି not ଜଳ, ମାଟି not ମୃତ୍ତିକା, କ୍ଷେତ not କ୍ଷେତ୍ର, "
            "ଗଛ not ଉଦ୍ଭିଦ, ଦରକାର not ଆବଶ୍ୟକ, ପାଇଁ not ନିମନ୍ତେ. This is spoken aloud to "
            "someone with little schooling — bookish Odia sounds like a government notice "
            "and the advice is lost.\n"
            "1b. Every sentence must be COMPLETE and stand on its own when heard once. "
            "Translate the MEANING, not word by word: rebuild the sentence the way an Odia "
            "speaker would say it, with the verb and ending it needs, even if that changes "
            "the English word order. Never leave a clause dangling or a verb missing. Read "
            "each sentence back to yourself — if it would sound unfinished or odd spoken "
            "aloud, rewrite it.\n"
            "1c. NEVER translate an English abstract noun literally. Say what physically "
            "happens instead. 'water management' is NOT ପାଣି ପରିଚାଳନା — say କ୍ଷେତରେ ପାଣି "
            "କେତେ ରଖିବେ. 'recovery' for a crop is NOT ମରାମତି (repairing a machine) and NOT "
            "ଆରୋଗ୍ୟ (a person getting over an illness) "
            "— say ଫସଲ ଭଲ ହେବା. 'support the plants' is NOT ଗଛମାନେ ସହାୟତା — say ଗଛକୁ ଶକ୍ତି "
            "ଦେବା. 'stress' is NOT ଚାପ — say ଗଛ ଦୁର୍ବଳ ହେବା.\n"
            "1d. Get the GRAMMAR of the object right. If something is done TO the plants, "
            "they take the object ending (ଗଛମାନଙ୍କୁ), never the subject form (ଗଛମାନେ) — "
            "otherwise the farmer hears that the plants are doing it. Use ସେଗୁଡ଼ିକୁ for "
            "plants and things, never the honorific ତାଙ୍କୁ, which is for people.\n"
            # A mixed register is what makes the answer sound machine-made: the
            # reported output said ଥାମିବେ (future tense, "will halt") in one
            # sentence and ଲଗାନ୍ତୁ (imperative) in the next.
            "1e. Every instruction must be a POLITE IMPERATIVE ending in ...ନ୍ତୁ, and the "
            "whole answer must keep that one form. Address the farmer as ଆପଣ throughout. "
            # ଆପଣଙ୍କୁ is the dative. Odia allows it as the subject of ଉଚିତ
            # ("ଆପଣଙ୍କୁ ... କରିବା ଉଚିତ"), and English "You should ..." pulls the
            # translator straight to it — but an imperative takes the plain
            # nominative, so the two cannot appear together.
            "The subject of an instruction is ଆପଣ, NEVER the dative ଆପଣଙ୍କୁ: write "
            "ଆପଣ ପ୍ରୟୋଗ କରନ୍ତୁ, never ଆପଣଙ୍କୁ ପ୍ରୟୋଗ କରନ୍ତୁ, which reads as applying "
            "the fertiliser TO the farmer. Keep ଆପଣଙ୍କୁ only where the farmer is the "
            "OBJECT (ଏହା ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବ). "
            "Say ବନ୍ଦ କରନ୍ତୁ ('stop'), ଲଗାନ୍ତୁ ('apply'), ରଖନ୍ତୁ ('keep'), ଦିଅନ୍ତୁ ନାହିଁ "
            "('do not give'). NEVER use the future tense (ଥାମିବେ, କରିବେ, ରଖିବେ) for an "
            "instruction — that tells the farmer what will happen, not what to do.\n"
            "1e-i. ONLY THE MAIN VERB of a sentence takes ...ନ୍ତୁ. A verb inside the "
            "sentence stays a verbal noun: write ଲଗାଇବା ବନ୍ଦ କରନ୍ତୁ ('stop applying'), "
            "NEVER ଲଗାନ୍ତୁ ବନ୍ଦ କରନ୍ତୁ. Two ...ନ୍ତୁ verbs in a row is always wrong.\n"
            "1f. Say lodging the way the farmer said it: ପଡ଼ିଯାଇଛି / ପଡ଼ିଗଲା. Do not use "
            "ଝୁକିଗଲା ('leaned over') and do not invent another word. If the crop or variety "
            "is already named, do NOT add ଫସଲ again — write ଆପଣଙ୍କର ପୂଜା ଧାନ ପଡ଼ିଯାଇଛି, not "
            "ପୂଜା ଧାନ ଫସଲ ପଡ଼ିଯାଇଛି.\n"
            "2. Preserve agricultural meaning exactly.\n"
            "3. Keep fertilizer PRODUCT names unchanged: DAP, MOP, NPK, SSP.\n"
            "4. Write nutrient ELEMENTS by their full Odia name for the farmer, NEVER a "
            "bare letter symbol: N -> ନାଇଟ୍ରୋଜେନ, P -> ଫସଫରସ, K -> ପୋଟାସିୟମ, S -> ଗନ୍ଧକ, "
            "Zn -> ଜିଙ୍କ, Fe -> ଲୌହ. A single letter like P or S means nothing to a farmer.\n"
            "5. Use natural Odia sentence structure, not word-by-word translation.\n"
            "6. Use correct Odia spelling and grammar.\n"
            "7. Preserve numbers, percentages and fertilizer grades exactly.\n"
            "8. Return only the translated Odia text.\n"
            "9. Do not explain or add extra information.\n"
            "10. NEVER leave English words in Latin script in the output. Every English "
            "agricultural term MUST either:\n"
            "    a) Be translated to its natural Odia equivalent if one exists, OR\n"
            "    b) Be written in Odia script (transliterated) ONLY if no Odia word exists.\n"
            "    c) Do NOT transliterate (spell out the English sound in Odia letters) a "
            "word that has an Odia translation — TRANSLATE it. For example 'tillering' must "
            "become its Odia term (କୁଆଁରା ବାହାରିବା), never the phonetic spelling ଟିଲେରିଂ. "
            "Phonetic transliteration is allowed ONLY for names with no Odia word, such as "
            "chemical names.\n"
            "11. The ONLY Latin-script characters allowed in the output are: numbers, "
            "fertilizer abbreviations (DAP, MOP, NPK, N, P, K), and units (kg, g, ha, L).\n"
            + glossary_intro
            + glossary_block
            + "\nTEXT TO TRANSLATE:"
        )
    elif source_lang == "Odia" and target_lang == "English":
        # Pin TECHNICAL TOKENS first — before the glossary, before the LLM.
        # A dictated acronym arrives as Odia letter names (BGA -> ବିଜିଏ) and has
        # no glossary entry, so the translator free-translates it into a string
        # the documents never contain ("blue green algae" appears 0 times in the
        # corpus; "BGA" appears 10 times). Retrieval then misses a chunk that
        # plainly holds the answer and the question is reported as
        # "no information" — an answerable question lost silently.
        # protect_input_terms() rewrites the token to its canonical Latin form,
        # so the model only has to leave it alone. See technical_terms.py.
        # EVERY STAGE IS LOGGED. Two production faults were invisible from the
        # final English alone: a spoken range collapsed into a sum
        # ('ବାର ଦଶ ପନ୍ଦର' -> 37) and a term with no glossary entry
        # (ରୋଗପୋକ) rendered as something else entirely. Which stage did it
        # is now readable in the log instead of reconstructible by hand.
        log.info("translate_in | 1 original odia = %r", text)
        text, pinned_tokens = protect_input_terms(text)
        log.info("translate_in | 2 after protect_input_terms = %r (pinned=%s)",
                 text, pinned_tokens)

        # Collapse spelling variants to one canonical term BEFORE translating
        # (ଅଜୋଲା / ଅଜୋଲ୍ଲା / ଅଝୋଲା -> Azolla). See synonyms.py.
        text, syn_terms = canonicalise(text)
        log.info("translate_in | 3 after canonicalise = %r", text)
        pinned_tokens = pinned_tokens + [t for t in syn_terms if t not in pinned_tokens]

        # BEFORE the numbers are read: a sibilant the speech model misheard can
        # hide a number word from odia_numbers entirely (ସହେ for ଶହେ).
        text = _repair_odia_mishearings(text)

        # Convert spoken Odia numbers to digits IN CODE, before the model sees
        # them. Odia fractional words carry a value with no digit to copy
        # (ଅଢ଼େଇ = 2.5), and the model was dropping them outright — turning a
        # 2.5-acre dose question into a 1-acre one. See odia_numbers.py.
        text, num_conversions = convert_odia_numbers(text)
        log.info("translate_in | 4 after convert_odia_numbers = %r (converted=%s)",
                 text, num_conversions or "nothing")

        # Translate the farmer's Odia question into the EXACT standard English terms
        # the documents use (reverse glossary), so retrieval matches well.
        rev = relevant_reverse_glossary(text)
        log.info("translate_in | 5 glossary matches = %s", rev or "none")
        glossary_terms = list(rev.values())
        rev_block = ("\n".join(f"    - {od} -> {en}" for od, en in rev.items())
                     if rev else "")
        sys = (
            "Translate the following Odia agricultural text into clear, SIMPLE English. "
            "Use common everyday words. Return ONLY the English translation.\n"
            # This section previously taught number conversion with two worked
            # examples, both containing 7061. The model ANCHORED on it: asked
            # about ସି ଆର୍ ହଜାରେ ନଅ (CR 1009) it answered "CR Dhan 7061" —
            # copying the digits straight out of its own instructions. Numbers
            # are now converted deterministically in odia_numbers.py, so the
            # prompt no longer teaches conversion and carries NO sample digits
            # for the model to copy.
            "NUMBERS: every number in the text is ALREADY written in digits and is "
            "already correct. Copy each one EXACTLY as it appears. Never recompute, "
            "round, re-spell, merge or drop a number, and never invent one that is "
            "not in the text.\n"
            "RICE VARIETIES: variety names usually end in a number (for example the "
            "pattern 'CR Dhan <number>' or 'MTU <number>'). Keep the number attached "
            "to its name exactly as written, and never read it as a quantity of "
            "varieties.\n"
            "'DHAN' IS PART OF THE NAME: in a variety name, Dhan (ଧାନ) is a proper "
            "noun, not the word for rice. Write 'CR Dhan <number>', never "
            "'CR rice <number>'. Keep the prefix capitalised: CR, MTU, NRRI.\n"
            # A farmer reports crop damage as a VERB, and a dropped verb silently
            # changes the question. "ପୂଜାଧାନ ୫୫ ଦିନରେ ପଡ଼ିଗଲା, ସାର ଦେଇପାରିବି?"
            # became "Pooja Dhan 55 is ready in 55 days, can we apply manuring?"
            # — the lodging vanished, the number was glued onto the variety name,
            # and "is ready" was invented. The answer then advises fertiliser,
            # while the documents say excess fertiliser CAUSES lodging.
            "CROP CONDITION VERBS ARE THE POINT OF THE QUESTION: never drop or "
            "soften a verb describing what happened to the crop, and never replace "
            "it with a growth-stage statement. If a crop (ଧାନ, ଫସଲ, ଗଛ, or a named "
            "variety) ପଡ଼ିଗଲା / ପଡ଼ିଛି / ପଡ଼ିଯାଇଛି / ନଇଁପଡ଼ିଛି, the crop FELL OVER — "
            "translate that as LODGING (e.g. 'the crop has lodged'). It does NOT "
            "mean the crop is ready, mature or harvested. Other condition verbs "
            "(ଶୁଖିଗଲା dried, ପଚିଗଲା rotted, ମରିଗଲା died, ହଳଦିଆ ହୋଇଗଲା turned yellow) "
            "must likewise be kept.\n"
            "A NUMBER IS NOT PART OF A VARIETY NAME WHEN A UNIT FOLLOWS IT: in "
            "'<variety> ୫୫ ଦିନରେ', the 55 belongs to ଦିନ (days) and is a duration — "
            "write '<variety> at 55 days', NEVER '<variety> 55'. Only attach a "
            "number to a variety name when no unit word follows it.\n"
            "VARIETY NAMES ARE PROPER NOUNS: Rice variety names are names, not descriptions. "
            "Keep them as the standard romanized variety name; do NOT translate the literal "
            "meaning of the words inside the name. Many variety names are built from ordinary "
            "words (colours, places, deities), so translating them corrupts the name and breaks "
            "search. For example କଳାଚମ୍ପା is the variety 'Kalachampa' — output 'Kalachampa', "
            "NEVER 'black champa'. Likewise keep names such as Lalat, Pooja, Sahbhagi Dhan, "
            "Swarna, Kalachampa unchanged. When unsure whether a word is a variety name, keep "
            "it romanized rather than translating its meaning.\n"
            # The rule above is right and it over-fires: told that variety names
            # are proper nouns, the model romanised the ADJECTIVE in front of
            # one. "ବିଳମ୍ବ କିସମ" (a late/long-duration variety) came back as
            # "the Bilamb variety" — a cultivar that does not exist, so the
            # documents had nothing to match and the answer described a
            # made-up variety. Duration words are descriptions, and the corpus
            # writes them in English, so they must be translated.
            "BUT A DURATION WORD IS NOT A NAME: a word in front of କିସମ / ପ୍ରଜାତି / "
            "ଧାନ that describes how LONG the crop takes is a description, never a "
            "variety name. Translate it: ବିଳମ୍ବ (or ଡେରି) = late-maturing / long "
            "duration, ମଧ୍ୟମ = medium-duration, ଅଳ୍ପ କାଳୀନ (or ଶୀଘ୍ର) = "
            "early-maturing / short duration. Write 'late-maturing variety', never "
            "'Bilamb variety'. The same goes for any other ordinary adjective before "
            "କିସମ (ଉନ୍ନତ = improved, ଅଧିକ ଅମଳକ୍ଷମ = high-yielding)."
            # THE HINTS WERE ADVISORY AND THE MODEL TREATED THEM THAT WAY.
            # "Use these EXACT English terms" lost to the model's own reading of
            # the sentence: told ହଳ କରିବା -> ploughing, it still wrote "cut",
            # because it had already mistranslated a NEIGHBOURING word
            # (ରୋଇବା, transplanting) as "harvesting" and then made the rest of
            # the sentence agree with that. A farmer was told to cut the crop
            # when they had asked when to plough it. The block below is worded
            # as a constraint, not a suggestion, and names the one rewrite that
            # actually happens: turning one field operation into another.
            + (("\n\nGLOSSARY MATCHES — AUTHORITATIVE TERMINOLOGY CONSTRAINTS:\n"
                + rev_block
                + "\nFor every Odia term listed above you MUST use the English "
                "meaning given, in the final translation. These are constraints, "
                "not suggestions:\n"
                "  - Do NOT replace a matched term with a synonym.\n"
                "  - Do NOT reinterpret or omit a matched term.\n"
                "  - Do NOT change a matched term to make the sentence read "
                "more naturally, and do NOT change one because of what you "
                "think a neighbouring word means.\n"
                "  - ALL of the matches must survive together; keeping one is "
                "not enough.\n"
                "  - NEVER turn one field operation into another. Transplanting "
                "is not harvesting, ploughing is not cutting, sowing is not "
                "harvesting, and ploughing is not tillage when the text says "
                "ploughing.\n"
                "Natural English word order and grammar are welcome. The "
                "meanings above are not negotiable.\n")
               if rev else "")
            + pin_list_for_prompt(pinned_tokens)
            + conversions_for_prompt(num_conversions)
        )
    else:
        sys = (
            f"You are a translator. Translate from {source_lang} to {target_lang}. "
            f"Return ONLY the translated text. Preserve agricultural terminology."
        )
    
    result = ""

    # Answer translation (English -> Odia) now uses the Google Translate API.
    # The deterministic glossary + transliteration passes below still normalise
    # agricultural terms, so Google's plain translation is fine as the base.
    # Glossary terms lifted out of the English before translating. The markers
    # stay in the text through EVERY post-pass and are swapped for the glossary's
    # Odia at the very end — otherwise the later passes silently overrule the
    # glossary: _correct_odia rewrites ବୀଜ->ବିହନ, undoing the forced "seed rate",
    # and _fix_transliterations replaces the forced ବର୍ମିକମ୍ପୋଷ୍ଟ.
    glossary_marks: dict = {}

    if target_lang == "Odia":
        try:
            # Lift the glossary terms out so Google translates AROUND them. This
            # is the only thing that makes the glossary bind on the Google path.
            log.info("translate_out | 1 english answer = %r", text)
            marked, glossary_marks = _force_glossary_terms(text)
            log.info("translate_out | 2 glossary forced = %s",
                     dict(zip(glossary_marks, glossary_marks.values()))
                     if glossary_marks else "none")
            log.info("translate_out | 3 marked input to google = %r", marked)
            # The same English, translated with NOTHING forced, for comparison.
            _log_baseline_translation(text, source_lang, target_lang,
                                      "translate_out")
            result = _google_translate(marked, source_lang, target_lang)
            log.info("translate_out | 4 raw google output = %r", result)
            # A MARKER GOOGLE DROPS IS A CLAUSE THE FARMER NEVER HEARS. Live:
            # "for effective ZQX2" came back as a stranded ପ୍ରଭାବଶାଳୀ with ZQX2
            # gone, so "nutrient uptake" vanished from the answer. Restoring
            # cannot bring back what is not there, and _restore_glossary_terms
            # only catches markers that survived MANGLED. So the sentence is
            # translated once more with the dropped terms left inline: the
            # meaning is kept, at the cost of that one term's exact wording.
            dropped = [m for m in glossary_marks if m not in result]
            if dropped:
                lost = [glossary_marks[m] for m in dropped]
                log.warning("translate_out | google dropped glossary marker(s) %s "
                            "%s; translating again with them inline", dropped, lost)
                marked, glossary_marks = _force_glossary_terms(text, exclude=lost)
                result = _google_translate(marked, source_lang, target_lang)
                log.info("translate_out | 4b raw google output, retry = %r", result)
            # Substitute here, NOT at the end of the function. Carrying markers
            # any further sends them through _detransliterate_odia, which is an
            # LLM pass: it treats ZQX0XQZ as a word it should improve and swaps
            # in a plausible Odia term, so the glossary value is lost and
            # sometimes the term disappears altogether.
            # The approved Odia the glossary just put into the text. Handed to
            # the fluency pass so it knows which words are not its to change —
            # it runs AFTER the markers are gone, so without this it sees them
            # as ordinary wording and "improves" them.
            forced_odia = list(glossary_marks.values())
            result = _restore_glossary_terms(result, glossary_marks)
            glossary_marks = {}
            log.info("translate_out | 5 glossary restored = %r", result)
            # Google keeps English word order on longer sentences and the
            # restored glossary terms sit in the English verb slot (see
            # _fix_odia_fluency). Repair the syntax NOW, so every later pass
            # (_correct_odia, _detransliterate_odia, to_spoken) operates on
            # well-formed sentences. The LLM fallback below does NOT need this:
            # its prompt already demands natural Odia word order.
            # Action-verb entries were not frozen; their approved word is still
            # required, inflected. Passed only when there are any, so the call is
            # unchanged for every answer that has none.
            action_stems = list(glossary_action_stems(text).values())
            if action_stems:
                log.info("translate_out | 2b action verbs (inflect, not frozen) = %s",
                         action_stems)
            result = _fix_odia_fluency(
                result, text, glossary_terms=forced_odia,
                **({"glossary_stems": action_stems} if action_stems else {}))
            log.info("translate_out | 6 after fluency rewrite = %r", result)
        except Exception as exc:
            glossary_marks = {}
            log.warning("Google Translate Odia failed (%s); falling back to LLM", exc)

    # --- previous model-based Odia translation (Gemini 2.5 flash -> OpenAI) ---
    # Commented out in favour of Google Translate above. Kept for reference /
    # quick rollback.
    # if target_lang == "Odia" and settings.GEMINI_API_KEY:
    #     try:
    #         result = _gemini_text(sys, text)
    #     except Exception as exc:
    #         log.warning("Gemini Odia translation failed (%s); falling back to OpenAI", exc)

    # ODIA -> ENGLISH: GOOGLE FIRST, OPENAI AS THE FALLBACK.
    #
    # Both engines are good at different halves of this. On "ଡିଏପି ଗୋଟେ ଏକରରେ
    # କେତେ ପକେଇବା?" the LLM produced "How much can be harvested per acre in
    # DAP 1" — an invented operation and a variety name made from a quantity —
    # where Google gives "How much DAP to plant in 1 acre?". Google's weakness is
    # the opposite one: it does not know this project's glossary, so ରୋଇବା came
    # back as "planting" rather than the glossary's "transplanting".
    #
    # So Google runs first with the glossary substituted INTO the sentence, and
    # the result is only accepted if every substituted term actually survived.
    # A dropped term means the sentence was rebuilt around it, and the OpenAI
    # path — whose prompt carries the same glossary as authoritative constraints
    # — is the better answer for that question.
    if not result and source_lang == "Odia" and target_lang == "English":
        try:
            marked, applied = _force_reverse_glossary_terms(text)
            if applied:
                log.info("translate_in | glossary matches = %s", applied)
                log.info("translate_in | protected input = %r", marked)
            log.info("translate_in | 6 protected input to google = %r", marked)
            # The same Odia question with NOTHING forced, for comparison: stage
            # 7 below is always the glossary-influenced translation.
            _log_baseline_translation(text, "Odia", "English", "translate_in")
            candidate = _google_translate(marked, "Odia", "English")
            google_candidate = candidate
            log.info("translate_in | 7 raw google output = %r", candidate)
            missing = _missing_glossary_terms(candidate, applied)
            money = _money_intrusion(text, candidate)
            if money:
                log.warning("translate_in | Google turned this into a COST "
                            "question (%r) that the Odia never asked; falling "
                            "back to OpenAI | google output = %r",
                            money, candidate)
            if (candidate or "").strip() and not missing and not money:
                result = candidate
                accepted_from_google = True
                log.info("translate_in | 8 glossary-restored output = %r", result)
                log.info("translate_in | backend = google | output = %r", result)
            else:
                log.warning("translate_in | Google dropped %s; falling back to "
                            "OpenAI | google output = %r",
                            missing or "the whole translation", candidate)
        except Exception as exc:                            # noqa: BLE001
            log.warning("translate_in | Google failed (%s); falling back to OpenAI",
                        exc)

    # Fallback for non-Odia targets, and a safety net if Google Translate is
    # unavailable (missing key / network error) so a response is still produced.
    if not result:
        try:
            result = _chat([{"role": "system", "content": sys}, {"role": "user", "content": text}])
        except Exception as exc:
            log.warning("OpenAI translation failed (%s)", exc)
            result = ""
        # A VAGUE TRANSLATION BEATS A FABRICATED ONE. The fallback runs because
        # Google dropped a glossary term — a real but small loss. If it answers
        # by inventing a proper noun Google never produced, that is the bigger
        # harm: retrieval goes looking for a variety that does not exist, and the
        # farmer is answered about it. Google's wording is used instead.
        if source_lang == "Odia" and target_lang == "English" and google_candidate:
            invented = _invented_name(result, google_candidate)
            if invented:
                log.warning("translate_in | fallback invented %r, which Google did "
                            "not produce; using Google's wording instead (%r)",
                            invented, google_candidate)
                result = google_candidate

    # Resilience for the farmer's Odia question -> English (used for retrieval):
    # the OpenAI call rarely fails or returns empty. Fall back to Google Translate,
    # then to the original text, so the pipeline never dies at this step.
    if not (result or "").strip() and source_lang == "Odia" and target_lang == "English":
        try:
            result = _google_translate(text, "Odia", "English")
            log.info("Odia->English used Google Translate fallback")
        except Exception as exc:
            log.warning("Google Translate Odia->English fallback failed (%s)", exc)
    if not (result or "").strip():
        result = text   # last resort: pass the original through unchanged

    # A pinned token can still be paraphrased away by the model. Retrieval only
    # needs the token PRESENT, so re-assert it rather than paying for a repair
    # round-trip.
    if pinned_tokens:
        result = reassert_tokens(result, pinned_tokens)

    # Canonicalise the ENGLISH to the corpus's own vocabulary.
    #
    # Re-asserting the reverse-glossary terms here was tried and made things
    # WORSE: for "ପୂଜାଧାନ ୫୫ ଦିନରେ ପଡ଼ିଗଲା ... ସାର ପ୍ରୟୋଗ" the only term the
    # glossary resolves is ସାର ପ୍ରୟୋଗ, so it appended "fertilizer application"
    # and pushed the lodging chunks out of the top 10 altogether. Amplifying the
    # terms we already match is not the fix — the missing one is the problem.
    if source_lang == "Odia" and target_lang == "English":
        # DOES THE ENGLISH STILL SAY WHAT THE FARMER SAID?
        #
        # Everything above this line chose between ENGINES — Google unless it
        # dropped a glossary term or invented a cost question, OpenAI otherwise.
        # None of it asked whether the sentence still means the same thing, so
        # two engines agreeing on a wrong reading passed straight through. This
        # is the answer direction's own pattern — detect, retry once naming the
        # fault, validate or keep the original — applied to the question.
        #
        # Runs on the PROCESSED Odia, which is what was actually translated:
        # numbers are already digits here, so they can be compared directly.
        # See odia_semantics for why every check needs evidence from both sides.
        faults = validate_odia_to_english(text, result)
        if faults:
            log.warning("translate_in | 8b semantic validation: %s",
                        "; ".join(str(f) for f in faults))
            # REPAIRED ONLY ON THE GOOGLE PATH, and the asymmetry is deliberate.
            # Reaching the OpenAI fallback already means two opinions have been
            # taken: Google's, and then OpenAI's under a prompt that carries the
            # SAME constraints this repair would impose — the glossary as
            # authoritative, never swap one field operation for another, keep
            # every number and variety name. Asking a third time is a weaker
            # prompt second-guessing a stronger one at the price of another
            # call. The gap this closes is the Google path, where the existing
            # guards passed the sentence and nobody looked at its meaning.
            # The fault is logged either way, so the fallback path stays visible.
            if not accepted_from_google:
                log.info("translate_in | not repairing: the OpenAI fallback "
                         "already applied the same constraints")
            elif _semantic_repair_enabled():
                repaired = _repair_translation(text, result, faults)
                if repaired:
                    remaining = validate_odia_to_english(text, repaired)
                    if remaining:
                        # KEEPING THE ORIGINAL IS THE SAFE FAILURE. A repair
                        # that is still wrong may be wrong in a NEW way, and
                        # the first translation at least came from an engine
                        # whose failure modes the guards above already cover.
                        log.warning("translate_in | 8d repair still faulty (%s); "
                                    "keeping the original translation = %r",
                                    "; ".join(str(f) for f in remaining), result)
                    else:
                        # THE REPAIR REWRITES THE SENTENCE, SO THE PINS MUST BE
                        # RE-ASSERTED. reassert_tokens ran above, on the text
                        # the repair has just replaced: a dictated acronym that
                        # was rescued as "... (BGA)" is dropped again the moment
                        # the model rewrites the sentence, and retrieval needs
                        # that token PRESENT — it is the whole reason the token
                        # was pinned. Cheap, idempotent, and a no-op when the
                        # repair kept them.
                        if pinned_tokens:
                            repaired = reassert_tokens(repaired, pinned_tokens)
                        log.info("translate_in | 8c semantic repair accepted = %r",
                                 repaired)
                        result = repaired
            else:
                log.info("translate_in | semantic repair is switched off; "
                         "keeping the translation as it is")
        # Second pass on the LATIN side. The translator invents its own
        # romanisation ("Ajola", "Azola", "blue green algae") that no Odia-side
        # rule can predict, so the variants have to be collapsed again after
        # translation — this is where most of them actually appear.
        result, _ = canonicalise(result)
        result = _canonicalize_english(result)
        log.info("translate_in | 9 final output = %r", result)

    if target_lang == "Odia":
        result = apply_glossary(result)        # fast local replacements first
        latin_words = re.findall(r'\b[a-zA-Z]{3,}\b', result)
        # The old hardcoded allow-list was {DAP, MOP, NPK, kg, ha, pH}, so every
        # OTHER acronym the documents use — BGA, SSP, DSR, SSNM, LCC, FYM, IRRI,
        # MTU — was handed to an LLM to be "fixed" into Odia script, destroying
        # the acronym in the answer the farmer hears. is_allowed_latin() covers
        # the corpus-grounded set (technical_terms.py) and is case-insensitive.
        # Glossary markers are Latin-looking but must not be "fixed" — they are
        # placeholders waiting to become Odia at the end of this function.
        unexpected = [w for w in latin_words
                      if not is_allowed_latin(w) and not _MARK_RE.fullmatch(w)]
        if unexpected:
            log.warning("Odia translation contains Latin words: %s", unexpected)
            fix_prompt = (f"Fix this Odia text — these words are still in Latin script: "
                        f"{unexpected}. Transliterate them into Odia script. "
                        f"Return ONLY the corrected text.\n\n{result}")
            result = _chat([{"role": "user", "content": fix_prompt}])
        result = _correct_odia(result)
    
    # Apply Odia corrections if translating to Odia
    if target_lang == "Odia":
        result = _correct_odia(result)
        result = _fix_transliterations(result)
        result = _detransliterate_odia(result, text)
        # Re-apply deterministically: the LLM cleanup above can reintroduce a
        # known transliteration while rephrasing — normalise it again for free.
        result = _fix_transliterations(result)
        # Bookish -> spoken. Deterministic, because rule 1 above does not hold on
        # its own: the model keeps producing ଜଳ / କ୍ଷେତ୍ର / ମୃତ୍ତିକା. Runs late so
        # it also catches words reintroduced by the cleanup passes.
        result = to_spoken(result)
        # Register: "should apply" → "apply". The translator loves "କରିବା ଉଚିତ";
        # farmers want direct instructions. Field feedback: ଉଚିତ reads as a
        # hedge and makes advice sound optional.
        result = to_imperative(result)
        result = _expand_element_symbols(result)
        # Show numbers in Odia numerals on screen too (e.g. 20-25 -> ୨୦-୨୫).
        result = result.translate(_ODIA_DIGITS)

    # LAST: swap the markers for the glossary's Odia, after every pass that could
    # have overruled them has already run.
    if glossary_marks:
        result = _restore_glossary_terms(result, glossary_marks)

    if target_lang == "Odia":
        # One canonical encoding leaves this function. The answer is stored on
        # the Query row and is the input to the TTS path, so a decomposed vowel
        # here becomes a mispronounced word later (see compose_odia).
        result = compose_odia(result)

    return result


def classify_and_enrich(query_text_en: str) -> dict:
    """
    Extract task_type, problem_entity, category, season, crop_stage, missing_context as JSON.
    """
    sys = (
        "You are an agricultural query classifier. Given a farmer's rice-related question, "
        "respond ONLY with a JSON object having these fields:\n"
        '{"category": one of ["Rice Variety","Disease","Fertiliser","Water Mgmt","Others"],\n'
        ' "task_type": one of ["Diagnosis","Variety Suitability","Fertiliser Rec.","Management Advice","Other"],\n'
        ' "problem_entity": short phrase (e.g. "Blast", "Urea dose"),\n'
        ' "season": one of ["Kharif","Rabi","Zaid",""] (empty if unknown),\n'
        ' "crop_stage": one of ["Nursery","Tillering","Flowering","Mature Grain",""] (empty if unknown),\n'
        ' "missing_context": short comma-separated list of context the farmer did not provide '
        '(e.g. "variety, land type") or empty string}'
    )
    raw = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": query_text_en}],
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# _QUERY_EXPANSION_PROMPT = (
#     "You rewrite a farmer's agriculture question into alternative search queries "
#     "so a document search can find the answer even when the wording differs.\n"
#     "Given the question, return ONLY a JSON array of 3 short alternative phrasings "
#     "in English. Keep all key terms (crop names, variety codes, chemicals, growth "
#     "stages, numbers). Vary the wording/synonyms, not the meaning. No markdown, no "
#     "explanation."
# )

_QUERY_EXPANSION_PROMPT = (
    "You are rewriting a farmer's agriculture question only to improve document retrieval.\n"
    "Return ONLY a JSON array of exactly 3 alternative queries in English.\n\n"
    "Rules:\n"
    "- The meaning and intent MUST remain exactly the same as the original question.\n"
    "- Do NOT add, remove, assume, or change any information.\n"
    "- Preserve all crop names, variety names/codes, chemicals, fertilizers, pesticides, diseases, pests, growth stages, numbers, dates, and locations exactly as given.\n"
    "- Only make small wording changes such as reordering words, replacing with equivalent synonyms, or changing sentence structure.\n"
    "- Never broaden or narrow the question.\n"
    # A REWRITE MAY SOFTEN A SUPERLATIVE; IT MAY NEVER ADD ONE. A farmer says
    # "which is the best seed for high land" meaning "which one suits high
    # land", and the documents rate varieties as fit or suitable, never as best.
    # Rewriting that to "best variety" makes the search look for a ranking no
    # document contains — and then the answer either refuses or invents one.
    "- A farmer's 'best', 'good' or 'better' means SUITABLE or RECOMMENDED. You "
    "may rewrite it that way. You must NEVER add 'best', 'highest', 'top' or "
    "any other ranking word that the original question did not use.\n"
    "- If a safe alternative cannot be created, repeat the original question.\n"
    "- Output ONLY a JSON array of 3 strings. No markdown or explanation."
)


# Farmer wording -> the EXACT term the documents use. When a farmer phrase on the
# left appears in the (English) query, we add a search variant containing the
# document term on the right, so both vector and keyword search can match. This
# is the deterministic safety net for vocabulary gaps the LLM rephrasing misses
# (e.g. "dry land" / "no standing water" -> the documents say "upland").
import re as _re
_DOMAIN_SYNONYMS = [
    # "high land" / "highland" is how farmers here say upland, and it was the
    # one phrasing missing from this list: the question went to the index with
    # the farmer's word and the documents' word ("upland") was never searched
    # for. The canonical terms on the right are the corpus's own vocabulary —
    # see entity_intent.LAND_TYPES, which holds the same equivalence for the
    # ANSWER side.
    (_re.compile(r"\b(high\s*land|highland|up\s+land)\b", _re.I), "upland"),
    (_re.compile(r"\b(dry\s*land|dry\s*field|dryland|no standing water|"
                 r"without standing water|stays dry|field stays dry|rainfed|"
                 r"un-?irrigated|drought[- ]?prone)\b", _re.I), "upland"),
    (_re.compile(r"\b(low\s*land|lowland|water[- ]?logged|"
                 r"submerged|flood[- ]?prone)\b", _re.I), "lowland"),
    # The farmer says "drought-resistant"; the documents say "drought tolerant" /
    # "stress tolerant" (e.g. the Sahbhagi Dhan factsheet). Bridge the wording so
    # retrieval finds those variety documents.
    (_re.compile(r"\b(drought[- ]?(resistant|tolerant|hardy|proof)|"
                 r"(survives?|withstands?|resist(s|ant to)?)\s+drought|"
                 r"stress[- ]?tolerant)\b", _re.I),
     "drought tolerant stress tolerant variety"),
    (_re.compile(r"\b(medium land|medium-?deep)\b", _re.I), "medium land"),
    (_re.compile(r"\b(cow\s*dung|farmyard|fym)\b", _re.I), "farmyard manure"),
    # The farmer asks for "potash"; the documents NEVER use that word. The only
    # potassium dose in the corpus is written as "MOP 15-20 kg/acre" (the DSR
    # nutrient chunk), and "potash" matches 0 chunks on a keyword search. So a
    # correctly translated potash question still came back "no information
    # available". Bridge it to the documents' own abbreviation.
    (_re.compile(r"\b(potash|pot\s*ash|muriate of potash|k2o|"
                 r"potassium fertili[sz]er)\b", _re.I),
     "MOP muriate of potash"),
    (_re.compile(r"\b(green manur\w*|dhanicha|dhaincha|sesbania)\b", _re.I),
     "green manuring"),
    # Dhanicha/Dhaincha/Sesbania are the SAME crop spelled differently in the
    # documents; search all spellings so benefit/usage chunks are not missed.
    (_re.compile(r"\bdhanicha\b", _re.I), "dhaincha"),
    (_re.compile(r"\bdhaincha\b", _re.I), "dhanicha"),
    (_re.compile(r"\b(seed\s*amount|how much seed|quantity of seed)\b", _re.I),
     "seed rate"),
    (_re.compile(r"\b(spacing|gap between plants|plant to plant)\b", _re.I),
     "spacing"),
    # ponytail: duration class bridging for fertiliser questions
    (_re.compile(r"\b(short[\s-]*duration|early[\s-]*maturing|early[\s-]*variety"
                 r"|early[\s-]*rice|early[\s-]*maturity)\b", _re.I),
     "short duration variety 100-110 days"),
    (_re.compile(r"\b(medium[\s-]*duration|mid[\s-]*season|medium[\s-]*maturing"
                 r"|medium[\s-]*time|medium[\s-]*maturity|medium[\s-]*period)\b", _re.I),
     "medium duration variety 120-135 days"),
    (_re.compile(r"\b(long[\s-]*duration|late[\s-]*maturing|late[\s-]*variety"
                 r"|long[\s-]*time|long[\s-]*maturity|long[\s-]*period)\b", _re.I),
     "long duration variety 140-160 days"),
]


def _synonym_variants(query_text_en: str) -> list:
    """Add query variants that contain the document's own term for any farmer
    synonym present, so retrieval matches the documents' vocabulary."""
    out = []
    low = query_text_en.lower()
    for pattern, canonical in _DOMAIN_SYNONYMS:
        if pattern.search(query_text_en) and canonical.lower() not in low:
            out.append(f"{query_text_en} {canonical}")
            out.append(f"rice variety for {canonical}"
                       if canonical in ("upland", "lowland", "medium land")
                       else canonical)
    return out


def _expand_queries(query_text_en: str) -> list:
    """Generate a few alternative phrasings of the English query for multi-query
    retrieval. Always includes the original + deterministic synonym variants.
    Fails safe to just the original."""
    queries = [query_text_en]
    try:
        raw = _chat(
            [
                {"role": "system", "content": _QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": query_text_en},
            ],
        )
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        import json as _json
        for alt in _json.loads(raw):
            if isinstance(alt, str):
                alt = alt.strip()
                if alt and alt.lower() != query_text_en.lower() and alt not in queries:
                    queries.append(alt)
    except Exception as exc:
        log.warning("retrieve: query expansion failed (%s); using original only", exc)

    # Deterministic synonym variants always run (independent of the LLM).
    for v in _synonym_variants(query_text_en):
        if v not in queries:
            queries.append(v)
    # The same question with the entity noun in its other number, plus the
    # documents' own word for any land type or trait the farmer phrased
    # differently. This is what makes "which variety" and "which varieties"
    # search identically instead of merely being ANSWERED identically.
    # THE CANONICAL VARIANTS ARE NOT NEGOTIABLE. They are the searches written
    # in the corpus's own words ("... for upland?"), and they were losing their
    # slots to LLM rephrasings of the farmer's wording — a trace of "which rice
    # variety is suitable for high land?" showed "rice variety suitable for
    # upland" dropped by the cap outright. The farmer's original is first, the
    # canonical variants come next, and the rewrites fill whatever is left.
    canonical = [v for v in entity_intent.retrieval_variants(query_text_en)
                 if v not in (queries[:1] + [])]
    ordered = queries[:1] + [v for v in canonical if v.lower() != queries[0].lower()]
    for query in queries[1:]:
        if query not in ordered:
            ordered.append(query)
    return ordered[:6 + min(len(canonical), 4)]


def _hit_key(hit: dict):
    meta = hit.get("metadata") or {}
    key = (meta.get("document_id"), meta.get("chunk_index"))
    if key == (None, None):
        return (hit.get("chunk") or "")[:200]
    return key


def _merge_vector_hits(catalogue, queries: list, n_results: int) -> list:
    """Run vector search for every query phrasing; keep the closest copy of each
    chunk; return ranked by distance (closest first). All phrasings are embedded
    in ONE batch call (one embeddings round-trip instead of one per phrasing)."""
    best, order = {}, []
    for hits_for_query in search_catalogue_batch(catalogue, queries, n_results=n_results):
        for hit in hits_for_query:
            key = _hit_key(hit)
            prev = best.get(key)
            if prev is None:
                best[key] = hit
                order.append(key)
            else:
                pd, hd = prev.get("distance"), hit.get("distance")
                if hd is not None and (pd is None or hd < pd):
                    best[key] = hit
    hits = [best[k] for k in order]
    hits.sort(key=lambda h: (h.get("distance") is None, h.get("distance") or 0.0))
    return hits


def _merge_keyword_hits(catalogue, queries: list, n_results: int) -> list:
    """Run BM25 keyword search for every query phrasing; keep the highest-scoring
    copy of each chunk; return ranked by score (highest first)."""
    best, order = {}, []
    for qx in queries:
        try:
            kw_hits = keyword_search(catalogue, qx, n_results=n_results)
        except Exception as exc:
            log.warning("retrieve: keyword search failed (%s)", exc)
            kw_hits = []
        for hit in kw_hits:
            key = _hit_key(hit)
            prev = best.get(key)
            if prev is None:
                best[key] = hit
                order.append(key)
            else:
                if (hit.get("score") or 0) > (prev.get("score") or 0):
                    best[key] = hit
    hits = [best[k] for k in order]
    hits.sort(key=lambda h: -(h.get("score") or 0.0))
    return hits


_DOC_VOCAB = (
    "fertilizer", "fertiliser", "nutrient", "dose", "npk", "urea", "dap",
    "mop", "potash", "phosphorus", "nitrogen", "potassium", "basal",
    "top dressing", "top-dressing", "seed rate", "spacing", "pesticide",
    "fungicide", "insecticide", "irrigation",
)


def _canonical_from_expansions(original: str, queries: list) -> Optional[str]:
    """Pick the single best 'clean' rephrasing to show the answer step, or None.

    `queries` is the output of _expand_queries: index 0 is the farmer's original
    question, the LLM rephrasings follow, then deterministic synonym variants.
    We want a genuine rewrite in DOCUMENT vocabulary (e.g. 'manuring' -> 'fertilizer
    recommendation'), NOT a synonym variant that is just 'original + appended
    terms', and NOT a rewrite that merely echoes the farmer's colloquial word
    ('manure') when a cleaner phrasing exists.

    So we first drop candidates that still contain the original text, then prefer
    the remaining candidate richest in document vocabulary. This is what stops
    the answer step from receiving 'amount of manure for Swarna' (which the model
    treats as different from the 'fertilizer' context) when a 'fertilizer'
    phrasing is available. Returns None if there's no genuine rewrite, in which
    case the caller falls back to the farmer's original wording, unchanged."""
    orig = (original or "").strip().lower()
    # Non-agricultural, non-question input ("You are wrong", "Hello", "OK"):
    # the expansion LLM will hallucinate a topic. Don't rewrite.
    if not _QUESTION_WORD_RE.match(orig) and not any(
            t in orig for t in _DOC_VOCAB):
        return None
    candidates = []
    for q in queries[1:]:
        qs = (q or "").strip()
        if qs and qs.lower() != orig and orig not in qs.lower():
            candidates.append(qs)
    if not candidates:
        return None

    def _vocab_score(q: str) -> int:
        low = q.lower()
        return sum(1 for term in _DOC_VOCAB if term in low)

    # Stable sort: ties keep original (LLM) order, so the first clean rewrite wins
    # when none is vocabulary-richer.
    candidates.sort(key=_vocab_score, reverse=True)
    return candidates[0]


def _ranked_lists(catalogue, queries: list, pool: int) -> list:
    """One ranked list per (query, modality) — the input Reciprocal Rank Fusion
    is meant to be given.

    WHY NOT ONE MERGED LIST. The old path merged every phrasing's hits into a
    single list sorted by raw distance (vector) or raw BM25 score (keyword), and
    fused THAT. Both orderings are meaningless across queries: a short canonical
    query like "rice varieties recommended for upland" produces larger cosine
    distances and smaller BM25 scores than a long natural question, simply
    because it has fewer terms — so its hits sank to the bottom of the merged
    list and RRF, reading position, scored them as though the search had ranked
    them poorly. The canonical variants were added to reach rows written in the
    documents' words, and the fusion was quietly discounting exactly those.

    Keeping the lists separate is what makes rank comparable: rank 1 for one
    phrasing counts the same as rank 1 for another, which is the entire premise
    of RRF.
    """
    lists = []
    try:
        per_query = search_catalogue_batch(catalogue, queries, n_results=pool)
    except Exception as exc:                                # noqa: BLE001
        log.warning("retrieve: vector search failed (%s)", exc)
        per_query = [[] for _ in queries]
    for query, hits in zip(queries, per_query):
        if hits:
            lists.append({"query": query, "modality": "vector", "hits": hits})
    for query in queries:
        try:
            hits = keyword_search(catalogue, query, n_results=pool)
        except Exception as exc:                            # noqa: BLE001
            log.warning("retrieve: keyword search failed (%s)", exc)
            hits = []
        if hits:
            lists.append({"query": query, "modality": "keyword", "hits": hits})
    return lists


def _diversify(ranked: list, holder: dict, n_results: int,
               per_document_cap: Optional[int] = None) -> list:
    """Take the top `n_results`, but let no single document fill the window.

    A corpus with many near-identical FAQ rows can hand back twenty chunks from
    the same source, crowding out the one varietal sheet that actually answers
    the question. The cap is deliberately loose — a real varietal sheet SHOULD
    contribute several of its rows — it only stops one document owning the whole
    context. Anything held back is appended afterwards, so nothing is lost while
    there is room.

    `per_document_cap` overrides the default for the one case where the default
    is actively wrong: a question asking WHICH ENTITIES EXIST. The answer to
    "which 100-day varieties are there?" lives in a single varietal sheet, and
    the default cap of n_results//3 let it contribute only 6 chunks of 20 —
    roughly 28 rows of a 183-row table — so the farmer was shown two or three
    names out of dozens. Measured: at n=20 the sheet gets 6 chunks; at n=30 with
    a cap of 20 it gets 20, while ten slots still go to other documents so the
    individual variety sheets and FAQ rows are not starved. Removing the cap
    entirely was rejected for exactly that reason — it took ALL twenty slots.
    """
    if per_document_cap is None:
        per_document_cap = max(3, n_results // 3)
    counts, chosen, overflow = {}, [], []
    for key in ranked:
        hit = holder[key]
        document = (hit.get("metadata") or {}).get("document_id")
        seen = counts.get(document, 0) if document is not None else 0
        if document is not None and seen >= per_document_cap:
            overflow.append(key)
            continue
        counts[document] = seen + 1
        chosen.append(key)
        if len(chosen) >= n_results:
            return chosen
    return (chosen + overflow)[:n_results]


_DOSE_QUERY_RE = re.compile(
    r"(?:dose|rate|quantity|amount|how much|kg|hectare|acre|urea|dap|mop|potash"
    r"|fertiliz|nutrient|npk|application)",
    re.IGNORECASE,
)


def retrieve(catalogue, query_text_en: str, n_results: int = 6,
             rrf_k: Optional[int] = None,
             queries: Optional[list] = None,
             per_document_cap: Optional[int] = None):
    """Hybrid retrieval with RRF fusion, table-content boost, and optional
    cross-encoder reranking."""
    if rrf_k is None:
        rrf_k = getattr(settings, "RRF_K", 60)
    if queries is None:
        queries = _expand_queries(query_text_en)
    pool = max(n_results * 2, 12)

    lists = _ranked_lists(catalogue, queries, pool)

    fused, holder, first_seen = {}, {}, {}
    for ranked in lists:
        for rank, hit in enumerate(ranked["hits"]):
            key = _hit_key(hit)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            first_seen.setdefault(key, len(first_seen))
            if key not in holder or (holder[key].get("distance") is None
                                     and hit.get("distance") is not None):
                holder[key] = hit

    if not fused:
        return []

    # Boost table chunks for dose/quantity queries.
    if _DOSE_QUERY_RE.search(query_text_en):
        for key in fused:
            ct = (holder[key].get("metadata") or {}).get("content_type", "")
            if ct == "table":
                fused[key] *= 1.3

    reserved, budget = [], max(1, n_results // 2)
    for ranked in lists:
        if len(reserved) >= budget:
            break
        for hit in ranked["hits"][:1]:
            key = _hit_key(hit)
            if key not in reserved:
                reserved.append(key)

    ranked_keys = sorted(fused, key=lambda k: (-fused[k], first_seen[k]))
    ordered = reserved + [k for k in ranked_keys if k not in reserved]

    # Optional cross-encoder reranking before diversification.
    if getattr(settings, "USE_CROSS_ENCODER", False):
        try:
            from queries.services.reranker import cross_encoder_rerank
            rerank_pool = ordered[:n_results * 3]
            reranked = cross_encoder_rerank(
                query_text_en,
                [(k, holder[k]) for k in rerank_pool],
            )
            tail = [k for k in ordered if k not in rerank_pool]
            ordered = reranked + tail
        except Exception as exc:
            log.warning("cross-encoder rerank failed: %s", exc)

    chosen = _diversify(ordered, holder, n_results, per_document_cap)
    log.info("retrieve | %d phrasing-list(s) -> %d distinct chunk(s) -> %d kept",
             len(lists), len(fused), len(chosen))
    return [holder[k] for k in chosen]


def _chunk_key(hit: dict) -> str:
    """What makes two hits the SAME chunk, whichever catalogue returned it.

    Whitespace-insensitive on the text itself: the same page ingested into two
    catalogues produces two rows with two ids, and comparing ids would call them
    different while the farmer reads the same paragraph twice.
    """
    text = re.sub(r"\s+", " ", (hit.get("chunk") or "")).strip().lower()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def retrieve_across_catalogues(query_text_en: str, catalogue_ids: list,
                               n_results: int = 10,
                               queries: Optional[list] = None) -> list:
    """Hybrid retrieval over SEVERAL catalogues, merged into one ranked list.

    Each catalogue is searched INDEPENDENTLY by the existing retrieve(), so
    every ranking rule that applies to one corpus still applies here — this
    layer only decides how the separate lists become one.

    Three things it has to get right:

      * NO CATALOGUE DOMINATES. Scores from different collections are not
        comparable — one corpus can be larger, better written, or simply closer
        in wording — so merging by score would let the biggest corpus fill the
        context and starve the one that answers the other half of the question.
        The lists are interleaved by RANK instead: every catalogue's best hit
        comes first, then every catalogue's second, and so on.
      * THE SAME CHUNK APPEARS ONCE. Documents are ingested into more than one
        catalogue, and a duplicate paragraph wastes context and reads as
        corroboration it is not.
      * PROVENANCE SURVIVES. Each hit carries the catalogue it came from, so
        the prompt can show the boundary and the trace can show what was read.

    Falls back to plain retrieve() for a single catalogue, so the one-catalogue
    path is byte-for-byte what it was.
    """
    from catalogues.models import Catalogue

    ids = [int(i) for i in (catalogue_ids or [])]
    catalogues = list(Catalogue.objects.filter(pk__in=ids))
    by_id = {c.pk: c for c in catalogues}
    ordered = [by_id[i] for i in ids if i in by_id]     # caller's order = priority
    if not ordered:
        return []
    if len(ordered) == 1:
        return retrieve(ordered[0], query_text_en, n_results=n_results,
                        queries=queries)

    # Per catalogue, ask for a fair share plus a little slack, so the merge has
    # something to choose from without any one list being able to fill the
    # context on its own.
    share = max(2, -(-n_results // len(ordered)))
    per_catalogue = []
    for catalogue in ordered:
        try:
            hits = retrieve(catalogue, query_text_en, n_results=share + 2,
                            queries=queries)
        except Exception as exc:                         # noqa: BLE001
            # One unreadable collection must not lose the other half of the
            # answer; the rest of the question is still answerable.
            log.warning("multi-catalogue retrieval: %s (id=%s) failed (%s)",
                        catalogue.name, catalogue.pk, exc)
            continue
        for hit in hits:
            meta = dict(hit.get("metadata") or {})
            meta["catalogue_id"] = catalogue.pk
            meta["catalogue_name"] = catalogue.name
            hit["metadata"] = meta
        per_catalogue.append((catalogue, hits[:share]))
        log.info("multi-catalogue retrieval: %s (id=%s) -> %d chunk(s)",
                 catalogue.name, catalogue.pk, len(hits[:share]))

    merged, seen = [], {}
    for rank in range(share):
        for catalogue, hits in per_catalogue:
            if rank >= len(hits):
                continue
            hit = hits[rank]
            key = _chunk_key(hit)
            if key in seen:
                # Same paragraph from a second catalogue: keep the copy already
                # placed and record that both hold it, rather than printing it
                # twice.
                also = seen[key]["metadata"].setdefault("also_in_catalogues", [])
                name = hit["metadata"].get("catalogue_name")
                if name and name not in also:
                    also.append(name)
                continue
            seen[key] = hit
            merged.append(hit)
            if len(merged) >= n_results:
                break
        if len(merged) >= n_results:
            break

    log.info("multi-catalogue retrieval: %d catalogue(s) -> %d chunk(s) after "
             "dedupe, cap %d per catalogue", len(per_catalogue), len(merged), share)
    return merged[:n_results]


def _document_links(document_ids: list) -> dict:
    """{document_id: {"document_url", "file_url"}} for the retrieved documents.

    One bulk query for the whole result set. Tracing must never be able to break
    answer generation, so every failure here degrades to "no link" rather than
    raising: a missing URLconf entry or a deleted Document is not a reason to
    lose an answer the farmer is waiting for.
    """
    ids = [i for i in {d for d in document_ids} if i is not None]
    if not ids:
        return {}
    links: dict = {}
    try:
        from django.urls import reverse
        from documents.models import Document
        for doc in Document.objects.filter(id__in=ids).only("id", "file"):
            entry = {"document_url": "", "file_url": ""}
            try:
                entry["document_url"] = reverse("documents:detail", args=[doc.id])
            except Exception:
                pass
            try:
                if doc.file:
                    entry["file_url"] = doc.file.url
            except Exception:
                pass
            links[doc.id] = entry
    except Exception as exc:
        log.warning("could not build document links for trace: %s", exc)
    return links


def _trace_retrieved_chunks(hits: list) -> list:
    chunks = []
    links = _document_links([(h.get("metadata") or {}).get("document_id") for h in hits])
    for position, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        chunk_text = (hit.get("chunk") or "").strip()
        link = links.get(meta.get("document_id")) or {}
        chunks.append({
            "rank": position,
            "document_id": meta.get("document_id"),
            "document_title": meta.get("document_title") or "Untitled document",
            "chunk_index": meta.get("chunk_index"),
            "category": meta.get("category") or "",
            "subcategory": meta.get("subcategory") or "",
            "score": hit.get("score"),
            "distance": hit.get("distance"),
            # Where this chunk came from. document_url is the in-app document
            # page (always resolvable); file_url opens the uploaded file itself;
            # source_url is the external origin recorded at index time and is
            # often blank.
            "document_url": link.get("document_url", ""),
            "file_url": link.get("file_url", ""),
            "source_url": meta.get("source_url") or "",
            "chunk": chunk_text[:1600],
            "truncated": len(chunk_text) > 1600,
        })
    return chunks


FARMER_PERSONA = (
    "You are PaddyMitra, talking to a small rice farmer who may have little schooling. "
    "Talk like an experienced neighbour standing in the field with them, not a scientist "
    "and not a manual. Give simple, do-this advice.\n"
    # The answer is spoken aloud to a person, so it has to sound like a person
    # said it. Terse imperatives ("Apply X. Use Y. Maintain Z.") are accurate but
    # read like a machine issuing orders, which is what farmers reported.
    "HOW TO SOUND HUMAN (as important as being correct):\n"
    " - Speak TO the farmer. Say \"you\" and \"your field\", not \"the farmer\" or "
    "\"the field\". It is a conversation, not an instruction sheet.\n"
    " - If they told you about a problem, show you heard it in the FIRST few words "
    "before advising (\"If your crop has fallen over, stop the urea now\"). Never open "
    "with a bare command when a problem was reported.\n"
    " - Do NOT start every answer with a verb. Vary the opening naturally, the way a "
    "person talking would.\n"
    " - Join the steps the way speech does — \"first\", \"after that\", \"once it is "
    "done\" — instead of listing separate orders.\n"
    " - Warmth in a few plain words is welcome (\"do not worry\", \"this is common\"). "
    "Never flatter, never pad, never waste the farmer's time. No greetings, no "
    "sign-offs, no \"I hope this helps\".\n"
    "Rules for your answer:\n"
    " - Keep it SHORT: 2-4 short sentences of flowing prose. Sounding human must not "
    "make the answer longer.\n"
    # A farmer asked how to apply fertiliser to their soil test. The answer began
    # by telling them to go and get a soil test, then added an unasked-for
    # micronutrient spray and a note about spraying on moist soil. Three
    # sentences, none of them the answer. Extra advice is not a bonus when it is
    # spoken aloud: it buries the one thing they rang up to find out.
    " - ANSWER THE QUESTION ASKED, and stop. Do not add advice they did not ask "
    "for, do not add a step they have clearly already taken, and do not tack on a "
    "general tip at the end. If the whole answer is one sentence, give one "
    "sentence.\n"
    " - Do not pad the answer with what a careful farmer would do anyway (testing "
    "the soil, watching for pests, watering well) unless that IS the question. Say "
    "the specific thing they asked for: the amount, the time, or the method.\n"
    # The CONTEXT stores doses per variety, so the top chunk for a general
    # question is usually about one particular variety. Reporting it faithfully
    # is what produced: "How to use potassium in rice?" -> "For Kalachampa rice,
    # put 25 kg/ha ...". The farmer grows something else and cannot tell that the
    # figure was not meant for them.
    " - ANSWER AT THE SAME BREADTH AS THE QUESTION. If they asked in general, "
    "answer in general. NEVER name a variety, district, season or soil type the "
    "farmer did not mention, even when the CONTEXT is about one — that turns "
    "their question into a different question and gives them a figure that may "
    "not be theirs.\n"
    " - If the CONTEXT only holds a figure for one specific variety and they did "
    "not name it, give the rate WITHOUT tying it to that variety. Do not quietly "
    "answer for whichever one the documents happened to mention, and do not ask "
    "them which variety they grow — just give the rate that applies.\n"
    # This is spoken down a phone. A question at the end is one the farmer cannot
    # answer, so it lands as the machine dodging the question they DID ask.
    " - NEVER END WITH A QUESTION. Do not ask the farmer which variety, district, "
    "season or stage they have. Answer with what you know. If you genuinely have "
    "nothing, say that plainly in one sentence and stop — do not turn it back "
    "into a question.\n"
    " - Write PLAIN SENTENCES ONLY. Do NOT use bullet points, numbered lists, dashes, "
    "asterisks, headings, or any markdown/symbols — the answer is read aloud, so those "
    "get mispronounced or change the meaning. Separate ideas with full stops, not bullets.\n"
    " - Plain everyday words only (words a 10-year-old knows). One idea per sentence; "
    "sentences about 8-12 words.\n"
    # Every sentence is translated on its own and then SPOKEN once. A clause that
    # borrows its verb from the clause before it survives in written English and
    # collapses in Odia: "Apply urea at tillering and again at panicle initiation"
    # came back as "... ପ୍ରୟୋଗ କରନ୍ତୁ ଏବଂ ପୁଣି ଶିଷ ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥାରେ।" — the
    # second half has no verb at all, so the farmer hears a phrase that stops
    # mid-air and never learns what to do at that stage.
    " - Every sentence must be COMPLETE on its own, with its own verb. NEVER write a "
    "clause that leans on a verb from an earlier clause (\"and again at the panicle "
    "stage\", \"then 20 kg at flowering\"). Write the verb out again, as its own "
    "sentence: \"Apply urea again when the panicle starts to come out.\"\n"
    # "the nitrogen your plants need" is correct English — the "that" is simply
    # left out. But two nouns landing next to each other with no joining word is
    # hard to follow when it is heard once, and the translator has to guess where
    # the clause starts. Keeping the "that", or turning the sentence around, costs
    # nothing and removes the guess.
    " - Never drop the joining word from a clause. Write \"the nitrogen THAT your "
    "plants need\", not \"the nitrogen your plants need\" — better still, turn it "
    "around: \"this gives your plants the nitrogen they need\". Two nouns side by "
    "side with no joining word are hard to follow when heard once.\n"
    # Stage names are the farmer's landmarks. If they cannot see it, they cannot
    # time the dose by it.
    " - Name a crop stage by what the farmer can SEE in the field — tillers coming out, "
    "the panicle starting to come out, flowering, grain filling. Do NOT invent a stage "
    "name, and never name a stage the CONTEXT does not name.\n"
    # This answer gets translated into Odia and spoken aloud. Abstract English
    # nominalisations have no Odia equivalent, so the translator renders them
    # word by word and the farmer hears nonsense: "proper water management to aid
    # recovery" came back as ପାଣି ପରିଚାଳନା ... ମରାମତି — literally "water
    # administration ... repair", as if the field were a machine being serviced.
    " - Say the PHYSICAL THING TO DO, never an abstract noun for it. Write \"keep only "
    "a little water in the field\", NOT \"maintain proper water management\". Write "
    "\"put MOP\", NOT \"focus on the application of MOP\". Banned words: management, "
    "application, maintenance, monitoring, recovery, optimisation, implementation, "
    "utilisation, \"focus on\", \"ensure that\", \"in order to\".\n"
    " - Use verbs the farmer's hands can do — put, spread, mix, drain, cut, wait, "
    "look at. Every sentence should describe something they could go and do now.\n"
    # Every one of these was a real failure. The English word has a farming sense
    # and an everyday sense; the translator takes the everyday one and the farmer
    # hears something absurd:
    #   management -> ପରିଚାଳନା (administering an office)
    #   recovery   -> ମରାମତି   (repairing a machine)
    #   stress     -> ଚାପ      (mental pressure)
    #   establish  -> ଗଢ଼ି ତୋଳିବା (constructing a building)
    #   standing water -> ପାଣି ଜମି (spelled like "water land")
    #   flood      -> ବନ୍ୟା     (the disaster that destroys the crop)
    #   recovery   -> ଆରୋଗ୍ୟ   (a person getting better from an illness)
    #   too high   -> ଉଚ୍ଚ      (tall / upland, not "too much" water)
    " - YOUR WORDS WILL BE TRANSLATED AND SPOKEN ALOUD IN ODIA. Never use an English "
    "word whose common meaning outside farming is different from the one you intend — "
    "it will be translated in the wrong sense. Say what physically happens instead: "
    "\"take root\" not \"establish\", \"get strong again\" not \"recover\", \"keep water\" "
    "not \"manage water\", \"gets weak\" not \"suffers stress\", \"puddle the field\" or "
    "\"put water in the field\" not \"flood the field\" — flooding is the disaster that "
    "destroys their crop. If a word would confuse someone hearing it once with no "
    "schooling, choose a plainer one.\n"
    " - NOT scientific and NOT calculative. Do NOT explain why it works, do NOT discuss "
    "chemistry, mechanisms, soil science, or nutrient theory. Do NOT make the farmer do "
    "maths or conversions, and do NOT give nutrient percentages or ratios.\n"
    " - Tell only WHAT to do, HOW MUCH, and WHEN, in the simplest way. Give one clear "
    "amount (a single number with its unit) - do not pile up many figures.\n"
    " - Where helpful, use familiar measures a farmer knows (a bag, a handful) alongside "
    "the amount.\n"
    " - Avoid jargon, English abbreviations, idioms, and long clauses. If a technical word "
    "is unavoidable, give the common local name.\n"
    " - Do NOT cite documents, research, or authors.\n"
    # "The context does not provide..." is the single clearest sign a machine
    # wrote the answer — the farmer has no idea what "the context" is.
    " - NEVER refer to your own sources or workings. Do not say \"the context\", "
    "\"the information provided\", \"the documents\", \"my data\", or \"based on what I "
    "have\". The farmer cannot see any of that and it makes you sound like a machine.\n"
    # NOTE: no quotable sentence here on purpose. An earlier version supplied one
    # ("I do not know the exact amount for your land") and the model copied it
    # verbatim, declining questions it could actually answer — the same anchoring
    # that made it echo the number 7061 out of a worked example.
    " - Only say you do not know when the CONTEXT really has nothing. Say it in your own "
    "plain words and stop. NEVER ask the farmer to supply the very figure they asked "
    "you for, and do not end by asking them something else instead.\n"
    # The farmer says "2.5 acres" and the documents give a per-acre rate. Refusing
    # is unhelpful and forcing them to multiply breaks the no-maths rule, so the
    # answer does the arithmetic.
    # DO NOT re-enable arithmetic here without a deterministic implementation.
    # Two attempts were made. Permitting "multiply the rate by the land size"
    # produced 90-100 kg of urea for 2.5 acres; the corpus contains no such rate
    # at all (its figures are 20-25 kg/acre at panicle, 10-15 kg/acre at grain
    # fill), so the number was invented by summing split doses and relabelling
    # them per hectare. Restricting it to "one multiplication only" did not stop
    # it. A confidently wrong fertiliser dose is the most harmful output this
    # system can produce, so the model is not allowed to compute one.
    " - NEVER calculate YOURSELF. Do not multiply, add, or convert units. If a COMPUTED "
    "AMOUNTS block appears below, the arithmetic for the farmer's land size has already "
    "been done for you — copy those figures exactly and do not recompute or adjust them. "
    "With no such block, give the rate exactly as the CONTEXT states it, with the unit it "
    "uses (per acre or per hectare) and the crop stage it applies to.\n"
    " - When the farmer gives a land size you have no figure for, still ANSWER: give the "
    "stated per-acre or per-hectare rate and name the stage. Do not refuse, and do not "
    "claim you have no information when the CONTEXT holds a rate for what they asked.\n"
    # A bare "give 20-25 kg of urea" is unusable and dangerous: the farmer asked
    # about half an acre, the figure is per acre, and nothing in the sentence
    # said so. The unit requirement already existed inside the NEVER-calculate
    # bullet above, but buried there the model kept dropping it. It is restated
    # as its own rule, with the land-size mismatch named explicitly, because
    # that is the case that was failing. Still NO arithmetic — the answer states
    # the basis and lets the farmer see it differs from their plot.
    " - EVERY quantity must carry the area it applies to, in the same sentence as the "
    "number. Never write an amount on its own: write \"20 to 25 kg per acre\", not \"20 to "
    "25 kg\". If the farmer's land size is different from the rate's basis, say the rate "
    "and its basis plainly (for example \"the recommended amount is 20 to 25 kg per acre\") "
    "so they can see what it is measured against. Do not work out their total for them.\n"
    # Vague filler is what makes advice sound generated rather than given.
    " - No empty filler. Never end with hollow lines like \"for the best results\", "
    "\"at the right time\", \"as needed\" or \"consult an expert\" unless you say exactly "
    "what time, what amount, or which expert. Every sentence must carry real information."
)

RESEARCHER_PERSONA = (
    "You are PaddyMitra Research, an AI advisor speaking to an agricultural researcher, "
    "extension officer, or agronomist. Answer in TECHNICAL, DETAILED language.\n"
    "Rules for your answer:\n"
    " - Give a thorough, detailed response written as flowing PROSE PARAGRAPHS. Do NOT "
    "use bullet points, numbered lists, dashes, asterisks, headings, tables, or any "
    "markdown/symbols — the answer may be read aloud, so those get mispronounced. Keep the "
    "technical depth, but express it in full sentences.\n"
    " - Use scientific names, active ingredients, mechanisms, and measured values.\n"
    " - Cite the source documents you used by title (bracketed, e.g. [Source: ...]).\n"
    " - Include trial numbers and percentages where the context provides them.\n"
    " - Where methods or concentrations differ between sources, note the disagreement.\n"
    " - If context is insufficient, state the gap precisely and suggest what additional data would resolve it."
)

# Applied to BOTH personas. Most numeric errors farmers complain about come
# from the LLM converting units (e.g. tons/ha → kg/acre) and losing precision
# or making outright wrong conversions. We disable conversion entirely.
UNIT_FIDELITY_RULES = (
    "\n\nNUMERIC / UNIT FIDELITY (MANDATORY):\n"
    " - Quote numbers and units EXACTLY as they appear in the CONTEXT. "
    "Do not convert between dissimilar units (e.g. do not change kg to lb, "
    "do not change litres to gallons, do not change tons/ha to kg/acre).\n"
    " - Keep the same unit string the source used (e.g. \"5-6 tons per hectare\", "
    "\"120 kg N/ha\", \"40 grams per litre\").\n"
    " - Do not round, average, or rephrase numbers. If the source says \"5-6 tons/ha\", "
    "write \"5-6 tons/ha\" — not \"about 5 tons\" and not \"2000-2400 kg/acre\".\n"
    "\nAREA UNIT CONVERSION (ALLOWED):\n"
    "You MAY convert between these AREA units when the farmer asks in a "
    "different unit than the CONTEXT provides:\n"
    "  1 hectare = 2.471 acres = 100 decimals (approx)\n"
    "  1 acre = 0.405 hectare = 40 gunthas = 100 decimals\n"
    "When converting, show BOTH the original and converted figure, e.g. "
    "\"the recommendation is 80 kg N/ha, which is about 32 kg N per acre\".\n"
    "You may also SUM split doses when the farmer explicitly asks for the "
    "TOTAL amount, e.g. \"the total urea across all stages is X kg/acre "
    "(basal Y + tillering Z + PI W)\" — always show the breakdown.\n"
    "Answer in the UNIT the farmer asked. If the farmer says 'acre', give "
    "the answer per acre. If the farmer says 'hectare', give per hectare.\n"
)


# Closing sentences that sound like advice but carry none. The persona forbids
# them and the model still produces them, so they are removed deterministically —
# the same lesson as the numbers and the acronyms: when correctness depends on the
# model choosing particular words, do it in code instead.
_FILLER_TAIL_RE = re.compile(
    r"\b(for (the )?(best|better|good) results?|at the right time|as (needed|required|per your need)"
    r"|for (a )?(better|good|healthy) (yield|growth|crop)|as (soon as )?possible"
    r"|for optimal (growth|results?|yield)|to (ensure|get) (the )?best"
    # "Apply them at the recommended rate for your field size" names no rate and
    # no field size — it points at advice instead of giving it, which is the
    # emptiest thing an answer can end on when it is being read aloud.
    r"|at the recommended (rate|dose|amount)|as recommended"
    r"|as per (the )?recommendation|according to the recommendation"
    r"|for your field size|based on your field size"
    r"|(the )?recommended (rate|dose|amount) for your)\b",
    re.IGNORECASE,
)
# A sentence naming an amount, a time, or an input is real advice even if it also
# contains one of the phrases above — only strip sentences that say nothing else.
_HAS_SUBSTANCE_RE = re.compile(r"\d|\b(kg|g|ha|acre|cm|ton|bag|day|week|urea|MOP|DAP|NPK|SSP)\b",
                               re.IGNORECASE)

# Served FAQ answers are stored with line breaks and the odd bullet; read aloud,
# a one-item-per-line schedule becomes a string of fragments. Farmers asked for
# paragraphs, so collapse a served answer into flowing prose. Presentation only —
# no figure, unit or word is changed, only the line breaks between them.
_BULLET_LINE_RE = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])\s+")


def _to_paragraph(text: str) -> str:
    """Flatten a bullet / one-line-per-item answer into a single paragraph."""
    if not text:
        return text
    t = _BULLET_LINE_RE.sub("", text)
    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# Abstract English that has no Odia equivalent, so the translator renders it word
# by word and the farmer hears nonsense ("water management" -> ପାଣି ପରିଚାଳନା,
# "recovery" -> ମରାମତି, i.e. repairing a machine). The persona forbids these and
# the model produces them anyway, so they are rewritten in code — the same reason
# the numbers, acronyms and filler are handled here rather than in a prompt.
_GERUND_TO_IMPERATIVE = {
    "applying": "apply", "putting": "put", "using": "use", "adding": "add",
    "keeping": "keep", "checking": "check", "maintaining": "keep",
    "spreading": "spread", "mixing": "mix", "draining": "drain",
}
_CONCRETISE: list[tuple[re.Pattern, object]] = [
    # "Instead, focus on applying MOP" -> "Instead, apply MOP"
    (re.compile(r"\bfocus on (" + "|".join(_GERUND_TO_IMPERATIVE) + r")\b", re.I),
     lambda m: _GERUND_TO_IMPERATIVE[m.group(1).lower()]),
    # "the right water level" (not bare "the water level") so the replacement also
    # reads correctly after a verb: "the field has the right water level".
    (re.compile(r"\b(?:proper|good|correct)\s+water\s+management\b", re.I), "the right water level"),
    (re.compile(r"\bwater\s+management\b", re.I), "the water level"),
    (re.compile(r"\bnutrient\s+management\b", re.I), "the fertiliser you give"),
    (re.compile(r"\bto\s+aid\s+(?:in\s+)?(?:the\s+)?recovery\b", re.I), "to help the crop recover"),
    (re.compile(r"\b(?:aid|support)\s+(?:the\s+)?recovery\b", re.I), "help the crop recover"),
    # Those two only ever caught "aid/support the recovery". "for quick recovery"
    # matched neither and came back as ଆରୋଗ୍ୟ — a PERSON getting better from an
    # illness, which is not something a rice plant does. The word has to go in
    # every form it appears in, not in the two that happened to be reported.
    (re.compile(r"\bfor\s+(?:a\s+)?(?:quick|quicker|fast|faster|speedy|better|good)?\s*"
                r"recovery\b", re.I), "to help the crop get strong again"),
    # Before the generic form, so "helps the recovery of the crop" comes out as
    # "helps the crop GET strong again" rather than "...crop GETTING strong again".
    (re.compile(r"\bhelps?\s+(?:in\s+)?(?:the\s+)?recovery\s+of\s+"
                r"(?:the\s+|your\s+)?(?:crop|plants?|rice|paddy)\b", re.I),
     "helps the crop get strong again"),
    (re.compile(r"\b(?:the\s+)?recovery\s+of\s+(?:the\s+|your\s+)?(?:crop|plants?|rice|paddy)\b",
                re.I), "the crop getting strong again"),
    (re.compile(r"\brecovery\b", re.I), "getting strong again"),
    # A crop does not show "signs of life" — it starts growing again. The longer
    # form is matched first, or "start to show signs of life" would become the
    # stuttering "start to start growing again".
    (re.compile(r"\bstart(?:s|ing|ed)?\s+to\s+show\s+signs?\s+of\s+life\b", re.I),
     "start growing again"),
    (re.compile(r"\bshow(?:s|ing)?\s+signs?\s+of\s+life\b", re.I), "start growing again"),
    # "the water level is too HIGH" becomes ଉଚ୍ଚ — high as in TALL, the word for
    # upland. Water in a field is too MUCH, not too tall. Scoped to a sentence
    # that is already talking about water, so a genuinely high thing (a high
    # yield, high temperature) is untouched.
    (re.compile(r"(\bwater\b[^.!?]{0,60}?)\btoo\s+high\b", re.I), r"\1too much"),
    (re.compile(r"(\bwater\b[^.!?]{0,60}?)\btoo\s+low\b", re.I), r"\1too little"),
    # "stress" on a crop becomes ଚାପ — mental/physical pressure, meaningless here.
    (re.compile(r"\bto avoid (?:any )?further stress\b", re.I), "so the plants do not get weaker"),
    (re.compile(r"\b(?:further |additional )?stress on the (?:plants|crop)\b", re.I),
     "the plants getting weaker"),
    (re.compile(r"\bfor potassium support\b", re.I), "to make the stems strong"),
    # "rice paddy" is redundant in English and lethal in translation: `rice` and
    # `paddy` are BOTH ଧାନ in the glossary, so it comes out as ଧାନ ଧାନ.
    (re.compile(r"\brice paddy\b", re.I), "rice field"),
    (re.compile(r"\bpaddy rice\b", re.I), "rice"),
    # "standing water" translates to ପାଣି ଜମି. ଜମି there is the participle of
    # ଜମିବା (to collect), but it is spelled exactly like ଜମି = LAND, so spoken
    # aloud the farmer hears "water land keep". In a paddy field water is
    # standing by definition, so the word earns nothing. Keep it only where it
    # means stagnant water to be drained (handled separately below).
    (re.compile(r"\b(\d[\d\s.,–-]*(?:cm|centimetre|centimeter|inch)(?:\s+of)?)\s+standing water\b", re.I),
     r"\1 water"),
    (re.compile(r"\bmaintain(?:ing)?\s+standing water\b", re.I), "keep water"),
    (re.compile(r"\bkeep(?:ing)?\s+standing water\b", re.I), "keep water"),
    # Drainage sense — here "standing" carries the meaning, so name it plainly.
    (re.compile(r"\b(avoid|prevent|remove|drain)\s+(?:any\s+)?standing water\b", re.I),
     r"\1 water sitting in the field"),
    # "establish" has a farming sense (take root) and a general one (set up a
    # body, build something). The translator picks the general one: seedlings
    # "establishing" came out as ଗଢ଼ି ତୋଳିବା — to CONSTRUCT — so the farmer hears
    # the seedlings being built. "take root" gives the natural ମୂଳ ଧରିବା.
    # "are established" / "get established" too — the noun is often not adjacent,
    # and that form came back as ପ୍ରତିଷ୍ଠା (founding an institution).
    (re.compile(r"\b(seedlings?|plants?|crop|transplants?)\s+(?:are|is|get|gets|have|has)\s+"
                r"(?:well\s+)?established\b", re.I), r"\1 have taken root"),
    (re.compile(r"\b(seedlings?|plants?|crop|transplants?)\s+establish(?:es|ed)?\b", re.I),
     r"\1 take root"),
    # NOT a bare "well-established" rule: a "well-established variety" is a PROVEN
    # one, nothing to do with roots. The plant sense is already covered above.
    # No article inserted: "Good crop establishment" must not become
    # "Good THE crop taking root".
    (re.compile(r"\b(crop|seedling|plant)\s+establishment\b", re.I), r"\1 rooting"),
    (re.compile(r"\bestablishment\b", re.I), "rooting"),
    (re.compile(r"\bhelps?\s+(?:it|them)\s+establish\b", re.I), "helps them take root"),
    # Over-technical product names. The translator has no Odia word for the
    # qualifier so it spells it out phonetically: "neem seed kernel extract"
    # produced ନିମ ବିହନ କର୍ଣ୍ଣେଲ ନିଷ୍କାସନ — "kernel" written in Odia letters,
    # which is just an English word the farmer cannot read. The qualifier adds
    # nothing for them, so it goes before translation.
    (re.compile(r"\bneem seed kernel extract\b", re.I), "neem seed extract"),
    (re.compile(r"\bseed kernel extract\b", re.I), "seed extract"),
    (re.compile(r"\bkernel extract\b", re.I), "extract"),
    # CROP STAGE NAMES. "jointing" appears ZERO times in the indexed corpus and
    # is not a stage any Odia rice farmer names — the model coined it for a
    # question that had asked, in the farmer's own words, about ଗଣ୍ଠି ଛାଡିବା
    # (the stem letting out its nodes), and the farmer heard back a stage they
    # could not picture.
    #
    # synonyms.py already collapses these spellings on the QUESTION side, but it
    # never sees the ANSWER, so the coined term survived all the way to the
    # farmer. Normalising here sends it into the one form the glossary knows
    # (_BUILTIN_FALLBACK: "internode elongation stage" -> ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା),
    # which puts the farmer's own words back in their ear.
    (re.compile(r"\b(?:jointing|stem\s+elongation)\s+stage\b", re.I),
     "internode elongation stage"),
    (re.compile(r"\b(?:jointing|stem\s+elongation)\b", re.I),
     "internode elongation stage"),
    # "flood the field" is a DISASTER in Odia. ବନ୍ୟା is the flood that ruins a
    # crop, so "mix it into the soil before you flood the field" came back as
    # "... ପୂର୍ବରୁ କ୍ଷେତକୁ ବନ୍ୟା କରିବା" — telling the farmer to flood their own
    # land. The glossary is right (flooding -> ବନ୍ୟା); the English is what is
    # wrong, so it is fixed here, before the translator ever sees it.
    #
    # Working fertiliser into the soil and then putting water on it IS puddling,
    # and the glossary already carries puddling -> କାଦୁଅ କରିବା, so naming the
    # step correctly also gets the right Odia for free.
    (re.compile(r"\bbefore\s+(?:you\s+)?flood(?:ing)?\s+(?:the|your)\s+"
                r"(?:field|land|plot)s?\b", re.I), "before puddling the field"),
    # Any other deliberate flooding is filling the field with water — still not
    # ବନ୍ୟା. Kept separate because it is a water-level instruction, not puddling.
    (re.compile(r"\bflooding\s+(the|your)\s+(field|land|plot)s?\b", re.I),
     r"filling \1 \2 with water"),
    (re.compile(r"\bflood\s+(the|your)\s+(field|land|plot)s?\b", re.I),
     r"fill \1 \2 with water"),
    (re.compile(r"\bmake sure to\s+", re.I), ""),
    (re.compile(r"\bensure that\s+", re.I), ""),
    (re.compile(r"\bin order to\b", re.I), "to"),
]


# ── Answering a question NARROWER than the one that was asked ─────────────
#
# Asked "How to use potassium in rice?", the model answered:
#
#     "... For KALACHAMPA rice, put 25 kg/ha during the panicle initiation
#      stage. Mix it well into the soil before the last puddling."
#
# The farmer never mentioned Kalachampa. It came from the retrieved chunks: the
# corpus stores doses per variety, so the top hit was variety-specific and the
# model reported it faithfully — attributing a general question to one variety.
#
# A farmer growing something else now has a dose that may not be theirs, and no
# way to tell. This is not the model padding the answer; it is the model
# NARROWING it, so the "don't add extra advice" rules do not catch it.
#
# Detected by proper nouns: a capitalised word in the middle of an English
# sentence is a name, and a name the QUESTION does not contain is something the
# farmer did not ask about. Variety names (Kalachampa, Swarna, Pooja, Lalat,
# Sahbhagi) are exactly that shape.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:-\d+)?\b")
_SENTENCE_SPLIT_EN = re.compile(r"(?<=[.!?])\s+")

# Capitalised words that are NOT an entity the farmer must have named: products,
# nutrients, seasons and the openers of a sentence written in title case.
_ENTITY_ALLOWED = frozenset({
    "urea", "muriate", "potash", "potassium", "nitrogen", "phosphorus",
    "phosphate", "zinc", "sulphur", "sulfur", "boron", "iron", "manganese",
    "magnesium", "calcium", "molybdenum", "compost", "vermicompost", "manure",
    "gypsum", "lime", "azolla", "dhaincha", "sesbania", "rhizobium",
    "trichoderma", "azotobacter", "azospirillum",
    "kharif", "rabi", "zaid", "monsoon",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # Ordinary words that happen to start a clause or follow a full stop.
    "the", "this", "that", "these", "those", "there", "then", "your", "you",
    "for", "and", "but", "apply", "put", "use", "mix", "keep", "give", "spread",
    "after", "before", "during", "once", "first", "next", "also", "make",
    "check", "look", "wait", "drain", "spray", "start", "stop", "add", "avoid",
    "when", "where", "what", "how", "why", "which", "since", "because", "while",
    "rice", "paddy", "soil", "water", "field", "crop", "seed", "plant",
})


# ── "Which variety" vs "which varieties" ──────────────────────────────────
#
# A farmer asking for NAMES expects the list the documents hold, and a farmer
# asking in the singular expects a recommendation out of that SAME list. Which
# is why none of this lives here any more: entity detection, the singular/plural
# fold, land-type equivalence, the candidate collector and its ranking are one
# reusable layer in services/entity_intent.py, used by retrieval and generation
# alike so the two can never read a question differently.
#
# What remains below are the names the rest of this module has always called.
_TOLERANCE_WORDS = entity_intent.TOLERANCE_WORDS
_NEGATED_TOLERANCE_RE = entity_intent.NEGATED_SUPPORT_RE
_VARIETY_NAME_RE = entity_intent.CULTIVAR_NAME_RE
_NOT_A_VARIETY_NAME = entity_intent.NOT_A_CULTIVAR_NAME
_CONDITION_STOPWORDS = entity_intent.CONDITION_STOPWORDS


def _variety_request(question: str):
    """"multiple", "single", or None — how many varieties were asked for."""
    return entity_intent.variety_request(question)


def _is_variety_question(question: str) -> bool:
    """Unchanged contract for existing callers: is this about varieties at all?"""
    return _variety_request(question) is not None


def _condition_terms(question: str) -> set:
    """The traits a variety has to have, as words to look for beside its name."""
    land_type, _traits, _season, matched = entity_intent.detect_attributes(question)
    return set(entity_intent.condition_terms(question, land_type, matched))


def _varieties_supported_by(question: str, retrieved_hits: list) -> list:
    """Every DISTINCT variety the context explicitly ties to what was asked,
    strongest evidence first."""
    return entity_intent.supported_names(question, retrieved_hits)


def _unrequested_entities(answer: str, question: str) -> list[str]:
    """Proper nouns the ANSWER introduces that the QUESTION never mentioned.

    Sentence-initial words are skipped — a capital there is grammar, not a name.
    Anything the technical-term layer already protects (corpus acronyms, variety
    PREFIXES like CR/MTU, scientific names, units) is allowed, because those are
    vocabulary rather than a specific case being singled out.
    """

    if _is_variety_question(question):
        return []
    
    asked = (question or "").lower()
    found: list[str] = []
    for sentence in _SENTENCE_SPLIT_EN.split(answer or ""):
        for m in _PROPER_NOUN_RE.finditer(sentence):
            if m.start() == 0:
                continue                       # start of the sentence
            word = m.group(0)
            low = word.lower()
            if low in _ENTITY_ALLOWED or low in asked:
                continue
            if is_allowed_latin(word) or word.isupper():
                continue                       # DAP, MOP, BGA, CR, MTU …
            if low not in [f.lower() for f in found]:
                found.append(word)
    return found


def _concretise_english(answer: str) -> str:
    """Replace abstract office-English with the physical thing to do.

    Runs on the ENGLISH answer, before translation, because that is where the
    damage starts: fix the English and the Odia comes out right by itself.
    """
    out = (answer or "")
    for rx, repl in _CONCRETISE:
        out = rx.sub(repl, out)
    # Re-capitalise where a leading phrase was removed ("Make sure to check" -> "check").
    out = re.sub(r"(^|(?<=[.!?]\s))([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out != (answer or "").strip():
        log.info("concretised English answer for translation")
    return out


def _drop_filler_tail(answer: str) -> str:
    """Remove a trailing sentence that is pure filler.

    Only the LAST sentence, only when it carries no number, unit or input name,
    and never if it would empty the answer — an answer that is only a caveat must
    survive intact.
    """
    text = (answer or "").strip()
    if not text:
        return answer
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) < 2:
        return answer
    last = parts[-1]
    if _FILLER_TAIL_RE.search(last) and not _HAS_SUBSTANCE_RE.search(last):
        log.info("dropped filler tail: %r", last)
        return " ".join(parts[:-1]).strip()
    return answer


def _drop_trailing_question(answer: str) -> str:
    """Remove question(s) the answer ends with.

    The farmer asked something; the reply is the answer, not a new question back.
    A trailing "Which variety do you grow?" reads as the machine deflecting, and
    over the phone the farmer has no way to reply to it anyway — the audio just
    ends on a question they cannot answer.

    Only TRAILING questions are dropped, and never the whole answer: a reply that
    is nothing but a question means the model had nothing else to say, and an
    empty answer is worse than an awkward one.
    """
    text = (answer or "").strip()
    if not text:
        return answer
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    kept = list(parts)
    dropped = []
    while len(kept) > 1 and kept[-1].rstrip().endswith("?"):
        dropped.append(kept.pop())
    if dropped:
        remainder = " ".join(kept).strip()
        has_dose = bool(_DOSE_RE.search(remainder))
        short = len(remainder.split()) < 25
        if short and not has_dose:
            log.info("kept trailing question(s) — remainder is a clarification, "
                     "not an answer with doses")
            return answer
        log.info("dropped trailing question(s): %s", list(reversed(dropped)))
        return remainder
    return answer


_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s*(kg|kilogram|g|gram|ton|t|ml|l|litre|liter)\b",
                      re.IGNORECASE)


# A dose is only safe if the NUMBER and the SUBSTANCE match the documents
# together. Checking the number alone passed "90-100 kg of urea", because
# 90-100 kg appears in the context attached to NITROGEN — and urea is 46%
# nitrogen, so restating one as the other is out by more than half.
_SUBSTANCES = (
    "urea", "mop", "dap", "npk", "ssp", "muriate of potash", "potash", "potassium",
    "nitrogen", "phosphorus", "phosphate", "zinc", "zinc sulphate", "zinc sulfate",
    "fym", "farmyard manure", "vermicompost", "azolla", "bga", "lime", "compost",
    "gypsum", "sulphur", "sulfur", "iron", "boron",
)
# "kg N/ha", "kg N per", "60 kg N total" — documents use "N" as nitrogen shorthand
_N_ABBREV_RE = re.compile(r"\bkg\s+n\b", re.IGNORECASE)


def _dose_pairs(text: str) -> set:
    """{(number, substance)} for every dose in `text`.

    The window spans BOTH sides of the figure, because the documents write it
    either way — "apply 20-25 kg Urea per acre" and "Urea (20-25 kg/acre)" are
    the same fact, and looking only forwards missed the second form.

    The window is truncated at sentence boundaries (". ") so a substance from
    an adjacent sentence does not contaminate the pairing.
    """
    flat = re.sub(r"\s+", " ", (text or "")).lower()
    pairs = set()
    for m in _DOSE_RE.finditer(flat):
        num = m.group(1).replace(" ", "").replace("–", "-")
        start = max(0, m.start() - 80)
        end = m.end() + 60
        # Truncate at the nearest sentence boundary so substances from
        # adjacent sentences don't bleed into this number's window.
        dot_after = flat.find(". ", m.end())
        if dot_after != -1 and dot_after < end:
            end = dot_after
        dot_before = flat.rfind(". ", start, m.start())
        if dot_before != -1:
            start = dot_before + 2
        window = flat[start:end]
        for s in _SUBSTANCES:
            if s in window:
                pairs.add((num, s))
        if _N_ABBREV_RE.search(window):
            pairs.add((num, "nitrogen"))
    return pairs


# Words that mark a figure as something to APPLY rather than something to expect.
# A yield ("2400 kg per acre") and an application rate ("apply 2400 kg per acre")
# look identical to the dose regex; only the verb tells them apart.
_APPLICATION_VERB_RE = re.compile(
    r"\b(appl(y|ied|ication)|use|using|put|give|giving|spread|broadcast|"
    r"top.?dress(ing)?|dose|recommendation is to|rate of)\b", re.IGNORECASE)


def _substance_free_doses(text: str) -> list[str]:
    """Application figures that name no input at all.

    `_dose_pairs` can only judge a figure it can pair with a SUBSTANCE, so a dose
    with none is invisible to it — which is how "The general recommendation for
    rice is to apply 2400 kg per acre" reached a farmer. Two things are wrong
    with that sentence and the pair check catches neither: nothing verifies the
    number, and the farmer cannot act on it because it never says 2400 kg OF WHAT.

    A yield figure is deliberately left alone — "a yield of 2400 kg per acre" is
    a legitimate substance-free number, so an application verb must be nearby
    before the figure counts as a dose.
    """
    flat = re.sub(r"\s+", " ", (text or "")).lower()
    out = []
    for m in _DOSE_RE.finditer(flat):
        window = flat[max(0, m.start() - 60): m.end() + 60]
        if any(s in window for s in _SUBSTANCES) or _N_ABBREV_RE.search(window):
            continue                       # the pair check already covers it
        if not _APPLICATION_VERB_RE.search(window):
            continue                       # a yield / harvest figure, not a dose
        out.append(m.group(0).strip())
    return out


# ---------------------------------------------------------------------------
# AREA SCALING. Deterministic, in code, because the LLM cannot be trusted with
# it: permitting "multiply the rate by the land size" produced 90-100 kg of urea
# for 2.5 acres, and restricting it to one multiplication did not stop it. The
# arithmetic therefore happens HERE, the model is handed the finished figure to
# copy, and `_unverified_doses` is told the computed numbers are legitimate.
# Without that last step the scaled figure is stripped as invented — the dose
# validator's retry prompt says in as many words "you invented or CALCULATED
# them", so a bolt-on scaler would be undone one function later.
#
# Deliberately narrow:
#   - Scales ONLY when the farmer's unit matches the rate's basis (acre->acre,
#     hectare->hectare). Unit conversion stays forbidden; a per-hectare rate for
#     an acre question is left alone for the prompt rules to handle.
#   - Needs an explicit per-acre / per-hectare basis in the document text. A
#     bare "20 kg" is not scaled, because nothing says what it is per.
_AREA_WORDS = {"half": 0.5, "quarter": 0.25, "one": 1.0, "two": 2.0,
               "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0,
               "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
# ponytail: decimal = 1/100 acre, guntha = 1/40 acre (Odisha local units)
_AREA_TO_ACRE = {
    "acre": 1, "acres": 1, "ac": 1,
    "hectare": 2.471, "hectares": 2.471, "ha": 2.471,
    "decimal": 0.01, "decimals": 0.01,
    "dismil": 0.01, "dismils": 0.01,
    "decimil": 0.01, "decimils": 0.01,
    "dismal": 0.01, "desimal": 0.01,
    "guntha": 0.025, "gunthas": 0.025,
    "gunta": 0.025, "guntas": 0.025,
}
_AREA_RE = re.compile(
    r"\b(?:(\d+(?:\.\d+)?)|(half|quarter|one|two|three|four|five|six|seven|eight|nine|ten))\s*"
    r"(?:an?\s+)?(acre|acres|ac|hectare|hectares|ha"
    r"|decimal|decimals|dismil|dismils|decimil|decimils|dismal|desimal"
    r"|guntha|gunthas|gunta|guntas)\b",
    re.IGNORECASE)

# Standard rice fertilizer schedule (Odisha Kharif Manual / medium-duration
# transplanted rice). Used as fallback when retrieved chunks don't contain
# enough parseable rates for the deterministic schedule to fire.
_STANDARD_RICE_SCHEDULE_ACRE = (
    "Basal: apply DAP 35 kg/acre, Urea 5 kg/acre, MOP 16 kg/acre, "
    "Zinc Sulphate 10 kg/acre. "
    "Active tillering (21 DAT): apply Urea 30-35 kg/acre. "
    "Panicle initiation (45 DAT): apply Urea 30-35 kg/acre, MOP 15-20 kg/acre."
)
_STANDARD_RICE_SCHEDULE_HA = (
    "Basal: apply DAP 87.5 kg/hectare, Urea 12.5 kg/hectare, MOP 40 kg/hectare, "
    "Zinc Sulphate 25 kg/hectare. "
    "Active tillering (21 DAT): apply Urea 75-87.5 kg/hectare. "
    "Panicle initiation (45 DAT): apply Urea 75-87.5 kg/hectare, MOP 37.5-50 kg/hectare."
)

# "15-20 kg/acre", "15 to 20 kg per acre", "35 kg/acre".
_RATE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[-–]|to)\s*(\d+(?:\.\d+)?)?\s*"
    r"(kg|kilogram|g|gram|ton|tons|t|litre|liter|l|ml)\s*"
    r"(?:/|\s*per\s+)\s*(acre|ac|hectare|ha)\b", re.IGNORECASE)
_RATE_SINGLE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|kilogram|g|gram|ton|tons|t|litre|liter|l|ml)\s*"
    r"(?:/|\s*per\s+)\s*(acre|ac|hectare|ha)\b", re.IGNORECASE)


# How close a product name must be to a figure for the pairing to be trusted.
# Wide enough for "apply Urea 30-35 kg/acre" and "MOP 15-20 kg/acre"; tight
# enough that a figure whose own product is unnamed is dropped instead of being
# attached to whichever product happens to sit further along the sentence.
_SUBSTANCE_MAX_GAP = 25


def _basis(unit: str) -> str:
    return "hectare" if unit.lower() in ("hectare", "hectares", "ha") else "acre"


def _fmt_qty(value: float) -> str:
    """7.5 -> '7.5', 10.0 -> '10'. Two decimals at most; farmers weigh in kg."""
    return f"{round(value, 2):g}"


def farmer_area(question: str) -> tuple[float, str, str] | None:
    """(size_in_rate_basis, rate_basis, display_label).

    Local units (decimal, guntha) are converted to acres so the scaler can
    use per-acre rates from the corpus directly. display_label keeps the
    farmer's original wording for the prompt, e.g. '50 decimal (0.5 acre)'."""
    m = _AREA_RE.search(question or "")
    if not m:
        return None
    raw_size = float(m.group(1)) if m.group(1) else _AREA_WORDS[m.group(2).lower()]
    if raw_size <= 0:
        return None
    unit = m.group(3).lower()
    factor = _AREA_TO_ACRE.get(unit, 1)
    if unit in ("hectare", "hectares", "ha"):
        if raw_size == 1.0:
            return None
        return raw_size, "hectare", f"{_fmt_qty(raw_size)} hectare"
    size_in_acres = raw_size * factor
    if size_in_acres == 1.0:
        return None
    # ponytail: show original unit if not acres
    if unit not in ("acre", "acres", "ac"):
        label = f"{_fmt_qty(raw_size)} {unit} ({_fmt_qty(size_in_acres)} acre)"
    else:
        label = f"{_fmt_qty(size_in_acres)} acre"
    return size_in_acres, "acre", label


def _unit_area(question: str) -> tuple[float, str, str] | None:
    """(1.0, basis, label) for an exactly one-acre / one-hectare question.

    farmer_area() returns None for a size of 1 because there is nothing to SCALE,
    but the deterministic schedule still needs the per-unit basis so it can answer
    a "1 acre" or "1 hectare" question with the standard per-unit schedule instead
    of refusing. Only the size-1 case is handled here; every other size stays with
    farmer_area().
    """
    m = _AREA_RE.search(question or "")
    if not m:
        return None
    raw = float(m.group(1)) if m.group(1) else _AREA_WORDS.get((m.group(2) or "").lower(), 0.0)
    unit = (m.group(3) or "").lower()
    if unit in ("hectare", "hectares", "ha"):
        return (1.0, "hectare", "1 hectare") if raw == 1.0 else None
    if raw * _AREA_TO_ACRE.get(unit, 1) == 1.0:
        return 1.0, "acre", "1 acre"
    return None


def _context_rates(context: str) -> list[tuple[str, str, float, float | None, str]]:
    """(substance, unit, low, high, basis) for every rate with an explicit basis."""
    flat = re.sub(r"\s+", " ", (context or ""))
    out = []
    seen = set()
    for rx in (_RATE_RE, _RATE_SINGLE_RE):
        for m in rx.finditer(flat):
            if rx is _RATE_RE:
                low, high, unit, area = m.group(1), m.group(2), m.group(3), m.group(4)
            else:
                low, high, unit, area = m.group(1), None, m.group(2), m.group(3)
            if any(s <= m.start() < e for s, e in seen):
                continue          # already taken by the range pattern
            seen.add((m.start(), m.end()))
            # NEAREST substance, not the first one in _SUBSTANCES order. Taking
            # the first mislabelled every figure in a multi-product sentence:
            # "Apply DAP 35 kg/acre, MOP 15-20 kg/acre" made 35 kg a MOP rate
            # and 15-20 a urea rate, because "urea" appeared later in the same
            # window and sorts earlier in the tuple. A dose attached to the
            # wrong fertiliser is the exact harm this whole module guards.
            win_start = max(0, m.start() - 60)
            window = flat[win_start: m.end() + 60].lower()
            rel = m.start() - win_start
            substance, best = "", _SUBSTANCE_MAX_GAP + 1
            for s in _SUBSTANCES:
                for sm in re.finditer(re.escape(s), window):
                    gap = abs(sm.start() - rel)
                    if gap < best:
                        substance, best = s, gap
            if not substance:
                continue
            row = (substance, unit.lower(), float(low),
                   float(high) if high else None, _basis(area))
            if row not in out:
                out.append(row)
    # ponytail: table format — "Per acre" in header, rows like "Urea 5 kg"
    # e.g. "Fertilizer sources Per acre ... Urea 5 kg 12.5 kg"
    _TABLE_HDR_RE = re.compile(r"\bper\s+(acre|hectare|ha)\b", re.IGNORECASE)
    _TABLE_ROW_RE = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in _SUBSTANCES)
        + r")\s+(\d+(?:\.\d+)?)\s*(kg|g|ton|t|ml|l)\b", re.IGNORECASE)
    hdr = _TABLE_HDR_RE.search(flat)
    if hdr:
        basis = _basis(hdr.group(1))
        for m in _TABLE_ROW_RE.finditer(flat):
            substance = m.group(1).lower()
            low = float(m.group(2))
            unit = m.group(3).lower()
            row = (substance, unit, low, None, basis)
            if row not in out:
                out.append(row)
    return out


def area_scaled_doses(question: str, context: str):
    """Doses rescaled to the farmer's land size.

    Returns `(note, allowed_pairs)`. `note` is prompt text listing the computed
    amounts for the model to COPY; `allowed_pairs` are the (number, substance)
    pairs `_unverified_doses` must accept even though they are not written in
    the documents, because this function produced them.
    """
    area = farmer_area(question)
    if area is None:
        return "", set()
    size, basis, display_label = area

    rates = _context_rates(context)
    matching_count = sum(1 for *_, rb in rates if rb == basis)
    if matching_count < 3:
        fallback = (_STANDARD_RICE_SCHEDULE_HA if basis == "hectare"
                    else _STANDARD_RICE_SCHEDULE_ACRE)
        rates = _context_rates(fallback)

    lines, allowed = [], set()
    for substance, unit, low, high, rate_basis in rates:
        if rate_basis != basis:
            continue              # no unit conversion, ever
        lo = _fmt_qty(low * size)
        if high is not None:
            hi = _fmt_qty(high * size)
            scaled = f"{lo}-{hi}"
            src = f"{_fmt_qty(low)}-{_fmt_qty(high)} {unit} per {rate_basis}"
            allowed |= {(scaled, substance), (lo, substance), (hi, substance)}
        else:
            scaled = lo
            src = f"{_fmt_qty(low)} {unit} per {rate_basis}"
            allowed.add((lo, substance))
        lines.append(f"  - {substance}: documents say {src}; for "
                     f"{display_label} that is {scaled} {unit}")

    if not lines:
        return "", set()
    # ponytail: total across stages when multiple rates exist for a substance
    totals: dict[str, list] = {}
    for substance, unit, low, high, rate_basis in rates:
        if rate_basis != basis:
            continue
        totals.setdefault((substance, unit), []).append((low, high))
    for (substance, unit), entries in totals.items():
        if len(entries) < 2:
            continue
        total_lo = sum(lo for lo, _ in entries) * size
        total_hi = sum((hi or lo) for lo, hi in entries) * size
        tlo = _fmt_qty(total_lo)
        thi = _fmt_qty(total_hi)
        total_str = f"{tlo}-{thi}" if tlo != thi else tlo
        lines.append(f"  - {substance} TOTAL across all stages: "
                     f"{total_str} {unit} for {display_label}")
        allowed |= {(tlo, substance), (thi, substance),
                    (total_str, substance)}
    note = ("\nCOMPUTED AMOUNTS FOR THIS FARMER'S LAND "
            f"({display_label}) — these were calculated for you, "
            "already correct, and you must COPY them rather than work anything "
            "out yourself:\n" + "\n".join(lines)
            + "\nGive the computed amount for what the farmer asked about, and "
            f"say it is for {display_label}. You may also state the "
            f"per-{basis} rate it came from. Do NOT do any arithmetic of your "
            "own and do NOT use a computed amount for a substance the farmer "
            "did not ask about.\n"
            "IMPORTANT: present these amounts STAGE BY STAGE as the CONTEXT "
            "organizes them (basal, active tillering, panicle initiation). "
            "State which fertilizer and how much at each stage. Do NOT give "
            "only a flat total — farmers need to know WHEN to apply WHAT.\n")
    log.info("area scaling | %s %s | %d rate(s) scaled",
             _fmt_qty(size), basis, len(lines))
    return note, allowed


# Stage-grouped standard schedule for the deterministic fallback.
# Each entry: (stage_name, [(substance, per_acre_low, per_acre_high_or_None, unit)])
_STANDARD_STAGES_ACRE = [
    ("basal", [("DAP", 35, None, "kg"), ("Urea", 5, None, "kg"),
               ("MOP", 16, None, "kg"), ("Zinc Sulphate", 10, None, "kg")]),
    ("active tillering", [("Urea", 30, 35, "kg")]),
    ("panicle initiation", [("Urea", 30, 35, "kg"), ("MOP", 15, 20, "kg")]),
]
_STANDARD_STAGES_HA = [
    ("basal", [("DAP", 87.5, None, "kg"), ("Urea", 12.5, None, "kg"),
               ("MOP", 40, None, "kg"), ("Zinc Sulphate", 25, None, "kg")]),
    ("active tillering", [("Urea", 75, 87.5, "kg")]),
    ("panicle initiation", [("Urea", 75, 87.5, "kg"), ("MOP", 37.5, 50, "kg")]),
]


def _deterministic_schedule(question: str, context: str) -> str | None:
    """Render a fertilizer schedule from structured data, no LLM arithmetic.

    Returns None when there's nothing to render (no area or no rates).
    Falls back to the standard Odisha rice schedule when the retrieved context
    has fewer than 3 distinct substance+rate pairs (incomplete table parse).
    """
    area = farmer_area(question) or _unit_area(question)
    if area is None:
        return None
    size, basis, display_label = area
    # Always render a clean, stage-grouped PARAGRAPH from the standard Odisha rice
    # schedule. The earlier ">=3 context rates" branch dumped the raw context
    # rates as a repetitive, bulleted and sometimes mislabelled list ("- UREA: x
    # for 2 acre" several times over, "- DAP: 32 for 2 acre") with no stage
    # context — Q-3012 / Q-3044. The standard schedule is the canonical
    # whole-schedule answer and scales cleanly to the farmer's area.
    stages = _STANDARD_STAGES_HA if basis == "hectare" else _STANDARD_STAGES_ACRE
    sentences = []
    for i, (stage_name, items) in enumerate(stages):
        dosages = []
        for substance, low, high, unit in items:
            lo_s = _fmt_qty(low * size)
            if high is not None:
                hi_s = _fmt_qty(high * size)
                dosages.append(f"{lo_s}-{hi_s} {unit} of {substance}")
            else:
                dosages.append(f"{lo_s} {unit} of {substance}")
        joined = ", and ".join([", ".join(dosages[:-1]), dosages[-1]]) if len(dosages) > 1 else dosages[0]
        if i == 0:
            sentences.append(f"For {display_label} of rice, apply {joined} at the {stage_name.lower()} stage.")
        else:
            sentences.append(f"At the {stage_name.lower()} stage, apply {joined}.")
    return " ".join(sentences)


# Stage-dose questions that name NO area ("how much urea at tillering") used to
# refuse: the context carries the dose only as a RANGE ("30-40 kg/ha", "50% of N
# at tillering"), the model answers with a point value ("35 kg") the grounding
# guard cannot trace, and _deterministic_schedule needs an area to fire. This
# fallback answers such a question from the SAME standard tables, per acre (or
# per hectare if asked), so a stage question gets a grounded amount instead of a
# refusal. It is a last resort reached only on the refusal path.
_STAGE_Q_RE = [
    ("basal", re.compile(r"\bbasal\b", re.I)),
    ("active tillering", re.compile(r"\btiller\w*\b", re.I)),
    ("panicle initiation", re.compile(r"\bpanicle\b|\bpanicle\s+initiation\b", re.I)),
]
_STAGE_PRODUCT_RE = {
    "urea": re.compile(r"\burea\b", re.I),
    "dap": re.compile(r"\bdap\b", re.I),
    "mop": re.compile(r"\bmop\b|\bmuriate\b|\bpotash\b", re.I),
    "zinc sulphate": re.compile(r"\bzinc\b|\bznso4\b", re.I),
}


def _deterministic_stage_dose(question: str) -> str | None:
    """Standard per-acre dose for a named growth stage when NO area is given."""
    q = question or ""
    stages = [name for name, rx in _STAGE_Q_RE if rx.search(q)]
    if not stages:
        return None
    if farmer_area(q) or _unit_area(q):
        return None                         # an area was named -> schedule handles it
    per_ha = bool(re.search(r"\b(hectare|hectares|ha)\b", q, re.I))
    table = dict(_STANDARD_STAGES_HA if per_ha else _STANDARD_STAGES_ACRE)
    basis = "hectare" if per_ha else "acre"
    wanted = [p for p, rx in _STAGE_PRODUCT_RE.items() if rx.search(q)]
    out = []
    for stage in stages:
        items = table.get(stage, [])
        if wanted:
            items = [it for it in items if it[0].lower() in wanted]
        if not items:
            continue
        parts = []
        for substance, low, high, unit in items:
            qty = (f"{_fmt_qty(low)}-{_fmt_qty(high)} {unit}" if high is not None
                   else f"{_fmt_qty(low)} {unit}")
            parts.append(f"{qty} of {substance}")
        joined = (", and ".join([", ".join(parts[:-1]), parts[-1]])
                  if len(parts) > 1 else parts[0])
        out.append(f"At the {stage} stage, apply {joined} per {basis}.")
    if not out:
        return None
    return (" ".join(out) + " (This is the standard recommendation for rice; "
            "adjust to your variety's duration and a soil test where available.)")


def _deterministic_product_dose(question: str) -> str | None:
    """Per-acre schedule for a named product across all stages, when the question
    names a product (urea/DAP/MOP/zinc) but NO stage and NO area count — e.g.
    "how much urea per acre?". Without a stage the stage fallback can't fire and
    without a count the schedule can't, so this returned a refusal although the
    standard schedule has the answer. Last resort on the refusal path only."""
    q = question or ""
    if any(rx.search(q) for _, rx in _STAGE_Q_RE):
        return None                         # a stage -> _deterministic_stage_dose
    if farmer_area(q) or _unit_area(q):
        return None                         # a count -> _deterministic_schedule
    wanted = [p for p, rx in _STAGE_PRODUCT_RE.items() if rx.search(q)]
    if not wanted:
        return None
    per_ha = bool(re.search(r"\b(hectare|hectares|ha)\b", q, re.I))
    table = _STANDARD_STAGES_HA if per_ha else _STANDARD_STAGES_ACRE
    basis = "hectare" if per_ha else "acre"
    disp = {"urea": "Urea", "dap": "DAP", "mop": "MOP", "zinc sulphate": "Zinc Sulphate"}
    out = []
    for product in wanted:
        parts, tot_lo, tot_hi, unit = [], 0.0, 0.0, "kg"
        for stage_name, items in table:
            for substance, low, high, u in items:
                if substance.lower() != product:
                    continue
                unit = u
                tot_lo += low
                tot_hi += high if high is not None else low
                qty = (f"{_fmt_qty(low)}-{_fmt_qty(high)} {u}" if high is not None
                       else f"{_fmt_qty(low)} {u}")
                parts.append(f"{qty} at {stage_name}")
        if not parts:
            continue
        name = disp.get(product, product)
        if len(parts) > 1:
            total = (f"{_fmt_qty(tot_lo)}-{_fmt_qty(tot_hi)}"
                     if tot_lo != tot_hi else _fmt_qty(tot_lo))
            out.append(f"{name}: apply {'; '.join(parts)} per {basis} "
                       f"(about {total} {unit} per {basis} in total across the season).")
        else:
            out.append(f"{name}: apply {parts[0]} per {basis}.")
    if not out:
        return None
    return (" ".join(out) + " (Standard recommendation for rice; adjust to your "
            "variety's duration and a soil test where available.)")


# A hypothetical rate stated in the QUESTION ("if the rate is 80 kg/ha, how much
# for 2 acres?") converted across ha<->acre. area_scaled_doses deliberately does
# "no unit conversion, ever" (it scales a per-acre rate for an acre question,
# never a per-hectare rate), so this cross-unit arithmetic was left to the model,
# which mis-multiplied (80 x 2 = 160, instead of 80/2.471 x 2 ~= 65). Done in
# code, like every other dose figure in this pipeline.
def _rate_conversion_answer(question: str) -> str | None:
    q = question or ""
    rm = _RATE_SINGLE_RE.search(q)
    if not rm:
        return None
    rate, runit, rarea = float(rm.group(1)), rm.group(2), rm.group(3).lower()
    am = _AREA_RE.search(q)
    if am:
        traw = (float(am.group(1)) if am.group(1)
                else _AREA_WORDS.get((am.group(2) or "").lower()))
        tunit = am.group(3).lower()
    else:
        pm = re.search(r"\bper\s+(acre|acres|ac|hectare|hectares|ha)\b", q, re.I)
        if not pm:
            return None
        traw, tunit = 1.0, pm.group(1).lower()
    if traw is None:
        return None
    if _basis(rarea) == _basis(tunit):
        return None                        # same basis: not the ha<->acre bug
    rate_per_acre = rate / _AREA_TO_ACRE[rarea]
    area_in_acres = traw * _AREA_TO_ACRE[tunit]
    total = rate_per_acre * area_in_acres
    rbasis = _basis(rarea)                  # "hectare" or "acre"
    area_in_rbasis = area_in_acres / _AREA_TO_ACRE[rarea]
    tgt_label = f"{_fmt_qty(traw)} {tunit}"
    return (f"At {_fmt_qty(rate)} {runit} per {rbasis}, {tgt_label} "
            f"(about {_fmt_qty(round(area_in_rbasis, 2))} {rbasis}) needs about "
            f"{_fmt_qty(round(total))} {runit} in total — roughly "
            f"{_fmt_qty(round(rate_per_acre, 1))} {runit} per acre. "
            f"(1 hectare = 2.471 acres.)")


def _area_converted_nums(ctx_nums: set) -> set:
    """Expand context numbers with plausible hectare↔acre conversions."""
    FACTOR = 2.471
    extra = set()
    for n in ctx_nums:
        try:
            parts = n.split("-")
            converted = []
            for p in parts:
                v = float(p)
                converted.append(_fmt_qty(v / FACTOR))  # ha→acre
                converted.append(_fmt_qty(v * FACTOR))  # acre→ha
            extra.add("-".join(converted[:len(parts)]))
            extra.add("-".join(converted[len(parts):]))
        except (ValueError, ZeroDivisionError):
            continue
    return extra


def _unverified_doses(answer: str, context: str, allowed_pairs=None) -> list[str]:
    """Doses in `answer` whose (number, substance) pair is absent from `context`.

    Prompt rules could not hold this down — three separate wordings each produced
    a different wrong dose for the same question. This verifies the output
    instead of asking the model to behave.

    `allowed_pairs` carries the figures `area_scaled_doses` computed in code.
    They are absent from the documents by definition — that is the whole point
    of computing them — so without this they would be reported as invented.
    """
    ctx_pairs = _dose_pairs(context) | set(allowed_pairs or ())
    ctx_nums = {n for n, _ in ctx_pairs}
    ctx_nums |= _area_converted_nums(ctx_nums)
    bad = []
    for num, sub in sorted(_dose_pairs(answer)):
        if (num, sub) in ctx_pairs:
            continue
        why = "wrong substance" if num in ctx_nums else "not in the documents"
        bad.append(f"{num} of {sub} ({why})")
    for dose in _substance_free_doses(answer):
        m = _DOSE_RE.search(dose)
        if m:
            num = m.group(1).replace(" ", "").replace("–", "-")
            if num in ctx_nums:
                log.info("substance-free dose %r tolerated — number %s "
                         "exists in context", dose, num)
                continue
        bad.append(f"{dose} (does not say of WHAT — unusable and unverifiable)")
    return bad


def _drop_substance_free_dose_sentences(answer: str,
                                       context: str = "") -> str:
    """Remove sentences carrying an application figure that names no input.

    Never empties the answer: if every sentence is one of these the answer is
    returned unchanged, because saying something questionable beats saying
    nothing at all when the farmer is waiting.

    When *context* is given, a substance-free dose whose number appears in the
    context paired with a known substance is tolerated — the LLM dropped the
    substance name but the figure itself is grounded.
    """
    ctx_nums = {n for n, _ in _dose_pairs(context)} if context else set()
    parts = [p for p in re.split(r"(?<=[.!?])\s+", (answer or "").strip()) if p.strip()]

    def _has_real_bad_dose(sentence: str) -> bool:
        for d in _substance_free_doses(sentence):
            m = _DOSE_RE.search(d)
            if m:
                num = m.group(1).replace(" ", "").replace("–", "-")
                if num in ctx_nums:
                    continue
            return True
        return False

    kept = [p for p in parts if not _has_real_bad_dose(p)]
    if not kept or len(kept) == len(parts):
        return answer
    log.warning("dropped sentence(s) with an input-less dose: %s",
                [p for p in parts if p not in kept])
    return " ".join(kept).strip()


_FERT_NAME_RE = re.compile(
    r"\b(urea|dap|mop|potash|muriate|zinc|npk|ssp|dhaincha|azolla|fym|compost|"
    r"manure|vermicompost)\b", re.I)


def _has_fertiliser_dose(text: str) -> bool:
    """True if the text still carries a usable fertiliser amount — a number, a
    weight unit, and a fertiliser name. Used to tell a real answer from the
    fragment left behind after dose-stripping."""
    t = text or ""
    return bool(re.search(r"\d", t)
                and re.search(r"\b(kg|kilo\w*|bag|quintal|tonne|ton)\b", t, re.I)
                and _FERT_NAME_RE.search(t))


def _context_from(retrieved_hits: list) -> tuple:
    """(context, catalogue names). Blocks grouped by catalogue when there are two.

    The [Source: ...] marker is left exactly as it was — everything downstream
    finds a chunk's document by walking up to the nearest one — and the
    catalogue boundary is a separate header line above it. With a single
    catalogue the text is byte-for-byte what it has always been, so nothing
    about the ordinary path changes.
    """
    blocks, names = [], []
    for hit in retrieved_hits:
        name = (hit.get("metadata") or {}).get("catalogue_name") or ""
        if name and name not in names:
            names.append(name)
    grouped = len(names) > 1
    current = None
    for hit in retrieved_hits:
        meta = hit.get("metadata") or {}
        if grouped:
            name = meta.get("catalogue_name") or "other sources"
            if name != current:
                blocks.append(f"===== CATALOGUE: {name} =====")
                current = name
        blocks.append(f"[Source: {meta.get('document_title', 'doc')}]\n{hit['chunk']}")
    return "\n\n".join(blocks), names


# ── Did the answer keep every variety it was given? ───────────────────────
#
# The pool is right, the prompt lists all of them and says "give ALL of these,
# not a selection" — and the model still returns five of six. A farmer reading
# that has no way to know a seed was left out, and the one omitted may be the
# only one their dealer stocks. This is the same shape as the dose and
# claim-strength checks below it: detect, retry ONCE naming what was dropped,
# and accept the retry only if it actually fixed it without inventing a name.
#
# NOTHING HERE DECIDES WHICH VARIETIES ARE SUPPORTED. It only compares the
# answer against the list generation was already handed, so it cannot widen or
# narrow the pool, and it is skipped entirely unless the farmer asked for
# options and more than one was supported.


def _name_pattern(name: str) -> "re.Pattern":
    """A matcher for one variety name that punctuation and case cannot fool.

    "MTU-1140" is written MTU 1140, MTU1140 and mtu-1140 in the same corpus, and
    all three are the same seed. What it must NOT do is match a different name
    that merely starts the same: the boundaries stop "CR Dhan 31" matching
    inside "CR Dhan 310", which plain substring matching would.
    """
    parts = [re.escape(p) for p in re.split(r"[^A-Za-z0-9]+", name or "") if p]
    if not parts:
        return re.compile(r"(?!x)x")            # matches nothing
    return re.compile(r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]*".join(parts)
                      + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _missing_candidates(answer: str, names: list) -> list:
    """The supported varieties the answer failed to name."""
    text = answer or ""
    return [name for name in names if not _name_pattern(name).search(text)]


def _unsupported_varieties(answer: str, allowed: list, question: str = "") -> list:
    """Variety names the answer introduces that were not on the supported list.

    The guard on the retry: being told to add the missing ones must not become
    licence to add others. A name the FARMER used is not an addition.
    """
    permitted = list(allowed) + entity_intent.named_varieties(question or "")
    found = []
    for name in entity_intent._names_in(answer or "", "cultivar"):
        # `search` as well as `fullmatch`, because the cultivar shape absorbs a
        # capitalised word in front of "Dhan": the question "Is CR Dhan 100 any
        # good?" yields the permitted name "Is CR Dhan 100", and the answer's
        # "CR Dhan 100" has to be recognised inside it. Extraction is left
        # exactly as it is; this guard is simply not fooled by it.
        if any(_name_pattern(ok).fullmatch(name)
               or _name_pattern(name).search(ok) for ok in permitted):
            continue
        if name not in found:
            found.append(name)
    return found


# ── How much of the CONTEXT the question actually asked for ───────────────
#
# "When and how much fertiliser should be applied to rice?" came back with
# three stages, four products and seven figures — every row of the schedule the
# retrieval returned. Nothing was wrong with the facts; the ANSWER was wrong,
# because it answered a bigger question than the farmer asked.
#
# The cause was a rule conflict, not a missing rule. "ANSWER ONLY WHAT WAS
# ASKED" sat a few lines above "IF THE QUESTION ASKS GENERALLY WHAT FERTILISER A
# CROP NEEDS, with no stage named ... give the COMPLETE schedule". The second
# rule exists for a real harm — a farmer asking what to apply in DSR was once
# told the basal dose alone, one seventh of the timetable — and this question
# names no stage, so the model obeyed it. Both rules were right; nothing said
# which one governs.
#
# So the scope is decided HERE, before the prompt is built, and the prompt is
# told the answer rather than left to weigh two rules. Nothing is truncated
# afterwards: a scope decision made after generation is just deletion, and it
# removes exactly the facts a short answer needed.
SCOPE_COMPLETE = "complete_schedule"   # asked for all of it, in so many words
SCOPE_BROAD = "broad"                  # open-ended "how do I manage X"
SCOPE_TIMING_QUANTITY = "timing_and_quantity"
SCOPE_QUANTITY = "quantity"            # one figure
SCOPE_SINGLE_FACT = "single_fact"      # one property of one thing
SCOPE_DEFAULT = "direct"

_SCOPE_COMPLETE_RE = re.compile(
    r"\b(?:complete|full|entire|whole|all\s+the|every)\s+"
    r"(?:fertili[sz]er\s+)?(?:schedule|timetable|programme|program|list|"
    r"application|applications|recommendations?|doses?|stages?)\b"
    r"|\blist\s+(?:all|every)\b|\ball\s+(?:the\s+)?(?:stages?|applications?|"
    r"fertili[sz]ers?|doses?)\b|\bstage[- ]wise\b|\bat\s+each\s+stage\b|"
    r"\bfor\s+each\s+stage\b", re.IGNORECASE)
_SCOPE_BROAD_RE = re.compile(
    r"\bhow\s+(?:should|do|can)\s+i\s+manage\b|\bmanagement\s+of\b"
    r"|\bhow\s+to\s+manage\b|\btell\s+me\s+about\b|\bexplain\b|\boverview\b",
    re.IGNORECASE)
_SCOPE_WHEN_RE = re.compile(r"\bwhen\b|\bwhat\s+time\b|\bwhich\s+stage\b"
                            r"|\btiming\b|\bhow\s+long\s+before\b", re.IGNORECASE)
_SCOPE_HOWMUCH_RE = re.compile(r"\bhow\s+much\b|\bhow\s+many\b|\bquantity\b"
                               r"|\bdose\b|\brate\b|\bper\s+(?:acre|hectare)\b",
                               re.IGNORECASE)
_SCOPE_SINGLE_FACT_RE = re.compile(
    r"\bwhat\s+is\s+the\s+(?:duration|yield|height|maturity|grain\s+type|"
    r"seed\s+rate|spacing|name)\b|\bhow\s+long\s+does\b|\bhow\s+many\s+days\b",
    re.IGNORECASE)


def response_scope(question: str) -> str:
    """How much of the CONTEXT this question asked for.

    Deliberately conservative and ordered widest-first: anything that asks for
    the whole schedule keeps the whole schedule, and a question this does not
    recognise falls through to the ordinary "answer what was asked" behaviour.
    """
    text = question or ""
    if _SCOPE_COMPLETE_RE.search(text):
        return SCOPE_COMPLETE
    if _SCOPE_BROAD_RE.search(text):
        return SCOPE_BROAD
    if _SCOPE_SINGLE_FACT_RE.search(text):
        return SCOPE_SINGLE_FACT
    when, how_much = _SCOPE_WHEN_RE.search(text), _SCOPE_HOWMUCH_RE.search(text)
    if when and how_much:
        return SCOPE_TIMING_QUANTITY
    if how_much:
        return SCOPE_QUANTITY
    return SCOPE_DEFAULT


# What the prompt is told for each scope. The wording never mentions a length:
# a limit invites the model to drop a figure the farmer needed, and the point is
# to answer a smaller QUESTION, not to give a shorter answer to a big one.
_SCOPE_NOTES = {
    SCOPE_COMPLETE: (
        "The farmer asked for the COMPLETE set. Give every stage and every "
        "application the CONTEXT supports, in the order the document states "
        "them, leaving none out."),
    SCOPE_BROAD: (
        "The farmer asked an open question, so a broader answer is right. Cover "
        "what the CONTEXT supports for it, without reciting figures that "
        "nothing in the question asked about."),
    SCOPE_SINGLE_FACT: (
        "The farmer asked for ONE property of ONE thing. Give that property and "
        "stop. Do not recite the rest of its profile — not its yield, its land "
        "type, its grain, its doses — however much of that the CONTEXT holds."),
    SCOPE_QUANTITY: (
        "The farmer asked HOW MUCH. Give the figure for what they named, with "
        "its unit and the stage it belongs to, and stop. Do not add the other "
        "products, the other stages, or the rest of the schedule."),
    SCOPE_TIMING_QUANTITY: (
        "The farmer asked WHEN and HOW MUCH. Both belong in the answer, and "
        "each stage the CONTEXT supports for what they named should carry its "
        "timing and its figure — said compactly, one short line per stage. Do "
        "not add the explanation of why, the benefits, or applications for "
        "something they did not ask about."),
    # Every question used to fall through this one with NO scope note at all,
    # which meant the most ordinary questions — the bulk of what farmers ask —
    # got no instruction about scope beyond the general rules. It says nothing
    # about how MUCH to say, because a question with no recognised shape may
    # legitimately need the whole schedule (see the DSR rule below); it governs
    # only the padding around the answer.
    SCOPE_DEFAULT: (
        "Answer this one directly. Everything the CONTEXT supports FOR WHAT WAS "
        "ASKED belongs in the answer — but nothing around it: no restating of "
        "the question, no preamble, no summary of what you are about to say, and "
        "no closing advice."),
}


def _scope_note(question: str) -> str:
    """The ANSWER SCOPE block for this question, or "" when nothing applies."""
    scope = response_scope(question)
    detail = _SCOPE_NOTES.get(scope)
    if not detail:
        return ""
    log.info("answer scope | %s", scope)
    return f"\n- ANSWER SCOPE for THIS question — {scope}. {detail}\n"


def _requested_criterion(intent) -> str:
    """The facet the farmer asked for, in the documents' own words, or "".

    "medium duration", "upland", "drought" — what the answer has to be ABOUT.
    """
    if intent is None:
        return ""
    parts = []
    for group, value in (intent.attributes or {}).items():
        values = value if isinstance(value, list) else [value]
        for canonical in values:
            term = entity_intent.canonical_term(group, canonical)
            if term and term not in parts:
                parts.append(term)
    return ", ".join(parts)


def _candidate_lines(pool: list) -> list:
    """"NAME  [its row states: <conditions>]" for each candidate.

    The conditions are the facets read off the candidate's OWN evidence line, in
    the document's wording. Carrying them into the prompt is what lets an answer
    say what a variety is recommended FOR, instead of implying it suits whatever
    field the farmer happens to have.

    THE BRACKETS ARE THE POINT. This used to read "NAME - the CONTEXT says: ..."
    and the model copied that wording straight through, so farmers were reading
    "CR Dhan 100 - the CONTEXT says: Fit for upland" — a phrase from the
    machinery, in an answer meant to be read aloud to someone standing in a
    field. A bracketed note reads as an aside to the model rather than a
    sentence to repeat, and the rule beside it says so outright. The scrub in
    _strip_internal_wording() is what makes it certain.
    """
    lines = []
    for candidate in pool:
        stated = candidate.get("stated_phrases") or []
        if stated:
            lines.append(f"{candidate['name']}  [its row states: "
                         + ", ".join(stated) + "]")
        else:
            lines.append(candidate["name"])
    return lines


# ── Wording that belongs to the machinery, not to the farmer ──────────────
#
# The answer is read aloud in Odia to someone with a phone in a paddy field.
# "The CONTEXT says" is not a phrase from their world; it is a label from this
# module's prompt, and hearing it tells them nothing except that a machine
# wrote it. The persona already forbids these words, and the prompt now says so
# twice — but an instruction is a request, and this is a guarantee. Every
# substitution below preserves the surrounding sentence: labels come off, and
# nothing that carries meaning is touched.
#
# NOTHING HERE MAY WEAKEN A CLAIM. No rule rewrites an adjective, a number or a
# qualifier — "moderately resistant" and "up to 2 weeks" pass through these
# patterns untouched, because none of them matches an adjective or a figure.
_INTERNAL_WORDING = [
    # The reported leak: "CR Dhan 100 - the CONTEXT says: Fit for upland".
    (re.compile(r"\s*[-–—]?\s*\b(?:the\s+|this\s+)?context\s+"
                r"(?:says|states|shows|mentions|indicates|reads)\s*:\s*", re.I), " – "),
    # The same thing as a clause: "the context says that X is fit for upland".
    (re.compile(r"\b(?:the\s+|this\s+)?context\s+"
                r"(?:says|states|shows|mentions|indicates)\s+that\s+", re.I), ""),
    # Provenance openers, which add nothing the farmer can act on.
    (re.compile(r"\b(?:as\s+)?(?:per|according\s+to|based\s+on)\s+the\s+"
                r"(?:context|retrieved\s+\w+|available\s+context|sources?|documents?)"
                r"\s*[,:]?\s*", re.I), ""),
    (re.compile(r"\bthe\s+(?:sources?|documents?)\s+"
                r"(?:says?|states?|mentions?|indicates?|shows?)\s+that\s+", re.I), ""),
    (re.compile(r"\s*[-–—]?\s*\bthe\s+(?:sources?|documents?)\s+"
                r"(?:says?|states?|mentions?|indicates?)\s*:\s*", re.I), " – "),
    # A refusal must survive, so this one SWAPS the label instead of cutting it:
    # "the context does not specify" still has to say that nothing is known.
    (re.compile(r"\bthe\s+(?:retrieved\s+)?(?:context|documents?|sources?)\s+"
                r"(?:does\s+not|doesn't|do\s+not|don't)\b", re.I),
     "the information available does not"),
    (re.compile(r"\bthe\s+(?:retrieved\s+)?(?:context|chunks?|passages?)\b", re.I),
     "the information available"),
    # Bare machinery words. "Context" and "chunk" have no farming sense here;
    # "source" and "document" are left alone unless one of the rules above
    # already identified them as provenance, because a farmer's answer may
    # legitimately talk about a source of water or a nutrient source.
    (re.compile(r"\b(?:retrieved\s+)?(?:chunks?|candidate\s+pool|grounding\s+rules?|"
                r"retrieval|the\s+prompt)\b", re.I), "the information available"),
    (re.compile(r"\bcontext\b", re.I), "information"),
    # The bracketed note from _candidate_lines(), if it is ever echoed.
    (re.compile(r"\s*\[\s*its\s+row\s+states\s*:\s*([^\]]*)\]", re.I), r" – \1"),
]


def _strip_internal_wording(answer: str) -> str:
    """Remove the machinery's own vocabulary from an answer, meaning intact."""
    text = answer or ""
    for pattern, replacement in _INTERNAL_WORDING:
        text = pattern.sub(replacement, text)
    # Tidy what the substitutions leave behind: doubled spaces, a dash that
    # ended up against punctuation, and a sentence now starting mid-word.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?:^|(?<=[.!?]\s))\s*[–—-]\s*", "", text)
    text = re.sub(r"([–—-])\s*\1+", r"\1", text)
    text = re.sub(r"(^|(?<=[.!?]\s))([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text != (answer or "").strip():
        log.info("stripped internal wording from the answer")
    return text


# ---------------------------------------------------------------------------
# COMPLETE VARIETY LISTS BY DURATION.
#
# "Name the 150-day varieties" has one correct answer: all nine of them. Four
# separate things stopped that happening, every one of them a prose heuristic
# meeting a spreadsheet:
#   1. candidate_pool() returned [] for the table rows.
#   2. detect_attributes() read "Submergence" off the Stress column but not
#      "Long" off the Duration Category column, so the maturity facet the
#      question asked for was never stated and the rows were filtered out.
#   3. _names_in() split "CR 1009 SUB-1" into "CR 1009" and "SUB-1".
#   4. "name some 150 day rice" did not even parse as a variety question.
# Teaching all four to read a table is a lot of blast radius for one shape of
# question. The table is already structured, so it is read as a table here and
# the result is handed over finished — the same approach as the dose scaler.

# ──────────────── Follow-up question detection (module-level) ────────────────
# ponytail: pronoun references that signal a follow-up to a prior question
_FOLLOWUP_RE = re.compile(
    r"\b(this|that|these|those|it|its|the same|above|previous)\b",
    re.IGNORECASE)
# Subset: references that point at a LIST of entities in the previous answer.
_ENTITY_REF_RE = re.compile(
    r"\b(these|those|above|the above|which one|which of|from these|from those"
    r"|the first one|the second one|the third one|the last one"
    r"|first one|second one|third one|last one"
    r"|that variety|that one|this variety|this one"
    r"|all these|all those|more about this|more about it)\b",
    re.IGNORECASE)
# A short message that is itself a complete question (starts with a question
# word) is standalone, not a follow-up — even if it's only 3 words.
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|how|which|when|where|why|who|is|are|can|do|does|should|will|tell)\b",
    re.IGNORECASE)


def _is_followup(query_en: str, chat_hist: list) -> bool:
    """Decide whether `query_en` is a follow-up to the prior conversation.

    True when the question contains an anaphoric reference ("this variety",
    "that one"), OR when it is a short non-question reply to a clarifying
    question the system just asked ("irrigated", "medium land").
    """
    if not chat_hist:
        return False
    # Explicit pronoun reference — always a follow-up regardless of length.
    if _FOLLOWUP_RE.search(query_en):
        return True
    # Entity reference ("which one", "which of these") — a contextual follow-up
    # even though it starts with a question word.
    if _ENTITY_REF_RE.search(query_en):
        return True
    # The last assistant message asked a clarifying question — any non-question
    # reply is an answer to it, regardless of length ("MTU 1010, 120 days").
    # Also catches polite requests ("please tell me", "please provide") that
    # the LLM sometimes phrases without a question mark.
    last_assistant = (chat_hist[-1].get("assistant") or "").strip()
    _last_is_clarifying = (
        "?" in last_assistant
        or re.search(r"\bplease\s+(tell|provide|share|give)\b", last_assistant,
                     re.IGNORECASE)
    )
    if _last_is_clarifying and not _QUESTION_WORD_RE.match(query_en):
        return True
    # Short reply (1-3 words) that isn't a standalone question.
    words = query_en.split()
    if len(words) > 3:
        return False
    if _QUESTION_WORD_RE.match(query_en):
        return False
    return False


#
# ponytail: keyed to the "A | B | C" row layout the xlsx/csv extractors emit
# (see chunking.extract_xlsx_text). A varietal sheet published as prose will not
# be picked up; the ranked path still answers those, just less completely.
_DURATION_ROW_RE = re.compile(
    r"^\s*\d+\s*\|\s*([^|]+?)\s*\|\s*(\d{2,3})\s*\|", re.MULTILINE)

# The question must name a day count AND actually be ASKING FOR VARIETIES.
# Both halves earn their place:
#   - without the day count, "20 days after transplanting" is a timing phrase;
#   - without the asking-for-names test, "how much urea for 120 day rice" would
#     force 53 variety names into a fertiliser answer, which is a far worse
#     failure than the short list this is meant to fix.
_DURATION_LIST_RE = re.compile(
    r"\b(\d{2,3})\s*[-– ]?\s*day(?:s)?\b", re.IGNORECASE)
_WANTS_VARIETIES_RE = re.compile(
    # variet* plus the common misspellings varit*/vareit* ("varities", "varity")
    r"\b(?:variet\w*|varit\w*|vareit\w*|kisam|cultivar\w*)\b"
    # A listing verb and the crop word, with room for the duration between
    # them — "name some 150 day rice", "list 145 day dhan".
    r"|\b(?:which|name|names|list|suggest|give|recommend)\s+"
    r"(?:\w+\s+){0,4}(?:rice|paddy|dhan)\b"
    r"|\b(?:rice|paddy|dhan)\s+(?:names?|types?)\b", re.IGNORECASE)
# A quantity question is never a request for a variety list, however many crop
# words it contains.
_IS_DOSE_QUESTION_RE = re.compile(
    r"\b(how much|how many kg|dose|dosage|quantity|kg|urea|fertili[sz]er|mop|dap"
    r"|npk|appl(y|ication)|spray)\b", re.IGNORECASE)
_DURATION_MIN, _DURATION_MAX = 70, 200


def duration_requested(question: str) -> int | None:
    """The crop duration in days a VARIETY-LIST question asks about, or None."""
    text = question or ""
    if _IS_DOSE_QUESTION_RE.search(text):
        return None
    if not _WANTS_VARIETIES_RE.search(text):
        return None
    for m in _DURATION_LIST_RE.finditer(text):
        days = int(m.group(1))
        if _DURATION_MIN <= days <= _DURATION_MAX:
            return days
    return None


def varieties_by_duration(catalogue, days: int) -> list:
    """Every variety in the catalogue whose Duration column is exactly `days`.

    Exact, not banded: a farmer asking for 150-day varieties is not asking for
    the 145s. Returned in catalogue order with duplicates collapsed.
    """
    from catalogues.services.vector_store import chunks_containing

    names, seen = [], set()
    # The substring narrows the fetch; the regex is what actually decides, so a
    # "150" sitting in a yield or a serial number cannot get in.
    for chunk in chunks_containing(catalogue, str(days)):
        for m in _DURATION_ROW_RE.finditer(chunk):
            if int(m.group(2)) != days:
                continue
            name = m.group(1).strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    if names:
        log.info("duration list | %d-day varieties found: %d (%s)",
                 days, len(names), ", ".join(names))
    return names


def _variety_has_advisory(variety: str, hits: list) -> bool:
    """True if a duration-based fertiliser advisory (doc-90) for this variety is
    in the retrieved context. Such an advisory is by crop DURATION, not land
    type, so it answers the question without the land-type clarification."""
    key = re.sub(r"[^a-z0-9]", "", re.sub(r"\bdhan\b", "", (variety or "").lower()))
    if len(key) < 3:
        return False
    for h in hits:
        m = h.get("metadata") or {}
        if m.get("content_type") != "fertilizer_advisory":
            continue
        txt = re.sub(r"[^a-z0-9]", "", (h.get("chunk") or "").lower())
        mv = re.sub(r"[^a-z0-9]", "", str(m.get("variety", "")).lower())
        if key in txt or (mv and (key in mv or mv in key)):
            return True
    return False


def _variety_input_note(intent, retrieved_hits: list) -> str:
    """The note for "which fertiliser for <variety>?" — a relationship question.

    The documents hold this answer in two halves: one page says which land a
    variety is for, another says what to apply on that land. Answered without
    the first half, the figures come from whichever ecosystem retrieval happened
    to return, and a dose for upland goes to a farmer with a lowland field.

    So the variety's own ecosystem is established FIRST, from its own rows only
    (entity_intent.variety_conditions), and the answer is scoped to it. When the
    documents do not establish it, that is the answer: no mapping, no dose.

    Returns "" for every other question, so nothing else is affected.
    """
    if intent is None or intent.name_kind != "input":
        return ""
    named = entity_intent.named_varieties(intent.question)
    if not named:
        return ""                       # a general fertiliser question, unchanged

    variety = named[0]
    conditions = entity_intent.variety_conditions(variety, retrieved_hits)

    # The main retrieval is optimised for the question, not the variety name.
    # When the variety metadata row is not in the top hits, do a small targeted
    # lookup so the land type can still be established.
    if not conditions:
        from catalogues.models import Catalogue
        try:
            variety_cat = Catalogue.objects.filter(
                name__icontains="variet").first()
            if variety_cat:
                extra = retrieve(variety_cat, variety, n_results=3)
                conditions = entity_intent.variety_conditions(variety, extra)
                if conditions:
                    log.info("variety_input_note: found %s conditions via "
                             "targeted retrieval from catalogue %s",
                             variety, variety_cat.name)
        except Exception:
            pass

    phrases = [p for p in (conditions.get("_phrases") or [])]
    land = conditions.get("land_type", "")

    if not land:
        # A duration-based advisory (doc-90) answers this without a land type.
        if _variety_has_advisory(variety, retrieved_hits):
            log.info("fertiliser for %s: duration-based advisory in context — "
                     "answering without asking land type", variety)
            return (
                f"\n\nTHIS IS A QUESTION ABOUT {variety.upper()}. The CONTEXT "
                f"contains a fertiliser/nutrient advisory for {variety} based on "
                "its crop DURATION (short / medium / long duration). Give that "
                "advisory — the per-acre schedule (basal, active tillering, "
                f"panicle initiation) the CONTEXT states for {variety}. It is by "
                "duration and does NOT depend on land type, so do NOT ask which "
                "land type the farmer has. Use ONLY the figures the CONTEXT gives "
                f"for {variety}.\n")
        log.info("fertiliser for %s: land type not established — asking farmer",
                 variety)
        return (
            f"\n\nTHIS IS A QUESTION ABOUT {variety.upper()}. The CONTEXT "
            "contains fertiliser recommendations for different land types but "
            f"does not say which land type {variety} is grown on.\n"
            "DO THIS: Ask the farmer which type of land they have — upland, "
            "medium land, lowland (irrigated or rainfed) — so you can give "
            "the correct recommendation. Keep the question SHORT (one sentence). "
            "Do NOT guess the land type, and do NOT give a fertiliser "
            "recommendation until the farmer answers.\n"
            "If the CONTEXT contains a GENERAL rice fertiliser recommendation "
            "that is not land-type-specific, you may give that instead and "
            "note that it is a general recommendation.\n")

    stated = ", ".join(phrases) if phrases else land.replace("_", " ")
    log.info("fertiliser for %s: scoped to its stated land type %s (%s)",
             variety, land, stated)
    return (
        f"\n\nTHIS IS A QUESTION ABOUT {variety.upper()}, ANSWERED THROUGH THE "
        f"LAND IT IS GROWN ON. The CONTEXT states that {variety} is for: "
        f"{stated}. FIRST preference: give the fertiliser the CONTEXT "
        f"recommends specifically for {stated}. Do NOT use a recommendation "
        "the CONTEXT gives for a DIFFERENT land type or ecosystem, and do NOT "
        "mix figures from two of them.\n"
        "SECOND preference: if the CONTEXT has NO land-type-specific "
        "recommendation but DOES have a GENERAL rice fertiliser "
        "recommendation (one that is not tied to any particular land type), "
        "you may give that general recommendation and note that it is a "
        f"general recommendation, not specific to {stated}.\n"
        f"If the CONTEXT has NEITHER, say that plainly.\n")


# ── Dose basis: fertiliser PRODUCTS vs nutrient ELEMENTS ──────────────────
#
# "Fertilisation" names the PRODUCT a farmer buys and spreads — Urea, DAP, MOP,
# SSP, Zinc sulphate. "Nutrient" names the ELEMENT a crop takes up — N, P (P2O5),
# K (K2O). A farmer who asks "how much fertiliser" wants products to buy; one who
# asks "how much nutrient / NPK / nitrogen" wants the elemental requirement. The
# two are NOT interchangeable, and converting between them (DAP is 18-46-0, Urea
# 46-0-0, MOP 0-0-60) is arithmetic the model must not improvise — a wrong factor
# is a wrong dose. So the answer is steered to the basis the farmer asked in,
# using figures the CONTEXT already states in that basis; it is never asked to
# convert. Decided only when the wording leans ONE way; "potash" is in neither
# group because it is both the K2O nutrient and the corpus's word for MOP.
_PRODUCT_WORD_RE = re.compile(
    r"\b(fertili[sz]\w*|manur\w*|urea|dap|mop|muriate|ssp|tsp|znso4|gypsum|"
    r"zinc\s*sulphate|zinc\s*sulfate|di[-\s]?ammonium|single\s*super)\b", re.I)
_NUTRIENT_WORD_RE = re.compile(
    r"\b(nutrient\w*|micronutrient\w*|npk|n\s*[:\-]?\s*p\s*[:\-]?\s*k|nitrogen|"
    r"phosphor\w*|phosphate|potassium|k2o|p2o5)\b", re.I)


def _dose_basis(question: str) -> str:
    """'product', 'nutrient' or '' — the basis a dose answer should be given in."""
    q = question or ""
    prod, nutr = bool(_PRODUCT_WORD_RE.search(q)), bool(_NUTRIENT_WORD_RE.search(q))
    if prod == nutr:                      # both named, or neither -> do not steer
        return ""
    return "product" if prod else "nutrient"


def generate_answer(query_text_en: str, retrieved_hits: list, user_type: str = "farmer",
                    chat_history: list | None = None,
                    must_name: list | None = None) -> str:
    """`must_name` is a COMPLETE list the answer has to reproduce in full —
    currently the varieties matching a duration the farmer asked for, read
    straight off the catalogue table rather than off the ranked chunks."""
    # A pure ha<->acre rate-conversion probe ("if the rate is 80 kg/ha, how much
    # for 2 acres?") is answered in code — the arithmetic, not the documents, is
    # the question, and the model gets the cross-unit multiply wrong.
    conv = _rate_conversion_answer(query_text_en)
    if conv:
        log.info("answered by deterministic ha<->acre rate conversion")
        return conv
    context, catalogue_names = _context_from(retrieved_hits)
    history_note = format_history_for_prompt(chat_history) if chat_history else ""

    # When the farmer references entities from the previous answer ("these",
    # "those", "the above", "which one"), constrain the LLM to those entities.
    followup_constraint = ""
    if chat_history and _ENTITY_REF_RE.search(query_text_en):
        prev_answer = (chat_history[-1].get("assistant") or "").strip()
        if prev_answer:
            followup_constraint = (
                "\n\nFOLLOW-UP ENTITY CONSTRAINT:\n"
                "The farmer is referring to items from the PRIOR CONVERSATION "
                "HISTORY above ('these', 'those', 'the above', 'which one'). "
                "The previous PaddyMitra answer listed specific items "
                "(varieties, fertilizers, quantities). Those are the REFERENCE "
                "SET for this question.\n"
                "- Answer ONLY about the items named in the previous PaddyMitra "
                "answer. Do NOT substitute different varieties, fertilizers, or "
                "other entities.\n"
                "- Look carefully through ALL of the CONTEXT, including any "
                "tabular data with columns like name | days | duration class. "
                "A row such as '50 | INDRA(MTV-1061) | 150 | Long' tells you "
                "that MTV-1061 matures in 150 days. Extract and compare these "
                "numbers for the varieties in the REFERENCE SET.\n"
                "- Only say the information is not available if, after checking "
                "every row and paragraph in the CONTEXT, you truly cannot find "
                "the requested data for ANY of the reference items.\n"
                "- Do NOT change quantities from the previous answer unless the "
                "CONTEXT explicitly provides a different figure for the exact "
                "scenario being discussed.\n"
            )
            log.info("query | follow-up entity constraint applied")

    # ASKED WHICH VARIETY, GIVEN THE OPTIONS - IN EITHER WORDING.
    # Which entities the evidence supports is a question about the retrieved
    # text, not a judgement call, so it is answered here and handed over. The
    # POOL is the same for "which variety" and "which varieties", and so is the
    # ANSWER: both list every grounded option. Only a farmer who explicitly
    # asked for one — best, top, only one — gets a single pick, and it comes out
    # of that same pool.
    variety_note = ""
    # The varieties the answer is REQUIRED to name, set only on the
    # multiple-options path below. Empty everywhere else, which is what keeps
    # the completeness check off every other kind of question.
    required_names: list = []
    intent = entity_intent.analyse(query_text_en)
    # A fertiliser question that names a variety is answered through the land
    # that variety is grown on; every other fertiliser question is untouched.
    variety_note = _variety_input_note(intent, retrieved_hits)

    # A duration list read off the catalogue table. It REPLACES the ranked pool
    # for this question rather than adding to it: the table is the authority on
    # which varieties have that duration, and a name the ranked chunks happened
    # to surface is not evidence of a duration the table did not record.
    if must_name:
        required_names = list(must_name)
        variety_note = (
            "\n\nCOMPLETE LIST — the catalogue records exactly these "
            f"{len(must_name)} varieties for the duration the farmer asked "
            "about. This list was read from the catalogue table directly, so it "
            "is COMPLETE and you must name EVERY ONE of them. Do not shorten "
            "it, do not pick a few, do not add a variety that is not here, and "
            "do not say there are others:\n- "
            + "\n- ".join(must_name) + "\n")
        log.info("variety request (duration table): requiring all %d names",
                 len(must_name))
    # `must_name` already answered this one from the table; the ranked pool
    # below would only overwrite a complete list with a partial one.
    if not must_name and intent is not None and intent.name_kind == "cultivar":
        pool = entity_intent.candidate_pool(intent, retrieved_hits)
        supported = [c["name"] for c in pool]
        listed = "\n- ".join(_candidate_lines(pool))

        # THE FARMER NEVER SAID WHICH LAND THEY HAVE. A varietal sheet
        # recommends a variety FOR an ecosystem — shallow favourable lowland,
        # irrigated late, upland — and a question that names only a duration
        # class ("a medium duration variety") does not say which of those the
        # farmer's field is. Answering "this variety is suitable for your land"
        # is then a claim no document made. Either ask, or carry the source's
        # own condition into the answer; both are honest, and the second is
        # what a farmer who is standing in the field can actually use.
        # WHAT THE ANSWER IS ABOUT, SAID OUT LOUD.
        #
        # The farmer asked for a medium-DURATION variety and got back "Suitable
        # varieties for medium land are: …" — the right names under the wrong
        # heading, which tells them something no document says. Two things
        # caused it: this note used to carry a hardcoded example heading
        # ("Suitable varieties for medium land are: ...") that the model copied
        # verbatim whatever the question was, and nothing anywhere stated the
        # requested facet as the criterion the answer must be framed on. The
        # example is now built from the question, and the criterion is spelled
        # out. The pool was already correct; only the framing was not.
        criterion = _requested_criterion(intent)
        criterion_note = ""
        if criterion:
            criterion_note = (
                f"\nPRIMARY ANSWER CRITERION: the farmer asked for varieties "
                f"that are {criterion.upper()}. Answer THAT. Frame the answer "
                f"around {criterion} — the varieties below are here because "
                f"their own rows say they are {criterion}. The CONTEXT also "
                "states other attributes for some of them, such as a land type, "
                "a grain size or a plant height: those may be mentioned as extra "
                "detail, but they are NOT what was asked and must NOT become the "
                f"heading or the subject of the answer. Never write 'suitable "
                f"varieties for medium land' — or for any other attribute — when "
                f"the question was about {criterion}.\n")

        ecosystem_note = ""
        if intent.land_type_missing:
            asked_for = ", ".join(f"{k.replace('_', ' ')}: {v}"
                                  for k, v in intent.attributes.items()) or "none"
            log.info("variety request: land type NOT given (other facets: %s)",
                     asked_for)
            ecosystem_note = (
                "\n\nTHE FARMER DID NOT SAY WHICH LAND TYPE THEY HAVE. Do NOT "
                "say a variety is suitable for their land, their field or their "
                "conditions — the CONTEXT cannot support that, because it "
                "recommends each variety FOR a stated ecosystem and the farmer "
                "has not said which is theirs. Instead, for every variety you "
                "name, state the ecosystem or condition it is recommended for, "
                "keeping the wording's meaning and every qualifier (for example "
                "'CR Dhan 910 is recommended for shallow favourable lowland and "
                "irrigated late conditions'). Say it as a fact about the "
                "variety, never as a report about where you read it. "
                "Then ask them, in one short question at the end, which land "
                "type they have - upland, medium land, irrigated lowland, "
                "rainfed lowland or shallow lowland - so the recommendation can "
                "be narrowed.\n")

        if supported and intent.wants_list:
            # THE ORDINARY CASE, IN EITHER WORDING. "Which variety is suitable
            # for medium land?" is a farmer asking what they could sow, not a
            # request to be told about one seed while the others go unmentioned.
            # A plain list of what the CONTEXT supports is the answer; there is
            # no lead recommendation and no "other options", because nothing in
            # the question asked anyone to choose.
            log.info("variety request (options): %d supported by the context: %s",
                     len(supported), ", ".join(supported))
            # Only here: the farmer asked for the options and more than one is
            # supported. The explicit-single branch below never sets this, so
            # "the best variety" is never forced to list them all.
            if len(supported) > 1:
                required_names = list(supported)
            variety_note = (
                "\n\nEVERY VARIETY THE CONTEXT SUPPORTS FOR WHAT WAS ASKED - "
                "give ALL of these, not a selection. The note in [square "
                "brackets] after each name is FOR YOU, not for the farmer: it "
                "is what that variety's own row states. Say it back in plain "
                "farmer English beside the name - 'Fit for upland' becomes "
                "'suitable for upland land' - keeping every qualifier and "
                "figure exactly as written ('moderately resistant' stays "
                "moderately resistant, 'up to 2 weeks' stays up to 2 weeks). "
                "NEVER copy the brackets, and never write 'the context', 'the "
                "source says' or any other words about where you got it:\n- "
                + listed
                + "\nPresent them as the supported options, plainly - for "
                + (f"example 'Suitable {criterion} varieties are: ...' - "
                   if criterion else "example 'Suitable varieties are: ...' - ")
                + "as a "
                "numbered list when there are three or more. Do NOT single one "
                "out as the recommendation and do NOT introduce the rest as "
                "'other options': the farmer did not ask anyone to choose. Do "
                "not add a variety that is not in this list, do not rank them, "
                "and do not call any of them the best unless the CONTEXT says "
                "so.\n"
                + criterion_note
                + ecosystem_note)
        elif supported:
            # THE FARMER ASKED FOR ONE. Only an explicit request gets here — the
            # best, the top one, only one — and even then the pick comes out of
            # the SAME pool and is defended from the CONTEXT, never asserted.
            #
            # RANKING IS THE DOCUMENTS' TO MAKE, NOT OURS. The pool is ordered
            # by how explicitly each candidate is recommended for what was
            # asked, which is a fact about the retrieved text; that is enough to
            # lead with one, and not enough to call it best. When the top two
            # are equally well evidenced, the honest answer says so rather than
            # picking by coin-toss and dressing it up as a recommendation.
            best, alternatives = supported[0], supported[1:]
            ranking_supported = bool(
                pool and pool[0]["recommended"]
                and (len(pool) == 1 or pool[0]["score"] > pool[1]["score"]))
            log.info("variety request (explicit single): best=%s, %d other(s), "
                     "ranking supported by the context: %s",
                     best, len(alternatives), ranking_supported)
            variety_note = (
                "\n\nTHE VARIETIES THE CONTEXT SUPPORTS FOR WHAT WAS ASKED, "
                "best-evidenced first. The note in [square brackets] after each "
                "name is FOR YOU, not for the farmer: it is what that variety's "
                "own row states. Say it back in plain farmer English, keeping "
                "every qualifier and figure exactly as written, and NEVER copy "
                "the brackets or mention where the information came from:\n- "
                + listed
                + "\nThe farmer asked for ONE option, so give ONE: "
                + best
                + ". Say plainly what is known about it — that evidence is why "
                "it was chosen."
                + ("" if ranking_supported else
                   " The CONTEXT does NOT rank these varieties against each "
                   "other, so say plainly that the documents do not identify a "
                   "single best one and that " + best + " is given because it "
                   "is the one the CONTEXT supports most directly"
                   + (", with " + ", ".join(alternatives[:4])
                      + " equally supported" if alternatives else "") + ".")
                + " Do not name a variety that is not in this list, and never "
                "call it 'the best', 'the highest yielding' or better than the "
                "others unless the CONTEXT itself ranks them.\n"
                + criterion_note
                + ecosystem_note)
        elif ecosystem_note:
            # Nothing in the CONTEXT supports a recommendation, and the farmer
            # never said which land they have. Asking is the whole answer.
            variety_note = ecosystem_note

        # NOTHING SUPPORTS THE FACET THAT WAS ASKED FOR.
        #
        # The pool is empty and the question named a specific attribute, so the
        # honest answer is that the documents do not identify one. Without this,
        # the model was left holding the raw CONTEXT — which for a
        # medium-DURATION question is full of medium-LAND rows — plus the rule
        # telling it to answer a which-variety question with names. It duly
        # listed the medium-land varieties and asserted they were medium
        # duration, a claim no row makes. The pool already knows better; this is
        # what tells the model.
        if not supported and intent.attributes:
            asked = ", ".join(f"{facet.replace('_', ' ')}: "
                              + (", ".join(value) if isinstance(value, list) else value)
                              for facet, value in intent.attributes.items())
            log.info("variety request: nothing in the context supports %s", asked)
            variety_note = (
                "\n\nNO VARIETY IN THE CONTEXT IS SUPPORTED FOR WHAT WAS ASKED "
                "(" + asked + "). Say plainly that the information available "
                "does not identify a variety for it, and stop. Do NOT name a "
                "variety from the CONTEXT and do NOT say that a variety has "
                "this property: the CONTEXT describes those varieties by "
                "OTHER attributes — a land type, a grain size, a plant height — "
                "and none of those is what was asked for. Naming them here "
                "would tell the farmer something no document says.\n")
    # Pest-topic guard: when the question is about pests/diseases/insects,
    # prevent the LLM from substituting unrelated fertilizer advice.
    _pest_re = re.compile(
        r"\b(pest|bug|insect|disease|blight|blast|borer|worm|hopper|"
        r"thrips|midge|rot|smut|infection|fungus|spray|control)\b", re.IGNORECASE)
    pest_note = ""
    if _pest_re.search(query_text_en):
        pest_note = (
            "\n\nPEST/DISEASE TOPIC GUARD:\n"
            "This question is about a pest, disease, or crop protection issue. "
            "Answer ONLY with pest/disease management information from the CONTEXT. "
            "Do NOT give fertilizer recommendations, nutrient doses, or nitrogen "
            "splits unless the CONTEXT explicitly links them to THIS pest/disease. "
            "If the CONTEXT does not contain information about the specific pest "
            "or disease asked about, say plainly that the available information "
            "does not cover it.\n")

    # Timeline disambiguation: inject a note when the query involves
    # transplanting timelines so the LLM knows which timeline to answer about.
    timeline_note = ""
    timeline_intent = entity_intent.classify_timeline(query_text_en)
    if timeline_intent == entity_intent.TIMELINE_AMBIGUOUS:
        timeline_note = (
            "\n\nTIMELINE AMBIGUITY DETECTED:\n"
            "This question mentions both transplanting and a day count but does "
            "not clearly specify whether it means seedling age (days after "
            "sowing), days after transplanting, or transplanting method criteria. "
            "Answer the most directly supported interpretation from the CONTEXT "
            "and explicitly state which timeline you are describing. If the "
            "CONTEXT supports multiple interpretations, briefly mention both.\n")
    elif timeline_intent == entity_intent.TIMELINE_SEEDLING_AGE:
        timeline_note = (
            "\n\nTIMELINE: SEEDLING AGE QUESTION.\n"
            "The farmer is asking about seedling age at transplanting (days "
            "after sowing). Do NOT confuse this with days after transplanting.\n")
    elif timeline_intent == entity_intent.TIMELINE_DAT:
        timeline_note = (
            "\n\nTIMELINE: DAYS AFTER TRANSPLANTING.\n"
            "The farmer is asking about an activity at a specific number of days "
            "after transplanting. Do NOT confuse this with seedling age.\n")

    # Fertiliser (products) vs nutrient (elements): answer in the basis asked,
    # never by converting figures the CONTEXT does not already give in that basis.
    basis_note = ""
    _basis = _dose_basis(query_text_en)
    if _basis == "product":
        basis_note = (
            "\n\nDOSE BASIS — FERTILISER (PRODUCTS):\n"
            "The farmer asked about fertiliser. Give the dose as the fertiliser "
            "PRODUCTS a farmer buys and applies — for example Urea, DAP, MOP "
            "(muriate of potash), SSP, Zinc sulphate — with the quantities the "
            "CONTEXT states for them. Do NOT convert the answer into elemental N, "
            "P2O5, K2O yourself, and do not add an N/P/K breakdown unless the "
            "CONTEXT gives it for this question. If the CONTEXT states the dose "
            "only as elements, give that and say the product amounts are not "
            "specified.\n")
    elif _basis == "nutrient":
        basis_note = (
            "\n\nDOSE BASIS — NUTRIENT (ELEMENTS):\n"
            "The farmer asked about nutrients. Give the answer as the nutrient "
            "ELEMENTS nitrogen (N), phosphorus (P2O5) and potassium (K2O), with "
            "the quantities the CONTEXT states for them. Do NOT convert fertiliser "
            "products (Urea, DAP, MOP) into N/P/K yourself. If the CONTEXT gives "
            "only product amounts and no elemental figures, give the product "
            "amounts and say the elemental N/P/K breakdown is not specified.\n")

    persona = RESEARCHER_PERSONA if (user_type or "").lower() == "researcher" else FARMER_PERSONA
    # Arithmetic for the farmer's land size, done in code. Empty for the common
    # case (no land size named, or exactly one acre, where the stated rate IS
    # the answer). `scale_allowed` must reach every _unverified_doses call below
    # or the computed figures are stripped as invented.
    scale_note, scale_allowed = area_scaled_doses(query_text_en, context)
    sys = (
        persona
        + UNIT_FIDELITY_RULES
        + "\n\nGROUNDING RULES (MANDATORY):\n"
        + "- Answer using ONLY the CONTEXT below. Do NOT use outside knowledge.\n"
        # NOT outside knowledge — a naming equivalence. The documents write the
        # potassium dose only as "MOP" (e.g. "MOP 15-20 kg/acre"); the word
        # "potash" appears nowhere in the corpus. So a correctly translated
        # potash question landed on the MOP rate and was still answered "no
        # information available", because the model treated a different name as
        # a different fertiliser. Stating the identity is what makes the rate
        # above usable; it does NOT license a dose that is not in the CONTEXT.
        + "- FERTILISER NAMES. MOP, muriate of potash and potash are the SAME "
        + "fertiliser. If the question asks about potash and the CONTEXT gives "
        + "an MOP rate, that rate IS the answer — give it, with its area and "
        + "stage, and do not say you have no information. Call it potash when "
        + "the farmer called it potash.\n"
        # THE ORDER THESE ARE APPLIED IN, stated first because the rules below
        # can pull against each other and the model has to know which governs.
        # An answer that recites the whole retrieved document is not a thorough
        # answer, it is a different answer: asked when and how much fertiliser
        # to apply, a farmer was given three stages, four products and seven
        # figures, and had to find their own question inside it.
        + "\nANSWER SCOPE (apply in this order):\n"
        + "  1. Answer the question the farmer actually asked.\n"
        + "  2. Use ONLY the CONTEXT for every fact in that answer.\n"
        + "  3. Match the SCOPE of the question. The CONTEXT will hold more "
        + "than the question asked for — that is normal, and it is not a reason "
        + "to say all of it.\n"
        + "  4. Add nothing further. A related dose, stage, product or "
        + "condition the question did not ask about does not belong in the "
        + "answer, however relevant it looks.\n"
        + "  5. Never invent what is missing. If the CONTEXT does not hold what "
        + "was asked, say so plainly.\n"
        + "Include more than the question asked ONLY when leaving it out would "
        + "make the answer wrong or misleading on its own terms — never merely "
        + "because the CONTEXT contains it.\n"
        + _scope_note(query_text_en)
        + scale_note
        # SHAPE, not length. Every rule above governs WHAT may be in the answer;
        # none of them governed how it opens, and that is where the padding was
        # coming in: a farmer who asked how much urea got a sentence of
        # throat-clearing, then their figure, then a paragraph of good practice.
        # The answer is read ALOUD, once, in a field — the first thing heard has
        # to be the thing that was asked for.
        + "\nCLARIFYING QUESTIONS:\n"
        + "- If the CONTEXT contains DIFFERENT recommendations for DIFFERENT "
        + "land types (irrigated, rainfed, upland, lowland, medium land) and the "
        + "farmer's question does NOT specify which land type they have, ASK them: "
        + "'Which type of land do you have — irrigated, rainfed, or upland?' "
        + "before giving a recommendation. Do NOT guess the land type.\n"
        + "- Similarly, if the CONTEXT has different answers for different rice "
        + "duration classes (short, medium, long) and the farmer did not specify, "
        + "ask which duration they grow.\n"
        + "- Keep the clarifying question SHORT (one sentence). Do not give a "
        + "partial answer alongside it — just ask.\n"
        + "\nANSWER SHAPE:\n"
        + "  - Open with the answer itself. The variety name, the figure, the "
        + "timing — whatever was asked for goes in the FIRST sentence.\n"
        + "  - Do not restate or rephrase the question, and do not describe what "
        + "you are about to answer. No 'regarding your question', no 'here is "
        + "the information about', no 'as per the recommendation'.\n"
        + "  - No closing line. Do not end with general advice, encouragement, "
        + "a caution nobody asked for, or an offer of further help.\n"
        + "  - Say each fact once. Do not repeat a figure in words after giving "
        + "it in numerals, and do not summarise at the end what you just said.\n"
        + "  - Use as many words as the facts need and no more. Every sentence "
        + "must carry a fact the farmer asked for; a sentence that only frames, "
        + "introduces or softens another sentence must be deleted.\n"
        # The answer is read aloud to a farmer in a field. Every word below is
        # this module's own vocabulary, and hearing it tells them nothing.
        + "- NEVER NAME THE MACHINERY. The words 'context', 'the CONTEXT says', "
        + "'source', 'document', 'retrieved', 'chunk', 'candidate' and 'prompt' "
        + "must not appear in your answer, and neither must square brackets or "
        + "any note addressed to you. State what is true of the crop or the "
        + "variety as a plain fact: write 'CR Dhan 100 is suitable for upland "
        + "land', never 'the CONTEXT says CR Dhan 100 is fit for upland'.\n"
        # SCOPE. Everything below this comment answers one failure: an answer
        # that is true in every sentence and still wrong as a whole, because it
        # carried doses, stages and fertilisers nobody asked about. A farmer
        # reading a list of seven applications when they asked about one cannot
        # tell which line is theirs.
        + "- ANSWER ONLY WHAT WAS ASKED, then stop. Do not add a dose, a stage, "
        + "a fertiliser or a recommendation the question did not ask for, however "
        + "relevant it looks in the CONTEXT.\n"
        # Kept SEPARATE from the rule above because it is a different habit: the
        # answer is correct and complete, and then a sentence about why it helps
        # is appended. A farmer asked how much urea, not what nitrogen does.
        + "- NO EXPLANATIONS, BENEFITS OR EXTRA ADVICE unless the question asks "
        + "for them, or the answer cannot be understood without them. Do not say "
        + "what the input does for the crop, why the timing matters, or what good "
        + "practice would add. If the farmer DOES ask why, or what something is "
        + "for, then that IS the question and the CONTEXT's own explanation is "
        + "the answer.\n"
        + "- IF THE QUESTION NAMES A CROP STAGE, give that stage's recommendation "
        + "and no other. Basal, tillering, panicle initiation and grain filling "
        + "are different applications; listing the others answers a question that "
        + "was not asked.\n"
        # THE WHOLE SCHEDULE IS ALSO THE ANSWER TO A QUESTION THAT NAMES NO
        # STAGE. "What fertiliser should I apply in DSR?" is not a question about
        # basal — it is the timetable, asked in the ordinary way a farmer asks
        # it, and answering with the first application alone sends them to the
        # field with one seventh of what the document says. Naming a stage
        # narrows; naming none does not.
        # NARROWED, NOT REMOVED. This rule was written for a real harm — a
        # farmer asking what to apply in DSR was told the basal dose alone, one
        # seventh of the timetable — but as written it also fired for "when and
        # how much fertiliser for rice?", and the whole document came back. It
        # now applies to a question that asks WHAT to apply without naming a
        # quantity; the ANSWER SCOPE block above governs the rest.
        + "- IF THE QUESTION ASKS WHAT FERTILISER A CROP OR SYSTEM NEEDS, "
        + "naming neither a stage nor a quantity (for example 'what fertiliser "
        + "should I apply in DSR?'), give the COMPLETE schedule the CONTEXT "
        + "supports: every stage, in the order the document states them. This "
        + "does NOT apply when the farmer asked how much of one input, or about "
        + "one stage, or anything narrower — see ANSWER SCOPE above, which "
        + "governs.\n"
        + "- IF THE QUESTION ASKS FOR THE WHOLE SCHEDULE, or for all the "
        + "fertiliser applications, do the same — every stage the CONTEXT "
        + "supports, in document order, leaving none out.\n"
        # PER-AREA QUESTIONS. Three ways to manufacture a per-acre number that
        # no document states, all of them arithmetic the model should not be
        # doing: divide a hectare rate, add the stages together, or read out a
        # schedule and let the farmer add it up.
        + "- IF THE QUESTION ASKS HOW MUCH PER ACRE (or per hectare), answer "
        + "with a figure for THAT area. If the CONTEXT only has a rate in a "
        + "different area unit (e.g. per hectare when the farmer asked per acre), "
        + "convert using 1 hectare = 2.471 acres and show both figures. "
        + "Do NOT add the stages together into a season total UNLESS the farmer "
        + "explicitly asks for the TOTAL amount.\n"
        + "- A QUESTION ASKING HOW MUCH OF ONE INPUT IS NOT A REQUEST FOR THE "
        + "SCHEDULE. Give that input's figure and stop. The complete-schedule "
        + "rule above applies to a question that asks WHAT to apply without "
        + "naming a quantity, not to one that asks HOW MUCH.\n"
        + "- If the CONTEXT holds no figure for the area that was asked about, "
        + "say plainly that you do not have it — UNLESS a COMPUTED AMOUNTS block "
        + "below supplies it, in which case give that figure. Do not otherwise "
        + "fill the gap from a related recommendation, a different stage, or "
        + "another area basis.\n"
        + "- Every figure must come from the CONTEXT, with the meaning the "
        + "CONTEXT gives it. Never invent a value, never infer one from a related "
        + "figure, and never substitute one figure for another — a yield is not a "
        + "seed rate, a per-hectare rate is not a per-acre rate, and a rate for "
        + "one stage is not the rate for the next.\n"
        + "\nPRODUCTION-QUANTITY vs AREA-RATE MISMATCH:\n"
        + "If the question mentions PRODUCING a quantity of rice (e.g. "
        + "'1900 tons', '500 quintals') BUT ALSO provides a cultivated "
        + "area (e.g. '2 hectares', '5 acres'), IGNORE the production "
        + "quantity and answer using the area — give the per-area "
        + "recommendation for that land size.\n"
        + "If the question mentions a production quantity but does NOT "
        + "provide a cultivated area, do NOT calculate or invent a total. "
        + "Explain that recommendations are per unit area and ask the "
        + "farmer in ONE question for all missing details — cultivated "
        + "area, land type (irrigated / rainfed / upland), and variety — "
        + "so the farmer can reply once. Always end with a question mark.\n"
        + "Similarly, do NOT convert an elemental nutrient "
        + "rate (kg N/ha) into a commercial fertilizer quantity (kg urea) "
        + "unless the CONTEXT itself makes that conversion.\n"
        + "\nSTAGE-WISE ANSWER STRUCTURE:\n"
        + "When the farmer asks which fertilizer to apply and when, organize "
        + "the answer by crop growth stage in the order the CONTEXT states "
        + "them (basal → active tillering → panicle initiation → any other). "
        + "For each stage, state: the stage name, each fertilizer, its "
        + "quantity, and the area basis. Do not omit a stage or fertilizer "
        + "that is present in the CONTEXT. Do not combine different stages "
        + "into one undifferentiated list.\n"
        + "\nDO NOT MIX DIFFERENT RECOMMENDATIONS:\n"
        + "The CONTEXT may contain recommendations for different varieties, "
        + "land types, durations, or farming systems. Never combine figures "
        + "from unrelated sources to create a complete-looking answer.\n"
        + "- Do not use a long-duration variety's rates for an early-duration "
        + "variety when a separate recommendation exists.\n"
        + "- Do not mix a general recommendation with a land-type-specific "
        + "one without source support.\n"
        + "- Do not combine nutrient quantities (N, P, K) with commercial "
        + "fertilizer quantities (urea, DAP, MOP) as though they are the "
        + "same measurement.\n"
        + "- If two relevant sources conflict, prefer the one that most "
        + "closely matches the farmer's question. If the conflict cannot be "
        + "resolved, explain the uncertainty.\n"
        + "\nTIMELINE PRECISION (MANDATORY):\n"
        + "Agricultural timelines are NOT interchangeable. NEVER confuse:\n"
        + "- SEEDLING AGE: how old seedlings are at transplanting (days after sowing).\n"
        + "- DAYS AFTER TRANSPLANTING (DAT): time elapsed after seedlings are planted in the main field.\n"
        + "- DAYS AFTER SOWING (DAS): time elapsed since seeds were sown in the nursery.\n"
        + "If the CONTEXT says seedlings are '10-12 days old' at transplanting, that is seedling AGE, "
        + "NOT '10-12 days after transplanting'. Do not rewrite one as the other.\n"
        + "If the CONTEXT gives a recommendation for a specific method (SRI, mat nursery, conventional), "
        + "name the method. Do not present method-specific criteria as universal.\n"
        + "If the question is ambiguous about which timeline it means, answer the most supported "
        + "interpretation and state which timeline you are describing.\n"
        + "\nSUBSTANCE-FREE DOSES ARE FORBIDDEN:\n"
        + "Every application figure MUST name the substance it refers to "
        + "(e.g. '60 kg N/ha', '35 kg DAP/acre', '20 kg urea'). NEVER write "
        + "a bare dose like '100 kg per hectare' without saying OF WHAT. If "
        + "the CONTEXT says '60-120 kg N/ha', write '60-120 kg N/ha' — do not "
        + "drop the 'N' and do not change the number.\n"
        # FORMAT. Farmers hear the answer read aloud once. Bullet points and
        # one-fact-per-line schedules become a stuttering list of fragments, and
        # the same dose gets repeated (a stage line plus a total line). It has to
        # be spoken prose.
        + "\nFORMAT — PLAIN PARAGRAPH (MANDATORY):\n"
        + "  - Write the whole answer as flowing sentences in a short paragraph. "
        + "Do NOT use bullet points, numbered lists, a dash or hyphen starting a "
        + "line, or one item per line. A stage-wise fertiliser schedule is still "
        + "sentences, e.g. 'At the basal stage apply 35 kg DAP, 5 kg urea and 16 "
        + "kg MOP per acre; at active tillering apply 30-35 kg urea; at panicle "
        + "initiation apply 30-35 kg urea and 15-20 kg MOP.'\n"
        + "  - Say each amount once. Do not repeat a fertiliser, and do not add a "
        + "season TOTAL unless the farmer explicitly asked for the total.\n"
        # PERCENTAGES. A farmer measures fertiliser by weight in a bag, not as a
        # share of nitrogen. "25% of N at tillering" is unactionable in a field.
        + "\nAMOUNTS IN KILOGRAMS, NOT PERCENTAGES (MANDATORY):\n"
        + "  - Give fertiliser amounts in kg (and bags only where the CONTEXT "
        + "gives bags). Do NOT state a dose as a percentage of nitrogen or an N "
        + "split in percent. If the CONTEXT expresses the split only as "
        + "percentages (e.g. 25% / 50% / 25% of N by stage), give the kg amount "
        + "of urea per stage that the CONTEXT supports instead, and leave the "
        + "percentages out of the answer.\n"
        # BASAL. "Just transplanted, what now" is the basal dose, not a topdress.
        # A nearby 'N days after transplanting' passage pulled the answer to urea
        # alone and dropped DAP.
        + "\nBASAL AT TRANSPLANTING:\n"
        + "  - If the farmer says the crop has just been transplanted or sown and "
        + "asks which fertiliser to apply NOW, with no day-count or later stage "
        + "named, that is the BASAL application: give the basal dose the CONTEXT "
        + "supports (commonly DAP, urea, MOP and zinc sulphate together). Do NOT "
        + "drop DAP, and do not answer with a later top-dressing.\n"
        # SEED ADVICE. "Tell me about high-yielding seeds" is a which-variety
        # question in plain clothes; a paragraph on 'use certified seed' names
        # nothing the farmer can buy.
        + "\nSEED / VARIETY ADVICE:\n"
        + "  - If the farmer asks about seed or varieties in general terms — high "
        + "yielding, early or short duration, needing less water, suited to their "
        + "area — and the CONTEXT names specific varieties that meet it, NAME "
        + "several of those varieties rather than giving only general seed advice "
        + "or a single name. List the ones the CONTEXT supports.\n"
        # A WHICH-VARIETY QUESTION WANTS NAMES. The failure this prevents is an
        # answer that is true and useless: the sheets name four varieties rated
        # resistant to blast, and the farmer is told "some varieties are
        # moderately resistant" — nothing they can take to a seed dealer.
        + "- IF THE QUESTION ASKS WHICH VARIETY OR VARIETIES — including questions "
        + "asking for 'some names', 'which varieties', or varieties that are "
        + "resistant/tolerant/suitable for a condition — ANSWER WITH THE VARIETY "
        + "NAMES that the CONTEXT explicitly supports. Never replace named varieties "
        + "with a general statement such as 'some varieties are drought tolerant'.\n"

        + "- FOR A MULTIPLE-VARIETY QUESTION, GIVE EVERY VARIETY THE CONTEXT "
        + "EXPLICITLY STATES MEETS THE ASKED CONDITION, and ONLY those. A variety "
        + "mentioned nearby is not enough; it must have suitability, tolerance, "
        + "resistance or other evidence explicitly associated with the condition. "
        + "Do not require the CONTEXT to identify one single best variety. If the "
        + "CONTEXT supports several varieties, list the supported varieties. If it "
        + "supports only one, give that one. If it supports none, say plainly that "
        + "the information available does not specify any variety.\n"
        # Both directions, and sideways. Downgrading looks cautious and is not:
        # a farmer told their variety is only "tolerant" may replace seed that
        # was fine. The rating is the document's word, and it travels whole.
        # THE OTHER FAILURE, and the one the rules above can cause. Told often
        # enough not to go beyond the evidence, a model starts refusing what the
        # evidence plainly says: the sheet reads "fit for upland", the farmer
        # asked about high land, the words differ, and a variety that answers
        # them perfectly is withheld. A refusal is only honest when there is
        # nothing there.
        + "- VARIETY MATCHING MUST USE LAND-TYPE MEANING, NOT EXACT WORD "
        + "MATCHING. A variety is supported when the CONTEXT explicitly says it "
        + "is suitable, recommended, fit, adapted or appropriate for the land "
        + "type, ecology, season or condition the farmer asked about. Treat "
        + "equivalent land-type terms as matching where the agricultural meaning "
        + "is the same: 'upland' matches 'high land', 'lowland' matches 'low "
        + "land', 'rainfed lowland' matches 'rainfed low land'. If the CONTEXT "
        + "says 'Fit for upland', that IS sufficient evidence to answer a "
        + "question about which variety suits high land. Do NOT refuse merely "
        + "because the farmer's wording and the source's wording differ.\n"
        # "BEST" IS A QUESTION WORD, NOT AN EVIDENCE WORD. Farmers ask for the
        # best seed the way anyone asks for a recommendation; the sheets rate
        # varieties as fit or suitable and rank nothing. Requiring the word
        # produced refusals; echoing it produced a ranking no document states.
        + "- A FARMER'S 'BEST', 'GOOD' OR 'BETTER' IS A REQUEST FOR A "
        + "RECOMMENDATION, not a demand for those words. The CONTEXT does not "
        + "have to contain them: if it says a variety is suitable, recommended, "
        + "fit, adapted or appropriate for what was asked, that variety is the "
        + "answer. Do not refuse because nothing is called 'best'.\n"
        + "- BUT DO NOT CLAIM A RANKING THE CONTEXT DOES NOT MAKE. Never say a "
        + "variety is 'the best', 'the highest yielding', or better than others "
        + "unless the CONTEXT ranks it that way. Say what the CONTEXT says — "
        + "'CR Dhan 100 is suitable for upland' — not 'CR Dhan 100 is the best "
        + "variety for high land'.\n"
        + "- DO NOT INVENT A VARIETY. Having matched the farmer's land type or "
        + "condition to the CONTEXT by meaning, name ONLY the varieties for "
        + "which the CONTEXT provides that suitability evidence. Matching by "
        + "meaning widens which WORDS count as the same condition; it never "
        + "widens which varieties count as answers.\n"
        # THE SINGULAR IS NOT A REQUEST FOR ONE. A farmer says "which variety
        # is good for my land" the way anyone says it, and answering with one
        # name while the sheets support three hides two seeds they could have
        # bought. The old rule here said one variety "may be returned" for a
        # singular question, and that permission was the whole bug.
        + "- 'WHICH VARIETY' AND 'WHICH VARIETIES' ARE THE SAME REQUEST. Both "
        + "ask which varieties are suitable, and both are answered with ALL the "
        + "varieties in the CONTEXT that explicitly satisfy the requested "
        + "condition. The same goes for 'seed' and 'seeds'. Never return fewer "
        + "varieties because the farmer used a singular word, and never "
        + "arbitrarily select one of several supported varieties.\n"
        + "- GIVE ONE VARIETY ONLY WHEN THE FARMER ASKED FOR ONE in so many "
        + "words — 'the best', 'the top one', 'only one', 'just one variety'. "
        + "Then name one and say what the CONTEXT says about it. If the CONTEXT "
        + "does not rank the varieties against each other, say so plainly "
        + "instead of inventing a winner.\n"
        + "- Never include a variety unless the CONTEXT explicitly supports the "
        + "requested condition.\n"
        + "- KEEP A RESISTANCE OR TOLERANCE LEVEL EXACTLY AS THE CONTEXT WRITES "
        + "IT. Never upgrade it, never downgrade it, never reword it: "
        + "'resistant', 'moderately resistant' and 'tolerant' are three "
        + "different ratings and none of them stands for another. Never invent a "
        + "variety name or a rating. If the CONTEXT names no variety that fits "
        + "what was asked, say plainly that the information available does not "
        + "specify one, and stop.\n"
        # THE QUALIFIER IS THE RATING. "Moderately resistant" promoted to
        # "resistant" is not a tidier sentence, it is a different agronomic
        # claim: the farmer buys that seed for a blast-prone field on the
        # strength of a word the sheet never used.
        + "- A HEDGE IS PART OF THE RATING, NOT A STYLE CHOICE. If the CONTEXT "
        + "says 'moderate tolerance', 'moderately resistant', 'mildly "
        + "tolerant' or 'partially resistant', the answer must carry that same "
        + "qualifier. NEVER write 'resistant' or 'tolerant' plainly where the "
        + "CONTEXT qualifies it, and never write 'immune' at all.\n"
        # The blanket claim. No varietal sheet says this, and the pest it does
        # not cover is the one that takes the crop.
        + "- NEVER GENERALISE A LIST INTO A BLANKET. If the CONTEXT names the "
        + "specific pests or diseases a variety resists, name THOSE. Do not "
        + "write 'resistant to major pests and diseases', 'tolerant to most "
        + "diseases' or any similar sweep — it claims cover the CONTEXT does "
        + "not give.\n"
        + "- DO NOT INVENT A SUITABILITY CONDITION. A variety is for the "
        + "ecosystem, season, land type and stress the CONTEXT states for it, "
        + "and for nothing else. Never widen 'recommended for shallow "
        + "favourable lowland' into 'suitable for all lowlands' or into "
        + "'suitable for your field', and never add a condition the CONTEXT "
        + "does not state for that variety.\n"
        # A farmer who reports a problem and then asks "can I do X?" is asking
        # whether X is safe GIVEN the problem. Answering the literal question
        # produced real harm: "my crop has lodged, can I apply fertilizer?" was
        # answered "Yes, apply Urea and MOP" while the retrieved context said
        # "Stop nitrogen application immediately ... to prevent lodging". Urea is
        # nitrogen, so the answer told the farmer to do the one thing that makes
        # lodging worse.
        + "- IF THE FARMER REPORTS A PROBLEM (crop lodged/fallen, yellowing, drying, "
        + "rotting, pest or disease damage, stunted growth), the problem GOVERNS the "
        + "answer. Before saying yes to any action they propose, check the CONTEXT for "
        + "what causes or worsens that problem. If the CONTEXT says an input causes or "
        + "worsens it, you MUST say to stop or avoid that input — never reply 'yes' to "
        + "applying it. Name the specific input to avoid (e.g. nitrogen/urea), then give "
        + "what the CONTEXT recommends instead.\n"
        + "- Do not answer a yes/no question with 'yes' merely because the CONTEXT "
        + "describes the action in general. It must be safe for the situation the "
        + "farmer described.\n"
        + "- Name a fertilizer, chemical, or input ONLY if it is explicitly in the CONTEXT. "
        + "Never invent product names (e.g. do not say SOP/Sulphate of Potash unless it is "
        + "in the context).\n"
        + "- Never claim a fertilizer contains a nutrient unless the CONTEXT says so "
        + "(e.g. do NOT say MOP contains sulphur).\n"
        + "- Prefer a variety-specific answer when the CONTEXT has one. If the CONTEXT has no "
        + "figure specific to the variety, but it DOES contain a general recommendation for "
        + "rice/paddy (a standard NPK dose, a stage-wise fertiliser schedule, per-acre or per-"
        + "hectare amounts, etc.), GIVE that recommendation and clearly say it is the general "
        + "recommendation for rice, not specific to the variety. Only advise a soil test / local "
        + "agriculture officer when the CONTEXT contains NEITHER a variety-specific NOR a general "
        + "recommendation for what was asked. Never invent or estimate numbers — use only figures "
        + "that appear in the CONTEXT.\n"
        + "- The farmer's everyday words map to the documents' terms: 'manure'/'manuring' means "
        + "fertiliser / nutrient application; 'medicine'/'spray' means pesticide or fungicide. "
        + "Answer from the CONTEXT using this mapping — do NOT refuse merely because the farmer's "
        + "word differs from the document's word, as long as the CONTEXT contains the information.\n"
        + "- The farmer's VARIETY NAME may be misspelled, transliterated, or translated literally. "
        + "Use your own understanding of the language to match it to a variety NAMED IN THE CONTEXT: "
        + "e.g. the farmer's 'black champa' is the variety 'Kalachampa' (kala = black); a colour or "
        + "common word in the farmer's wording may be the literal translation of part of a variety "
        + "name. If a variety in the CONTEXT clearly refers to the same one, answer from that "
        + "context and do NOT refuse over a spelling or translation difference. Only say the "
        + "variety isn't covered if NO variety in the CONTEXT plausibly matches.\n"
        + "\nSYNONYM AND SPELLING MATCHING:\n"
        + "Odia words may be spelled differently in different contexts. Treat these "
        + "as the SAME thing: ଧଣିଚା / ଧନିଚା / dhaincha / dhanicha (Sesbania), "
        + "ଆଜୋଲା / ଅଜୋଲା / Azolla, ୟୁରିଆ / urea, ପୋଟାସ / potash / MOP. "
        + "Do NOT refuse because the farmer's spelling differs from the document's.\n"
        + "\nORGANIC + CHEMICAL INPUTS:\n"
        + "When the farmer mentions using organic inputs (FYM, compost, vermicompost, "
        + "green manure, Azolla, dhaincha/dhanicha) AND asks about chemical fertilizer "
        + "doses, do NOT refuse or say 'could not verify'. Instead:\n"
        + "  1. Give the standard chemical fertilizer recommendation from the CONTEXT.\n"
        + "  2. If the CONTEXT mentions how organic inputs affect the chemical dose "
        + "(e.g. reduce nitrogen by X%), include that guidance.\n"
        + "  3. If the CONTEXT does not specify any adjustment, give the standard "
        + "dose and note that the farmer applied organic input — do not refuse.\n"
        # CAUSE AND EFFECT. A symptom has more than one explanation, and the
        # farmer acts on whichever one they are given: told their yellow leaves
        # are nitrogen deficiency, they buy urea — and if it was actually zinc,
        # or waterlogging, or a disease, the urea is money spent and a week lost
        # while the crop gets worse.
        + "\n\nCAUSE AND EFFECT:\n"
        + "- PRESERVE CAUSAL UNCERTAINTY. Where the CONTEXT describes a cause as "
        + "possible, suspected or associated with a symptom — 'could be due to', "
        + "'may indicate', 'may be caused by' — keep that SAME level of "
        + "uncertainty in the answer. Do not present the cause as confirmed, and "
        + "do not recommend a corrective fertiliser as though it were.\n"
        + "- IF THE CONTEXT GIVES SEVERAL POSSIBLE CAUSES, mention the relevant "
        + "alternatives rather than choosing one of them without evidence. Two "
        + "or three plain alternatives the farmer can look for in the field are "
        + "worth more than one confident guess.\n"
        # The two shortcuts a model takes to reach a deficiency it was never
        # told about: reasoning backwards from what the farmer applied, and
        # reading a symptom as its most familiar cause.
        + "- A SYMPTOM IS NOT A DIAGNOSIS, AND NEITHER IS WHAT THE FARMER "
        + "ALREADY APPLIED. That they have used a fertiliser, or that a symptom "
        + "is present, does not by itself establish a nutrient deficiency. "
        + "Recommend a corrective fertiliser ONLY where the CONTEXT explicitly "
        + "supports THAT fertiliser for the stated symptom, crop stage, "
        + "deficiency or condition. Where it does not, say what the documents do "
        + "say about the symptom and stop.\n"
        + basis_note
        + variety_note
        + pest_note
        + timeline_note
        + history_note
        + followup_constraint
        + (("\n\nTHE CONTEXT BELOW COMES FROM MORE THAN ONE COLLECTION ("
            + ", ".join(catalogue_names) + "), because the question touches "
            "both. Use ONLY what answers the question. You are NOT required to "
            "say something from each collection, and a collection that turns "
            "out to be irrelevant should simply go unused.\n")
           if len(catalogue_names) > 1 else "")
        + "\nCONTEXT:\n"
        + context
    )
    answer = _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": query_text_en},
    ])

    # Numeric provenance check. Every dose the farmer acts on must be traceable to
    # the documents; retry ONCE naming the offending figures, then give up on the
    # numbers rather than state an invented dose.
    bad = _unverified_doses(answer, context, scale_allowed)
    if bad:
        log.warning("answer contained doses not in the context: %s", bad)
        # Build explicit corrections for computed-amount substance swaps
        _swap_hints = []
        if scale_allowed:
            _allowed_by_num: dict[str, set] = {}
            for n, s in scale_allowed:
                _allowed_by_num.setdefault(n, set()).add(s)
            for entry in bad:
                if " of " not in entry:
                    continue
                num, rest = entry.split(" of ", 1)
                num = num.strip()
                wrong_sub = rest.split("(")[0].strip()
                correct = _allowed_by_num.get(num)
                if correct and wrong_sub not in correct:
                    _swap_hints.append(
                        f"You wrote {num} kg {wrong_sub}, but the COMPUTED "
                        f"AMOUNTS say {num} kg is "
                        f"{' or '.join(sorted(correct))}, not {wrong_sub}.")
        swap_note = (" ".join(_swap_hints) + " ") if _swap_hints else ""
        answer = _chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query_text_en},
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                "These amounts are NOT in the CONTEXT: " + ", ".join(bad)
                + ". You invented or calculated them. " + swap_note
                + "Rewrite the answer using ONLY "
                "amounts written in the CONTEXT, copied exactly with their unit, "
                "substance name, and stage. Do not convert a nutrient (nitrogen) "
                "into a product (urea) and do not add up doses. If a COMPUTED "
                "AMOUNTS block was given, its figures ARE allowed — copy them "
                "exactly; otherwise do not work out a total for the land size. "
                "The CONTEXT likely still has relevant figures — look for them "
                "and cite them with the EXACT number and substance (e.g. "
                "'60-120 kg N/ha', '15 kg phosphorus/acre'). Only say the "
                "information is not available if the CONTEXT truly has NO "
                "relevant fertilizer figures at all."},
        ])
        still = _unverified_doses(answer, context, scale_allowed)
        if still:
            log.error("dose figures still unverified after retry: %s", still)
            ctx_pairs = _dose_pairs(context) | set(scale_allowed or ())
            # Computed-amount substance swap: the LLM wrote the right
            # number from COMPUTED AMOUNTS but paired it with the wrong
            # fertilizer. Fix by replacing the wrong substance with the
            # correct one from the computed amounts.
            if scale_allowed:
                _allowed_by_num: dict[str, set] = {}
                for n, s in scale_allowed:
                    _allowed_by_num.setdefault(n, set()).add(s)
                _swaps: list[tuple[str, str, str]] = []
                for entry in still:
                    if " of " not in entry or "wrong substance" not in entry:
                        continue
                    num = entry.split(" of ")[0].strip()
                    wrong = entry.split(" of ")[1].split("(")[0].strip()
                    correct = _allowed_by_num.get(num)
                    if correct and wrong not in correct and len(correct) == 1:
                        _swaps.append((num, wrong, next(iter(correct))))
                if _swaps and len(_swaps) == len(
                        [s for s in still if "wrong substance" in s]):
                    for num, wrong, right in _swaps:
                        pattern = re.compile(
                            r"\b" + re.escape(num) + r"\s*kg\s+(?:of\s+)?"
                            + re.escape(wrong) + r"\b", re.IGNORECASE)
                        answer = pattern.sub(f"{num} kg {right}", answer)
                    log.info("fixed computed-amount substance swap(s): %s",
                             [(w, r) for _, w, r in _swaps])
                    still = _unverified_doses(answer, context, scale_allowed)
            # When every remaining error is a nitrogen→urea conversion
            # (number exists in context as nitrogen, LLM wrote urea),
            # fix by replacing "urea" with "N" in those sentences rather
            # than stripping them — the number is correct, only the
            # substance name is wrong.
            if still:
                n_to_urea = all(
                    "of urea (wrong substance)" in s
                    and (s.split(" of ")[0].strip(), "nitrogen") in ctx_pairs
                    for s in still if " of " in s
                )
                if n_to_urea and all(" of " in s for s in still):
                    answer = re.sub(
                        r"\b(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s*kg\s+"
                        r"(?:of\s+)?urea\b",
                        r"\1 kg N", answer, flags=re.IGNORECASE)
                    log.info("fixed nitrogen→urea conversion in answer")
                else:
                    # Normalise en/em dashes so a ranged dose the guard flagged
                    # ("40-50 of nitrogen") still matches the sentence that wrote
                    # it with an en-dash ("40–50 kg/ha"); otherwise the unverified
                    # range slips through and a "how much" answer can end up as a
                    # lone timing sentence.
                    _norm_dash = lambda t: t.replace("–", "-").replace("—", "-")
                    bad_nums = {_norm_dash(s.split(" of ")[0].strip()) for s in still
                                if " of " in s}
                    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip())
                                 if s.strip()]
                    clean = [s for s in sentences
                             if not any(n in _norm_dash(s) for n in bad_nums)]
                    if clean and len(clean) > len(sentences) // 2:
                        answer = " ".join(clean).strip()
                        log.info("kept %d/%d sentences after stripping "
                                 "unverified doses", len(clean), len(sentences))
                    else:
                        det = (_deterministic_schedule(query_text_en, context)
                               or _deterministic_stage_dose(query_text_en)
                               or _deterministic_product_dose(query_text_en))
                        if det:
                            answer = det
                            log.info("used deterministic schedule/stage dose as fallback")
                        else:
                            answer = (
                                "I could not verify a reliable fertilizer amount "
                                "for this question from the available agricultural "
                                "information. Please confirm with your local "
                                "agriculture extension officer or soil-test "
                                "guidance."
                            )
    # Claim-strength check. Same shape as the dose check above, for the other
    # thing an answer can get wrong while every sentence stays plausible: a
    # rating promoted past what the sheet says ("moderately resistant" ->
    # "resistant"), or a named list swept into a blanket ("resistant to major
    # pests and diseases"). Detect, retry once naming the offending claims, and
    # keep the retry only if it actually fixed them without inventing a dose.
    overstated = entity_intent.overstated_claims(answer, context)
    if overstated:
        retried = _chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query_text_en},
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                "These claims are stronger than the CONTEXT: "
                + "; ".join(overstated)
                + ". Rewrite the answer keeping every rating exactly as the "
                "CONTEXT writes it — carry 'moderate', 'moderately' or any "
                "other qualifier through, name the specific pests or diseases "
                "the CONTEXT names instead of a blanket statement, and do not "
                "add a suitability condition the CONTEXT does not state. Change "
                "nothing else."},
        ])
        if (retried.strip()
                and not entity_intent.overstated_claims(retried, context)
                and not _unverified_doses(retried, context, scale_allowed)):
            answer = retried
        else:
            log.error("claim-strength retry rejected; the answer still overstates "
                      "the source: %s", overstated)

    # Completeness check. The pool was handed over in full and the instruction
    # said to give ALL of it; the model still returns five names out of six. The
    # farmer cannot tell a seed was left out, and the missing one may be the
    # only variety their dealer has. Same shape as the two checks above: detect,
    # retry once naming exactly what was dropped, and keep the retry only if it
    # is complete AND has not invented a variety along the way.
    #
    # COMPLETE MEANS "ANSWERED THE QUESTION", NOT "SAID EVERYTHING RETRIEVED".
    # `required_names` is set in ONE place — the branch where the farmer asked
    # which varieties are suitable and more than one qualified — so this can
    # only ever require the OPTIONS THE QUESTION ASKED FOR. It is empty for a
    # question about a dose, a stage or a single variety, which is why the
    # ANSWER SCOPE rules above cannot be undone here: asked how much DAP, this
    # check requires nothing, and a context holding six other fertilisers stays
    # out of the answer.
    missing = _missing_candidates(answer, required_names)
    if missing:
        log.warning("answer omitted %d of %d supported varieties: %s",
                    len(missing), len(required_names), ", ".join(missing))
        retried = _chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query_text_en},
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                "Your answer left out " + ", ".join(missing)
                + ". Every one of these varieties is supported by the CONTEXT "
                "for what the farmer asked, and the farmer needs to know about "
                "all of them — the one you dropped may be the only seed their "
                "dealer has. Rewrite the answer naming ALL of these, and only "
                "these: " + ", ".join(required_names)
                + ". Do not add any other variety, do not drop any of them, and "
                "keep everything else about the answer as it is."},
        ])
        still_missing = _missing_candidates(retried, required_names)
        invented = _unsupported_varieties(retried, required_names, query_text_en)
        if retried.strip() and not still_missing and not invented:
            answer = retried
        else:
            # ONE RETRY, NEVER A LOOP. The original answer is kept: it is
            # incomplete, but every variety in it is supported, and a retry that
            # is still missing names — or has invented one — is not an
            # improvement on that.
            log.error("completeness retry rejected; keeping the original answer "
                      "(still missing: %s; invented: %s)",
                      ", ".join(still_missing) or "none",
                      ", ".join(invented) or "none")

    # LAST RESORT for a figure that never says what it is OF. Unlike a wrong
    # number, there is nothing to salvage here: "apply 2400 kg per acre" carries
    # no instruction a farmer could follow, so dropping the sentence loses
    # nothing and leaves a shorter answer that is at least all true.
    if _substance_free_doses(answer):
        _before_strip = answer
        answer = _drop_substance_free_dose_sentences(answer, context)
        # If stripping an elemental / substance-free dose left a fragment with no
        # usable fertiliser amount for a per-area or schedule question, fall back
        # to the clean deterministic schedule rather than a dangling sentence like
        # "This means you put all these fertilizers in the field." (Q-3011)
        if _before_strip != answer and not _has_fertiliser_dose(answer):
            det = (_deterministic_schedule(query_text_en, context)
                   or _deterministic_stage_dose(query_text_en)
                   or _deterministic_product_dose(query_text_en))
            if det:
                answer = det
                log.info("input-less strip gutted the answer; "
                         "used deterministic fallback")

    # Scope check. The retrieved chunks are often variety-specific, so a GENERAL
    # question comes back answered for one variety the farmer never named. Same
    # shape as the dose check above: detect, retry once naming the problem, and
    # keep the better of the two rather than pay for a second round.
    unasked = _unrequested_entities(answer, query_text_en)
    if unasked:
        log.warning("answer narrowed the question to %s, which was not asked about",
                    unasked)
        retried = _chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query_text_en},
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                "The farmer did not ask about " + ", ".join(unasked)
                + ". You narrowed a general question to a specific case taken from "
                "the CONTEXT. Answer the question they actually asked: give the "
                "general recommendation that applies to their crop. Do not name "
                + ", ".join(unasked) + " or any other variety, district or season "
                "they did not mention. If the CONTEXT only has a figure for one "
                "specific case, give the figure without tying it to that case. Do "
                "NOT ask them which variety they grow and do not end with a "
                "question. Keep every amount exactly as it appears in the CONTEXT."},
        ])
        # Only accept the retry if it actually dropped the unasked-for names and
        # did not invent a new dose along the way.
        if (retried.strip()
                and not _unrequested_entities(retried, query_text_en)
                and not _unverified_doses(retried, context, scale_allowed)):
            answer = retried
        else:
            log.info("scope retry rejected; keeping the original answer")

    return _strip_internal_wording(
        _concretise_english(_drop_trailing_question(_drop_filler_tail(answer))))


# def tts_synthesize(text: str, voice: Optional[str] = None, language: str = "English") -> bytes:
#     """Synthesize text-to-speech using OpenAI TTS only."""
#     return _openai_tts_synthesize(text, voice)

def tts_synthesize(text, voice=None, language="English"):

    if language == "Odia":
        cleaned_text = prepare_for_tts(text, language)
        log.info("Odia TTS text sent to Narakeet: %s", cleaned_text)
        # Always-visible console print so you can verify numbers became words.
        print("\n========== ODIA TEXT SENT TO NARAKEET ==========", flush=True)
        print(cleaned_text, flush=True)
        print("================================================\n", flush=True)

        from .narakeet_tts import synthesize
        return synthesize(cleaned_text)

    return _openai_tts_synthesize(text, voice)

def _openai_tts_synthesize(text: str, voice: Optional[str] = None) -> bytes:
    """Synthesize text using OpenAI TTS."""
    resp = _client().audio.speech.create(
        model=settings.OPENAI_TTS_MODEL,
        voice=voice or settings.OPENAI_TTS_VOICE,
        input=text[:4000],
    )
    # openai>=1.0 returns a streamable response
    return resp.read() if hasattr(resp, "read") else resp.content


# ---------- High-level orchestrators ----------

def _answer_from_faq(q, faq, score, trace, start, *, farmer_language,
                     generate_tts, hook, whatsapp_number):
    """Finish a Query from a FAQ entry, skipping retrieval and generation.

    The stored answer is used EXACTLY as approved. The one thing that can still
    run is a translation, and only when the entry has no answer in the language
    being asked in — a FAQ written in English, asked in Odia. Translating then is
    right; re-translating an entry that already holds the Odia answer would let
    the approved wording drift on every hit, which is the opposite of the point.
    """
    # AN APPROVED ANSWER CAN STILL CARRY THE OLD LEAK. Entries are promoted
    # from queries whose feedback was Good, so an answer that said "CR Dhan 100
    # - the CONTEXT says: Fit for upland" before this was fixed is now stored and
    # served verbatim, to every farmer who asks again, for as long as the entry
    # lives. This path never touches generate_answer, so the scrub has to be
    # applied here too. It is presentation only: the approved WORDING and every
    # figure and qualifier in it are untouched, and nothing about matching,
    # storage or promotion changes.
    answer_en = _to_paragraph(_strip_internal_wording(faq.answer_en or faq.answer_local))
    q.ai_response_text = answer_en
    answer_local = _to_paragraph(_strip_internal_wording(faq.answer_local))

    if farmer_language and farmer_language != "English":
        if not answer_local:
            t0 = time.monotonic()
            # The CLEANED English is what gets translated, so the leak cannot
            # reach the farmer in Odia either.
            answer_local = translate(answer_en, "English", farmer_language)
            trace["steps"].append({
                "stage": "translate_out",
                "ms": int((time.monotonic() - t0) * 1000),
                "note": "FAQ entry had no answer in this language",
            })
    else:
        answer_local = answer_en

    q.ai_response_text_local = answer_local
    trace["faq"] = {"id": faq.pk, "score": round(score, 4),
                    "question": faq.question_en[:200]}

    if generate_tts:
        t0 = time.monotonic()
        try:
            audio_bytes = tts_synthesize(answer_local, language=farmer_language)
            q.ai_response_voice_file.save(
                f"q{q.pk}.mp3", ContentFile(audio_bytes), save=False)
            trace["steps"].append({"stage": "tts", "ms": int((time.monotonic() - t0) * 1000)})
        except Exception as exc:
            trace["steps"].append({"stage": "tts", "error": str(exc)})

    q.response_time_seconds = round(time.monotonic() - start, 2)
    q.status = "Responded"
    q.pipeline_trace = trace
    q.save()

    # Carry over the documents the approved answer was originally built from, so
    # the farmer still sees where it came from.
    try:
        if faq.source_query_id:
            doc_ids = list(faq.source_query.source_documents.values_list("id", flat=True))
            if doc_ids:
                q.source_documents.set(doc_ids)
    except Exception as exc:
        log.warning("faq: could not copy source documents (%s)", exc)

    faq_record_hit(faq)
    log.info("query %s answered from FAQ-%s (score %.3f) — retrieval skipped",
             q.pk, faq.pk, score)

    # Save conversation turn to .pkl chat history for follow-up questions.
    # ENGLISH, not Odia: this history is pasted into the English answer prompt,
    # so storing the Odia bypassed every term the translation side had already
    # settled (the glossary, the pinned terms, ପଟାସ -> potash) and handed the
    # model a script the retrieved chunks are not written in. English is also
    # the form a follow-up needs if the history is ever fed into retrieval.
    # The Odia is NOT lost — it stays on the Query row.
    if whatsapp_number:
        try:
            save_chat_turn(
                whatsapp_number,
                # Fall back to Odia only if translation produced nothing; an
                # empty turn is worse than an untranslated one.
                user_text=q.translated_query_text or q.original_query_text,
                assistant_text=answer_en or answer_local,
            )
        except Exception as exc:
            log.warning("Failed to save chat history for %s: %s", whatsapp_number, exc)

    if hook and whatsapp_number:
        try:
            from .whatsapp import send_via_picky_assist
            send_via_picky_assist(hook, whatsapp_number, answer_local, audio_url=None)
        except Exception as exc:
            log.warning("WhatsApp send failed: %s", exc)

    return q


# Which language a question is WRITTEN in. Odia and Hindi have their own Unicode
# blocks, so the script settles it — no model call, no guessing.
_SCRIPT_LANGUAGES = (
    ("Odia", re.compile(r"[଀-୿]")),
    ("Hindi", re.compile(r"[ऀ-ॿ]")),
)


def language_of_question(text: str, declared: str = "English") -> str:
    """The language to treat this question as, given what the caller declared.

    ONLY an impossible declaration is overridden, and only in one direction:

        declared English, written in Odia   -> Odia      (the reported bug)
        declared Odia, written in Latin     -> Odia      (declaration KEPT)

    The second case is deliberate and must not change. A farmer who types in
    English or romanised Odia through an Odia hook still wants the answer in
    Odia, and their declaration is the only thing that says so — `looks_english`
    already skips the inbound translation for them, which is all that is needed.
    So this function corrects a label that the script contradicts, and otherwise
    keeps its hands off.
    """
    if not (text or "").strip():
        return declared
    if declared and declared != "English":
        return declared                      # a non-English label is never overridden
    for language, pattern in _SCRIPT_LANGUAGES:
        if pattern.search(text):
            log.info("language | declared=%r but the question is written in %s "
                     "— treating it as %s", declared, language, language)
            return language
    return declared or "English"


def run_pipeline_for_text(
    catalogue, farmer_language: str, query_text: str,
    district: str = "", generate_tts: bool = False, hook=None,
    whatsapp_number: str = "", user_type: str = "farmer",
    initial_trace: dict | None = None, source_audio: tuple | None = None,
    catalogue_ids: list | None = None,
) -> Query:
    """Full pipeline entry point for a text-only query (demo form or webhook).

    `source_audio` is (bytes, suffix) when the farmer asked by voice. It is
    stored on the Query so a question asked aloud can be listened to later —
    a transcript that reads oddly is the one time the recording matters, and
    the gateway's own media URL has expired by then.
    """
    # Some webhook/test clients pass URL-encoded text (e.g. How+much...%3F).
    # Decode only when percent-escapes are present so ordinary literal plus signs
    # in normal messages are not changed.
    if query_text and re.search(r"%[0-9A-Fa-f]{2}", query_text):
        query_text = unquote_plus(query_text)

    # Start trace from initial_trace (e.g. STT backend info) if provided
    trace = (initial_trace.copy() if isinstance(initial_trace, dict) else {})
    if "steps" not in trace:
        trace.setdefault("steps", [])
    trace["user_type"] = user_type
    start = time.monotonic()
    # BELIEVE THE SCRIPT, NOT THE LABEL. The caller declares the language, and a
    # declaration of "English" for a question written in Odia disabled the whole
    # Odia pipeline at both ends: the question was never translated (so retrieval
    # ran on Odia text and matched almost nothing), and the answer was never
    # translated either — target_lang was "English", so the fluency repair, the
    # de-transliteration, the spoken register and the imperative pass were all
    # skipped, and whatever Odia the model happened to emit went straight to the
    # farmer. Only an impossible declaration is overridden; see the function.
    declared = farmer_language
    farmer_language = language_of_question(query_text, farmer_language)
    if farmer_language != declared:
        trace["declared_language"] = declared
        trace["detected_language"] = farmer_language
    q = Query.objects.create(
        whatsapp_number=whatsapp_number,
        district=district,
        hook=hook,
        trigger_keyword=(hook.trigger_keyword if hook else ""),
        catalogue=catalogue,
        original_query_language=farmer_language,
        original_query_text=query_text,
        status="Processing",
    )
    if source_audio and getattr(settings, "WHATSAPP_SAVE_INBOUND_AUDIO", True):
        try:
            data, suffix = source_audio
            q.original_query_voice_file.save(
                f"q{q.pk}{suffix or '.ogg'}", ContentFile(data), save=True)
        except Exception as exc:                 # noqa: BLE001
            # Keeping the recording must never cost the answer.
            log.warning("could not store the inbound voice note: %s", exc)

    try:
        # 1. translate farmer -> English (if needed)
        t0 = time.monotonic()
        # query_en = translate(query_text, farmer_language, "English")
        def looks_english(text):
            english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            total_chars = sum(1 for c in text if c.isalpha())

            if total_chars == 0:
                return True

            return (english_chars / total_chars) > 0.8

        if looks_english(query_text):
            query_en = query_text
        else:
            query_en = translate(query_text, farmer_language, "English")

        # Canonicalise spelling variants HERE, not only inside translate().
        # looks_english() short-circuits the translate() call for any question
        # already in Latin script, so a question typed in English — or a
        # romanised Odia transcript from the Whisper fallback — skipped the
        # synonym layer entirely: "When should Ajwala be applied to rice"
        # reached retrieval as "Ajwala" and returned "I don't have information".
        # This line is on the path BOTH branches take. canonicalise() is
        # idempotent, so the already-translated branch is unaffected.
        if query_en and re.search(r"%[0-9A-Fa-f]{2}", query_en):
            query_en = unquote_plus(query_en)

        query_en, syn_terms = canonicalise(query_en)
        if syn_terms:
            trace["synonyms"] = syn_terms

        # For short follow-up replies ("medium land", "lowland") the raw
        # query gives the router no signal.  Peek at chat history and prepend
        # the previous question so the router sees the real topic.
        _routing_query = query_en
        _early_hist = None
        if whatsapp_number:
            _early_hist = load_chat_history(whatsapp_number)
            if _early_hist and not _QUESTION_WORD_RE.match(query_en):
                _last_asst = (_early_hist[-1].get("assistant") or "")
                _prev_q = (_early_hist[-1].get("user") or "").strip()
                # Follow-up reply to a clarifying question, or short reply
                _asst_is_clarifying = (
                    "?" in _last_asst
                    or re.search(r"\bplease\s+(tell|provide|share|give)\b",
                                 _last_asst, re.IGNORECASE)
                )
                if _prev_q and (
                    _asst_is_clarifying or len(query_en.split()) <= 3
                ):
                    _routing_query = f"{_prev_q} {query_en}"

        # Route WhatsApp questions AFTER translation/canonicalisation. The web
        # form user already chose a catalogue; overriding it is wrong.
        if not whatsapp_number:
            catalogue_ids = catalogue_ids or ([catalogue.pk] if catalogue else [])
            trace["routing"] = {"skipped": "web_form_catalogue_selected",
                                "catalogue_ids": catalogue_ids}
        try:
            from hooks.routing import plan as plan_catalogues

            if query_en and re.search(r"%[0-9A-Fa-f]{2}", query_en):
                query_en = unquote_plus(query_en)
            if whatsapp_number:
                # Preserve the default catalogue before routing can override it.
                _wa_catalogue = catalogue
                routing_plan = plan_catalogues(_routing_query, default=catalogue)
                # Keep routing classification for topic guards, but force
                # WhatsApp retrieval to the default Rice Knowledge Base
                # catalogue.  The router must not redirect to a topic-specific
                # collection — all evidence lives in the single default index.
                catalogue = _wa_catalogue
                catalogue_ids = [_wa_catalogue.pk] if _wa_catalogue else []
                trace["routing"] = {
                    "primary": routing_plan.get("primary", ""),
                    "secondary": routing_plan.get("secondary", []),
                    "catalogue_ids": catalogue_ids,
                    "forced_to_default": True,
                    "query": _routing_query,
                }
                log.info(
                    "catalogue routing: primary=%s secondary=%s -> forced to default catalogue %s",
                    routing_plan.get("primary") or "default",
                    routing_plan.get("secondary") or "none",
                    catalogue_ids,
                )
        except Exception as exc:
            log.exception("Catalogue routing failed; using default catalogue")
            trace["routing_error"] = str(exc)
            catalogue_ids = [catalogue.pk] if catalogue else []

        trace["steps"].append({"stage": "translate_in", "ms": int((time.monotonic() - t0) * 1000)})
        q.translated_query_text = query_en

        # 1a. WhatsApp clarification workflow MUST run before FAQ lookup or RAG.
        # Otherwise a generic FAQ/RAG answer can bypass the client's required
        # clarification questions entirely.
        clarification_completed = False
        if whatsapp_number:
            try:
                from .clarification_workflow import handle_message
                clarification = handle_message(whatsapp_number, query_en)
            except Exception as exc:
                # A workflow error should be visible in the trace; continue with
                # the established answer path rather than failing the farmer's query.
                log.exception("WhatsApp clarification workflow failed")
                trace["clarification_error"] = str(exc)
                clarification = {"action": "continue"}

            action = clarification.get("action", "continue")
            if action == "ask":
                clarification_en = clarification["question"]
                t0 = time.monotonic()
                clarification_local = (
                    clarification_en if farmer_language == "English"
                    else translate(clarification_en, "English", farmer_language)
                )
                trace["steps"].append({
                    "stage": "clarification_asked",
                    "ms": int((time.monotonic() - t0) * 1000),
                    "scenario_initial_question": clarification.get("scenario_initial_question", ""),
                    "match_score": clarification.get("match_score"),
                })
                trace["clarification"] = {
                    "action": "ask",
                    "scenario_initial_question": clarification.get("scenario_initial_question", ""),
                }
                q.ai_response_text = clarification_en
                q.ai_response_text_local = clarification_local
                q.response_time_seconds = round(time.monotonic() - start, 2)
                q.status = "Responded"
                q.pipeline_trace = trace
                q.save()
                if hook and whatsapp_number:
                    try:
                        from .whatsapp import send_via_picky_assist
                        send_via_picky_assist(hook, whatsapp_number, clarification_local, audio_url=None)
                    except Exception as exc:
                        log.warning("WhatsApp clarification send failed: %s", exc)
                        trace["clarification_delivery_error"] = str(exc)
                        q.pipeline_trace = trace
                        q.save()
                return q
            elif action == "complete":
                # Use the original question plus the farmer's reply for retrieval
                # and generation; don't let a generic FAQ short-circuit it.
                query_en = clarification["combined_query"]
                q.translated_query_text = query_en
                clarification_completed = True
                trace["clarification"] = {
                    "action": "complete",
                    "scenario_initial_question": clarification.get("scenario_initial_question", ""),
                    "scenario_question": clarification.get("scenario_question", ""),
                }

        # 1b. FAQ — a question already answered and approved skips the rest.
        #
        # Placed AFTER translation and canonicalisation on purpose: those are what
        # turn "ଅଜୋଲା କେତେ ଦେବି" and "how much Azola to apply" into the same
        # English string, and matching before them would only ever hit on the
        # exact original wording.
        #
        # Everything below this block is skipped on a hit: no retrieval, no
        # generation, no answer translation. That is the point — the approved
        # answer is served verbatim, so the farmer who asks twice hears the same
        # thing twice, and it costs no LLM call.
        if clarification_completed:
            faq_hit, faq_score = None, 0.0
            duration_query = None
            trace["steps"].append({"stage": "faq_lookup", "hit": False,
                                  "skipped": "clarification_completed"})
        else:
            t0 = time.monotonic()
            faq_hit, faq_score = faq_lookup(query_en, catalogue=catalogue)
            # DAYS -> DURATION CLASS. The farmer names a day count; the approved
            # answers are written per class. Only tried when the question's own
            # wording found nothing, so it can never override a direct match.
            duration_query = duration_probe(query_en)
            if not duration_query:
                # THE VARIETY KNOWS ITS OWN CLASS. A question naming Sahbhagi Dhan
                # names no class and no days, so nothing was probed and the generic
                # entry answered it with the wrong dose. The class is read from the
                # entries that name the variety — see faq.duration_for_variety.
                by_variety = faq_duration_for_variety(query_en, catalogue=catalogue)
                if by_variety:
                    duration_query = duration_probe(query_en, label=by_variety)
            # THE PROBE MAY ADD CONTEXT. IT MAY NOT SERVE.
            #
            # It used to be promoted straight to faq_hit, which meant an approved
            # answer was returned VERBATIM — skipping retrieval and generation — off
            # a sentence the farmer never said. Query 1030: "Talk about the 100-day
            # rice variety" scored 0.0 against every entry on its own words, the
            # probe scored 0.891 against FAQ-137, and the farmer was sent a
            # fertiliser schedule with chunks=0.
            #
            # The docstring on duration_probe already claimed this could not happen,
            # on the strength of one measurement (0.787, under the 0.80 serve
            # threshold). That was an observation, not a guarantee, and FAQ-137
            # scores 0.891. The guarantee now lives here, where it is enforced for
            # every entry and every future phrasing.
            #
            # Nothing is lost: the probe is picked up again below as FAQ CONTEXT
            # (see `probe_near`), which is where a sentence we composed belongs —
            # evidence the model may use, not an approved answer in its own right.
            if not faq_hit and duration_query:
                probe_hit, probe_score = faq_lookup(duration_query, catalogue=catalogue)
                if probe_hit:
                    log.info("faq | no hit for %r; the duration phrasing %r matches "
                             "FAQ-%s (%.3f) — offered as CONTEXT, not served, "
                             "because the farmer did not ask it",
                             query_en, duration_query, probe_hit.pk, probe_score)
            step = {
                "stage": "faq_lookup",
                "ms": int((time.monotonic() - t0) * 1000),
                "hit": bool(faq_hit),
                "score": round(faq_score, 4),
            }
            # WHICH QUESTION WAS MATCHED. Without this the trace shows a confident
            # 0.89 FAQ match and no sign that the question had been rewritten — a
            # reviewer reading query 1030 could not tell the farmer never asked it.
            if duration_query:
                step["duration_probe"] = duration_query
            trace["steps"].append(step)
            # ponytail: duration variety list needs the full table, not an FAQ
            # snippet that names only 2-3. Skip the FAQ when the gate fires.
            if faq_hit and not duration_requested(query_en):
                return _answer_from_faq(
                    q, faq_hit, faq_score, trace, start,
                    farmer_language=farmer_language, generate_tts=generate_tts,
                    hook=hook, whatsapp_number=whatsapp_number,
                )

        # 2. classify + enrich — kicked off in the BACKGROUND so it overlaps with
        #    retrieve/generate/translate. Its output is only metadata on the Query
        #    record (it is NOT used to build the answer), so it should not sit on
        #    the critical path. Collected just before the final save (step 5b).
        import concurrent.futures
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _classify_timed():
            _t = time.monotonic()
            try:
                data = classify_and_enrich(query_en)
            except Exception as exc:
                log.warning("classify_and_enrich failed: %s", exc)
                data = {}
            return data, int((time.monotonic() - _t) * 1000)

        cls_future = _executor.submit(_classify_timed)

        # 2b. follow-up enrichment — load chat history BEFORE retrieval so a
        # bare follow-up ("how much for half an acre?", "what land suits this
        # variety?") inherits the previous question's topic for search.
        chat_hist = _early_hist if _early_hist is not None else (
            load_chat_history(whatsapp_number) if whatsapp_number else [])
        if chat_hist:
            trace["chat_history_turns"] = len(chat_hist)
            log.info("query | using %d turns of chat history for %s",
                     len(chat_hist), whatsapp_number)
        # Only use previous turns when this message genuinely depends on them.
        # A standalone question must not inherit unrelated conversation context.
        use_history_for_answer = (
            bool(chat_hist)
            and not clarification_completed
            and _is_followup(query_en, chat_hist)
        )
        log.info(
            "query | answer_history=%s | turns_available=%d | clarification_completed=%s",
            use_history_for_answer, len(chat_hist), clarification_completed,
        )

        retrieval_query = query_en
        _followup_candidates = []
        _followup_has_entity_ref = False
        if use_history_for_answer:
            _followup_has_entity_ref = bool(_ENTITY_REF_RE.search(query_en))
            for turn in reversed(chat_hist):
                candidate = (turn.get("user") or "").strip()
                if candidate:
                    _followup_candidates.append(candidate)
                    if len(_followup_candidates) >= 2:
                        break
            if _followup_candidates:
                prev_asst = (chat_hist[-1].get("assistant") or "").strip()
                if _followup_has_entity_ref and prev_asst:
                    # "which of these" / "the above" — retrieval needs the
                    # entity names from the previous answer, not just the
                    # previous question's topic.  Truncate to keep the
                    # embedding focused.
                    retrieval_query = (f"{_followup_candidates[0]} "
                                       f"{prev_asst[:300]} {query_en}")
                    trace["followup_entity_ref"] = True
                    log.info("query | follow-up with entity ref, retrieval "
                             "enriched with prev answer: %r",
                             retrieval_query[:200])
                else:
                    retrieval_query = f"{_followup_candidates[0]} {query_en}"
                trace["followup_enriched"] = True
                log.info("query | follow-up detected, retrieval query enriched "
                         "with last turn: %r", retrieval_query[:200])

        # 3. retrieve
        # Expand the query ONCE here so the same phrasings drive BOTH retrieval
        # and the answer step (no second LLM call). retrieve() would expand on its
        # own if we didn't pass `queries`, so this only reuses work, never adds it.
        t0 = time.monotonic()
        expanded = _expand_queries(retrieval_query)

        # --- query-rewrite visibility -----------------------------------------
        # original question we received, how many phrasings it became, and the
        # exact list used for retrieval (index 0 is the original).
        log.info("query | original=%r | rewrites=%d (1 original + %d variants)",
                 query_en, len(expanded), len(expanded) - 1)
        for i, qx in enumerate(expanded):
            log.info("query | retrieval[%d] = %r%s",
                     i, qx, "  <- original" if i == 0 else "")

        # ONE CATALOGUE OR SEVERAL. `catalogue_ids` arrives from the router
        # when a question spans two domains ("which variety resists blast?").
        # With one id, or none, this is the single-catalogue call it has always
        # been — the multi path is not a replacement, it is a branch.
        
        routed_ids = [i for i in (catalogue_ids or []) if i]

        log.info(
            "RETRIEVAL ROUTING DEBUG: query=%r catalogue=%s "
            "catalogue_ids=%r routed_ids=%r",
            query_en,
            getattr(catalogue, "name", None),
            catalogue_ids,
            routed_ids,
        )

        if len(routed_ids) > 1:
            log.info(
                "RETRIEVAL MODE: multi-catalogue; ids=%s",
                routed_ids,
            )
            hits = retrieve_across_catalogues(
                query_en,
                routed_ids,
                n_results=10,
                queries=expanded,
            )
            trace["catalogues"] = routed_ids
        else:
            log.info(
                "RETRIEVAL MODE: single-catalogue; id=%s",
                routed_ids[0] if routed_ids else getattr(catalogue, "pk", None),
            )

            is_entity = entity_intent.analyse(query_en) is not None
            wanted = 30 if is_entity else 10

            hits = retrieve(
                catalogue,
                query_en,
                n_results=wanted,
                queries=expanded,
                per_document_cap=20 if is_entity else None,
            )


        # Follow-up retry: if retrieval found nothing useful and we have a
        # second-last turn to try, re-run retrieval with that context instead.
        # ponytail: only compare vector scores (0-1 cosine); BM25 scores are
        # on a different scale and would incorrectly skip the retry.
        _vector_scores = [h.get("score") or 0 for h in hits
                          if not h.get("keyword")]
        _best_vscore = max(_vector_scores, default=0)
        if (len(_followup_candidates) >= 2 and _best_vscore < 0.3
                and trace.get("followup_enriched")):
            retrieval_query = f"{_followup_candidates[1]} {query_en}"
            expanded = _expand_queries(retrieval_query)
            log.info("query | follow-up retry with second-last turn: %r",
                     retrieval_query)
            trace["followup_retry"] = True
            if len(routed_ids) > 1:
                hits = retrieve_across_catalogues(query_en, routed_ids,
                                                  n_results=10, queries=expanded)
            else:
                hits = retrieve(catalogue, query_en, n_results=wanted,
                                queries=expanded,
                                per_document_cap=20 if is_entity else None)

        # WHAT THE QUESTION WAS READ AS, AND WHAT THE EVIDENCE DID. Recorded
        # where a reviewer can see it, because the two failures this pipeline
        # keeps having are invisible from the answer alone: a facet read as the
        # wrong one ("medium"), and evidence that was retrieved and then not
        # matched ("Fit for upland" reached the context and no candidate came
        # out). The first is answered by `attributes`, the second by the
        # accepted/rejected lists — each rejection says which test dropped it.
        _intent = entity_intent.analyse(query_en)
        if _intent is not None:
            _report: dict = {}
            entity_intent.candidate_pool(_intent, hits, _report)
            trace["intent"] = {
                "entity": _intent.entity,
                "quantity": _intent.quantity,
                "attributes": _intent.attributes,
                "land_type": _intent.land_type,          # the canonical term
                "land_type_missing": _intent.land_type_missing,
                "condition_terms": _report.get("condition_terms", []),
                "retrieval_variants": entity_intent.retrieval_variants(query_en,
                                                                       _intent),
            }
            trace["candidates"] = {
                "evidence_found": _report.get("evidence_found", False),
                "accepted": _report.get("accepted", []),
                "rejected": _report.get("rejected", [])[:10],
            }

        # THE FAQ THAT ALMOST MATCHED IS STILL EVIDENCE. The FAQ used to be
        # all-or-nothing: an entry either replaced the whole pipeline or was
        # never seen, so a question a colleague had already answered in a
        # slightly different form contributed nothing. A near miss enters here
        # as one more source, with no special status. Entries that CONTRADICT
        # the question - a different season, product, stage or area unit - never
        # reach this point; see faq.nearest().
        near, near_score = faq_nearest(query_en, catalogue=catalogue)
        contexts = [(near, near_score)] if near else []

        # THE DURATION PHRASING IS TRIED EVEN WHEN THE QUESTION FOUND SOMETHING.
        # It used to be a fallback, and that was wrong. Live, for "How many kg of
        # urea should be applied for the 100-day short variety?", the farmer's
        # own wording found the GENERIC entry — "How much urea I have to apply in
        # my rice crop?" — which clears the floor and says nothing about short
        # duration. Because something was found, the probe was skipped, the entry
        # that actually answers the question never reached the model, and the
        # answer came back as "the information available does not specify the
        # amount of urea for a 100-day short variety."
        #
        # So both are offered, highest score first, and the class guard in
        # faq.nearest() keeps the wrong duration out of either of them.
        if duration_query:
            probe_class = (duration_class(query_en)
                           or faq_duration_for_variety(query_en,
                                                       catalogue=catalogue))
            probe_near, probe_score = faq_nearest(
                duration_query, catalogue=catalogue, duration=probe_class)
            if not probe_near and near:
                # NOTHING IS WRITTEN FOR THIS CLASS, so the question has no FAQ
                # answer — and the entry its own wording found is a CLASSLESS
                # one. Live: "୧୨୦ ଦିନିଆ ଧାନରେ କେତେ କିଲୋ ୟୁରିଆ?" is a medium
                # duration question, there is no medium entry, and the generic
                # "How much urea I have to apply in my rice crop?" (0.741) was
                # handed over — so its schedule went out headed "For medium
                # duration rice varieties". A dose wearing a class it was never
                # written for is the failure this whole path exists to prevent,
                # so the near miss is dropped and the documents answer instead.
                log.info("faq | no entry written for the %s duration class; "
                         "dropping FAQ-%s (%.3f), which names no class",
                         probe_class or "requested", near.pk, near_score)
                contexts = []
            if probe_near:
                # AND IT REPLACES THE QUESTION'S OWN NEAR MISS. Passing both was
                # measured WORSE than passing neither: the generic entry ("How
                # much urea I have to apply in my rice crop?") is a near-exact
                # lexical match for "how many kg of urea", so the model took its
                # 70 kg / 10-15 / 30-35 figures and applied them to the 100-day
                # SHORT variety — a confident wrong dose, where the refusal had
                # at least been harmless. An entry written for the class the
                # farmer asked about outranks one written for no class at all.
                if near and probe_near.pk != near.pk:
                    log.info("faq | FAQ-%s (%.3f) names no duration class; using "
                             "FAQ-%s (%.3f), matched on %r, instead",
                             near.pk, near_score, probe_near.pk, probe_score,
                             duration_query)
                contexts = [(probe_near, probe_score)]

        contexts.sort(key=lambda pair: pair[1], reverse=True)
        for entry, score in contexts:
            hits.append({
                "chunk": f"{entry.question_en}\n{entry.answer_en}",
                "metadata": {
                    "document_title": f"Previously answered: {entry.question_en[:70]}",
                    "faq_id": entry.pk,
                    "faq_score": round(score, 4),
                },
            })
        if contexts:
            best, best_score = contexts[0]
            trace["faq_context"] = {"id": best.pk, "score": round(best_score, 4),
                                    "question": best.question_en}
            if len(contexts) > 1:
                trace["faq_context_extra"] = [
                    {"id": e.pk, "score": round(s, 4), "question": e.question_en}
                    for e, s in contexts[1:]]

        trace["steps"].append({
            "stage": "retrieve",
            "ms": int((time.monotonic() - t0) * 1000),
            "hits": len(hits),
            "faq_context": contexts[0][0].pk if contexts else None,
        })
        trace["retrieved_chunks"] = _trace_retrieved_chunks(hits)

        # 4. generate  (persona: farmer | researcher)
        # QUERY REWRITING: the farmer's wording (e.g. "How much manuring...") often
        # doesn't match the documents' vocabulary ("fertilizer recommendation"),
        # so the generator can fail to connect the question to the retrieved
        # chunks even when retrieval found them. We therefore hand generate_answer
        # the ORIGINAL question PLUS the canonical (document-vocabulary) rewrite —
        # reused for free from the expansion above. The original is kept so the
        # farmer's actual intent, numbers and names are never lost.
        # A DAY COUNT NEEDS ITS CLASS SPELLED OUT. With the class-matched entry
        # in context but nothing in the CONTEXT saying that a 100-day variety IS
        # a short duration variety, the model refused outright — "I do not have
        # that information" — because the grounding rules forbid it from making
        # that link itself. Saying it here, as an interpretation of the QUESTION
        # rather than as a retrieved fact, is what turns the right entry into the
        # right answer. The farmer's own wording is still above it, unchanged.
        canonical = duration_query or _canonical_from_expansions(query_en, expanded)
        # When the farmer answered a clarifying question (e.g. "lowland" after
        # "Which type of land?"), the bare reply is not a question the LLM can
        # answer on its own. Combine it with the original question so the LLM
        # sees "What is the fertilizer dose for MTU 1010? — lowland".
        _last_asst_text = (chat_hist[-1].get("assistant") or "") if chat_hist else ""
        _last_asst_was_question = (
            "?" in _last_asst_text
            or bool(re.search(r"\bplease\s+(tell|provide|share|give)\b",
                              _last_asst_text, re.IGNORECASE))
        )
        _has_entity_ref = bool(_ENTITY_REF_RE.search(query_en))
        if (use_history_for_answer and _followup_candidates
                and (_last_asst_was_question or _has_entity_ref)
                and (not _QUESTION_WORD_RE.match(query_en) or _has_entity_ref)):
            gen_query = f"{_followup_candidates[0]} ({query_en})"
            log.info("query | follow-up merged into gen_query: %r",
                     gen_query)
        else:
            gen_query = query_en
        if canonical:
            gen_query = f"{query_en}\n\n(Interpreted for clarity as: {canonical})"
            trace["canonical_query"] = canonical
            log.info("query | canonical rewrite used for generation = %r", canonical)
        else:
            log.info("query | no clean rewrite found; using original for generation")
        t0 = time.monotonic()
        # A duration question is answered from the catalogue TABLE, not from
        # whichever rows the ranking happened to surface — the 9 varieties at
        # 150 days live in 7 near-identical chunks and "all of them" is not
        # something a top-N can promise.
        must_name = []
        _days = duration_requested(query_en)
        if _days:
            try:
                must_name = varieties_by_duration(catalogue, _days)
                if must_name:
                    trace["duration_list"] = {"days": _days, "count": len(must_name)}
            except Exception as exc:
                log.warning("duration variety lookup failed for %s days: %s",
                            _days, exc)

        # Crucial: do not pass all saved turns to generation for a standalone
        # question. For clarification-completed questions, the combined query
        # already carries the needed details, so history is not needed either.
        # Limit to last 3 turns to avoid old unrelated context bleeding in.
        _MAX_PROMPT_TURNS = 3
        answer_history = chat_hist[-_MAX_PROMPT_TURNS:] if use_history_for_answer else []
        answer_en = generate_answer(gen_query, hits, user_type=user_type,
                                    chat_history=answer_history, must_name=must_name)
        trace["steps"].append({
            "stage": "generate",
            "ms": int((time.monotonic() - t0) * 1000),
            "persona": user_type,
        })
        q.ai_response_text = answer_en

        # 5. translate English -> farmer's language
        t0 = time.monotonic()
        answer_local = translate(answer_en, "English", farmer_language)
        trace["steps"].append({"stage": "translate_out", "ms": int((time.monotonic() - t0) * 1000)})
        q.ai_response_text_local = answer_local

        # 5b. collect the classification that ran in parallel (usually already done)
        cls, cls_ms = cls_future.result()
        _executor.shutdown(wait=False)
        trace["steps"].append({"stage": "classify", "ms": cls_ms, "parallel": True})
        q.category = (cls.get("category") or "")[:40]
        q.task_type = (cls.get("task_type") or "")[:40]
        q.problem_entity = (cls.get("problem_entity") or "")[:200]
        q.season = (cls.get("season") or "")[:20]
        q.crop_stage = (cls.get("crop_stage") or "")[:30]
        q.missing_context = (cls.get("missing_context") or "")[:500]

        # 6. TTS (optional)
        if generate_tts:
            t0 = time.monotonic()
            try:
                audio_bytes = tts_synthesize(answer_local, language=farmer_language)
                q.ai_response_voice_file.save(
                    f"q{q.pk}.mp3", ContentFile(audio_bytes), save=False
                )
                trace["steps"].append({"stage": "tts", "ms": int((time.monotonic() - t0) * 1000)})
            except Exception as exc:
                trace["steps"].append({"stage": "tts", "error": str(exc)})

        # 7. attach source documents
        doc_ids = {h["metadata"].get("document_id") for h in hits if h["metadata"].get("document_id")}
        if doc_ids:
            q.save()  # must save before m2m set
            q.source_documents.set(doc_ids)

        q.response_time_seconds = round(time.monotonic() - start, 2)
        q.status = "Responded"
        q.pipeline_trace = trace
        q.save()

        # Save conversation turn to .pkl chat history for follow-up questions.
        # English for the same reason as the FAQ path above — this text is read
        # back into the English answer prompt, not shown to the farmer.
        if whatsapp_number:
            try:
                save_chat_turn(
                    whatsapp_number,
                    user_text=query_en or query_text,
                    assistant_text=answer_en or answer_local,
                )
            except Exception as exc:
                log.warning("Failed to save chat history for %s: %s", whatsapp_number, exc)

        # 8. WhatsApp send (only if we have a hook + number configured)
        if hook and whatsapp_number:
            try:
                from .whatsapp import send_via_picky_assist
                send_via_picky_assist(hook, whatsapp_number, answer_local, audio_url=None)
            except Exception as exc:
                log.warning("WhatsApp send failed: %s", exc)

        return q

    except Exception as exc:
        log.exception("Pipeline failure for query %s", q.pk)
        trace["error"] = str(exc)
        q.pipeline_trace = trace
        q.status = "Failed"
        q.response_time_seconds = round(time.monotonic() - start, 2)
        q.save()
        raise


def identify_from_image(image_path: str, catalogue=None) -> dict:
    """
    Run CLIP image search against the global image bank.
    Returns: {
        "matched": bool,                  # confident match?
        "prefix": str,                    # matched visual group name
        "description": str,               # sidecar visual description (English)
        "image_url": str,                 # representative image URL (top hit)
        "distance": float | None,
        "score": float | None,
        "low_confidence": bool,           # True when above the confidence threshold
        "all_hits": list,                 # raw top-k for debug
    }
    """
    from image_bank.services.image_vector_store import query_by_image_path

    threshold = float(getattr(settings, "IMAGE_BANK_DISTANCE_MAX", 0.35))
    top_k = int(getattr(settings, "IMAGE_BANK_TOP_K", 5))
    catalogue_id = getattr(catalogue, "id", None) if catalogue else None
    hits = query_by_image_path(image_path, top_k=top_k, catalogue_id=catalogue_id)
    if not hits:
        return {"matched": False, "prefix": "", "description": "", "image_url": "",
                "distance": None, "score": None, "low_confidence": True, "all_hits": []}

    top = hits[0]
    distance = top.get("distance")
    meta = top.get("metadata") or {}
    low_conf = distance is None or distance > threshold
    return {
        "matched": True,
        "prefix": meta.get("prefix", ""),
        "description": (meta.get("description") or top.get("description") or "").strip(),
        "image_url": meta.get("image_url", ""),
        "distance": distance,
        "score": top.get("score"),
        "low_confidence": low_conf,
        "all_hits": hits,
    }


def run_pipeline_for_image(
    catalogue, farmer_language: str, uploaded_image,
    district: str = "", generate_tts: bool = False, hook=None,
    whatsapp_number: str = "", user_type: str = "farmer",
    user_query_text: str = "",
) -> Query:
    """
    Image-as-query entry point.

    Flow:
      1. Save the uploaded image to a temp file on disk.
      2. CLIP-match against the global image bank to identify the topic.
      3. If matched, build a synthesized query from the matched prefix +
         visual description, then run the standard text pipeline against the
         catalogue (translate -> classify -> retrieve -> generate -> translate).
      4. If low confidence, prefix the answer with a warning.
    """
    import tempfile

    # Persist upload to a real path so PIL/CLIP can read it.
    suffix = Path(uploaded_image.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in uploaded_image.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        identification = identify_from_image(tmp_path, catalogue=catalogue)
    except Exception as exc:
        log.exception("Image identification failed")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    prefix = identification.get("prefix") or "(unknown)"
    visual_desc = identification.get("description") or ""
    low_conf = identification.get("low_confidence", True)

    # If the caller supplied a text question along with the image, translate it
    # into English first so we can blend it into the synthesized query.
    user_q_en = ""
    if user_query_text:
        try:
            user_q_en = translate(user_query_text, farmer_language or "English", "English")
        except Exception:
            user_q_en = user_query_text

    score = identification.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"

    if not identification.get("matched") or not visual_desc:
        if user_q_en:
            synth_query_en = (
                f"{user_q_en}\n\n"
                "(No matching reference image was found in the knowledge base. "
                "Answer the user's question using general knowledge and the retrieved "
                "documents. If the question depends on what is visually present, ask "
                "the user for a clearer photo or more details.)"
            )
        else:
            synth_query_en = (
                "An image was uploaded but no matching reference image was found in the "
                "knowledge base. Tell the user what extra information or a clearer photo "
                "would help."
            )
    else:
        topic = prefix.replace("_", " ")
        if user_q_en:
            # User has a specific question. The CLIP match is a HINT, not a fact —
            # the closest visual neighbour may still be a different concept (e.g.
            # growth stage when the user is asking about disease). Let the LLM
            # reconcile the two and disagree if the topics don't line up.
            confidence_note = (
                "low confidence — treat as a tentative hint only"
                if low_conf else
                f"similarity score {score_str}"
            )
            synth_query_en = (
                f"USER QUESTION:\n{user_q_en}\n\n"
                f"VISUAL HINT (from a CLIP image-similarity search; {confidence_note}):\n"
                f"The uploaded photo most closely resembles reference images of '{topic}'.\n\n"
                "INSTRUCTIONS:\n"
                "- Answer the USER QUESTION directly. Do NOT change the topic.\n"
                "- The VISUAL HINT may or may not be relevant to what the user is asking. "
                "If the hint's topic does not match the user's question (for example, the "
                "hint is a growth stage but the user asks about disease or nutrient "
                "deficiency), say so explicitly and do not assert the hint as a diagnosis.\n"
                "- Only use the hint if it is genuinely relevant to the user's question.\n"
                "- If the visual hint is low confidence, treat it cautiously.\n"
            )
        else:
            # No user question — pure image-as-query. Identification is the answer.
            confidence_note = (
                "low confidence" if low_conf else f"similarity score {score_str}"
            )
            synth_query_en = (
                f"An uploaded image was matched to reference images of '{topic}' "
                f"({confidence_note}) in the visual knowledge base. "
                "Briefly explain what this is and what the user should do about it. "
                "If the match is low confidence, say so and ask for a clearer photo."
            )

    # Run the existing text pipeline using the synthesized English query.
    q = run_pipeline_for_text(
        catalogue=catalogue,
        farmer_language="English",          # synthesized query is already English
        query_text=synth_query_en,
        district=district,
        generate_tts=generate_tts,
        hook=hook,
        whatsapp_number=whatsapp_number,
        user_type=user_type,
    )

    # Attach image-match metadata to the query record.
    q.original_query_text = (
        f"[IMAGE QUERY] {user_query_text}".strip() if user_query_text else "[IMAGE QUERY]"
    )
    q.original_query_language = farmer_language or "English"
    trace = q.pipeline_trace or {}
    trace["image_query"] = {
        "matched_prefix": identification.get("prefix"),
        "matched_description": (identification.get("description") or "")[:1000],
        "matched_image_url": identification.get("image_url"),
        "distance": identification.get("distance"),
        "score": identification.get("score"),
        "low_confidence": low_conf,
    }
    q.pipeline_trace = trace

    if low_conf and identification.get("matched"):
        warning = (
            "Note: the image match is low confidence. The closest visual group is "
            f"'{prefix.replace('_', ' ')}'. Please send a clearer photo if this is wrong.\n\n"
        )
        q.ai_response_text = warning + (q.ai_response_text or "")
        if q.ai_response_text_local:
            q.ai_response_text_local = warning + q.ai_response_text_local

    q.save()

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return q


def fetch_audio(url: str) -> tuple:
    """(bytes, suffix) for a voice note at `url`. One download, reused.

    Separate from transcription so the farmer's own recording can be KEPT.
    Before this, a WhatsApp voice note was written to a temporary file,
    transcribed and deleted — so a question asked by voice left only its
    transcript, and when a transcript looked wrong there was no way to hear what
    was actually said. The gateway's media URL is not a substitute: those links
    expire.
    """
    if not url:
        raise ValueError("audio_url missing")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    suffix = os.path.splitext(urlparse(url).path)[1].lower()
    if suffix not in (".ogg", ".opus", ".mp3", ".m4a", ".wav", ".amr", ".aac"):
        suffix = ".ogg"                    # what WhatsApp voice notes actually are
    return response.content, suffix


def transcribe_audio_bytes(data: bytes, suffix: str = ".ogg",
                           language_hint: str = "") -> tuple:
    """(text, stt_backend) for audio already in hand.

    The transcription itself is stt_transcribe(), which is where the Sarvam /
    local-Wav2Vec2 / Whisper routing lives; this only owns the temporary file it
    needs and the cleanup of it.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix or ".ogg", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return stt_transcribe(tmp_path, language_hint=language_hint)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def transcribe_audio_url(url: str, language_hint: str = "") -> tuple:
    """(text, stt_backend) for a voice note at `url`.

    Kept at this name and signature because both gateway adapters call it. It is
    now the two steps above composed, so a caller that wants to keep the audio
    can fetch once and hand the same bytes to both.
    """
    data, suffix = fetch_audio(url)
    return transcribe_audio_bytes(data, suffix, language_hint=language_hint)


def process_incoming_message(hook, payload: dict) -> Query:
    """
    Called by PickyAssistWebhookView. `payload` keys:
        from, message_type ("text"|"audio"), text, audio_url, district
    """
    lang = hook.primary_language
    district = payload.get("district", "")
    whatsapp = payload.get("from", "")
    msg_type = payload.get("message_type", "text")
    query_text = payload.get("text", "")

    source_audio = None
    if msg_type == "audio":
        data, suffix = fetch_audio(payload.get("audio_url", ""))
        source_audio = (data, suffix)
        query_text, stt_backend = transcribe_audio_bytes(
            data, suffix, language_hint=lang)

    # Pass the STT backend to the text pipeline so it can be recorded in trace.
    initial_trace = {"stt_backend": stt_backend} if msg_type == "audio" else None

    return run_pipeline_for_text(
        catalogue=hook.catalogue,
        farmer_language=lang,
        query_text=query_text,
        district=district,
        generate_tts=True,
        hook=hook,
        whatsapp_number=whatsapp,
        initial_trace=initial_trace,
        source_audio=source_audio,
    )
