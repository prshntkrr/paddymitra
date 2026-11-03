"""
End-to-end real AI pipeline for a farmer query:

    STT (voice only) -> Translate to English -> Retrieve from Chroma -> Classify
    -> Generate answer -> Translate to farmer's language -> TTS (optional)
    -> Send via WhatsApp (Picky Assist)

All OpenAI calls are real (model names configurable via settings).
The pipeline is resilient: on failure it records the error into Query.pipeline_trace.error
and attempts to send a graceful fallback text reply to the farmer.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import re
from pathlib import Path
from typing import List, Optional

import requests
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from .odia_glossary import (apply_glossary, build_prompt_glossary, relevant_glossary,
                            relevant_reverse_glossary)
from .odia_translit import apply_translit_fixes, apply_tts_spoken

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
_VARIETY_PREFIXES = (
    "ଧାନ", "ଏମଟିୟୁ", "ଏମ.ଟି.ୟୁ", "ସ୍ୱର୍ଣ୍ଣ", "ସ୍ୱର୍ଣ", "ନବୀନ", "ନଭୀନ",
    "ସବ୍", "ସବ", "ଆଇଆର", "ଆଇ.ଆର୍", "ବିନା", "ପୂଜା", "ପୁଜା", "ଲଳାଟ",
)
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
        is_after_variety = any(before.endswith(p) for p in _VARIETY_PREFIXES)
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
          temperature: float = 0.1) -> str:
    # temperature defaults to 0 so the SAME question yields the SAME answer every
    # time (previously 0.2, which made repeated queries drift).
    kwargs = {
        "model": model or settings.OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _gemini_text(system_prompt: str, user_text: str, model: Optional[str] = None) -> str:
    """Generate plain text with Gemini. Used as a cheap Odia translation path."""
    from google import genai

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model_names = [model or getattr(settings, "GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash-lite")]
    for fallback in ("gemini-2.5-flash", "gemini-2.0-flash-lite"):
        if fallback not in model_names:
            model_names.append(fallback)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        last_exc: Exception | None = None
        for model_name in model_names:
            for attempt in range(1, 4):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=[system_prompt, user_text],
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

    # Fertilizer grades (10-26-26) -> separate numbers FIRST, so the range step
    # below doesn't read them as "10 to 26 to 26".
    text = re.sub(r"\b\d+(?:-\d+){2,}\b",
                  lambda m: " ".join(m.group(0).split("-")), text)

    # Slash: unit/unit -> "per"; otherwise a choice -> "or".
    def _slash(m):
        a, b = m.group(1), m.group(2)
        if a.lower() in _SPEECH_UNITS and b.lower() in _SPEECH_UNITS:
            return f"{a} {c['per']} {b}"
        return f"{a} {c['or']} {b}"
    text = re.sub(r"\b([A-Za-z]{1,5})\s*/\s*([A-Za-z]{1,5})\b", _slash, text)

    # Numeric ranges: 25-30 / 25–30 / 25 — 30 -> "25 to 30".
    text = re.sub(r"(\d+)\s*[–—-]\s*(\d+)", rf"\1 {c['to']} \2", text)

    # Brackets: KEEP the content (it carries the dose) but drop the symbols,
    # using a comma so the aside is spoken as a natural pause.
    text = re.sub(r"\s*[\(\[\{]\s*", ", ", text)
    text = re.sub(r"\s*[\)\]\}]\s*", ", ", text)

    # Tidy the commas/spaces produced above.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:,\s*){2,}", ", ", text)       # ", , " -> ", "
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*([.!?;:])", r"\1", text)      # ", ." -> "."
    text = re.sub(r"\s+([.!?;:,])", r"\1", text)
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

        # Narakeet's Odia voice pronounces ANY Latin-script token in English.
        # For DISPLAY we keep abbreviations/units in Latin, but for SPEECH every
        # remaining Latin token must become Odia script (phonetic), otherwise the
        # farmer hears stray English words. Convert the known ones directly...
        TTS_SPOKEN_LATIN = {
            "DAP": "ଡ଼ି ଏ ପି", "MOP": "ଏମ ଓ ପି", "NPK": "ଏନ ପି କେ", "SSP": "ଏସ ଏସ ପି",
            "N": "ଏନ", "P": "ପି", "K": "କେ",
            "kg": "କିଲୋଗ୍ରାମ", "g": "ଗ୍ରାମ", "mg": "ମିଲିଗ୍ରାମ",
            "ha": "ହେକ୍ଟର", "ml": "ମିଲିଲିଟର", "L": "ଲିଟର", "pH": "ପିଏଚ",
        }
        for abbr, spoken in TTS_SPOKEN_LATIN.items():
            text = re.sub(rf"\b{re.escape(abbr)}\b", spoken, text)

        # ...and anything still in Latin: LEARN it. Translate the unknown words to
        # Odia (Gemini/OpenAI) and persist them into odia_glossary.json so the same
        # word is replaced instantly next time, then re-apply the glossary to fix
        # the CURRENT response too.
        leftover = re.findall(r"\b[a-zA-Z][a-zA-Z/-]*\b", text)
        if leftover:
            log.info("Odia TTS: leftover Latin words -> learning into glossary: %s", leftover)
            try:
                from .glossary_learn import learn_terms
                if learn_terms(leftover):
                    text = apply_glossary(text)
            except Exception as exc:
                log.warning("Odia TTS glossary learning failed: %s", exc)

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
        return text.strip()

    cleaned = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model="gpt-4.1-mini"
    )

    return _apply_tts_glossary(cleaned)

# ---------- Pipeline stages ----------

def stt_transcribe(audio_path: str, language_hint: str = "") -> tuple[str, str]:
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


# Odia spelling corrections for common LLM errors
ODIA_CORRECTIONS = {
    # Common misspellings and corrections
    "ସାରି": "ସାର",  # fertiliser (incorrect → correct)
    "ଚାଷୀ": "ଚାଷ",  # farming
    "ରୋଗୀ": "ରୋଗ",  # disease
    "ଦ୍ରବ୍ୟ": "ଦ୍ରବ",  # substance
    "ଜିବନ": "ଜୀବନ",  # life
    "ସମାଧାନ": "ସମାଧାନ",  # solution
    "ବୀଜ": "ବିହନ",  # seed (prefer ବିହନ in agricultural context)
    "ଅଧିକ": "ଅଧିକ",  # more (verify context)
}


def _correct_odia(text: str) -> str:
    """Apply common Odia spelling corrections."""
    for wrong, right in ODIA_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


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
        "ଟିଲେରିଂ -> କୁଆଁରା ବାହାରିବା (tillering); "
        "ୱାଟର -> ପାଣି (water).\n"
        "STRICT RULES:\n"
        "- Keep all real Odia words, numbers, percentages and punctuation EXACTLY as they are.\n"
        "- Do NOT change chemical names, fertilizer product names, or rice variety names that "
        "have no common Odia word (e.g. ଅମୋନିୟମ ସଲଫେଟ, ଡରି ଏ ପି) — leave them as they are.\n"
        "- Do NOT add, remove, or reorder information.\n"
        "- If nothing needs changing, return the text unchanged.\n"
        "- Return ONLY the corrected Odia text." + ref
    )

    def _dedupe_exact_repeat(value: str) -> str:
        value = value.strip()
        if len(value) % 2:
            return value
        mid = len(value) // 2
        left = value[:mid].strip()
        right = value[mid:].strip()
        return left if left and left == right else value

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


def translate(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    if source_lang == target_lang:
        return text
    
    # Use specialized prompt for Odia
    if target_lang == "Odia":
        # Give the model the EXACT Odia for every glossary term present in THIS text.
        # This is what stops it from transliterating a translatable word (e.g.
        # tillering -> ଟିଲେରିଂ, which then sounds like English in TTS).
        rel = relevant_glossary(text)
        if rel:
            glossary_intro = (
                "12. The text contains these terms. Output EXACTLY the Odia shown for "
                "each — do NOT transliterate them and do NOT keep them in English:\n"
            )
            glossary_block = "\n".join(f"    - {en} → {od}" for en, od in sorted(rel.items()))
        else:
            glossary_intro = "12. Common agricultural terms and their Odia equivalents:\n"
            glossary_block = build_prompt_glossary()   # loads from odia_glossary.json

        sys = (
            "You are an expert agricultural translator specializing in Odia (ଓଡ଼ିଆ). "
            "Translate the following agricultural text into natural, farmer-friendly Odia.\n\n"

            "CRITICAL RULES:\n"
            "1. Use simple, everyday SPOKEN Odia that a village farmer easily understands. "
            "Keep sentences short. Avoid bookish or heavily Sanskritized words; prefer the "
            "common spoken word.\n"
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
        # Translate the farmer's Odia question into the EXACT standard English terms
        # the documents use (reverse glossary), so retrieval matches well.
        rev = relevant_reverse_glossary(text)
        rev_block = ("\n".join(f"    - {od} -> {en}" for od, en in rev.items())
                     if rev else "")
        sys = (
            "Translate the following Odia agricultural text into clear, SIMPLE English. "
            "Use common everyday words. Return ONLY the English translation.\n"
            "NUMBERS: Convert any spelled-out number (in Odia or English words) into "
            "digits, keeping the exact value. For example ସାତ ହଜାର ଏକଷଠି -> 7061, "
            "ଏକ ଶହ କୁଡ଼ି -> 120. Never round or drop part of a number.\n"
            "RICE VARIETIES: Rice variety names often contain numbers (e.g. CR Dhan 7061, "
            "Naveen, Swarna Sub1). Keep variety numbers as digits exactly as said; do not "
            "turn them into a quantity of varieties (e.g. '7061 variety', NOT 'seven "
            "thousand varieties')."
            + ("\n\nUse these EXACT English terms for the Odia terms found in the text:\n"
               + rev_block if rev else "")
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
    if target_lang == "Odia":
        try:
            result = _google_translate(text, source_lang, target_lang)
        except Exception as exc:
            log.warning("Google Translate Odia failed (%s); falling back to LLM", exc)

    # --- previous model-based Odia translation (Gemini 2.5 flash -> OpenAI) ---
    # Commented out in favour of Google Translate above. Kept for reference /
    # quick rollback.
    # if target_lang == "Odia" and settings.GEMINI_API_KEY:
    #     try:
    #         result = _gemini_text(sys, text)
    #     except Exception as exc:
    #         log.warning("Gemini Odia translation failed (%s); falling back to OpenAI", exc)

    # Fallback for non-Odia targets, and a safety net if Google Translate is
    # unavailable (missing key / network error) so a response is still produced.
    if not result:
        try:
            result = _chat([{"role": "system", "content": sys}, {"role": "user", "content": text}])
        except Exception as exc:
            log.warning("OpenAI translation failed (%s)", exc)
            result = ""

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

    if target_lang == "Odia":
        result = apply_glossary(result)        # fast local replacements first
        latin_words = re.findall(r'\b[a-zA-Z]{3,}\b', result)
        allowed = {"DAP", "MOP", "NPK", "kg", "ha", "pH"}
        unexpected = [w for w in latin_words if w not in allowed]
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
        result = _expand_element_symbols(result)
        # Show numbers in Odia numerals on screen too (e.g. 20-25 -> ୨୦-୨୫).
        result = result.translate(_ODIA_DIGITS)

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


_QUERY_EXPANSION_PROMPT = """
    You are an agricultural search expert.

    Rewrite the farmer's question into 4 different search queries that are most likely
    to match agricultural documents.

    Rules:

    - Preserve the original meaning.
    - Use agricultural terminology used in manuals.
    - Expand farmer wording into standard terminology.
    - Prefer document vocabulary over conversational wording.
    - Keep crop names, variety names and numbers unchanged.
    - Do NOT answer the question.
    - Return ONLY a JSON array.

    Examples

    Farmer:
    How much manuring will you do for Swarna?

    Queries:
    [
    "What fertilizer recommendation is there for Swarna paddy?",
    "What fertilizer dose is recommended for Swarna rice?",
    "Recommended nutrient application for Swarna paddy",
    "Fertilizer schedule for Swarna rice"
    ]

    Farmer:
    Medicine for blast?

    Queries:
    [
    "Recommended pesticide for rice blast",
    "Rice blast disease control",
    "Fungicide recommendation for blast",
    "Blast disease management in rice"
    ]
    """


# Farmer wording -> the EXACT term the documents use. When a farmer phrase on the
# left appears in the (English) query, we add a search variant containing the
# document term on the right, so both vector and keyword search can match. This
# is the deterministic safety net for vocabulary gaps the LLM rephrasing misses
# (e.g. "dry land" / "no standing water" -> the documents say "upland").
import re as _re
_DOMAIN_SYNONYMS = [
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
    return queries[:6]


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


def retrieve(catalogue, query_text_en: str, n_results: int = 6, rrf_k: int = 60):
    """Hybrid retrieval. Combines:
      * vector (semantic) search over the original query + LLM rephrasings, and
      * BM25 keyword search over the same phrasings,
    then fuses the two ranked lists with Reciprocal Rank Fusion. Vector search
    handles 'same question, different words'; keyword search handles 'different
    vocabulary' (exact terms like variety codes, 'upland', 'seed rate') that the
    embedding can miss. Greatly improves recall when the answer IS in the docs."""
    queries = _expand_queries(query_text_en)
    pool = max(n_results * 2, 12)

    vector_ranked = _merge_vector_hits(catalogue, queries, pool)
    keyword_ranked = _merge_keyword_hits(catalogue, queries, pool)

    # Reciprocal Rank Fusion: score = sum 1/(k + rank) across both lists.
    fused, holder = {}, {}
    for ranked in (vector_ranked, keyword_ranked):
        for rank, hit in enumerate(ranked):
            key = _hit_key(hit)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            # prefer the vector hit (it carries a distance) when both exist
            if key not in holder or (holder[key].get("distance") is None
                                     and hit.get("distance") is not None):
                holder[key] = hit

    if not fused:                       # both empty -> fall back to plain vector
        return vector_ranked[:n_results]

    ranked_keys = sorted(fused, key=lambda k: fused[k], reverse=True)
    return [holder[k] for k in ranked_keys[:n_results]]


def _trace_retrieved_chunks(hits: list) -> list:
    chunks = []
    for position, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        chunk_text = (hit.get("chunk") or "").strip()
        chunks.append({
            "rank": position,
            "document_id": meta.get("document_id"),
            "document_title": meta.get("document_title") or "Untitled document",
            "chunk_index": meta.get("chunk_index"),
            "category": meta.get("category") or "",
            "subcategory": meta.get("subcategory") or "",
            "score": hit.get("score"),
            "distance": hit.get("distance"),
            "chunk": chunk_text[:1600],
            "truncated": len(chunk_text) > 1600,
        })
    return chunks


FARMER_PERSONA = (
    "You are PaddyMitra, talking to a small rice farmer who may have little schooling. "
    "Talk like a helpful neighbour, not a scientist. Give simple, do-this advice.\n"
    "Rules for your answer:\n"
    " - Keep it SHORT: 2-4 short sentences of flowing prose.\n"
    " - Write PLAIN SENTENCES ONLY. Do NOT use bullet points, numbered lists, dashes, "
    "asterisks, headings, or any markdown/symbols — the answer is read aloud, so those "
    "get mispronounced or change the meaning. Separate ideas with full stops, not bullets.\n"
    " - Plain everyday words only (words a 10-year-old knows). One idea per sentence; "
    "sentences about 8-12 words. Active voice, direct commands (\"Do this. Put 2 bags.\").\n"
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
    " - If the needed detail is missing, say so plainly and ask ONE simple question."
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
    "Do not convert between units (e.g. do not change tons/hectare to kg/acre, "
    "do not change kg/ha to lb/acre, do not change litres to gallons).\n"
    " - Keep the same unit string the source used (e.g. \"5-6 tons per hectare\", "
    "\"120 kg N/ha\", \"40 grams per litre\").\n"
    " - Do not round, average, or rephrase numbers. If the source says \"5-6 tons/ha\", "
    "write \"5-6 tons/ha\" — not \"about 5 tons\" and not \"2000-2400 kg/acre\".\n"
    " - If the question asks for a unit not present in the CONTEXT, say so plainly "
    "instead of converting.\n"
)


def generate_answer(query_text_en: str, retrieved_hits: list, user_type: str = "farmer") -> str:
    context = "\n\n".join(
        f"[Source: {h['metadata'].get('document_title', 'doc')}]\n{h['chunk']}"
        for h in retrieved_hits
    )
    persona = RESEARCHER_PERSONA if (user_type or "").lower() == "researcher" else FARMER_PERSONA
    sys = (
        persona
        + UNIT_FIDELITY_RULES
        + "\n\nGROUNDING RULES (MANDATORY):\n"

        # ==========================
        # SOURCE OF TRUTH
        # ==========================
        + "- Answer ONLY from the CONTEXT below. Do NOT use outside knowledge.\n"
        + "- Every factual statement must be supported by the retrieved CONTEXT.\n"
        + "- Never invent fertilizer names, chemical names, varieties, diseases, timings, dosages, or practices.\n"

        # ==========================
        # MULTI-CHUNK REASONING
        # ==========================
        + "- Read ALL retrieved chunks before answering.\n"
        + "- The complete answer may be spread across multiple chunks.\n"
        + "- Combine related information from multiple chunks into one complete answer.\n"
        + "- If one chunk identifies the crop, variety, disease, stage, or condition and another chunk contains the recommendation, combine them.\n"
        + "- If different chunks provide dosage, timing, and application method separately, merge them into one coherent answer.\n"
        + "- Do NOT assume information is missing simply because it is absent from one chunk.\n"
        + "- Only conclude that information is unavailable after considering ALL retrieved chunks together.\n"

        # ==========================
        # UNDERSTAND FARMER LANGUAGE
        # ==========================
        + "- Farmers may use different words than the documents. Match the INTENT of the question rather than identical wording.\n"
        + "- Treat equivalent agricultural terms as referring to the same concept whenever the CONTEXT supports it.\n"
        + "- Examples include:\n"
        + "    manure ≈ fertilizer recommendation\n"
        + "    fertilization ≈ nutrient application\n"
        + "    top dressing ≈ fertilizer application\n"
        + "    basal dose ≈ basal fertilizer\n"
        + "    plant food ≈ fertilizer\n"
        + "    medicine ≈ pesticide or fungicide when supported by the CONTEXT\n"

        # ==========================
        # SPECIFIC vs GENERAL
        # ==========================
        + "- Prefer variety-specific recommendations over general recommendations.\n"
        + "- If no variety-specific recommendation exists but a general recommendation for the same crop exists, use the general recommendation and clearly state that it is the general recommendation.\n"
        + "- Prefer crop-stage-specific recommendations over general recommendations.\n"
        + "- Prefer disease-specific recommendations over general recommendations.\n"

        # ==========================
        # CONFLICT HANDLING
        # ==========================
        + "- If two chunks conflict, prefer the more specific recommendation.\n"
        + "- If both recommendations are equally authoritative, mention both instead of guessing.\n"

        # ==========================
        # NUMERIC FIDELITY
        # ==========================
        + "- Copy all numbers, units, percentages, timings, and dosages exactly from the CONTEXT.\n"
        + "- Never calculate, estimate, round, average, or convert units.\n"

        # ==========================
        # SAFETY
        # ==========================
        + "- Name a fertilizer, pesticide, fungicide, herbicide, or chemical ONLY if it appears explicitly in the CONTEXT.\n"
        + "- Never claim a fertilizer contains a nutrient unless the CONTEXT explicitly states it.\n"
        + "- Never invent product names or active ingredients.\n"

        # ==========================
        # IF INFORMATION IS MISSING
        # ==========================
        + "- If the retrieved CONTEXT collectively does not contain enough information to answer the question, clearly say what information is missing.\n"
        + "- Ask only ONE simple follow-up question when clarification is required.\n"
        + "- Do not ask unnecessary clarification questions if the retrieved CONTEXT already contains enough information to answer.\n"

        + "\nCONTEXT:\n"
        + context
    )
    return _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": query_text_en},
    ])


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

def run_pipeline_for_text(
    catalogue, farmer_language: str, query_text: str,
    district: str = "", generate_tts: bool = False, hook=None,
    whatsapp_number: str = "", user_type: str = "farmer",
    initial_trace: dict | None = None,
) -> Query:
    """Full pipeline entry point for a text-only query (demo form or webhook)."""
    # Start trace from initial_trace (e.g. STT backend info) if provided
    trace = (initial_trace.copy() if isinstance(initial_trace, dict) else {})
    if "steps" not in trace:
        trace.setdefault("steps", [])
    trace["user_type"] = user_type
    start = time.monotonic()
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

        trace["steps"].append({"stage": "translate_in", "ms": int((time.monotonic() - t0) * 1000)})
        q.translated_query_text = query_en

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

        # 3. retrieve
        t0 = time.monotonic()
        hits = retrieve(catalogue, query_en, n_results=10)
        trace["steps"].append({
            "stage": "retrieve",
            "ms": int((time.monotonic() - t0) * 1000),
            "hits": len(hits),
        })
        trace["retrieved_chunks"] = _trace_retrieved_chunks(hits)

        # 4. generate  (persona: farmer | researcher)
        t0 = time.monotonic()
        answer_en = generate_answer(query_en, hits, user_type=user_type)
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

    if msg_type == "audio":
        url = payload.get("audio_url", "")
        if not url:
            raise ValueError("audio_url missing")
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            tmp.write(r.content)
            tmp_path = tmp.name
        try:
            query_text, stt_backend = stt_transcribe(tmp_path, language_hint=lang)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

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
    )
