"""
odia_translit.py
================
Deterministic Odia-transliteration -> canonical-Odia correction layer.

WHY THIS EXISTS
---------------
The English->Odia glossary (odia_glossary.py) can only enforce a term while it is
still written in LATIN script: apply_glossary() matches English keys. But the
translator frequently writes a term PHONETICALLY in Odia script instead of
translating it (e.g. tillering -> ଟିଲେରିଂ, fertilizer -> ଫର୍ଟିଲାଇଜର). Once that
happens the English-keyed glossary can no longer touch it, so the same term is
"fixed" in some answers and left transliterated in others.

This module closes that gap deterministically. It maps the Odia TRANSLITERATION
back to the proper Odia TERM, with:

  1. a fuzzy REGEX layer for the few high-frequency terms whose transliteration
     has many spelling variants (fertilizer / tillering / urea), and
  2. a literal MAP layer (odia_translit_map.json) for the long tail — one entry
     per known transliteration, applied longest-first in a single pass.

Runtime is pure string work: NO LLM call. The map is grown offline/by humans
(see the build_translit_map management command and upsert_translit()), mirroring
the odia_glossary draft-review workflow.

PUBLIC API
----------
    apply_translit_fixes(text)     -> normalise transliterations to canonical Odia
    upsert_translit(bad, good)     -> add/overwrite one entry (human edit)
    delete_translit(bad)           -> remove one entry
    get_translit_map()             -> read-only copy of the merged map
    reload_translit()              -> re-read the JSON after an external edit
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from .technical_terms import compose_odia, normalize_odia

log = logging.getLogger("irri")

_WRITE_LOCK = threading.Lock()
_MAP_PATH = Path(__file__).parent / "odia_translit_map.json"

# ── Encoding normalisation ────────────────────────────────────────────────
# Every pattern in this module is built from `normalize_odia()` (NFD) and every
# haystack is normalised the same way before matching, then composed back with
# `compose_odia()` (NFC) on the way out.
#
# This is not theoretical. The entries added for the reported mispronunciations
# (କିଲୋଗ୍ରାମ, ପୋଟାସିୟମ, ପୋକଜୋକ) were written PRECOMPOSED in
# odia_tts_spoken.json, while the text they were matched against arrived
# DECOMPOSED from `to_spoken()`. Not one of them ever fired, so the audio kept
# saying କିଲେଗ୍ରାମ and the JSON looked like it had no effect. Normalising both
# sides makes a hand-written entry work whichever way it was typed.


def _rx(pattern: str) -> re.Pattern:
    """Compile a pattern against the normal form the haystacks use."""
    return re.compile(normalize_odia(pattern))


def _entries(data: dict) -> dict[str, str]:
    """JSON entries only. A key starting with "_" is a note to whoever edits the
    file (JSON has no comments) and must never become a substitution rule."""
    return {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}


# ── Fuzzy layer ───────────────────────────────────────────────────────────
# A handful of English words are transliterated in MANY spelling variants, so a
# literal map entry per variant is impractical. These regexes catch them all.
# Applied BEFORE the literal map.
_FUZZY_FIXES: list[tuple[re.Pattern, str]] = [
    (_rx(r"ଇଉରିଆ"), "ୟୁରିଆ"),                                   # urea spelling fix
    (_rx(r"ଫର୍?ଟିଲ[ାି]?ଇ?ଜର୍?(?:ସ|ର୍ସ)?"), "ସାର"),             # fertilizer(s)
    (_rx(r"ଟିଲ(?:୍ଲ)?େ?ର[ିୀ](?:ଂ|ଙ୍ଗ୍?)"), "କୁଆଁରା ବାହାରିବା"),   # tillering
    # Muriate of Potash. Every part of this name gets spelled a different way each
    # time, and the wrong spellings were coming out of the GLOSSARY itself, not
    # the translator:
    #     muriate of potash -> ମୁରିଆଟ ଅଫ୍ ପୋଟାସ୍      (glossary value)
    #     murate of potash  -> ମ୍ୟୁରେଟ ଅଫ ପଟାସ        (glossary value)
    # Correct is ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ. Fixed here as well as in the glossary data,
    # because this pass runs over the answer AFTER the glossary term is restored,
    # so it holds whatever the deployed odia_glossary.json happens to say.
    # Matched as a whole phrase on purpose: bare ପୋଟାସ is the start of
    # ପୋଟାସିୟମ (potassium), which must not be touched.
    (_rx(r"ମ(?:୍ୟ)?ୁରି?(?:ଏ|ଆ|େ)?ଟ୍?\s*(?:ଅ|ଓ)ଫ୍?\s*ପ(?:ୋ)?ଟା(?:ଶ|ସ)୍?"),
     "ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ"),
]

# ── Literal map: built-in seed (used if the JSON file is missing) ──────────
# Keep seed entries SAFE and unambiguous: either a clear transliteration ->
# translation, or a trailing-halant cleanup. Grow the rest through the
# reviewed draft / web editor, never by guessing here.
_BUILTIN_SEED: dict[str, str] = {
    # transliteration -> real Odia word
    "ୱାଟର": "ପାଣି",            # water
    "ଵାଟର": "ପାଣି",            # water (variant)
    # trailing-halant cleanup (display consistency; meaning unchanged)
    "ଫସଫରସ୍": "ଫସଫରସ",
    "ପୋଟାସିୟମ୍": "ପୋଟାସିୟମ",
    "ନାଇଟ୍ରୋଜେନ୍": "ନାଇଟ୍ରୋଜେନ",
}


# ── Loader ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_map() -> dict[str, str]:
    """Load the literal map from JSON, merged over the built-in seed.

    Keys are NFD-normalised so an entry matches whichever encoding the text is
    in; values are left as authored and composed at the end of the pass.
    """
    merged = dict(_BUILTIN_SEED)
    if _MAP_PATH.exists():
        try:
            data = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                log.info("odia_translit: loaded %d entries from %s", len(data), _MAP_PATH)
                merged.update(_entries(data))
        except Exception as exc:
            log.error("odia_translit: JSON parse FAILED (%s); using built-in seed.", exc)
    return {normalize_odia(k): v for k, v in merged.items()}


@lru_cache(maxsize=1)
def _compiled_map() -> re.Pattern | None:
    """One alternation over every literal key, longest-first so the longest
    transliteration wins (e.g. a phrase before its sub-word)."""
    keys = [k for k in _load_map() if k]
    if not keys:
        return None
    keys.sort(key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in keys))


# ── TTS-only spoken forms ──────────────────────────────────────────────────
# These are how a term should be SPOKEN by the TTS voice, which differs from how
# it is DISPLAYED: on screen "DAP" stays "DAP", but the Odia voice must say
# "ଡି ଏ ପି ସାର". Because it differs from the glossary/display value it lives in
# its own file (odia_tts_spoken.json) and is applied ONLY on the TTS path — never
# on the on-screen answer.
_TTS_SPOKEN_PATH = Path(__file__).parent / "odia_tts_spoken.json"

_BUILTIN_TTS_SPOKEN: dict[str, str] = {
    "DAP": "ଡି ଏ ପି ସାର", "MOP": "ଏମ ଓ ପି ସାର", "NPK": "ଏନ ପି କେ ସାର",
    "SSP": "ଏସ ଏସ ପି ସାର", "Urea": "ୟୁରିଆ ସାର", "urea": "ୟୁରିଆ ସାର",
    "kg/ha": "କିଲୋଗ୍ରାମ ପ୍ରତି ହେକ୍ଟର", "g/L": "ଗ୍ରାମ ପ୍ରତି ଲିଟର",
    "ନାଇଟ୍ରୋଜେନ୍": "ଯବକ୍ଷାରଜାନ", "ନାଇଟ୍ରୋଜେନ": "ଯବକ୍ଷାରଜାନ",
}


@lru_cache(maxsize=1)
def _load_tts_spoken() -> dict[str, str]:
    merged = dict(_BUILTIN_TTS_SPOKEN)
    if _TTS_SPOKEN_PATH.exists():
        try:
            data = json.loads(_TTS_SPOKEN_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                merged.update(_entries(data))
                # Deliberately loud so the "did my JSON reach the server?" question
                # can be answered from the logs — count + a couple sample keys.
                sample = ", ".join(list(_entries(data))[:3])
                log.info("odia_tts_spoken: loaded %d entries from %s (sample: %s)",
                         len(data), _TTS_SPOKEN_PATH, sample)
            return {normalize_odia(k): v for k, v in merged.items()}
        except Exception as exc:
            log.error("odia_tts_spoken: JSON parse FAILED (%s); using built-in.", exc)
    log.warning("odia_tts_spoken: file missing or empty at %s — using %d built-in entries only",
                _TTS_SPOKEN_PATH, len(_BUILTIN_TTS_SPOKEN))
    return {normalize_odia(k): v for k, v in merged.items()}


@lru_cache(maxsize=1)
def _compiled_tts_spoken() -> re.Pattern | None:
    keys = [k for k in _load_tts_spoken() if k]
    if not keys:
        return None
    keys.sort(key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in keys))


def apply_tts_spoken(text: str) -> str:
    """Apply TTS-ONLY spoken replacements (e.g. 'DAP' -> 'ଡି ଏ ପି ସାର'). Managed in
    odia_tts_spoken.json. NEVER call this on the on-screen text."""
    if not text:
        return text
    pattern = _compiled_tts_spoken()
    if pattern is None:
        return compose_odia(text)
    mapping = _load_tts_spoken()
    hits: list[str] = []
    def _sub(m: re.Match) -> str:
        key = m.group(0)
        hits.append(key)
        return mapping.get(key, key)
    out = pattern.sub(_sub, normalize_odia(text))
    if hits:
        log.info("odia_tts_spoken: %d substitution(s): %s",
                 len(hits), ", ".join(sorted(set(hits))))
    return compose_odia(out)


def reload_translit() -> None:
    """Re-read the JSON files after an external edit."""
    _load_map.cache_clear()
    _compiled_map.cache_clear()
    _load_tts_spoken.cache_clear()
    _compiled_tts_spoken.cache_clear()
    _load_map()


# ── Public: apply ─────────────────────────────────────────────────────────

def apply_translit_fixes(text: str) -> str:
    """Replace Odia-script transliterations of English words with the real Odia
    term. Fuzzy regexes first, then the literal map in a single longest-first
    pass. Deterministic and cheap — safe to call on both display and TTS text."""
    if not text:
        return text
    text = normalize_odia(text)
    for rx, repl in _FUZZY_FIXES:
        text = rx.sub(repl, text)
    pattern = _compiled_map()
    if pattern is not None:
        mapping = _load_map()
        text = pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
    return compose_odia(text)


# ── Persistence helpers (mirror odia_glossary) ────────────────────────────

def _atomic_write(data: dict) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(_MAP_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, _MAP_PATH)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _read_disk() -> dict:
    if _MAP_PATH.exists():
        try:
            d = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception as exc:
            log.warning("odia_translit: read failed (%s)", exc)
    return {}


def add_translit_fixes(mapping: dict[str, str]) -> dict[str, str]:
    """Persist new transliteration -> canonical Odia entries (existing keys are
    kept; first correction wins). Rejects entries whose canonical value still
    contains Latin letters, or where key == value. Returns what was newly added."""
    clean = {
        str(k).strip(): str(v).strip()
        for k, v in mapping.items()
        if str(k).strip() and str(v).strip()
        and str(k).strip() != str(v).strip()
        and not re.search(r"[A-Za-z]", str(v))
    }
    if not clean:
        return {}
    with _WRITE_LOCK:
        existing = _read_disk()
        added = {k: v for k, v in clean.items() if k not in existing}
        if not added:
            return {}
        existing.update(added)
        _atomic_write(existing)
        log.info("odia_translit: learned %d new fix(es): %s", len(added), list(added))
    reload_translit()
    return added


def upsert_translit(bad: str, good: str) -> bool:
    """Add or overwrite ONE transliteration fix (human edit wins)."""
    k = (bad or "").strip()
    v = (good or "").strip()
    if not k or not v or k == v or re.search(r"[A-Za-z]", v):
        return False
    with _WRITE_LOCK:
        data = _read_disk()
        data[k] = v
        _atomic_write(data)
    reload_translit()
    return True


def delete_translit(bad: str) -> bool:
    """Remove ONE transliteration fix. Returns True if it existed."""
    k = (bad or "").strip()
    with _WRITE_LOCK:
        data = _read_disk()
        if k not in data:
            return False
        data.pop(k)
        _atomic_write(data)
    reload_translit()
    return True


def get_translit_map() -> dict[str, str]:
    """Read-only copy of the merged map (seed + JSON)."""
    return dict(_load_map())
