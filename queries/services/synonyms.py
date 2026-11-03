"""
synonyms.py
===========
QUESTION-LEVEL synonym layer: many ways of saying a thing -> ONE canonical term.

WHY THIS EXISTS
---------------
The glossary is a TRANSLATION map (english <-> odia, one value per key). It has
no way to say "these six strings all mean the same thing". But that is exactly
what a farmer's question needs, because the same term arrives spelled a
different way almost every time:

    ଅଜୋଲା / ଅଜୋଲ୍ଲା / ଅଝୋଲା / azola / ajwala / azzola   ->   Azolla

Those variants come from three places, none of which we control: how the farmer
pronounces it, how the STT engine spells what it heard, and how the translator
romanises it. A single missed variant costs the whole answer — "Azolla" reaches
the documents, "ajwala" reaches nothing.

WHERE IT RUNS
-------------
Both ends of the question, because variants appear in both scripts:

    odia question  --canonicalise(odia)--> translate --canonicalise(latin)--> retrieval

Applying it AFTER translation matters as much as before: the translator invents
its own romanisation ("Ajola", "Azola") that no Odia-side rule can predict.

CANONICAL TERMS MUST EXIST IN THE CORPUS
----------------------------------------
A canonical term the documents never use anchors the query to nothing — that
mistake was made earlier in this pipeline's history and cost a whole answer.
`unverified_terms()` reports any canonical form missing from the indexed text;
check it after editing.

MANAGING THE LIST
-----------------
Stored in `synonyms.json` next to this file, same shape and edit workflow as
odia_glossary.json:

    {"Azolla": ["ଅଜୋଲା", "ଅଜୋଲ୍ଲା", "azola", "ajwala"], ...}

    upsert_synonym("Azolla", "ajola")   add one variant
    delete_synonym("Azolla", "ajola")   remove one variant
    get_synonyms()                      read-only copy
    reload_synonyms()                   re-read after an external edit

PUBLIC API
----------
    canonicalise(text)        -> (text, [canonical terms found])
    canonical_for(word)       -> canonical term or "" for a single word
    unverified_terms(corpus)  -> canonical terms absent from the corpus
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

from .technical_terms import normalize_odia

log = logging.getLogger("irri")

_WRITE_LOCK = threading.Lock()
_PATH = Path(__file__).parent / "synonyms.json"

# Odia block, used as a "letter" class for word boundaries. Odia script has no
# case and no \b that the regex engine understands, so a term glued to more Odia
# must not match — the same rule the reverse glossary learned the hard way.
_ODIA_RANGE = r"଀-୿"

# ...but the strict trailing boundary alone was TOO strict, and it silently cost
# whole answers. Odia glues its case endings straight onto the noun, so a farmer
# never says the bare term:
#
#     ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା      the stage            (matched)
#     ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥାରେ    AT the stage         (did not match — ରେ is Odia)
#
# The second form is the one people actually say, and it is what the reported
# question used. With no match the term was never canonicalised, retrieval had
# nothing to anchor on, and the model invented the stage name "jointing" — a
# word that appears NOWHERE in the indexed corpus and that no farmer uses.
# The same miss applied to every Odia entry here (ଜିଆ ଖତରେ, ଅଣୁସାରକୁ …).
#
# One optional suffix is therefore allowed after an Odia variant, exactly as
# `_ODIA_SUFFIX` does in odia_glossary.py. Longest-first inside the group so
# ଙ୍କୁ beats ଙ୍କ. The LEADING boundary stays strict — that is the one that stops
# a fragment inside a compound from matching, and it is unaffected by this.
#
# The suffix is CONSUMED and dropped, not kept: the variant is being replaced by
# an English canonical term, and an English word cannot carry an Odia case
# ending. Leaving it would hand the translator "vermicompostରେ".
_ODIA_SUFFIX = (
    r"(?:ମାନଙ୍କୁ|ମାନଙ୍କର|ମାନଙ୍କ|ଗୁଡ଼ିକୁ|ଗୁଡ଼ିକର|ଗୁଡ଼ିକ|ଗୁଡ଼ାକ|ମାନେ"
    r"|ଙ୍କୁ|ଙ୍କର|ଙ୍କ|ରେ|ରୁ|କୁ|ଟି|ଟା|ର|େ)?"
)

# Seed list. Every canonical form here was checked against the indexed corpus.
# Variants are the spellings actually seen from STT, from farmers, and from the
# translator's romanisation — add to these rather than starting new entries.
_BUILTIN: dict[str, list[str]] = {
    "Azolla": [
        "ଅଜୋଲା", "ଅଜୋଲ୍ଲା", "ଅଝୋଲା", "ଅଜଓଲା", "ଆଜୋଲା",
        "azola", "azzola", "ajola", "ajwala", "ajolla", "azolaa",
    ],
    "BGA": [
        "ବିଜିଏ", "ବି ଜି ଏ", "ନୀଳ ହରିତ ଶୈବାଳ", "ନୀଳ-ହରିତ ଶୈବାଳ", "ନୀଳ ସବୁଜ ଶୈବାଳ",
        "bga", "b g a", "blue green algae", "blue-green algae", "bluegreen algae",
    ],
    "Rhizobium": [
        "ରାଇଜୋବିୟମ୍", "ରାଇଜୋବିୟମ", "ରାଇଜୋବିଅମ", "ରିଜୋବିୟମ",
        "rhizobium", "rizobium", "rhizobiam",
    ],
    "Azotobacter": [
        "ଅଜୋଟୋବାକ୍ଟର", "ଅଜୋଟୋବ୍ୟାକ୍ଟର", "ଅଜଟୋବାକ୍ଟର",
        "azotobacter", "azotobactor", "ajotobacter",
    ],
    "Azospirillum": [
        "ଅଜୋସ୍ପିରିଲମ୍", "ଅଜୋସ୍ପିରିଲମ", "ଅଜୋସ୍ପାଇରିଲମ",
        "azospirillum", "azospirilum", "ajospirillum",
    ],
    "Sesbania": ["ସେସବାନିଆ", "ସେସବାନିଯା", "sesbania", "sesbenia"],
    "Dhanicha": ["ଧଇଞ୍ଚା", "ଧନିଚା", "ଧଣିଚା", "dhanicha", "dhaincha", "dhainchaa"],
    "Trichoderma": ["ଟ୍ରାଇକୋଡର୍ମା", "ଟ୍ରାଇକୋଡରମା", "trichoderma", "tricoderma"],
    "vermicompost": [
        "ଜିଆ ଖତ", "ଜିଆଖତ", "ଜିଆ ସାର", "ଜିଆସାର", "ଜିଆ ଖାଦ", "କେଞ୍ଚୁଆ ଖତ",
        "ବର୍ମିକମ୍ପୋଷ୍ଟ", "ଭର୍ମି କମ୍ପୋଷ୍ଟ",
        "vermi compost", "vermicompost", "vermicomposting", "jiakhata", "jia khata",
    ],
    "micronutrient": [
        "ଅଣୁସାର", "ଅଣୁ ସାର", "ଅଣୁ ପୋଷକ", "ଅଣୁପୋଷକ", "ଅଣୁପୁଷ୍ଟିକର ଉପାଦାନ",
        "micro nutrient", "micronutrients", "micro-nutrient",
    ],
    "lodging": [
        "ଫସଲ ପଡ଼ିଗଲା", "ଧାନ ପଡ଼ିଗଲା", "ଫସଲ ପଡ଼ିଛି", "ନଇଁପଡ଼ିଛି", "ଗିରିଗଲା",
        "lodged", "fell over", "fallen over", "crop fall", "flattened",
    ],
    # Rice-variety duration classes. The translator was rendering "ବିଳମ୍ବ କିସମ"
    # as the pseudo-name "Bilamb variety", which then flows into the LLM as an
    # unknown cultivar and gets nonsense advice. Canonicalise so retrieval and
    # the LLM see the SEMANTIC category the farmer meant.
    "late-maturing variety": [
        "ବିଳମ୍ବ କିସମ", "ବିଲାମ୍ବ କିସମ", "ବିଳମ୍ବ ଧାନ", "ବିଳମ୍ବ କିସମର ଧାନ",
        "ବିଳମ୍ବ ପାଚ୍ୟ କିସମ", "ଡେରି ପାଚ୍ୟ କିସମ", "late maturing", "late duration",
        "long duration variety",
    ],
    # ମଧ୍ୟମ କିସମ ଜମି IS A LAND TYPE, NOT A DURATION CLASS. ଜମି is land, and
    # "ମଧ୍ୟମ କିସମ" ("medium sort") in front of it qualifies the FIELD, not the
    # seed. Without this entry the shorter "ମଧ୍ୟମ କିସମ" below matched first and
    # a question about medium land reached the pipeline as "medium-maturing
    # variety" — a different agronomic question, answered confidently and
    # wrongly. _compiled() orders patterns longest-first, so the land reading
    # wins wherever ଜମି actually follows, and only there.
    "medium land": [
        "ମଧ୍ୟମ କିସମ ଜମି", "ମଧ୍ୟମ କିସମର ଜମି", "ମଧ୍ୟମ ଜମି", "ମଧ୍ୟମ ପ୍ରକାର ଜମି",
        "medium type land", "medium land",
    ],
    "medium-maturing variety": [
        "ମଧ୍ୟମ କିସମ", "ମଧ୍ୟମ ପାଚ୍ୟ କିସମ", "medium duration variety",
    ],
    "early-maturing variety": [
        "ଅଳ୍ପ କାଳୀନ କିସମ", "ଶିଘ୍ର ପାଚ୍ୟ କିସମ", "ଅଳ୍ପ କାଳୀନ ଧାନ",
        "short duration variety", "early duration",
    ],
    # The seedbed. A farmer says ବିହନତଳା / ବିହନ ତଳି — the bed the seedlings are
    # raised in before transplanting. With no entry the translator treated it as
    # a NAME and romanised it: "The flood has destroyed the Bihantala", a place
    # that does not exist. Retrieval then had nothing to match and the answer
    # drifted onto fertiliser rates instead of re-sowing a nursery.
    #
    # "nursery" is the corpus's own word (9 chunks; "seedbed" only 2), so that is
    # the canonical form. The bare ତଳି is deliberately NOT a variant: on its own
    # it means the SEEDLING ("୩୫ ଦିନର ତଳି" = 35-day-old seedlings), not the bed.
    "nursery": [
        "ବିହନତଳା", "ବିହନ ତଳା", "ବିହନତଳି", "ବିହନ ତଳି", "ବିହନତଳ", "ବିହନ ତଳ",
        "ତଳିଘର", "ତଳି ଘର", "ବିହନ ବେଡ୍", "seed bed", "seedbed", "bihantala",
    ],
    # "jointing stage" is not a term Odia farmers use. The translator was
    # coining it from "ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା" (internode elongation), which the
    # LLM then failed to answer for. Canonicalise to the farmer term so the
    # reverse translation goes back to what they said.
    "internode elongation stage": [
        "ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା", "ଗଣ୍ଠି ଛାଡ଼ିବା ଅବସ୍ଥା", "ଗଣ୍ଠି ବାହାରିବା ଅବସ୍ଥା",
        "jointing", "jointing stage", "stem elongation", "stem elongation stage",
        "internode elongation",
    ],
}


# ── Storage ───────────────────────────────────────────────────────────────

def _read_disk() -> dict:
    if _PATH.exists():
        try:
            d = json.loads(_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception as exc:
            log.warning("synonyms: read failed (%s)", exc)
    return {}


def _atomic_write(data: dict) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, _PATH)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


@lru_cache(maxsize=1)
def _load() -> dict[str, list[str]]:
    """Disk entries merged over the built-in seed; disk wins per canonical term."""
    merged = {k: list(v) for k, v in _BUILTIN.items()}
    for term, variants in _read_disk().items():
        if isinstance(variants, list):
            merged[str(term)] = [str(v) for v in variants]
    return merged


def reload_synonyms() -> None:
    _load.cache_clear()
    _compiled.cache_clear()
    _load()


# ── Matching ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _compiled() -> list[tuple[re.Pattern, str]]:
    """(pattern, canonical) ordered LONGEST-FIRST.

    Longest-first so "blue green algae" wins over a shorter entry inside it, and
    so "ଜିଆ ଖତ" is taken before a bare "ଜିଆ" could be. Whitespace is flexible
    (`\\s*`) because Odia compounds are written spaced and unspaced
    interchangeably, and Latin variants get case-insensitive matching.
    """
    pairs: list[tuple[str, str, int]] = []
    for canonical, variants in _load().items():
        for v in variants:
            nv = normalize_odia(str(v).strip())
            if len(nv) < 3:
                continue          # too short to match safely
            body = r"\s*".join(re.escape(w) for w in nv.split())
            if re.search(r"[A-Za-z]", nv):
                rx = re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)
            else:
                # The suffix list is normalised too. Its ଗୁଡ଼ିକ / ଗୁଡ଼ାକ entries
                # carry a nukta, `body` and the haystack are both NFD, and a
                # precomposed alternative can never match decomposed text — the
                # plural forms would have been dead on arrival.
                rx = re.compile(
                    rf"(?<![{_ODIA_RANGE}]){body}{normalize_odia(_ODIA_SUFFIX)}"
                    rf"(?![{_ODIA_RANGE}])"
                )
            pairs.append((rx.pattern, canonical, len(nv)))
    pairs.sort(key=lambda p: p[2], reverse=True)

    out: list[tuple[re.Pattern, str]] = []
    for pattern, canonical, _ in pairs:
        flags = re.IGNORECASE if re.search(r"[A-Za-z]", pattern) else 0
        out.append((re.compile(pattern, flags), canonical))
    return out


def canonicalise(text: str) -> tuple[str, list[str]]:
    """Rewrite every known variant in `text` to its canonical term.

    Works on Odia script and on Latin, so the same call is useful before
    translation and after it. Returns `(text, canonical_terms_found)`; the term
    list is what the caller pins into the prompt and re-asserts afterwards.
    """
    if not (text or "").strip():
        return text or "", []
    out = normalize_odia(text)
    found: list[str] = []
    for rx, canonical in _compiled():
        if not rx.search(out):
            continue
        # Already canonical: record it, do not rewrite.
        if out.lower() != rx.sub(canonical, out).lower() or canonical.lower() not in out.lower():
            out = rx.sub(canonical, out)
        if canonical not in found:
            found.append(canonical)
    if found:
        log.info("synonyms: canonicalised %s", found)
    return out, found


def canonical_for(word: str) -> str:
    """Canonical term for a single word, or "" if it is not a known variant."""
    w = normalize_odia((word or "").strip())
    if not w:
        return ""
    for rx, canonical in _compiled():
        if rx.fullmatch(w):
            return canonical
    return ""


# ── Editing (mirrors odia_glossary / odia_translit) ───────────────────────

def upsert_synonym(canonical: str, variant: str) -> bool:
    """Add ONE variant to a canonical term. Creates the term if new."""
    c = (canonical or "").strip()
    v = (variant or "").strip()
    if not c or not v or c.lower() == v.lower() or len(v) < 3:
        return False
    with _WRITE_LOCK:
        data = _read_disk() or {k: list(x) for k, x in _BUILTIN.items()}
        variants = list(data.get(c) or _BUILTIN.get(c) or [])
        if any(v.lower() == e.lower() for e in variants):
            return False
        variants.append(v)
        data[c] = variants
        _atomic_write(data)
    reload_synonyms()
    log.info("synonyms: %r -> %r added", v, c)
    return True


def delete_synonym(canonical: str, variant: str) -> bool:
    """Remove ONE variant from a canonical term."""
    c = (canonical or "").strip()
    v = (variant or "").strip()
    with _WRITE_LOCK:
        data = _read_disk() or {k: list(x) for k, x in _BUILTIN.items()}
        variants = list(data.get(c) or _BUILTIN.get(c) or [])
        keep = [e for e in variants if e.lower() != v.lower()]
        if len(keep) == len(variants):
            return False
        data[c] = keep
        _atomic_write(data)
    reload_synonyms()
    return True


def get_synonyms() -> dict[str, list[str]]:
    """Read-only copy of the merged map."""
    return {k: list(v) for k, v in _load().items()}


def unverified_terms(corpus_text: str) -> list[str]:
    """Canonical terms that do NOT appear in the indexed corpus.

    A canonical term the documents never use anchors the query to nothing, so
    run this after editing and fix or remove whatever it reports.
    """
    low = (corpus_text or "").lower()
    return sorted(t for t in _load() if t.lower() not in low)
