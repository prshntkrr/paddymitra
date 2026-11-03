"""
technical_terms.py
==================
Protects TECHNICAL TOKENS — acronyms, genus / scientific names, product names —
from being translated or transliterated in EITHER direction.

WHY THIS EXISTS
---------------
A technical token is the one word in a question that decides whether retrieval
succeeds. The corpus says "BGA" 10 times and "Blue Green Algae" zero times, so
"BGA" is the only string that retrieves the answer. But the token is fragile at
both ends of the pipeline:

  INPUT  (farmer's Odia question -> English, for retrieval)
      The farmer says "BGA"; STT writes the Odia LETTER NAMES (ବିଜିଏ). There is
      no glossary entry for that, so the translator free-translates it into
      "Biji" / "blue green algae" / nothing at all. Retrieval then misses a
      chunk that plainly contains the answer, and an ANSWERABLE question is
      silently reported as "no information available".

  OUTPUT (English answer -> Odia)
      "BGA" survives translation as Latin text, then the Latin-script guard in
      pipeline.translate() sees an unknown Latin word and asks an LLM to
      transliterate it into Odia script — destroying the acronym in the answer
      the farmer hears.

Both failures are fixed by pinning a small, corpus-grounded set of tokens:

    reverse (input)   Odia letter-names / Odia spelling  ->  canonical token
    forward (output)  canonical token is ALLOWED to stay in Latin script

NUKTA NORMALISATION
-------------------
Odia nukta letters have two valid encodings and Unicode NFC does NOT unify them
(they are composition exclusions):

    NFC("ଡ" + "଼") != "ଡ଼"        NFD("ଡ଼") == "ଡ" + "଼"

Which form arrives depends on which STT backend ran (Sarvam / Wav2Vec2 /
Whisper), so a raw substring test matches for one backend and silently fails for
another — the "correct in some places, wrong in others" symptom. Every
comparison in this module goes through `normalize_odia()` (NFD), which is the
only form that unifies both spellings.

GROWING THE LIST
----------------
`_CORPUS_ACRONYMS` and `_SCIENTIFIC_NAMES` are grounded in the indexed corpus,
not guessed. To extend, confirm the token actually appears in the documents
(that is what makes it a useful retrieval key) and add it here.

PUBLIC API
----------
    normalize_odia(text)              -> NFD-normalised text (for MATCHING)
    compose_odia(text)                -> NFC-composed text (for SCREEN / TTS)
    protect_input_terms(odia_text)    -> (text_with_canonical_tokens, found)
    reassert_tokens(english, found)    -> restore tokens the LLM dropped
    allowed_latin_tokens()            -> set for the English->Odia Latin guard
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

log = logging.getLogger("irri")


# ── Unicode normalisation ─────────────────────────────────────────────────

def normalize_odia(text: str) -> str:
    """NFD-normalise Odia text so both nukta encodings compare equal.

    NFD is required, not NFC: ଡ଼ / ଢ଼ / ୟ are Unicode composition exclusions, so
    NFC leaves a decomposed sequence decomposed and the two spellings of the
    same letter stay unequal. NFD always splits them into base + U+0B3C.

    MATCHING ONLY. NFD text must never reach the screen or the TTS engine —
    see `compose_odia()` for why.
    """
    return unicodedata.normalize("NFD", text or "")


def compose_odia(text: str) -> str:
    """NFC-compose Odia text for OUTPUT (the on-screen answer and the TTS text).

    WHY THIS EXISTS — the mispronounced-vowel bug
    ---------------------------------------------
    `normalize_odia()` above is the right form for COMPARING, but it also
    decomposes the Odia vowel signs, which have nothing to do with nukta:

        ୋ  U+0B4B  ->  େ U+0B47  +  ା U+0B3E     (O)
        ୌ  U+0B4C  ->  େ U+0B47  +  ୗ U+0B57     (AU)
        ୈ  U+0B48  ->  େ U+0B47  +  ୖ U+0B56     (AI)

    Every one of them starts with େ (E). `to_spoken()` returns NFD text, so the
    ANSWER itself was leaving the pipeline decomposed and Narakeet — which reads
    the first vowel sign and drops the trailing one — pronounced the O as an E.
    That is exactly what farmers reported, and always on ୋ words:

        କିଲୋଗ୍ରାମ  heard as  କିଲେଗ୍ରାମ
        ପୋଟାସିୟମ   heard as  ପେଟାସିୟମ
        ପୋକଜୋକ     heard as  ପେକଜୋକ

    The same decomposition is why the hand-written entries for those three words
    in odia_tts_spoken.json never fired: the keys are precomposed and the text
    they were matched against was not.

    NUKTA IS DELIBERATELY LEFT ALONE. NFC does not recompose ଡ଼ / ଢ଼ (composition
    exclusions), so this call changes nothing about them — the decomposed nukta
    spelling is what the pipeline has always emitted and it is pronounced
    correctly. Only the vowel signs are repaired, which is the reported bug.
    """
    return unicodedata.normalize("NFC", text or "")


# ── Odia letter names, for reading out an acronym ─────────────────────────
# An acronym dictated in Odia arrives as the Odia spelling of each English
# letter name. Several letters have more than one common spelling (with or
# without the trailing halant, ଏ vs ଆ for "A"), so each letter lists variants.
_LETTER_NAMES: dict[str, tuple[str, ...]] = {
    "a": ("ଏ", "ଆ"),
    "b": ("ବି",),
    "c": ("ସି",),
    "d": ("ଡି", "ଡ଼ି"),
    "e": ("ଇ", "ଈ"),
    "f": ("ଏଫ୍", "ଏଫ"),
    "g": ("ଜି", "ଗି"),
    "h": ("ଏଚ୍", "ଏଚ"),
    "i": ("ଆଇ",),
    "j": ("ଜେ",),
    "k": ("କେ",),
    "l": ("ଏଲ୍", "ଏଲ"),
    "m": ("ଏମ୍", "ଏମ"),
    "n": ("ଏନ୍", "ଏନ"),
    "o": ("ଓ",),
    "p": ("ପି",),
    "q": ("କ୍ୟୁ",),
    "r": ("ଆର୍", "ଆର"),
    "s": ("ଏସ୍", "ଏସ"),
    "t": ("ଟି",),
    "u": ("ୟୁ", "ଉ"),
    "v": ("ଭି",),
    "w": ("ଡବଲୁ", "ଡବ୍ଲୁ"),
    "x": ("ଏକ୍ସ",),
    "y": ("ୱାଇ", "ଓ୍ୱାଇ", "ୱାଏ"),
    "z": ("ଜେଡ୍", "ଜେଡ", "ଜେଟ"),
    "0": ("ଶୂନ୍ୟ", "0", "୦"),
    "1": ("ଏକ", "1", "୧"),
    "2": ("ଦୁଇ", "2", "୨"),
    "3": ("ତିନି", "3", "୩"),
    "4": ("ଚାରି", "4", "୪"),
    "5": ("ପାଞ୍ଚ", "5", "୫"),
    "6": ("ଛଅ", "6", "୬"),
    "7": ("ସାତ", "7", "୭"),
    "8": ("ଆଠ", "8", "୮"),
    "9": ("ନଅ", "9", "୯"),
}

# Separators a speaker/STT may put between spelled-out letters: nothing, space,
# full stop, danda, hyphen, ZWNJ/ZWJ.
_SEP = r"[\s.।\-‌‍]*"

# Odia letters + the marks that bind to them. Used for a "not glued to another
# Odia word" boundary, so ପିଆଇ inside a longer native word does not match.
_ODIA_CHAR = r"଀-୿"


# ── Corpus-grounded token list ────────────────────────────────────────────
# Acronyms that appear in the INDEXED DOCUMENTS. The occurrence counts are the
# reason each one earns a slot: these are the strings retrieval needs to see.
#
#   token          corpus hits   what it is
_CORPUS_ACRONYMS: tuple[str, ...] = (
    "BGA",      # 10  blue-green algae biofertiliser  <- the reported failure
    "DAP",      # 26  di-ammonium phosphate
    "MOP",      # 37  muriate of potash
    "NPK",      # 20  nitrogen-phosphorus-potassium
    "SSP",      # 10  single super phosphate
    "FYM",      # 10  farmyard manure
    "DSR",      # 42  direct-seeded rice
    "SSNM",     # 12  site-specific nutrient management
    "LCC",      # 24  leaf colour chart
    "RCM",      #  6  rice crop manager
    "DAS",      #  6  days after sowing
    "PI",       # 10  panicle initiation (stage)
    "TL",       #  6  tillering (stage)
    "GHG",      #  3  greenhouse gas
    "IRRI",     # 70  institute
    "CRRI",     #  6  institute
    "NRRI",     #  4  institute
    "DRR",      #  8  institute
    "RKB",      # 36  rice knowledge bank
    "RKBO",     # 28  rice knowledge bank Odisha
)

# Variety-name prefixes. These sit next to a number (CR Dhan 7061, MTU 1010) and
# must survive verbatim or the variety cannot be looked up at all.
_VARIETY_PREFIXES: tuple[str, ...] = ("CR", "MTU", "MTR", "CRR")

# Genus / scientific / product names: canonical spelling -> Odia spellings seen
# in glossary values and in dictated speech. Unlike acronyms these are written
# as words, not letter-by-letter, so the variants are listed explicitly.
_SCIENTIFIC_NAMES: dict[str, tuple[str, ...]] = {
    "Azolla":       ("ଅଜୋଲା", "ଅଜୋଲ୍ଲା", "ଅଝୋଲା"),
    "Azotobacter":  ("ଅଜୋଟୋବାକ୍ଟର", "ଅଜୋଟୋବ୍ୟାକ୍ଟର"),
    "Azospirillum": ("ଅଜୋସ୍ପିରିଲମ୍", "ଅଜୋସ୍ପିରିଲମ"),
    "Rhizobium":    ("ରାଇଜୋବିୟମ୍", "ରାଇଜୋବିୟମ", "ରାଇଜୋବିଅମ"),
    "Sesbania":     ("ସେସବାନିଆ", "ସେସବାନିଯା"),
    "Dhanicha":     ("ଧଇଞ୍ଚା", "ଧନିଚା", "ଧଣିଚା"),
    "Gliricidia":   ("ଗ୍ଲିରିସିଡିଆ", "ଗ୍ଲିରିସିଡିଯା"),
}

# Multi-word Odia phrases that must retrieve as an ACRONYM rather than as their
# literal English translation. "Blue Green Algae" appears 0 times in the corpus
# while "BGA" appears 10 times, so translating the phrase literally guarantees a
# retrieval miss. Both forms are emitted: the acronym for the lexical match, the
# spelled-out name for the semantic/embedding match.
_PHRASE_TO_TOKEN: dict[str, str] = {
    "ନୀଳ ହରିତ ଶୈବାଳ": "BGA (blue-green algae)",
    "ନୀଳ-ହରିତ ଶୈବାଳ": "BGA (blue-green algae)",
    "ନୀଳ ସବୁଜ ଶୈବାଳ": "BGA (blue-green algae)",
}


# ── Pattern construction ──────────────────────────────────────────────────

def _acronym_pattern(token: str) -> str:
    """Regex matching `token` dictated as Odia letter names.

    'BGA' -> ବିଜିଏ, ବି ଜି ଏ, ବି.ଜି.ଏ. …  Every letter contributes its variant
    group; separators between letters are optional.
    """
    parts = []
    for ch in token.lower():
        variants = _LETTER_NAMES.get(ch)
        if not variants:
            return ""
        parts.append("(?:" + "|".join(re.escape(normalize_odia(v)) for v in variants) + ")")
    return _SEP.join(parts)


@lru_cache(maxsize=1)
def _compiled_rules() -> list[tuple[re.Pattern, str]]:
    """Build the ordered (pattern -> canonical token) rule list.

    Ordered LONGEST-FIRST so 'RKBO' wins over 'RKB' and a multi-word phrase wins
    over a single acronym inside it. All patterns are built from NFD-normalised
    text and are applied to NFD-normalised input.
    """
    rules: list[tuple[str, str, int]] = []   # (regex, replacement, sort length)

    # 1. Multi-word phrases -> acronym form (highest priority: longest strings).
    for phrase, token in _PHRASE_TO_TOKEN.items():
        norm = normalize_odia(phrase)
        # allow flexible whitespace / optional hyphen between the words
        flexible = _SEP.join(re.escape(w) for w in norm.split())
        rules.append((flexible, token, len(norm) + 1000))

    # 2. Scientific names, matched on their Odia spelling.
    for token, variants in _SCIENTIFIC_NAMES.items():
        for v in variants:
            norm = normalize_odia(v)
            rules.append((re.escape(norm), token, len(norm) + 500))

    # 3. Acronyms + variety prefixes, matched as dictated letter names.
    #    Guarded so the sequence is not glued to a surrounding Odia word.
    for token in _CORPUS_ACRONYMS + _VARIETY_PREFIXES:
        pat = _acronym_pattern(token)
        if not pat:
            log.warning("technical_terms: no letter-name spelling for %r", token)
            continue
        guarded = rf"(?<![{_ODIA_CHAR}])(?:{pat})(?![{_ODIA_CHAR}])"
        # weight by token length so RKBO outranks RKB
        rules.append((guarded, token, len(token)))

    rules.sort(key=lambda r: r[2], reverse=True)
    return [(re.compile(rx), repl) for rx, repl, _ in rules]


# ── Reverse direction: Odia question -> retrieval-ready English ───────────

# In a variety name, ଧାନ is a proper noun ("Dhan"), not the word for rice. Left
# in Odia script it is matched by the reverse glossary as ଧାନ -> "rice", which
# produced "CR rice 301" instead of "CR Dhan 301" — and that glossary hint
# CONTRADICTS the prompt rule telling the model to keep the name intact.
# Rewriting it to Latin here removes the contradiction at the source: once it is
# no longer Odia script, the glossary cannot match it at all.
# Also folds the trailing dot of a dictated "ସି.ଆର୍." into a clean "CR".
_VARIETY_DHAN_RE = re.compile(
    r"\b(" + "|".join(_VARIETY_PREFIXES) + r")\.?\s*" + normalize_odia("ଧାନ")
)

# Odia-script variety names written straight onto ଧାନ (ପୂଜାଧାନ = the Pooja
# variety). Left as one Odia blob, the translator glued the following number
# onto it — "ପୂଜାଧାନ ୫୫ ଦିନରେ" became the nonexistent variety "Pooja Dhan 55".
# Emitting "Pooja Dhan" in Latin separates the name from the number.
_ODIA_VARIETY_NAMES: dict[str, str] = {
    "ପୂଜା": "Pooja", "ପୁଜା": "Pooja",
    "ଲଳାଟ": "Lalat",
    "ନବୀନ": "Naveen", "ନଭୀନ": "Naveen",
    "ସ୍ୱର୍ଣ୍ଣା": "Swarna", "ସ୍ୱର୍ଣ୍ଣ": "Swarna", "ସ୍ୱର୍ଣ": "Swarna",
    "କଳାଚମ୍ପା": "Kalachampa",
    "ସହଭାଗୀ": "Sahbhagi",
    "ମନସ୍ୱରୀ": "Manaswari",
}
_ODIA_VARIETY_DHAN_RE = re.compile(
    "(" + "|".join(re.escape(normalize_odia(k)) for k in
                   sorted(_ODIA_VARIETY_NAMES, key=len, reverse=True))
    + r")\s*" + normalize_odia("ଧାନ")
)


# ── Crop-condition concepts ───────────────────────────────────────────────
# A farmer states the crop's problem as a VERB, and the glossary cannot help:
# glossary entries are noun phrases, and the subject is usually far from the
# verb ("ପୂଜାଧାନ ୫୫ ଦିନରେ ପଡ଼ିଗଲା"), so no adjacent-phrase entry ever matches.
#
# Pinning the concept HERE — before the translator runs — is what makes it
# reliable. Left to the LLM, the same question came back as "has lodged", "has
# lodging" and "fell over" on different runs; the corpus only ever writes
# "lodging" (9 chunks, and ZERO for "lodged"), so retrieval succeeded or failed
# depending on the wording the model happened to pick. Pinning removes the
# choice: the canonical term is inserted, the prompt is told to keep it, and
# reassert_tokens() restores it if the model drops it anyway.
#
# Every canonical term below is verified to appear in the indexed corpus — a
# term the documents never use would pin the query to nothing.
_CROP_NOUNS: tuple[str, ...] = (
    "ଧାନ", "ଫସଲ", "ଗଛ", "ଚାରା", "ବିହନ", "ପାଛିଆ", "Dhan",
)

# canonical English term -> Odia condition verbs / phrases
_CROP_CONDITIONS: dict[str, tuple[str, ...]] = {
    "lodging":  ("ପଡ଼ିଗଲା", "ପଡ଼ିଛି", "ପଡ଼ିଯାଇଛି", "ପଡ଼ିଯିବା", "ପଡ଼ିଲା",
                 "ନଇଁପଡ଼ିଛି", "ନଇଁ ପଡ଼ିଛି", "ନଇଁଗଲା", "ଶୋଇପଡ଼ିଛି",
                 "ଗିରିଗଲା", "ଉପୁଡ଼ିଗଲା", "ଅଛାଡ଼ି ହୋଇଗଲା"),
    "yellowing": ("ହଳଦିଆ ହୋଇଗଲା", "ହଳଦିଆ ପଡ଼ିଗଲା", "ହଳଦିଆ ହେଉଛି", "ପିଙ୍ଗଳା ହେଉଛି"),
    "drying":    ("ଶୁଖିଗଲା", "ଶୁଖିଯାଉଛି", "ଶୁଖିଯାଇଛି", "ଶୁଷ୍କ ହୋଇଗଲା"),
    "rot":       ("ପଚିଗଲା", "ପଚିଯାଉଛି", "ପଚିଯାଇଛି", "ସଢ଼ିଗଲା"),
    "stunted":   ("ବଢ଼ୁ ନାହିଁ", "ବଡ଼ ହେଉ ନାହିଁ", "ଛୋଟ ରହିଗଲା"),
}

# Words that, immediately before a condition verb, mean the subject is NOT the
# crop. "ବର୍ଷା ପଡ଼ିଗଲା" is "it rained", not a lodged crop.
_NON_CROP_SUBJECTS: tuple[str, ...] = ("ବର୍ଷା", "ପାଣି", "ଶିଳା", "ମେଘ", "ବରଫ", "କାନ୍ଥ")

_CROP_NOUN_RE = re.compile(
    "|".join(re.escape(normalize_odia(n)) for n in _CROP_NOUNS)
)


@lru_cache(maxsize=1)
def _condition_patterns() -> list[tuple[re.Pattern, str]]:
    out: list[tuple[re.Pattern, str]] = []
    for term, verbs in _CROP_CONDITIONS.items():
        for v in sorted(verbs, key=len, reverse=True):
            nv = normalize_odia(v)
            # capture what comes just before, to reject a non-crop subject
            rx = re.compile(r"(\S+)?\s*" + r"\s*".join(re.escape(w) for w in nv.split()))
            out.append((rx, term))
    return out


def detect_crop_conditions(odia_text: str) -> list[str]:
    """Canonical English terms for crop problems stated in `odia_text`.

    Fires when a crop noun appears ANYWHERE in the sentence and a condition verb
    is present — adjacency is not required, which is the whole point. Skips a
    verb whose immediate subject is rain/water so "ବର୍ଷା ପଡ଼ିଗଲା" stays weather.
    """
    text = normalize_odia(odia_text or "")
    if not _CROP_NOUN_RE.search(text):
        return []
    found: list[str] = []
    for rx, term in _condition_patterns():
        for m in rx.finditer(text):
            preceding = (m.group(1) or "").strip(" ,।")
            if any(normalize_odia(s) in preceding for s in _NON_CROP_SUBJECTS):
                continue
            if term not in found:
                found.append(term)
            break
    return found


def protect_input_terms(odia_text: str) -> tuple[str, list[str]]:
    """Replace Odia renderings of technical tokens with the canonical token.

    Runs BEFORE the translator so the LLM already sees `BGA` in Latin script and
    only has to leave it alone, instead of having to guess it from ବିଜିଏ.

    Returns `(text, found_tokens)`. `found_tokens` is handed to the prompt as a
    pin-list and to `reassert_tokens()` as a post-check.
    """
    if not (odia_text or "").strip():
        return odia_text or "", []

    text = normalize_odia(odia_text)
    found: list[str] = []

    for pattern, token in _compiled_rules():
        if not pattern.search(text):
            continue
        text = pattern.sub(token, text)
        # the phrase rules emit 'BGA (blue-green algae)'; pin the bare acronym
        bare = token.split(" ", 1)[0]
        if bare not in found:
            found.append(bare)

    # Runs AFTER the acronym pass, so the prefix is already Latin ("ସି.ଆର୍." has
    # become "CR" / "CR.") and can be matched here.
    text, n = _VARIETY_DHAN_RE.subn(r"\1 Dhan", text)
    if n and "Dhan" not in found:
        found.append("Dhan")

    def _odia_variety(m: re.Match) -> str:
        name = _ODIA_VARIETY_NAMES[m.group(1)]
        if name not in found:
            found.append(name)
        return f"{name} Dhan"

    text = _ODIA_VARIETY_DHAN_RE.sub(_odia_variety, text)

    # Crop problems are pinned as CONCEPTS, not substituted into the text: the
    # verb still has to be translated in place so the sentence reads naturally.
    # Appending the canonical term guarantees retrieval sees the corpus's word
    # regardless of which synonym the translator picks for the verb itself.
    for term in detect_crop_conditions(odia_text):
        if term not in found:
            found.append(term)

    if found:
        log.info("technical_terms: protected %s in the farmer's question", found)
    return text, found


def reassert_tokens(english_text: str, found: list[str]) -> str:
    """Append any pinned token the translator still dropped.

    The substitution above puts the token into the source text and the prompt
    pins it, but an LLM can still paraphrase it away. Retrieval only needs the
    token to be PRESENT, so appending it is enough to rescue the query — and is
    far cheaper than a repair round-trip.
    """
    if not found:
        return english_text
    missing = [t for t in found if t.lower() not in (english_text or "").lower()]
    if not missing:
        return english_text
    log.warning("technical_terms: translator dropped %s; re-asserting for retrieval", missing)
    return f"{english_text} ({', '.join(missing)})".strip()


def pin_list_for_prompt(found: list[str]) -> str:
    """Prompt fragment instructing the model to keep the tokens verbatim."""
    if not found:
        return ""
    return (
        "\nTECHNICAL TOKENS — the text already contains these in Latin script: "
        + ", ".join(found)
        + ". Reproduce each one EXACTLY as written, in Latin script. They are "
        "acronyms, institute names or scientific names. Never translate them, "
        "never spell them out, never write them in Odia script, never expand or "
        "abbreviate them."
    )


# ── Forward direction: keep the token Latin in the Odia answer ────────────

# Units and symbols that were already tolerated by the guard in translate().
_ALLOWED_UNITS: tuple[str, ...] = (
    "kg", "g", "mg", "ha", "ml", "L", "cm", "mm", "m", "pH", "ppm", "t",
)


@lru_cache(maxsize=1)
def allowed_latin_tokens() -> frozenset[str]:
    """Tokens allowed to remain in Latin script in an Odia answer.

    The Latin-script guard in translate() previously allowed only
    {DAP, MOP, NPK, kg, ha, pH}, so every other acronym in an answer — BGA, SSP,
    DSR, SSNM, LCC, FYM, IRRI, MTU — was sent to an LLM to be "fixed" into Odia
    script, destroying it. Comparison is case-insensitive at the call site.
    """
    tokens = set(_CORPUS_ACRONYMS) | set(_VARIETY_PREFIXES) | set(_ALLOWED_UNITS)
    tokens |= set(_SCIENTIFIC_NAMES)          # Azolla, Rhizobium, …
    tokens |= {"N", "P", "K", "S", "Zn", "Fe", "B", "Cu", "Mn", "Mo"}
    return frozenset(t.lower() for t in tokens)


def is_allowed_latin(word: str) -> bool:
    """True if `word` may stay in Latin script in an Odia answer."""
    return (word or "").lower() in allowed_latin_tokens()


# ── TTS: how a protected token should SOUND ───────────────────────────────

@lru_cache(maxsize=1)
def tts_spoken_forms() -> dict[str, str]:
    """Latin token -> Odia script spelling for the TTS voice.

    Necessary because `allowed_latin_tokens()` deliberately keeps these tokens in
    LATIN script for display, but the Odia voice pronounces any Latin token as
    English. Without an entry here a protected acronym falls through to the
    leftover-Latin LLM repair in prepare_for_tts — a paid call whose output is
    not deterministic (it produced "ଏଲଗୁ" from "ଲଗୁ" in testing).

    Acronyms are spelled out letter by letter (BGA -> ବି ଜି ଏ); scientific names
    are spoken as words (Azolla -> ଅଜୋଲା).
    """
    out: dict[str, str] = {}
    for token in _CORPUS_ACRONYMS + _VARIETY_PREFIXES:
        letters = [_LETTER_NAMES[c][0] for c in token.lower() if c in _LETTER_NAMES]
        if len(letters) == len(token):
            out[token] = " ".join(letters)
    for token, variants in _SCIENTIFIC_NAMES.items():
        if variants:
            out[token] = variants[0]
    return out
