"""
odia_numbers.py
===============
Deterministic Odia number-word -> digits conversion for the INPUT direction.

WHY THIS EXISTS
---------------
A farmer states quantities in native number words, and the translator drops or
mangles them:

    ଅଢ଼େଇ ଏକର ଜମିରେ କେତେ ୟୁରିଆ …   ->  "How much urea should be applied
                                          per acre of land?"

ଅଢ଼େଇ is 2½. The "2.5" vanished completely, turning a 2.5-acre dose question into
a 1-acre one. On a fertiliser dose that is a 2.5x error in the number the farmer
acts on, so this is the one failure class where a wrong answer is worse than no
answer.

Asking the LLM to do the conversion is what was already being tried, and it is
unreliable in exactly the cases that matter — Odia fractional forms are single
words with no digit in them, so there is nothing for the model to copy. Doing it
in code before the model sees the text removes the guesswork.

CONSERVATIVE BY DESIGN
----------------------
This module converts ONLY number words it is certain of, and leaves everything
else untouched for the translator to handle. A partial table that never guesses
is safe; a complete table with a wrong entry silently corrupts a dose. Add
entries only when the value is confirmed.

WHY STRICT BOUNDARIES (no inflection suffixes)
----------------------------------------------
Unlike the glossary matcher, number words take NO trailing-suffix allowance:
ଏକର ("acre") is ଏକ ("one") + ର, so a suffix allowance would rewrite
"ଏକ ଏକର" ("one acre") into "1 1". Numbers require both boundaries strict.

PUBLIC API
----------
    convert_odia_numbers(text) -> (text_with_digits, [(phrase, value), …])
"""
from __future__ import annotations

import logging
import re

from .technical_terms import normalize_odia

log = logging.getLogger("irri")

# Odia digits -> ASCII, so ୭୦୬୧ reaches retrieval as 7061.
_DIGIT_TABLE = {ord(o): a for o, a in zip("୦୧୨୩୪୫୬୭୮୯", "0123456789")}

# ── Number vocabulary ─────────────────────────────────────────────────────
# Standalone fractional words. These are the ones being DROPPED today: a single
# word carrying a fractional value, with no digit for the model to latch onto.
_FRACTIONS: dict[str, float] = {
    "ଅଧା": 0.5,
    "ଅଧେ": 0.5,
    "ଦେଢ଼": 1.5,       # one and a half
    "ଦେଢ": 1.5,
    "ଅଢ଼େଇ": 2.5,      # two and a half   <- the reported failure
    "ଅଢେଇ": 2.5,
    "ସିକି": 0.25,
    "ପାଅ": 0.25,
}

# Modifiers that shift the FOLLOWING cardinal.
_MODIFIERS: dict[str, float] = {
    "ସାଢ଼େ": 0.5,      # sadhe tini  = 3.5
    "ସାଢେ": 0.5,
    "ପୌଣେ": -0.25,     # paune dui   = 1.75
    "ସୱା": 0.25,       # sawa dui    = 2.25
    "ସବା": 0.25,
}

# Cardinals. Confirmed values only.
_CARDINALS: dict[str, int] = {
    "ଶୂନ୍ୟ": 0,
    "ଏକ": 1, "ଗୋଟିଏ": 1, "ଗୋଟେ": 1,
    "ଦୁଇ": 2, "ଦି": 2,
    "ତିନି": 3,
    "ଚାରି": 4,
    "ପାଞ୍ଚ": 5,
    "ଛଅ": 6,
    "ସାତ": 7,
    "ଆଠ": 8,
    "ନଅ": 9,
    "ଦଶ": 10,
    "ଏଗାର": 11,
    "ବାର": 12,
    "ତେର": 13,
    "ଚଉଦ": 14,
    "ପନ୍ଦର": 15,
    "ଷୋହଳ": 16,
    "ସତର": 17,
    "ଅଠର": 18,
    "ଉଣେଇଶ": 19,
    "କୋଡ଼ିଏ": 20, "କୋଡ଼ି": 20, "କୁଡ଼ି": 20,
    "ପଚିଶ": 25,
    "ତିରିଶ": 30,
    "ଚାଳିଶ": 40,
    "ପଚାଶ": 50,
    "ଷାଠିଏ": 60,
    "ଏକଷଠି": 61,
    "ସତୁରି": 70, "ସତୁରୀ": 70,
    "ଅଶୀ": 80,
    "ନବେ": 90,
}

# Multipliers. ହଜାରେ / ଶହେ carry an implied "one".
_MULTIPLIERS: dict[str, int] = {
    "ଶହ": 100, "ଶହେ": 100,
    "ହଜାର": 1000, "ହଜାରେ": 1000,
    "ଲକ୍ଷ": 100000,
}

_ALL_WORDS: dict[str, tuple[str, float]] = {}
for _w, _v in _FRACTIONS.items():
    _ALL_WORDS[normalize_odia(_w)] = ("frac", _v)
for _w, _v in _MODIFIERS.items():
    _ALL_WORDS[normalize_odia(_w)] = ("mod", _v)
for _w, _v in _CARDINALS.items():
    _ALL_WORDS[normalize_odia(_w)] = ("card", _v)
for _w, _v in _MULTIPLIERS.items():
    _ALL_WORDS[normalize_odia(_w)] = ("mult", _v)

_ODIA_RANGE = r"଀-୿"

# One run of number words = number words separated only by whitespace. Longest
# alternative first so ଗୋଟିଏ beats ଗୋଟେ and କୋଡ଼ିଏ beats କୋଡ଼ି.
_WORD_ALT = "|".join(
    re.escape(w) for w in sorted(_ALL_WORDS, key=len, reverse=True)
)
# Boundaries strict on BOTH sides — see the module docstring (ଏକର vs ଏକ).
_RUN_RE = re.compile(
    rf"(?<![{_ODIA_RANGE}])(?:(?:{_WORD_ALT})(?![{_ODIA_RANGE}])(?:\s+|$))+"
)
_ONE_RE = re.compile(rf"(?<![{_ODIA_RANGE}])(?:{_WORD_ALT})(?![{_ODIA_RANGE}])")


def _evaluate(words: list[str]) -> float | None:
    """Evaluate a run of Odia number words to a single value.

    Handles the additive-multiplicative shape Odia uses:
        ସାତ ହଜାର ଏକଷଠି -> 7*1000 + 61   = 7061
        ହଜାରେ ନଅ       -> 1*1000 + 9    = 1009
        ଏକ ଶହ କୋଡ଼ିଏ    -> 1*100 + 20    = 120
        ସାଢ଼େ ତିନି      -> 3 + 0.5       = 3.5
        ଅଢ଼େଇ          -> 2.5
    """
    # A RUN OF BARE CARDINALS IS A LIST, NOT A SUM.
    #
    # The additive rule above is what makes ଏକ ଶହ କୋଡ଼ିଏ into 120 — but in every
    # genuine composite the parts are joined by a MULTIPLIER (ଶହ, ହଜାର) or a
    # modifier (ସାଢ଼େ, ଅଢ଼େଇ). Odia writes its tens-and-units as single words
    # (61 = ଏକଷଠି), so two or more plain cardinals in a row are not one number:
    # they are a farmer speaking approximately.
    #
    #     ରୋଗ ବାର ଦଶ ପନ୍ଦର ଦିନ ହେଇଗଲାଣି
    #     "the disease has been there twelve, ten, fifteen days" = about 10-15
    #
    # was evaluated as 12 + 10 + 15 and the farmer was told their disease had
    # lasted 37 days. Left alone, the phrase reaches the translator as spoken
    # and comes out as the range it is.
    kinds = [_ALL_WORDS[w][0] for w in words]
    if kinds.count("card") > 1 and not {"mult", "frac", "mod"} & set(kinds):
        log.info("odia_numbers: %r is a spoken list/range, not a composite — "
                 "left unconverted", " ".join(words))
        return None

    total = 0.0
    current = 0.0
    extra = 0.0
    saw_value = False

    for w in words:
        kind, val = _ALL_WORDS[w]
        if kind == "card":
            current += val
            saw_value = True
        elif kind == "mult":
            current = (current or 1) * val
            total += current
            current = 0.0
            saw_value = True
        elif kind == "frac":
            extra += val
            saw_value = True
        elif kind == "mod":
            extra += val

    return (total + current + extra) if saw_value else None


def _format(value: float) -> str:
    """Render 2.0 as '2' and 2.5 as '2.5'."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def convert_odia_numbers(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite Odia number words and Odia digits as ASCII digits.

    Returns `(converted_text, [(original_phrase, digits), …])`. The list is for
    logging and for the prompt, so a reviewer can see exactly what was converted.
    """
    if not (text or "").strip():
        return text or "", []

    out = normalize_odia(text).translate(_DIGIT_TABLE)
    conversions: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        phrase = m.group(0)
        words = [w for w in _ONE_RE.findall(phrase.strip())]
        if not words:
            return phrase
        value = _evaluate(words)
        if value is None:
            return phrase
        digits = _format(value)
        conversions.append((phrase.strip(), digits))
        # keep whatever trailing whitespace the run carried
        trail = phrase[len(phrase.rstrip()):]
        return digits + (trail or " ")

    out = _RUN_RE.sub(_sub, out)

    if conversions:
        log.info("odia_numbers: converted %s",
                 ", ".join(f"{p!r}->{d}" for p, d in conversions))
    return out, conversions


def conversions_for_prompt(conversions: list[tuple[str, str]]) -> str:
    """Prompt fragment telling the model the digits are already correct."""
    if not conversions:
        return ""
    shown = ", ".join(f"{p} = {d}" for p, d in conversions)
    return (
        "\nNUMBERS ALREADY CONVERTED: the digits in the text were computed from "
        f"the farmer's spoken Odia numbers ({shown}). Copy every digit EXACTLY "
        "as it appears. Do not recompute, round, re-spell or drop them."
    )
