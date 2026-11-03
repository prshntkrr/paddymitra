"""Does the English still say what the farmer said?

The answer direction (English -> Odia) verifies its own output: naturalness
faults, glossary survival, numbers, symbols, then a retry with the fault named
and a validate-or-keep-original. The question direction had no equivalent. It
only ever chose between two ENGINES — `_missing_glossary_terms` switches to
OpenAI, `_money_intrusion` switches, `_invented_name` reverts to Google — and
if both engines agreed on a wrong reading, nothing looked again.

A wrong question is worse than a wrong answer, because it is invisible. The
farmer gets a fluent, confident English sentence about something else, retrieval
finds nothing for it, and the reply is an honest-looking "the information
available does not specify". It reads exactly like a gap in the knowledge base.

So this module asks the one question the pipeline never asked: does the English
still mean what the Odia said? It answers conservatively — every check here
needs evidence from BOTH sides before it reports anything.

NO CHECK FIRES ON A KEYWORD ALONE. The English containing "harvesting" is not a
fault; the Odia saying ରୁଆ (transplanting) while the English says harvesting and
never says transplanting is. A question that legitimately mentions both survives
untouched. That two-sided rule is what keeps a validator from rejecting good
translations, which would be a worse bug than the one it fixes.

GROUNDING. Every Odia marker below is taken from the project glossary or from a
production failure already documented in pipeline.py — not from guesswork. Terms
that could not be grounded are listed in UNVERIFIED at the bottom rather than
half-implemented, because a marker that is merely plausible produces false
faults, and a false fault sends a good translation to be "repaired".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .technical_terms import normalize_odia

_ODIA_RANGE = r"଀-୿"


@dataclass(frozen=True)
class Fault:
    """One reason to doubt the translation."""
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def _odia_has(text: str, markers) -> str:
    """The first marker present in `text`, or "". Normalised on both sides."""
    haystack = normalize_odia(text or "")
    for marker in markers:
        if normalize_odia(marker) in haystack:
            return marker
    return ""


def _en_has(text: str, pattern: re.Pattern) -> str:
    found = pattern.search(text or "")
    return found.group(0) if found else ""


# ── Field operations that must never be swapped for one another ──────────
#
# Each entry: the Odia that establishes what the farmer is asking about, the
# English that correctly expresses it, and the English that would mean a
# DIFFERENT operation. A fault needs the marker, a wrong word, and NO right
# word — all three.
@dataclass(frozen=True)
class _StageRule:
    code: str
    markers: tuple          # Odia; presence establishes the stage
    correct: re.Pattern     # English that expresses it properly
    wrong: re.Pattern       # English that means a different operation
    detail: str


_STAGE_RULES = (
    # ରୁଆ -> transplanting, ରୋଇ ସାରିବା -> "transplanting rice": both are in the
    # glossary. The live failure: a neighbouring word was misread and the model
    # made the rest of the sentence agree, so transplanting came back as
    # harvesting and a farmer asking when to plough was told to cut the crop.
    _StageRule(
        code="transplanting_became_harvesting",
        markers=("ରୁଆ", "ରୋଇ", "ରୋଇବା", "ରୋପଣ"),
        correct=re.compile(r"\b(transplant\w*|plant\w*|sow\w*|nursery)\b", re.I),
        wrong=re.compile(r"\b(harvest\w*|reap\w*|cut(?:ting)?|thresh\w*)\b", re.I),
        detail="the Odia is about transplanting but the English says harvesting",
    ),
    # ପଡ଼ିଗଲା and its forms = the crop FELL OVER. Documented in pipeline.py from
    # a production case: "ପୂଜାଧାନ ୫୫ ଦିନରେ ପଡ଼ିଗଲା" became "Pooja Dhan 55 is
    # ready in 55 days" — the lodging vanished and the answer recommended
    # fertiliser, which is what CAUSES lodging.
    _StageRule(
        code="lodging_became_maturity",
        markers=("ପଡ଼ିଗଲା", "ପଡ଼ିଛି", "ପଡ଼ିଯାଇଛି", "ନଇଁପଡ଼ିଛି", "ନଇଁପଡ଼ିଲା"),
        correct=re.compile(r"\b(lodg\w*|fell|fallen|falling|flatten\w*|bent|bending)\b", re.I),
        wrong=re.compile(r"\b(ready|mature\w*|matur\w*|ripe\w*|harvest\w*)\b", re.I),
        detail="the Odia says the crop lodged but the English says it is "
               "ready/mature",
    ),
    # ହଳ = tillage, ହଳ କରିବା = ploughing (both in the glossary). The same live
    # failure turned ploughing into cutting.
    _StageRule(
        code="ploughing_became_harvesting",
        markers=("ହଳ",),
        correct=re.compile(r"\b(plough\w*|plow\w*|till\w*|puddl\w*|land\s+prepar\w*)\b", re.I),
        wrong=re.compile(r"\b(harvest\w*|reap\w*|cut(?:ting)?)\b", re.I),
        detail="the Odia is about ploughing but the English says harvesting/cutting",
    ),
    # ଫୁଲ ଫୁଟା -> flowering (glossary).
    _StageRule(
        code="flowering_became_harvesting",
        markers=("ଫୁଲ ଫୁଟା", "ଫୁଲ ଫୁଟୁ", "ଫୁଲ ଆସୁ"),
        correct=re.compile(r"\b(flower\w*|bloom\w*|panicle|heading)\b", re.I),
        wrong=re.compile(r"\b(harvest\w*|reap\w*|ripe\w*)\b", re.I),
        detail="the Odia is about flowering but the English says harvesting",
    ),
    # Other crop-condition verbs the prompt already names. Losing them turns a
    # damage report into a routine question.
    _StageRule(
        code="damage_report_lost",
        markers=("ଶୁଖିଗଲା", "ଶୁଖିଯାଉଛି", "ପଚିଗଲା", "ମରିଗଲା", "ହଳଦିଆ ହୋଇଗଲା"),
        correct=re.compile(
            r"\b(dr(?:y|ied|ying)|wilt\w*|rot\w*|die[ds]?|dying|dead|yellow\w*|"
            r"damag\w*|disease\w*)\b", re.I),
        wrong=re.compile(r"\b(ready|mature\w*|harvest\w*|healthy|good)\b", re.I),
        detail="the Odia reports crop damage but the English does not",
    ),
)


# The farmer asked something; the English must still be asking it.
_ODIA_QUESTION_WORDS = ("କେତେ", "କେବେ", "କେଉଁ", "କିପରି", "କାହିଁକି", "କ'ଣ", "କଣ",
                        "କେମିତି", "କେତେବେଳେ")
# A WH-WORD COUNTS WHEREVER IT APPEARS, not only at the start. Requiring it in
# front reported a real query as broken: "ମୁଁ ଜାଣିବାକୁ ଚାହୁଁଛି ଯେ କେତେ ..." ->
# "I want to know how much fertiliser is needed for 1 acre and which fertiliser
# will give good yield." That is a question, politely framed as a statement, and
# it ends in a full stop. Only a sentence with no interrogative anywhere — such
# as the bare topic "Urea application in rice" — is still reported.
_EN_QUESTION_RE = re.compile(
    r"\?"
    r"|\b(what|which|when|how|why|where|who|whom|whose)\b"
    r"|^\s*(is|are|do|does|did|can|could|should|shall|will|would|may|might|"
    r"please|tell|give|suggest|advise)\b",
    re.I)

# Negation. A dropped "not" reverses the advice the farmer receives.
_ODIA_NEGATIONS = ("ନାହିଁ", "ନୁହେଁ", "ନକରନ୍ତୁ", "ନ କରନ୍ତୁ", "ନଦେବେ", "ନ ଦେବେ",
                   "ନଥାଏ", "ନହେଲେ")
_EN_NEGATION_RE = re.compile(
    r"\b(not|no|never|without|avoid|don'?t|doesn'?t|shouldn'?t|cannot|can'?t|"
    r"un\w+|stop)\b", re.I)

# Numbers are converted to digits BEFORE translation (odia_numbers.py), so by
# the time we compare, both sides should carry the same digit runs. Odia digits
# may still appear if conversion did not reach them.
_ODIA_DIGITS = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")
_DIGIT_RUN_RE = re.compile(r"\d+(?:\.\d+)?")


def _digits(text: str) -> list:
    return _DIGIT_RUN_RE.findall((text or "").translate(_ODIA_DIGITS))


def validate_odia_to_english(odia_text: str, english_text: str) -> list:
    """Reasons to doubt that `english_text` still means `odia_text`.

    Returns [] when nothing is suspicious, which is the common case and the one
    that must stay cheap: this is pure string work, no model call, and it runs
    on every Odia question.

    Deliberately NOT checked here, because the pipeline already checks them and
    a second opinion would only double-report:
      - money intrusion         -> pipeline._money_intrusion
      - dropped glossary terms  -> pipeline._missing_glossary_terms
      - invented variety names  -> pipeline._invented_name
    """
    odia = odia_text or ""
    english = english_text or ""
    faults: list = []

    if not odia.strip() or not english.strip():
        return faults

    for rule in _STAGE_RULES:
        marker = _odia_has(odia, rule.markers)
        if not marker:
            continue
        wrong = _en_has(english, rule.wrong)
        if not wrong:
            continue
        # THE THIRD CONDITION IS WHAT MAKES THIS SAFE. A question that asks
        # about transplanting AND harvesting contains both words legitimately;
        # only a translation that kept the wrong one and dropped the right one
        # is a fault.
        if rule.correct.search(english):
            continue
        faults.append(Fault(rule.code,
                            f"{rule.detail} (Odia {marker!r} -> English {wrong!r})"))

    # Numbers. Every figure the farmer gave must still be there — a dose, a day
    # count and an area all arrive as digits and all change the answer.
    lost = [d for d in _digits(odia) if d not in _digits(english)]
    if lost:
        faults.append(Fault("number_lost",
                            f"number(s) {lost} in the Odia are not in the English"))

    # Negation, only when the Odia clearly carries one.
    negation = _odia_has(odia, _ODIA_NEGATIONS)
    if negation and not _EN_NEGATION_RE.search(english):
        faults.append(Fault("negation_lost",
                            f"the Odia is negative ({negation!r}) but the English is not"))

    # Question type. Fires only when the Odia used an explicit question word and
    # the English is not asking anything — a statement cannot retrieve an answer
    # to a question that was never posed.
    asked = _odia_has(odia, _ODIA_QUESTION_WORDS)
    if asked and not _EN_QUESTION_RE.search(english.strip()):
        faults.append(Fault("question_type_lost",
                            f"the Odia asks {asked!r} but the English is not a question"))

    return faults


# ── Known limitations ────────────────────────────────────────────────────
#
# ଅମଳ IS AMBIGUOUS AND IS DELIBERATELY NOT A MARKER. The glossary maps it to
# BOTH 'harvesting' and 'yield' (the reverse map logs the collision on every
# load). "ଅମଳ କେତେ?" is a yield question and "ଅମଳ କେବେ?" is a harvest question,
# and nothing in the word itself separates them. Treating it as a harvest marker
# would report every yield question as a mistranslation.
#
# These are the stages named in the brief that could NOT be grounded in the
# glossary or in a documented production failure, so they are listed rather than
# implemented on a guess. Add each one only with a real Odia example to test it:
UNVERIFIED = (
    "sowing (ବୁଣିବା / ବୁଣା) — no glossary entry in this checkout",
    "weeding as an OPERATION — the glossary has ଘାସ (weed) and ଘାସ ନିୟନ୍ତ୍ରଣ "
    "(weed management) as nouns, not the act of weeding",
    "fruiting / grain filling — no glossary entry found",
    "ଦେବି read as 'pay' — covered by pipeline._money_intrusion, which compares "
    "against Odia money words rather than guessing from the verb",
)
