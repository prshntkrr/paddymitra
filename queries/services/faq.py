"""
faq.py
======
Answer a question from a settled FAQ entry instead of running retrieval.

WHY THIS EXISTS
---------------
A small set of questions comes back constantly. Re-running retrieval and
generation for them costs an LLM call every time and — worse — is
non-deterministic: the same farmer asking the same thing twice gets two
differently-worded answers, one of which may be the worse of the two. Once an
answer has been approved, it should be the answer.

HOW A QUESTION IS MATCHED
-------------------------
Two layers, cheapest first:

  1. NORMALISED EXACT. Lowercased, punctuation dropped, spacing collapsed. Free,
     runs against a DB index, and catches the "asked the identical thing" case.
  2. EMBEDDING SIMILARITY. Catches the paraphrase — "How much DAP per acre?" vs
     "Per acre how much DAP to apply?" — which layer 1 never will, because
     farmers do not phrase a question the same way twice.

A FAQ answer is served verbatim and nothing re-reads the documents behind it, so
a loose match does not degrade the answer — it replaces it with the answer to a
DIFFERENT question. The failure mode of matching too little is one wasted RAG
call. The failure mode of matching too much is a farmer acting on a dose that was
never meant for their question. Everything below is tuned for the second.

WHY SIMILARITY ALONE IS NOT ENOUGH (measured, not assumed)
----------------------------------------------------------
Cosine similarity on text-embedding-3-small does NOT separate these two classes.
Measured on real question pairs from this project:

    SHOULD match   "How many kg of DAP per acre for rice?"
                vs "What quantity of DAP is needed for one acre of paddy?"    0.83
    SHOULD match   "...not growing?" vs "...not growing well?"                0.65

    MUST NOT       "How much DAP per acre?"  vs "How much DAP per HECTARE?"   0.92
    MUST NOT       "When should I apply urea?" vs "When should I STOP ..."     0.89
    MUST NOT       "...urea at tillering?" vs "...urea at PANICLE stage?"      0.78

The two ranges overlap heavily. The dangerous pairs are near-identical sentences
differing in ONE decisive word — a unit, a product, a stage, a negation — and an
embedding under-weights exactly that word because everything around it matches.

So similarity is only half the test. `_decisive_tokens` extracts the words that
change what the farmer should actually DO, and both questions must agree on all
of them before the score is even consulted. That guard is what lets the threshold
sit low enough (0.80) to catch genuine rewordings without also catching
"per hectare" when the farmer asked "per acre".

WHY A LINEAR SCAN AND NOT A VECTOR DB
-------------------------------------
The corpus here is FAQ entries — hundreds, not the hundreds of thousands the
Chroma collection holds. A numpy dot product over every active row is well under
a millisecond at that size and costs no extra store to keep in sync, no separate
re-index step, and no second place for the data to be stale. Revisit if the table
ever reaches tens of thousands of rows.

PUBLIC API
----------
    lookup(question_en, catalogue=None)   -> (FAQ, score) or (None, 0.0)
    promote_from_query(query, user=None)  -> FAQ or None
    refresh_embedding(faq)                -> bool
    reembed_all()                         -> int
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata

from django.conf import settings
from django.db import transaction
from django.utils import timezone

log = logging.getLogger("irri")

# Cosine similarity a paraphrase must beat, ON TOP OF agreeing on every decisive
# token. 0.80 sits above the near-misses that survive the guard and below the
# genuine rewordings measured above (0.83 and up). It is only ever the SECOND
# test — see the module docstring for why it cannot be the only one.
#
# Local MiniLM produces a different, less spread-out distribution, so it gets its
# own floor. Override per deployment with FAQ_MATCH_THRESHOLD in settings/.env.
_DEFAULT_THRESHOLD = 0.80
_LOCAL_MODEL_THRESHOLD = 0.85


def _threshold() -> float:
    configured = getattr(settings, "FAQ_MATCH_THRESHOLD", None)
    if configured:
        return float(configured)
    if getattr(settings, "USE_LOCAL_EMBEDDINGS", False) or not getattr(settings, "OPENAI_API_KEY", ""):
        return _LOCAL_MODEL_THRESHOLD
    return _DEFAULT_THRESHOLD


# ── The decisive-token guard ──────────────────────────────────────────────
#
# Words that change what the farmer should DO. Two questions that disagree on any
# of these are different questions, however similar they read. Each group is
# canonicalised so a spelling or abbreviation difference is not itself a
# disagreement: "ha" and "hectare" are the same unit, "MOP" and "potash" the same
# product.
_PRODUCTS = {
    "dap": "dap", "mop": "mop", "potash": "mop", "muriate": "mop",
    "npk": "npk", "ssp": "ssp", "tsp": "tsp", "urea": "urea",
    "nitrogen": "n", "yuria": "urea",
    "phosphorus": "p", "phosphate": "p", "potassium": "k",
    "zinc": "zn", "sulphur": "s", "sulfur": "s", "boron": "b", "iron": "fe",
    "manganese": "mn", "magnesium": "mg", "calcium": "ca", "molybdenum": "mo",
    "fym": "fym", "manure": "manure", "compost": "compost",
    "vermicompost": "vermicompost", "azolla": "azolla", "dhaincha": "dhaincha",
    "lime": "lime", "gypsum": "gypsum", "biofertilizer": "biofertilizer",
}
# Units are grouped by DIMENSION, because absence and disagreement are not the
# same thing. "How many KG of DAP per acre" and "What QUANTITY of DAP for one
# acre" are the same question — the second simply does not name a mass unit — so
# comparing a flat set of units wrongly called that a conflict.
#
# AREA is the exception and is compared strictly, including "one side names none".
# The area unit is what the dose is measured against, so serving a per-acre answer
# to a per-hectare question is wrong by a factor of 2.5 — and those two questions
# score 0.92 against each other, so nothing else would stop it.
_UNITS = {
    "kg": ("mass", "kg"), "kilogram": ("mass", "kg"), "kilograms": ("mass", "kg"),
    "kilo": ("mass", "kg"), "g": ("mass", "g"), "gram": ("mass", "g"),
    "grams": ("mass", "g"), "mg": ("mass", "mg"), "quintal": ("mass", "quintal"),
    "ton": ("mass", "ton"), "tonne": ("mass", "ton"), "tons": ("mass", "ton"),
    "litre": ("volume", "l"), "liter": ("volume", "l"), "litres": ("volume", "l"),
    "liters": ("volume", "l"), "ml": ("volume", "ml"),
    "acre": ("area", "acre"), "acres": ("area", "acre"),
    "hectare": ("area", "ha"), "hectares": ("area", "ha"), "ha": ("area", "ha"),
    "decimal": ("area", "decimal"), "gunta": ("area", "gunta"),
    "bag": ("count", "bag"), "bags": ("count", "bag"),
    "day": ("time", "day"), "days": ("time", "day"),
    "week": ("time", "week"), "weeks": ("time", "week"),
    "month": ("time", "month"), "months": ("time", "month"),
    "year": ("time", "year"), "years": ("time", "year"),
}
_UNIT_DIMENSIONS = ("mass", "volume", "area", "count", "time")
# Dimensions where one side naming nothing is an omission, not a disagreement.
_UNIT_WILDCARD_OK = frozenset({"mass", "volume", "count", "time"})
_STAGES = {
    "nursery": "nursery", "sowing": "sowing", "seeding": "sowing",
    "transplanting": "transplanting", "transplant": "transplanting",
    "puddling": "puddling", "tillering": "tillering", "tiller": "tillering",
    "panicle": "panicle", "booting": "booting", "heading": "heading",
    "flowering": "flowering", "milking": "milking", "ripening": "ripening",
    "maturity": "maturity", "harvest": "harvest", "harvesting": "harvest",
    "basal": "basal", "topdressing": "topdressing",
}
# Polarity. "When should I apply urea" and "When should I STOP applying urea" are
# opposite instructions and score 0.89 against each other.
#
# CANONICALISED INTO CLASSES, exactly as _STAGES and _SEASONS are. It used to be
# a flat set, so the token kept was the WORD the farmer happened to use — and a
# set of words is not a polarity. Live: "If there is NO urea, what should be
# applied?" was blocked against "If Urea is NOT available, what should I apply?"
# with `polarity ['no'] != ['not']`, at a score of 0.881 — well over the serve
# threshold. Two ways of saying the same thing read as opposite meanings, and
# the farmer got a generic answer with no doses in it while the approved one
# (35 kg DAP, 124 kg Ammonium Sulphate, 27 kg MOP per acre) sat unused. See
# queries 1029 and 1030, FAQ-136/142.
#
# The classes stay SEPARATE because they are genuinely different instructions:
# "no urea" (absent) is not "stop urea" (prohibited) is not "excess urea"
# (too much). Only synonyms within one class collapse.
_POLARITY = {
    "not": "neg", "no": "neg", "never": "neg", "dont": "neg",
    "cannot": "neg", "cant": "neg", "without": "neg",
    "stop": "stop", "avoid": "stop",
    "excess": "excess", "too": "excess", "over": "excess",
    "harmful": "harm", "damage": "harm",
}
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# SEASON, compared as strictly as a stage and for the same reason. "How much
# fertilizer is to be applied in rice field in kharif" scores 0.808 against
# "What fertilizers need to be applied for rice?" — over the threshold, with
# nothing in products, stages, polarity, numbers or units to separate them. The
# stored answer is for one season and the farmer asked about none, so the
# qualifier the answer depends on is one they never hear.
_SEASONS = {
    "kharif": "kharif", "monsoon": "kharif", "wet": "kharif",
    "rabi": "rabi", "winter": "rabi",
    "summer": "summer", "boro": "summer",
}

# DURATION CLASS, decisive for exactly the reason season is. Measured against
# the operator's own entries, "For long duration rice varieties, how much dap to
# be applied and when?" scores 0.771 against the SHORT-duration entry and 0.713
# against the long one — the WRONG entry wins, because the two are phrased alike
# and nothing separated them. A dose written for a 100-day variety is not the
# answer for a 150-day one.
#
# Matched as a PHRASE, never as a bare word: "how long should I keep water" is
# not the long-duration class, and "medium land" is a land type.
_DURATION_HEAD_RE = re.compile(r"\b(?:duration|maturing|maturity)\b", re.I)
_DURATION_CLASS_RE = re.compile(r"\b(short|medium|mid|long|early|late)\b", re.I)
_DURATION_CANON = {"short": "short", "early": "short", "medium": "medium",
                   "mid": "medium", "long": "long", "late": "long"}


def _durations(text: str) -> frozenset:
    """Duration classes named in the text, canonicalised.

    Read BACKWARDS from the head word, because an enumeration binds only its
    last member to it: "classify the short, medium and long duration of rice
    varieties" names all three, and a rule anchored on "<class> duration" saw
    only `long` — which made the classification entry look like a long-duration
    entry and contradict every short-duration question.

    The window stops at a sentence boundary, and there is no head word in "how
    long should I keep water" or "medium land", so neither reads as a class.
    """
    out = set()
    body = text or ""
    for head in _DURATION_HEAD_RE.finditer(body):
        window = body[max(0, head.start() - 60):head.start()]
        window = re.split(r"[.?!।]", window)[-1]
        out.update(_DURATION_CANON[m.group(1).lower()]
                   for m in _DURATION_CLASS_RE.finditer(window))
    return frozenset(out)


# ── A VARIETY'S CLASS, LEARNED FROM THE ENTRIES THEMSELVES ───────────────
#
# "How much urea should be given for Sahbhagi Dhan?" names a variety, not a
# class, so no duration probe was built and the GENERIC urea entry (0.612) was
# handed over — its 70 kg went out as the dose for a short-duration variety.
# The entry that answers it, "For short duration rice varieties like Sahabhagi
# Dhan ...", scored 0.511: not close, and never seen.
#
# Nothing is hardcoded. The operator's own FAQ says the variety and the class in
# one sentence, so the mapping is read from there and improves as entries are
# added.
#
# SPELLING IS THE WHOLE DIFFICULTY. The question says "Sahbhagi", the entry says
# "Sahabhagi". Measured over this project's cultivar list:
#
#     true variants    Sahbhagi~Sahabhagi 0.941, Kalachampa~Kala Champa 0.952,
#                      Swarna Sub1~Swarna sub-1 0.957, Pooja Dhana~Puja Dhana 0.857
#     DIFFERENT names  Manaswini~Tapaswini 0.778, Naveen~Veena 0.727
#
# 0.85 sits in the gap with room either side. Bare "Pooja"~"Puja" (0.667) falls
# below it and cannot be separated by any threshold — it sits inside the
# different-cultivar band — so short names match only exactly.
_NAME_SIMILARITY = 0.85
_NAME_MIN_FUZZY = 6                      # shorter names must match exactly
_NAME_RUN_RE = re.compile(
    r"\b[A-Z][A-Za-z]{2,}(?:[ -](?:[A-Z][A-Za-z]{1,}|[Ss]ub-?\d+|\d+))*\b")
# Capitalised words that are never a variety: products, stages, and the words a
# question starts with.
_NOT_A_VARIETY = frozenset({
    "how", "what", "when", "which", "where", "why", "should", "the", "for",
    "apply", "application", "rice", "paddy", "crop", "acre", "hectare",
    "kharif", "rabi", "urea", "dap", "mop", "npk", "ssp", "zinc", "potash",
    "nitrogen", "fertilizer", "fertiliser", "manure", "compost", "basal",
    "tillering", "panicle", "initiation", "transplanting", "sowing", "stage",
    "active", "short", "medium", "long", "duration", "variety", "varieties",
})


def _candidate_names(question_en: str) -> list:
    """Capitalised runs in the question that could be a variety name."""
    out = []
    for match in _NAME_RUN_RE.finditer(question_en or ""):
        name = match.group(0).strip()
        words = [w for w in re.split(r"[ -]+", name) if w]
        keep = [w for w in words if w.lower() not in _NOT_A_VARIETY]
        if not keep:
            continue
        name = " ".join(keep)
        if len(name.replace(" ", "")) >= 4 and name not in out:
            out.append(name)
    return out


def _name_in_text(name: str, text: str) -> bool:
    """True if `text` names the same variety, allowing a spelling variant."""
    low_name, low_text = name.lower(), (text or "").lower()
    if low_name in low_text:
        return True
    if len(low_name.replace(" ", "")) < _NAME_MIN_FUZZY:
        return False                     # too short to match loosely
    words = re.findall(r"[a-z0-9-]+", low_text)
    span = len(low_name.split())
    for size in {span, max(1, span - 1), span + 1}:
        for i in range(len(words) - size + 1):
            window = " ".join(words[i:i + size])
            if difflib.SequenceMatcher(None, low_name, window).ratio() >= _NAME_SIMILARITY:
                return True
    return False


def duration_for_variety(question_en: str, catalogue=None) -> str:
    """The duration class of a variety this question names, or "".

    Read from the active FAQs: an entry naming both the variety and a class is
    what teaches the mapping. Returns "" unless exactly one class is found, so
    a variety described as two different classes never picks a side.
    """
    from queries.models import FAQ

    names = _candidate_names(question_en)
    if not names:
        return ""
    try:
        qs = FAQ.objects.filter(is_active=True).exclude(question_en="")
        if catalogue is not None:
            qs = qs.filter(models_Q_catalogue(catalogue))
        classes = set()
        for entry in qs.only("id", "question_en", "answer_en"):
            text = f"{entry.question_en} {entry.answer_en or ''}"
            if any(_name_in_text(name, text) for name in names):
                classes |= _durations(text)
        if len(classes) != 1:
            return ""
        found = next(iter(classes))
        log.info("faq: %s is a %s duration variety, per the entries naming it",
                 names[0], found)
        return found
    except Exception as exc:             # noqa: BLE001
        log.warning("faq: duration_for_variety failed (%s)", exc)
        return ""


def _varieties(text: str) -> set:
    """Variety names, folded to lower case. Never raises; "" on any trouble."""
    try:
        from .answer_grounding import _varieties_in
        return {v.lower() for v in _varieties_in(text or "")}
    except Exception:                                    # pragma: no cover
        return set()


# WHAT THE QUESTION ASKS FOR — a time or an amount. "When should tillering-stage
# fertilizer be applied?" (timing) and "How much tillering-stage fertilizer
# should I give?" (quantity) agree on every stage and product token and still
# want opposite answers: a date versus a dose. normalise_question() drops
# "when"/"how"/"much", so this interrogative is invisible to the token groups
# below — read it from the RAW text, like numbers and durations are.
# Compared with the OVERLAP rule (a clash only when both sides ask something and
# the two share nothing), so an entry phrased with no interrogative at all
# ("Fertilizer dose at tillering") still answers either wording. Live: "How much
# tillering stage fertilizer should I give?" matched FAQ-23 ("When should
# tillering-stage fertilizer be applied?" -> "At 20-25 days after transplanting")
# at 0.86 and served the timing answer to a quantity question.
_ASK_TIMING_RE = re.compile(
    r"\bwhen\b|\bwhat\s+time\b|\b(?:at\s+)?(?:which|what)\s+stage\b"
    r"|\btime\s+of\s+(?:application|applying)\b"
    r"|\bhow\s+many\s+days\b|\bhow\s+long\s+after\b", re.I)
_ASK_QUANTITY_RE = re.compile(
    r"\bhow\s+much\b"
    r"|\bhow\s+many\s+(?:kg|kilograms?|kilos?|quintals?|grams?|g|bags?"
    r"|ml|litres?|liters?|l|tonnes?|tons?)\b"
    r"|\bwhat\s+(?:quantity|amount|dose|dosage|rate)\b"
    r"|\b(?:quantity|dosage|dose)\b", re.I)


def _asks(text: str) -> frozenset:
    """{'timing'}, {'quantity'}, both, or empty — what the question asks for."""
    t = text or ""
    out = set()
    if _ASK_TIMING_RE.search(t):
        out.add("timing")
    if _ASK_QUANTITY_RE.search(t):
        out.add("quantity")
    return frozenset(out)


def _asks_conflict(a: str, b: str) -> str:
    """"asks ..." when one side asks ONLY a time and the other ONLY an amount.

    Overlap rule, never strict: silence on either side is not a clash, and an
    entry that asks both ("how much and when") still answers either."""
    aa, ab = _asks(a), _asks(b)
    if aa and ab and not (aa & ab):
        return f"asks {sorted(aa)} != {sorted(ab)}"
    return ""


# A FAQ written for after a FLOOD (submergence, waterlogging) answers a different
# question than plain leaf-yellowing — serving it to a farmer with no flood is the
# wrong advice (Q-3024). Treated as a strict decisive group below.
_DISASTER_RE = re.compile(
    r"\b(flood\w*|submerg\w*|waterlog\w*|inundat\w*|stagnant water)\b", re.I)

# An ORGANIC-fertiliser question ("how much organic fertiliser / FYM / compost /
# vermicompost per acre") must not be answered by the general CHEMICAL schedule
# FAQ (Urea/DAP/MOP) — Q-3014/3015. Treated as a strict decisive group below.
_ORGANIC_RE = re.compile(
    r"\b(organic|bio[-\s]?fertili\w+|farmyard manure|\bfym\b|compost|vermicompost|"
    r"green manure|cow\s?dung|azolla|dhaincha|jaibik\w*)\b", re.I)


def _decisive_tokens(text: str) -> dict:
    """The words that decide what the farmer should do, grouped and canonicalised."""
    words = normalise_question(text).split()
    out = {
        "products": frozenset(_PRODUCTS[w] for w in words if w in _PRODUCTS),
        "stages": frozenset(_STAGES[w] for w in words if w in _STAGES),
        "polarity": frozenset(_POLARITY[w] for w in words if w in _POLARITY),
        "numbers": frozenset(_NUMBER_RE.findall(text or "")),
        "seasons": frozenset(_SEASONS[w] for w in words if w in _SEASONS),
        # VARIETY, read with the same vocabulary the rest of the pipeline uses so
        # a name is a name in one place only. An entry written for BINA Dhan is
        # that variety's answer; a farmer who named no variety, or another one,
        # is asking a different question.
        "varieties": frozenset(_varieties(text)),
        # Read from the RAW text: normalise_question() may drop the hyphen or
        # the very qualifier this phrase depends on.
        "durations": _durations(text),
        # Flood / submergence context present on ONE side only means the two
        # questions are not the same (Q-3024: a post-flood FAQ served for a plain
        # leaf-yellowing question).
        "disaster": frozenset(["flood"]) if _DISASTER_RE.search(text or "") else frozenset(),
        # Organic vs chemical fertiliser: an "organic fertiliser / FYM / compost"
        # question is not answered by the general chemical schedule (Q-3014/3015).
        "organic": frozenset(["organic"]) if _ORGANIC_RE.search(text or "") else frozenset(),
    }
    by_dim: dict = {dim: set() for dim in _UNIT_DIMENSIONS}
    for w in words:
        found = _UNITS.get(w)
        if found:
            dim, canon = found
            by_dim[dim].add(canon)
    for dim, values in by_dim.items():
        out[f"unit:{dim}"] = frozenset(values)
    return out


def _decisive_conflict(a: str, b: str) -> str:
    """Name of the first decisive group the two questions disagree on, or "".

    Products, stages, polarity and numbers are compared STRICTLY — naming one
    where the other names none is itself a difference ("how much urea" is not the
    question "how much urea AT TILLERING" answers). Most unit dimensions allow
    one side to be silent; see `_UNIT_WILDCARD_OK`.
    """
    ta, tb = _decisive_tokens(a), _decisive_tokens(b)

    for group in ("products", "stages", "polarity", "numbers",
                  "seasons", "varieties", "durations", "disaster", "organic"):
        if ta[group] != tb[group]:
            return f"{group} {sorted(ta[group])} != {sorted(tb[group])}"

    # A timing question is not answered by an amount, nor an amount by a date.
    asks = _asks_conflict(a, b)
    if asks:
        return asks

    for dim in _UNIT_DIMENSIONS:
        sa, sb = ta[f"unit:{dim}"], tb[f"unit:{dim}"]
        if sa == sb:
            continue
        if dim in _UNIT_WILDCARD_OK and (not sa or not sb):
            continue                     # one side simply did not say — not a clash
        return f"unit:{dim} {sorted(sa)} != {sorted(sb)}"
    return ""


# ── Normalisation (layer 1) ───────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

# Question words and fillers that carry no topic. Dropping them means "how much
# DAP per acre" and "DAP per acre how much" normalise to the same key, which is
# most of what layer 1 is for.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "do", "does", "did", "should", "shall", "will",
    "can", "could", "would", "may", "i", "we", "you", "my", "our", "your",
    "to", "of", "in", "on", "for", "at", "and", "or", "be", "it", "this", "that",
    "how", "what", "when", "which", "who", "why", "much", "many", "please", "tell",
    "me", "about", "there", "any", "some",
})


def normalise_question(text: str) -> str:
    """Lowercase, strip punctuation and filler, sort nothing — a stable key.

    NFKC first so a question typed with full-width or composed characters keys
    the same as the plain one.
    """
    t = unicodedata.normalize("NFKC", (text or "")).lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    words = [w for w in _SPACE_RE.split(t) if w and w not in _STOP]
    return " ".join(words)[:500]


# ── Embeddings (layer 2) ──────────────────────────────────────────────────

def _cosine(a, b) -> float:
    """Cosine similarity of two equal-length vectors, 0.0 if either is unusable."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _embed(text: str):
    """Embed one question, or return [] if embedding is unavailable.

    Never raises: a FAQ lookup is an optimisation on the way to a working RAG
    answer, so an embedding outage must degrade to "no FAQ hit", never to a
    failed query.
    """
    try:
        from catalogues.services.embeddings import embed_query
        return embed_query(text) or []
    except Exception as exc:
        log.warning("faq: embedding failed (%s); falling back to exact match only", exc)
        return []


def refresh_embedding(faq) -> bool:
    """(Re)compute and store the entry's question vector. Returns True if stored."""
    from catalogues.services.embeddings import embedding_model_name
    vec = _embed(faq.question_en or faq.question_local)
    if not vec:
        return False
    faq.question_embedding = vec
    try:
        faq.embedding_model = embedding_model_name()[:80]
    except Exception:
        faq.embedding_model = ""
    faq.save(update_fields=["question_embedding", "embedding_model", "updated_at"])
    return True


def reembed_all() -> int:
    """Re-embed every active entry. Use after switching embedding model —
    vectors from two different models are not comparable."""
    from queries.models import FAQ
    n = 0
    for faq in FAQ.objects.filter(is_active=True):
        if refresh_embedding(faq):
            n += 1
    log.info("faq: re-embedded %d entries", n)
    return n


# ── Lookup ────────────────────────────────────────────────────────────────

def _contradicts(a: str, b: str) -> str:
    """Disagreement where BOTH sides named something and named different things.

    Stricter than _decisive_conflict in one direction and looser in the other,
    and the difference is what makes a near miss usable as evidence rather than
    as an answer. To ANSWER, an entry must match on every decisive token,
    silence included: an entry written for kharif is not the answer to a
    question that named no season, because the farmer never hears the
    qualifier. To be shown as CONTEXT it only has to not contradict — the
    entry's own wording carries "in kharif" into the prompt, so the model can
    use it and say so, or leave it.

    Kharif against rabi is still a contradiction, and still excluded. So is urea
    against MOP, tillering against basal, and per-acre against per-hectare.
    """
    ta, tb = _decisive_tokens(a), _decisive_tokens(b)
    # AN ENTRY THAT NAMES EVERY CLASS IS GENERIC. "According to number of days,
    # classify the short, medium and long duration of rice varieties" carries
    # all three, so strict inequality made it contradict every single-class
    # question — including the ones it is the answer to. Naming more than one
    # class is not a disagreement with any one of them.
    if len(ta["durations"]) > 1 or len(tb["durations"]) > 1:
        ta = {**ta, "durations": frozenset()}
        tb = {**tb, "durations": frozenset()}
    for group in ("products", "stages", "polarity", "numbers",
                  "seasons", "varieties", "durations", "disaster", "organic"):
        if ta[group] and tb[group] and ta[group] != tb[group]:
            return f"{group} {sorted(ta[group])} != {sorted(tb[group])}"
    for dim in _UNIT_DIMENSIONS:
        sa, sb = ta[f"unit:{dim}"], tb[f"unit:{dim}"]
        if sa and sb and sa != sb:
            return f"unit:{dim} {sorted(sa)} != {sorted(sb)}"
    asks = _asks_conflict(a, b)
    if asks:
        return asks
    return ""


def _context_floor() -> float:
    """Score above which a MISS is still worth showing the model as evidence."""
    configured = getattr(settings, "FAQ_CONTEXT_THRESHOLD", None)
    return float(configured) if configured else 0.60


def nearest(question_en: str, catalogue=None, duration: str = ""):
    """The closest entry that did NOT earn the right to answer, or (None, 0.0).

    `duration` restricts the search to entries WRITTEN FOR THAT CLASS. Without
    it, a probe built from a day count finds whatever is lexically closest:
    "For short duration rice varieties, how much urea to be applied and when?"
    matched the generic "How much urea I have to apply in my rice crop?" at
    0.813, ahead of the short-duration entry at 0.787 — and the farmer was given
    the generic dose under a short-duration heading.

    A near miss is not a wrong answer — it is a related question somebody has
    already answered, and throwing it away entirely is what made the FAQ
    all-or-nothing: either it replaced the whole pipeline or it was never seen.
    Handed to generation as one more source, the model can use it when it fits
    and ignore it when it does not, exactly as it does with a retrieved chunk.

    Two kinds are deliberately NOT returned. An entry that CONTRADICTS the
    question — both name a season, a product, a stage or an area unit and they
    differ — is a different question wearing similar words, and putting its
    figures in front of the model is how a kharif dose reaches a rabi farmer.
    And anything below the context floor, which is unrelated rather than near.
    An entry that is merely NARROWER than the question is offered: it carries
    its own qualifier in its wording, where the model and the farmer can see it.
    """
    from queries.models import FAQ

    question_en = (question_en or "").strip()
    if not question_en:
        return None, 0.0
    try:
        qs = FAQ.objects.filter(is_active=True).exclude(answer_en="", answer_local="")
        if catalogue is not None:
            qs = qs.filter(models_Q_catalogue(catalogue))
        vec = _embed(question_en)
        if not vec:
            return None, 0.0

        floor, ceiling = _context_floor(), _threshold()
        best, best_score = None, 0.0
        for faq in qs.exclude(question_embedding=[]).only(
                "id", "question_en", "question_embedding", "embedding_model"):
            score = _cosine(vec, faq.question_embedding)
            if score < floor:
                continue                   # unrelated, not near
            # ABOVE THE SERVE FLOOR IS NOT THE SAME AS "IT ANSWERED". An entry
            # can score 0.81 and still be refused as the answer because it
            # disagrees on a decisive token — and refusing it here as well left
            # the single most similar entry in the corpus contributing nothing
            # at all, which is the gap this function exists to close. Skip only
            # what lookup() would genuinely have served.
            if score >= ceiling and not _decisive_conflict(question_en,
                                                           faq.question_en):
                continue
            if _contradicts(question_en, faq.question_en):
                continue
            if duration and duration not in _durations(faq.question_en):
                continue                   # not written for the class asked about
            if score > best_score:
                best, best_score = faq, score
        if best is None:
            return None, 0.0
        hit = FAQ.objects.get(pk=best.pk)
        log.info("faq: FAQ-%s scored %.3f — below %.2f, offered as CONTEXT for %r",
                 hit.pk, best_score, ceiling, question_en[:60])
        return hit, best_score
    except Exception as exc:               # noqa: BLE001
        log.warning("faq: nearest() failed (%s); continuing without it", exc)
        return None, 0.0


def lookup(question_en: str, catalogue=None):
    """Best FAQ entry for this question, or (None, 0.0).

    `catalogue` narrows to entries for that catalogue plus the entries that apply
    to all of them (catalogue is null). Never raises.
    """
    from queries.models import FAQ

    question_en = (question_en or "").strip()
    if not question_en:
        return None, 0.0

    try:
        qs = FAQ.objects.filter(is_active=True).exclude(answer_en="", answer_local="")
        if catalogue is not None:
            qs = qs.filter(models_Q_catalogue(catalogue))

        # Layer 1 — normalised exact. The guard still applies: normalisation drops
        # filler words, so two questions CAN normalise alike while differing on a
        # decisive one ("apply" vs "stop applying" both reduce to "urea apply").
        norm = normalise_question(question_en)
        if norm:
            for hit in qs.filter(question_norm=norm)[:5]:
                conflict = _decisive_conflict(question_en, hit.question_en)
                if conflict:
                    log.info("faq: FAQ-%s matched wording but was BLOCKED (%s)",
                             hit.pk, conflict)
                    continue
                log.info("faq: exact match FAQ-%s for %r", hit.pk, question_en[:60])
                return hit, 1.0

        # Layer 2 — nearest vector that ALSO agrees on every decisive token.
        #
        # The guard is applied per candidate, not only to the winner: the highest
        # scorer is often the near-miss ("per hectare" against "per acre" scores
        # 0.92), and rejecting only that one would throw away a slightly lower
        # candidate that is the right answer.
        vec = _embed(question_en)
        if not vec:
            return None, 0.0

        best, best_score = None, 0.0
        blocked, blocked_score, blocked_why = None, 0.0, ""
        for faq in qs.exclude(question_embedding=[]).only(
                "id", "question_en", "question_embedding", "embedding_model"):
            score = _cosine(vec, faq.question_embedding)
            conflict = _decisive_conflict(question_en, faq.question_en)
            if conflict:
                if score > blocked_score:
                    blocked, blocked_score, blocked_why = faq, score, conflict
                continue
            if score > best_score:
                best, best_score = faq, score

        # Worth a log line of its own: a high-scoring entry stopped by the guard
        # is precisely the wrong answer that would otherwise have been served.
        if blocked is not None and blocked_score >= _threshold():
            log.info("faq: FAQ-%s scored %.3f but was BLOCKED (%s) for %r",
                     blocked.pk, blocked_score, blocked_why, question_en[:60])

        floor = _threshold()
        if best and best_score >= floor:
            # Re-fetch: the loop used .only() and the caller needs the answer.
            hit = FAQ.objects.get(pk=best.pk)
            log.info("faq: semantic match FAQ-%s (%.3f >= %.2f) for %r",
                     hit.pk, best_score, floor, question_en[:60])
            return hit, best_score

        if best:
            log.info("faq: best candidate FAQ-%s scored %.3f, below %.2f — using RAG",
                     best.pk, best_score, floor)
        return None, best_score
    except Exception as exc:
        log.warning("faq: lookup failed (%s); continuing with RAG", exc)
        return None, 0.0


def models_Q_catalogue(catalogue):
    """Entries for THIS catalogue plus the ones that apply to every catalogue."""
    from django.db.models import Q
    return Q(catalogue=catalogue) | Q(catalogue__isnull=True)


def record_hit(faq) -> None:
    """Count a served answer. Never lets bookkeeping break the response."""
    try:
        from django.db.models import F
        from queries.models import FAQ
        FAQ.objects.filter(pk=faq.pk).update(
            hit_count=F("hit_count") + 1, last_used_at=timezone.now())
    except Exception as exc:
        log.warning("faq: could not record hit for FAQ-%s (%s)", faq.pk, exc)


# ── Writing ───────────────────────────────────────────────────────────────

def save_entry(faq, *, embed: bool = True):
    """Persist an entry with its match keys up to date.

    The normalised key and the vector are DERIVED from question_en, so they are
    recomputed here rather than trusted from a form — otherwise an edited
    question would keep matching on its old wording.
    """
    faq.question_norm = normalise_question(faq.question_en)
    faq.save()
    if embed:
        refresh_embedding(faq)
    return faq


@transaction.atomic
def promote_from_query(query, user=None):
    """Turn a Query into a live FAQ entry. Returns the entry, or None.

    Called when feedback is set to Good. Idempotent per query: rating the same
    answer Good twice updates the existing entry rather than creating a second
    one that would compete with it in the match.

    Returns None — rather than raising — when there is nothing worth storing, so
    a feedback click never fails because of it.
    """
    from queries.models import FAQ

    question_en = (getattr(query, "translated_query_text", "") or "").strip()
    question_local = (getattr(query, "original_query_text", "") or "").strip()
    answer_en = (getattr(query, "ai_response_text", "") or "").strip()
    answer_local = (getattr(query, "ai_response_text_local", "") or "").strip()

    # A question with no answer is not an FAQ. Fall back to the local wording if
    # the question was never translated (an English question skips translate()).
    if not (answer_en or answer_local):
        log.info("faq: query %s has no answer to promote", getattr(query, "pk", "?"))
        return None
    if not question_en:
        question_en = question_local
    if not question_en:
        return None

    faq = FAQ.objects.filter(source_query=query).first()
    if faq is None:
        # Same question already covered by a hand-written or earlier entry? Update
        # that one instead of adding a near-duplicate that would split the match.
        norm = normalise_question(question_en)
        faq = FAQ.objects.filter(question_norm=norm).first() if norm else None
    created = faq is None
    if created:
        faq = FAQ()

    faq.question_en = question_en
    faq.question_local = question_local
    faq.language = getattr(query, "original_query_language", "") or "Odia"
    faq.answer_en = answer_en
    faq.answer_local = answer_local
    faq.catalogue = getattr(query, "catalogue", None)
    faq.source_query = query
    faq.is_active = True
    if created and user is not None and getattr(user, "is_authenticated", False):
        faq.created_by = user

    save_entry(faq)
    log.info("faq: %s FAQ-%s from query %s (feedback Good)",
             "created" if created else "updated", faq.pk, getattr(query, "pk", "?"))
    return faq


def demote_for_query(query) -> bool:
    """Take the entry promoted from `query` out of service.

    Used when the feedback is changed AWAY from Good — the answer that was
    approved is the same answer now being called Average or Bad, and it must stop
    being served to other farmers.
    """
    from queries.models import FAQ
    n = FAQ.objects.filter(source_query=query, is_active=True).update(is_active=False)
    if n:
        log.info("faq: deactivated %d entry(ies) from query %s after feedback change",
                 n, getattr(query, "pk", "?"))
    return bool(n)
