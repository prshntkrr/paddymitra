"""
hooks/routing.py
================
Choosing WHICH CORPUS answers a question on the generic WhatsApp endpoint.

WHY THIS EXISTS
---------------
A hook names its own catalogue, so a farmer who types NUTRIENT reaches the
nutrient corpus. The generic endpoint has no hook — it is the provider-
independent API — and until now it answered everything from one configured
default. With four catalogues on the server that meant a pest question was
answered from the rice-nutrition corpus, retrieved nothing useful, and came back
as a refusal that looked like missing documents rather than a misrouted question.

The farmer must not have to say which corpus they want. They ask a question; the
question itself says which one it is.

WHY NOT KEYWORD MATCHING
------------------------
A word list alone gets the easy half and fails the half that matters:

    "Which variety resists blast?"        names a disease, but wants a VARIETY
    "How much fertilizer for BINA Dhan?"  names a variety, but wants a DOSE
    "What causes leaf blast?"             names no product at all

So two kinds of evidence are read separately and then combined:

    SUBJECT  what the question is ABOUT   — urea, blast, variety, stem borer
    INTENT   what kind of answer it WANTS — how much, which one, how to control

A category needs both to score highly, and the winner must beat the runner-up by
a clear margin. "Which variety resists blast?" scores variety on subject AND
intent but pest on subject only, so it routes to varieties. A question that
splits evenly — "which fertilizer variety is best?" — routes nowhere and falls
back to the default, which is the honest outcome: guessing between two corpora
is how a farmer gets an answer from the wrong one.

ODIA
----
Farmers write Odia, and a router that only reads English would send every real
message to the default. The subject vocabulary is therefore expanded through the
glossary the rest of the pipeline already uses (odia_glossary), so "ୟୁରିଆ" counts
as urea and "ମହିଷା ରୋଗ" as blast. Intent phrases stay English-only, which is why
an unambiguous SUBJECT alone is enough to route — see the margin rule below.

CONFIGURATION
-------------
    WHATSAPP_CATALOGUE_ROUTES=nutrient:5,variety:6,pest:7
    WHATSAPP_DEFAULT_CATALOGUE_ID=4

A category with no route configured is never selected, so a deployment enables
routing by listing catalogues and disables it by listing none.
"""
from __future__ import annotations

import logging
import re

from django.conf import settings

log = logging.getLogger("irri")

# What a question is ABOUT. Terms here are looked up in the Odia glossary too, so
# each one matches in either language.
_SUBJECTS: dict[str, tuple] = {
    "nutrient": (
        "urea", "dap", "mop", "potash", "muriate of potash", "npk", "ssp", "tsp",
        "zinc", "zinc sulphate", "znso4", "gypsum", "lime", "boron", "sulphur",
        "nitrogen", "phosphorus", "potassium", "micronutrient",
        "fertilizer", "fertiliser", "manure", "compost", "vermicompost",
        "farmyard manure", "fym", "green manure", "organic manure",
        "organic", "organics", "azolla",
        "dhaincha", "dhanicha", "dhaincha seeds", "dhanicha seeds",
        "biofertilizer", "top dressing", "basal dose",
    ),
    "variety": (
        "variety", "varieties", "cultivar", "seed", "seeds", "hybrid",
        "duration", "maturity", "short duration", "long duration",
        # THE LAND A FARMER HAS is how they choose a variety — "what seed for
        # high land?" is a selection question with no selection word in it.
        # Only phrases that name a LAND TYPE or a TOLERANCE belong here: bare
        # "drought" or "flood" would turn "what do I do after the flood?" into
        # a variety question.
        "medium land", "lowland", "low land", "upland", "up land", "high land",
        "highland", "shallow land", "shallow lowland", "deep water",
        "irrigated", "rainfed", "rain fed",
        "drought tolerant", "drought resistant", "flood tolerant",
        "submergence tolerant", "saline tolerant", "salt tolerant",
    ),
    # SEED MANAGEMENT is what to DO with seed — treat it, soak it, sow it, get
    # it to come up evenly — and it lives in the general corpus, not in the
    # varietal sheets. "seed" appears here AND under variety on purpose: the
    # word alone does not say which question is being asked, and it was the
    # variety list owning it outright that sent "how should I manage rice
    # seeds?" to the varietal recommendations. What separates them is the
    # intent, below.
    "seed_management": (
        "seed", "seeds", "seedling", "seedlings", "nursery", "seed bed",
        "seedbed", "germination", "sprouting", "seed treatment", "seed rate",
    ),
    "pest": (
        "pest", "pests", "insect", "insects", "bug", "bugs", "disease", "diseases",
        "infection",
        "blast", "blight", "bacterial leaf blight", "sheath blight", "sheath rot",
        "brown spot", "false smut", "tungro", "stem borer", "leaf folder",
        "gall midge", "brown planthopper", "hopper", "thrips", "case worm",
        "rat", "snail", "weed", "weeds",
        "insecticide", "pesticide", "fungicide", "herbicide", "spray",
    ),
}

# What kind of answer the question WANTS. English only; see the module docstring.
_INTENTS: dict[str, tuple] = {
    "nutrient": (
        r"\bhow much\b", r"\bhow many (?:kg|kilo|bags?|quintal|seeds?)", r"\bwhat dose\b",
        r"\b(?:how many|how much)\b.{0,30}\bsow",
        r"\bdose\b", r"\bdosage\b", r"\bquantity\b", r"\bper acre\b", r"\bper hectare\b",
        r"\bapply\b", r"\bapplication\b", r"\btop\s*dress", r"\bsplit\b",
        r"\bwhen (?:should|do) i (?:apply|put|give)\b", r"\bhow to apply\b",
    ),
    "variety": (
        r"\bwhich (?:rice |paddy )?(?:variety|varieties|seed|seeds|cultivar)\b",
        r"\bwhat (?:rice |paddy )?(?:variety|varieties)\b",
        # "WHAT SEED", "what type of seed" — the same question as "which
        # variety", asked the way a farmer asks it. Written tightly so "what is
        # the seed rate" and "what should I do to seeds" are not caught: those
        # are handling questions, and the words between decide it.
        r"\bwhat (?:type of |kind of |sort of )?(?:rice |paddy )?seeds?\b",
        r"\bsuitable (?:for|variety)\b", r"\brecommend(?:ed)? (?:a )?variety\b",
        r"\bbest (?:variety|varieties|seed)\b", r"\bhow many days\b",
        r"\bduration of\b", r"\byield of\b",
        r"\bwhich one (?:should|to) (?:i )?(?:grow|plant|sow)\b",
        # CHOOSING, rather than handling: which one to put in the ground, and
        # what it has to cope with or produce once it is there.
        r"\bshould i (?:use|plant|grow|sow|choose|select)\b",
        r"\b(?:good|better|high|higher|more|maximum) yield\b",
        r"\bfor (?:high|up|low|medium|shallow|deep)\s?land\b",
        r"\bfor (?:upland|lowland|irrigated|rainfed|drought|flood|saline|salinity)\b",
    ),
    # HOW to handle seed, as opposed to WHICH seed to buy. Every pattern here is
    # about an operation performed on seed; none of them is a selection.
    "seed_management": (
        r"\bmanage\w*\b", r"\bhandl\w*\b", r"\bstor\w*\b",
        r"\btreat\w*\b", r"\bsoak\w*\b", r"\bprepar\w*\b", r"\bprim\w*\b",
        r"\bgerminat\w*\b", r"\bsprout\w*\b", r"\bemergence\b", r"\bqualit\w*\b",
        r"\bbefore (?:sowing|planting|transplanting)\b",
        r"\bseed rate\b", r"\bhow much seed\b",
        r"\bwhat should i do to\b", r"\buniform\w*\b",
    ),
    "pest": (
        r"\bcontrol\b", r"\bmanage\b", r"\btreat\b", r"\bcure\b", r"\bprevent\b",
        r"\bwhat causes?\b", r"\bwhy (?:is|are|do|does)\b", r"\bsymptom",
        r"\battack", r"\bdamag", r"\binfest", r"\baffected\b", r"\bdying\b",
        r"\bspray\b", r"\bremedy\b",
        # RESISTANCE is a disease property, and "which variety is resistant to
        # blast?" is the question that needs two corpora at once: the varietal
        # sheets know which variety, the pest sheets know what it resists.
        r"\bresistan\w*\b", r"\btoleran\w*\b", r"\bsusceptib\w*\b",
    ),
}

# SCORING. A subject is worth more than an intent because it says what the
# question is about; the bonus rewards a category that has both, which is what
# separates "which variety resists blast" (variety subject + variety intent)
# from the disease it happens to mention.
_SUBJECT_POINTS = 2.0
_INTENT_POINTS = 1.0
_BOTH_BONUS = 1.0
# A category is CONFIDENT when it has both kinds of evidence — the question
# names something from that domain AND asks the kind of question that domain
# answers. Only a confident category may be added as a SECOND corpus, which is
# what keeps "how much fertiliser for BINA Dhan?" (variety named, but only a
# dose asked for) on the nutrient corpus alone.
_CONFIDENT = _SUBJECT_POINTS + _INTENT_POINTS + _BOTH_BONUS
# The winner must reach this AND beat the runner-up by it. One clear subject with
# nothing against it is therefore enough — which is what lets an Odia question
# route at all, since its intent phrases are not matched.
_MARGIN = 2.0

# Recognise common structured rice-variety identifiers even when the farmer
# does not literally say "variety". Keep this conservative: uppercase cultivar
# prefixes followed by a numeric identifier (e.g. MTU 1010, IR 64, ADT 36).
_NAMED_VARIETY_RE = re.compile(
    r"\b(?:MTU|BINA|IR|CR|ADT|PUSA|Swarna|Samba\s+Mahsuri)"
    r"\s*[- ]?\s*\d{1,4}[A-Z]?\b",
    re.IGNORECASE,
)


def routes() -> dict:
    """{category: catalogue_id} from WHATSAPP_CATALOGUE_ROUTES, or {}.

    Accepts "nutrient:5,variety:6,pest:7". A malformed pair is skipped with a
    warning rather than taking the endpoint down: a typo in one route must not
    stop the other three answering.
    """
    raw = getattr(settings, "WHATSAPP_CATALOGUE_ROUTES", "") or ""
    if isinstance(raw, dict):
        return {str(k).strip().lower(): int(v) for k, v in raw.items()}
    out: dict = {}
    for pair in str(raw).split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition(":")
        name, value = name.strip().lower(), value.strip()
        if not name or not value.isdigit():
            log.warning("routing: ignoring malformed route %r in "
                        "WHATSAPP_CATALOGUE_ROUTES", pair)
            continue
        out[name] = int(value)
    return out


def _variants(term: str) -> list:
    """The term plus its Odia spellings, so one entry matches either language."""
    try:
        from queries.services.odia_glossary import get_variants
        return [term] + [v for v in get_variants(term) if v]
    except Exception:                                    # pragma: no cover
        return [term]


_matchers: dict = {}


def _subject_matchers() -> dict:
    """{category: [(surface, compiled)]}, built once.

    Word boundaries for Latin text; a plain substring test for Odia, which has
    no \\b the regex engine can see.
    """
    if _matchers:
        return _matchers
    for category, terms in _SUBJECTS.items():
        built = []
        for term in terms:
            for surface in _variants(term):
                if surface.isascii():
                    built.append((surface,
                                  re.compile(rf"\b{re.escape(surface)}\b", re.I)))
                else:
                    built.append((surface, None))
        _matchers[category] = built
    return _matchers


def classify(question: str) -> dict:
    """{category, score, margin, evidence} — category is "" when nothing wins."""
    text = (question or "").strip()
    if not text:
        return {"category": "", "score": 0.0, "margin": 0.0, "evidence": {}}
    lowered = text.lower()

    scores: dict = {}
    evidence: dict = {}
    # WHERE the evidence sits, not only that it exists. Two categories can both
    # be fully evidenced — "which rice variety is resistant to blast?" — and the
    # one the question LEADS with is the one being asked for; the other
    # qualifies it. Without this the tie broke alphabetically, which made
    # "pest" the primary of a question that plainly asks for a variety.
    first_at: dict = {}
    for category in _SUBJECTS:
        subjects, intents = [], []
        earliest = len(text) + 1
        for surface, rx in _subject_matchers()[category]:
            found = rx.search(text) if rx else None
            at = found.start() if found else (
                text.find(surface) if (not rx and surface in text) else -1)
            if at >= 0:
                subjects.append(surface)
                earliest = min(earliest, at)
        for pattern in _INTENTS[category]:
            found = re.search(pattern, lowered)
            if found:
                intents.append(found.group(0).strip())
                earliest = min(earliest, found.start())

        # A named cultivar is strong variety evidence even if the question asks
        # for a dose rather than asking which variety to grow. This lets plan()
        # combine the variety-specific corpus with nutrient guidance.
        named_variety = _NAMED_VARIETY_RE.search(text)
        if category == "variety" and named_variety:
            subjects.append(named_variety.group(0))
            intents.append("named variety reference")
            earliest = min(earliest, named_variety.start())

        first_at[category] = earliest
        score = (_SUBJECT_POINTS * bool(subjects)) + (_INTENT_POINTS * bool(intents))
        if subjects and intents:
            score += _BOTH_BONUS
        scores[category] = score
        if subjects or intents:
            evidence[category] = {"subjects": subjects[:4], "intents": intents[:4]}

    # Sorted by score, then by where the category's evidence starts, then by
    # name so a tie resolves the same way every time — an answer that changes
    # because two categories tied is the kind of drift nobody can reproduce.
    ranked = sorted(scores.items(),
                    key=lambda kv: (-kv[1], first_at.get(kv[0], 0), kv[0]))
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    confident = [name for name, score in ranked if score >= _CONFIDENT]

    # TWO CONFIDENT CATEGORIES IS NOT AMBIGUITY, IT IS A TWO-PART QUESTION.
    # "Which variety should I use and what fertiliser should I apply?" scores
    # full evidence for both, so the margin rule below would call it unclear and
    # send it to the default — where neither half is answered. When every
    # category that scored is confident, the question genuinely spans them.
    if len(confident) > 1:
        return {"category": confident[0], "score": best_score, "margin": margin,
                "confident": confident, "scores": scores, "evidence": evidence}
    if best_score < _MARGIN or margin < _MARGIN:
        return {"category": "", "score": best_score, "margin": margin,
                "confident": confident, "scores": scores, "evidence": evidence}
    return {"category": best, "score": best_score, "margin": margin,
            "confident": confident, "scores": scores, "evidence": evidence}


def plan(question: str, default=None) -> dict:
    """Which corpora answer this question.

        {"primary": "variety", "secondary": ["pest"], "catalogue_ids": [6, 7]}

    ONE catalogue is the normal case and stays the normal case. A second is
    added only when the question genuinely spans two domains — "which variety
    resists blast?" needs the varietal sheets for the variety and the pest
    sheets for the disease, and neither corpus alone can answer it. Searching
    everything by default would be the opposite trade: more noise in the
    context for every question, to help the few that need it.

    When nothing is confident the DEFAULT catalogue answers alone. An uncertain
    router that reaches for several corpora is guessing wider, not better.

    `secondary` and `catalogue_ids` are always present; `catalogue_ids[0]` is
    the primary, so a caller that only wants one can take the first.
    """
    from catalogues.models import Catalogue

    table = routes()
    default_id = getattr(default, "pk", None)
    empty = {"primary": "", "secondary": [], "catalogue_ids":
             [default_id] if default_id else [], "catalogues":
             [default] if default is not None else [], "evidence": {}}
    if not table:
        return empty

    verdict = classify(question)
    primary = verdict["category"]

    # If a specific cultivar is named in a dose/application question, use the
    # variety catalogue first (the question is explicitly variety-specific),
    # while retaining nutrient as a secondary corpus when configured/confident.
    lowered_question = (question or "").lower()
    has_named_variety = bool(_NAMED_VARIETY_RE.search(question or ""))
    asks_nutrient_dose = any(
        re.search(pattern, lowered_question) for pattern in _INTENTS["nutrient"]
    )
    log.info(
        "routing debug: question=%r named_variety=%s "
        "nutrient_dose=%s configured_categories=%s",
        question,
        has_named_variety,
        asks_nutrient_dose,
        list(table.keys()),
    )
    if has_named_variety and asks_nutrient_dose and "variety" in table:
        verdict["confident"] = list(dict.fromkeys(
            ["variety"] + [c for c in verdict.get("confident", []) if c != "variety"]
        ))
        primary = "variety"
        verdict["category"] = primary

    if not primary:
        log.info("routing: no clear category (best=%.1f margin=%.1f) — using "
                 "default catalogue %s | evidence=%s",
                 verdict["score"], verdict["margin"], default_id,
                 verdict["evidence"] or "none")
        return empty
    if primary not in table:
        # RECOGNISED, BUT NOT ROUTED. Seed management is the ordinary case: it
        # is a real category — knowing a question is about seed treatment
        # rather than variety choice is the whole point — but its evidence
        # lives in the general corpus, so the default IS its catalogue. Logged
        # by name, because "no clear category" would be a lie and would send
        # anyone debugging this to the classifier instead of the config.
        log.info("routing: category=%s has no configured catalogue — using the "
                 "default %s | evidence=%s", primary, default_id,
                 verdict["evidence"] or "none")
        return empty

    wanted = [primary] + [c for c in verdict.get("confident") or []
                          if c != primary and c in table]
    # (category, catalogue) pairs, so a category whose catalogue has been
    # deleted drops out of BOTH lists and cannot be reported as routed.
    chosen = []
    for category in wanted:
        catalogue = Catalogue.objects.filter(pk=table[category]).first()
        if catalogue is None:
            log.warning("routing: %s -> catalogue id=%s, which does not exist; "
                        "skipping that category", category, table[category])
            continue
        chosen.append((category, catalogue))

    if not chosen:
        log.warning("routing: no configured catalogue exists for %s; using "
                    "default catalogue %s", ", ".join(wanted), default_id)
        return empty

    categories = [category for category, _ in chosen]
    catalogues = [catalogue for _, catalogue in chosen]
    ids = [catalogue.pk for catalogue in catalogues]
    log.info("routing: primary=%s secondary=%s (score=%.1f margin=%.1f) -> "
             "catalogue ids=%s | evidence=%s",
             categories[0], categories[1:] or "none", verdict["score"],
             verdict["margin"], ids, verdict["evidence"])
    return {"primary": categories[0], "secondary": categories[1:],
            "catalogue_ids": ids, "catalogues": catalogues,
            "evidence": verdict["evidence"]}


def route(question: str, default=None):
    """(catalogue, category) — the single-catalogue view of plan().

    Kept because the hook path and anything else that wants exactly one corpus
    should not have to know about the multi-catalogue case.
    """
    decision = plan(question, default=default)
    catalogues = decision["catalogues"]
    return (catalogues[0] if catalogues else default), decision["primary"]
