"""
area_math.py
============
Scale a per-area recommendation to the field the farmer actually has —
DETERMINISTICALLY, in Python.

WHY IT IS DONE HERE AND NOT BY THE MODEL
----------------------------------------
The persona used to forbid arithmetic outright, and the comment above that rule
records why: twice the model was allowed to "multiply the rate by the land size"
and twice it produced a dose that existed nowhere in the corpus — 90-100 kg of
urea for 2.5 acres, arrived at by summing split doses and relabelling them per
hectare. Restricting it to "one multiplication only" did not help.

That rule ended with a condition: do not re-enable arithmetic WITHOUT A
DETERMINISTIC IMPLEMENTATION. This is that implementation. The model no longer
calculates; it is handed the finished figures and the working, and its only job
is to say them in the farmer's language.

Which is also the honest division of labour. Multiplying 80 by 0.607029 is not a
task that needs a language model, and it is the one part of an answer where being
approximately right is worthless.

WHAT IT REFUSES
---------------
Silence beats a confident wrong number, so it declines rather than guesses when:

  * the area unit is ambiguous — a bigha is not the same size in two districts,
    so it is never converted
  * the source rate is a range spanning a factor of more than 3 (that is not a
    recommendation, it is a table read badly)
  * the rate is a YIELD or a duration — those are not scaled per field here

PUBLIC API
----------
    parse_area(text)                 -> the field size the farmer gave, or None
    scale_for_area(question, ctx)    -> computed lines + evidence for grounding
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("irri")

# One hectare per unit. The acre factor is the one the agronomy documents and the
# extension services use; keeping the same constant everywhere means the arithmetic
# a farmer checks by hand agrees with ours.
_HA_PER = {
    "hectare": 1.0,
    "acre": 0.404686,
    "decimal": 0.00404686,      # 100 decimal = 1 acre
    "cent": 0.00404686,         # a cent is the same area as a decimal
    "gunta": 0.0101171,         # 40 gunta = 1 acre
    "guntha": 0.0101171,
    "square metre": 0.0001,
}

# Surface spellings, longest first so "square metre" is not eaten by "metre".
_AREA_WORDS = [
    ("hectare", ("hectares", "hectare", "hect", "ha", "ହେକ୍ଟର")),
    ("acre", ("acres", "acre", "ac", "ଏକର")),
    ("decimal", ("decimals", "decimal", "dismil", "ଡେସିମାଲ", "ଡେସିମିଲ")),
    ("cent", ("cents", "cent")),
    ("gunta", ("guntas", "gunta", "guntha", "ଗୁଣ୍ଠ")),
    ("square metre", ("square metres", "square meters", "sq m", "sqm", "m2")),
]

# Units that name an area but not a FIXED one. Converting these silently is how a
# farmer in one district gets a dose calculated for a plot in another.
_AMBIGUOUS_AREA = ("bigha", "katha", "kattha", "decimel", "ବିଘା", "କଠା")

_ODIA_DIGITS = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")

_AREA_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*("
    + "|".join(re.escape(w) for _, ws in _AREA_WORDS for w in ws)
    + r")\b", re.IGNORECASE)

_AMBIGUOUS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(_AMBIGUOUS_AREA) + r")\b", re.IGNORECASE)

_CANON_BY_WORD = {w.lower(): canon for canon, ws in _AREA_WORDS for w in ws}


def parse_area(text: str) -> dict | None:
    """The field size stated in `text`, converted to hectares.

    Returns None when no area is given, and {"ambiguous": unit} when one is given
    in a unit that has no fixed size — the caller must ask, not assume.
    """
    flat = re.sub(r"\s+", " ", (text or "")).translate(_ODIA_DIGITS)

    amb = _AMBIGUOUS_RE.search(flat)
    m = _AREA_RE.search(flat)
    if amb and (not m or amb.start() < m.start()):
        return {"ambiguous": amb.group(2).lower(), "text": amb.group(0).strip()}
    if not m:
        return None

    unit = _CANON_BY_WORD[m.group(2).lower()]
    value = float(m.group(1))
    if value <= 0:
        return None
    return {
        "value": value,
        "unit": unit,
        "hectares": value * _HA_PER[unit],
        "text": m.group(0).strip(),
    }


# ── Rates in the retrieved documents ──────────────────────────────────────

_RATE_UNIT = r"(?:kgs?|kilograms?|g|grams?|t|tons?|tonnes?|q|quintals?)"
_RATE_AREA = r"(?:ha|hectare|hectares|acre|acres)"

# "80:40:40 kg/ha" — an N:P:K triple, the commonest form a general rice
# recommendation takes in these documents.
_NPK_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*"
    rf"({_RATE_UNIT})\s*(?:/|per\s+)\s*({_RATE_AREA})\b", re.IGNORECASE)

# "25 kg/ha urea", "20-25 kg per acre of MOP", and — the form that broke this —
# "80 kg nitrogen per hectare", where the product sits BETWEEN the unit and the
# area. Requiring the area to follow the unit directly meant the commonest
# phrasing in these documents produced no rate at all, so nothing was ever
# scaled to the farmer's field.
# Named groups throughout: adding `mid` once silently shifted every positional
# index, so the area group picked up " nitrogen " instead of "hectare".
_RATE_RE = re.compile(
    rf"(?P<low>\d+(?:\.\d+)?)"
    rf"(?:\s*(?:-|–|—|to)\s*(?P<high>\d+(?:\.\d+)?))?\s*"
    rf"(?P<unit>{_RATE_UNIT})\b(?P<mid>[^.;|\d]{{0,30}}?)"
    rf"(?:/|per\s+)\s*(?P<area>{_RATE_AREA})\b",
    re.IGNORECASE)

_MASS_TO_KG = {"kg": 1.0, "kgs": 1.0, "kilogram": 1.0, "kilograms": 1.0,
               "g": 0.001, "gram": 0.001, "grams": 0.001,
               "t": 1000.0, "ton": 1000.0, "tons": 1000.0, "tonne": 1000.0,
               "tonnes": 1000.0, "q": 100.0, "quintal": 100.0, "quintals": 100.0}
_AREA_TO_HA_FACTOR = {"ha": 1.0, "hectare": 1.0, "hectares": 1.0,
                      "acre": 0.404686, "acres": 0.404686}

# What the figure is FOR. A yield is deliberately absent: multiplying an expected
# yield by a field size is a different claim from a fertiliser requirement, and
# nobody asked for it here.
# ZnSO4 was in _SUBSTANCE_NEAR but not here, so a clause whose only input is
# zinc sulphate carried no scalable cue and its rate was dropped before anything
# could label it — the two vocabularies have to know the same products.
_SCALABLE_NEAR = re.compile(
    r"\b(fertili[sz]er|dose|apply|applied|application|use[sd]?|using|basal|"
    r"top\s*dress\w*|urea|dap|mop|npk|ssp|tsp|potash|nitrogen|phosphorus|"
    r"potassium|zn\s*so\s*[4₄?]?|zinc|borax|gypsum|lime|fym|manure|compost|"
    r"seed\s*rate|seed\s*requirement|recommend\w*)\b", re.IGNORECASE)
_YIELD_NEAR = re.compile(r"\b(yield|productivity|production|harvest)\b", re.IGNORECASE)

# ZnSO4 is written as a formula as often as a name, and a PDF extractor mangles
# the subscript: the corpus really contains "ZnSO? 10 kg/acre". Unrecognised, the
# figure took the nearest name in front of it and was recorded as a MOP rate —
# the right number under the wrong product, which is the error a farmer acts on
# by buying the wrong bag. The trailing character is therefore optional and may
# be 4, a subscript ₄, or whatever the extractor put in its place.
_SUBSTANCE_NEAR = re.compile(
    r"\b(urea|dap|mop|muriate\s+of\s+potash|npk|ssp|tsp|potash|nitrogen|"
    r"phosphorus|potassium|zinc\s+sul(?:ph|f)ate|zn\s*so\s*[4₄?]?|"
    r"zinc|fym|farmyard\s+manure|vermicompost|compost|gypsum|lime|borax|seed)"
    r"(?![A-Za-z])", re.IGNORECASE)

# One name per product, so provenance and the answer agree however the document
# spelled it. Bare "zinc" is deliberately absent: the element is a NUTRIENT and
# the sulphate is a PRODUCT, and folding them would erase the distinction that
# stops a nutrient rate being handed over as a fertiliser quantity.
_PRODUCT_CANON = {
    "dap": "DAP", "mop": "MOP", "muriate of potash": "MOP", "potash": "MOP",
    "urea": "Urea", "ssp": "SSP", "tsp": "TSP", "npk": "NPK",
    "zinc sulphate": "ZnSO4", "zinc sulfate": "ZnSO4",
    "fym": "FYM", "farmyard manure": "FYM", "vermicompost": "vermicompost",
    "compost": "compost", "gypsum": "gypsum", "lime": "lime", "borax": "borax",
}
_ZINC_SULPHATE_RE = re.compile(r"^zn\s*so\s*[4₄?]?$", re.IGNORECASE)


def _canon_product(name: str) -> str:
    """The one name this product is recorded under, whatever the document wrote."""
    key = re.sub(r"\s+", " ", (name or "").strip()).lower()
    if _ZINC_SULPHATE_RE.match(key):
        return "ZnSO4"
    return _PRODUCT_CANON.get(key, name.strip())


# A token that LOOKS like a product name and is in no vocabulary: a formula, an
# acronym, a brand. It matters only as a BOUNDARY — see _unknown_product_between.
_PRODUCT_LIKE_RE = re.compile(r"\b(?=[A-Za-z]*[A-Z0-9₄?])[A-Za-z][A-Za-z0-9₄?]{1,14}\b")

# Words that may stand between a product and its rate without being a product.
_BETWEEN_PRODUCT_AND_RATE = frozenset({
    "and", "or", "with", "along", "plus", "of", "the", "a", "an", "at", "in",
    "apply", "applying", "applied", "use", "using", "add", "adding", "also",
    "then", "per", "each", "every", "about", "around", "approximately",
    "kg", "kgs", "g", "gm", "t", "ton", "tons", "tonne", "tonnes", "q",
    "quintal", "quintals", "acre", "acres", "ha", "hectare", "hectares",
    "dose", "rate", "basal", "top", "dressing", "sowing", "stage", "das",
    "is", "are", "be", "should", "must", "may", "can", "will", "to", "for",
})


def _unknown_product_between(text: str) -> str:
    """A product-like token in `text` that no vocabulary knows, or "".

    A PRODUCT NAME IS A BOUNDARY. Once one is passed, a later rate belongs to it
    and not to whatever was named before — and that has to hold for products the
    vocabulary has never heard of, or every OCR variant reopens the same bug. So
    an unrecognised token here blocks the earlier name from claiming the rate,
    and is reported as the product instead: naming it as the document wrote it is
    honest, and mislabelling it MOP is not.
    """
    for m in _PRODUCT_LIKE_RE.finditer(text or ""):
        token = m.group(0)
        if token.lower() in _BETWEEN_PRODUCT_AND_RATE:
            continue
        if _SUBSTANCE_NEAR.fullmatch(token):
            continue                       # a known product; handled by the caller
        if token.isdigit():
            continue
        return token
    return ""

# THE DOCUMENTS WRITE THE NUTRIENT AS A LETTER. "Apply 80 kg N per hectare" is
# the commonest form a nutrient rate takes here, and a bare N matched nothing
# above — so the scaled total came back as "the recommended input: 32.4 kg",
# a dose with no statement of what it is a dose OF. A farmer cannot act on that,
# and the grounding guard cannot check it either.
#
# Only where the letter stands ALONE between the unit and the area, which is
# what makes it a label rather than an initial in a word.
_NUTRIENT_LETTER = {"n": "nitrogen", "p": "phosphorus", "k": "potassium"}
_LETTER_ONLY_RE = re.compile(r"^\s*([NPK])\s*$")


def _substance_in(fragment: str, before: int | None = None) -> str:
    """The input named in `fragment` — spelled out, or as a bare N/P/K letter.

    NEAREST BEFORE `before`, when an offset is given. A clause of this corpus
    lists two products at once —

        "DAP - 35-40 kg/acre MOP - 15-20 kg/acre"

    — and taking the first match in the clause labelled the MOP rate as DAP: the
    right number under the wrong product, which is the most expensive kind of
    wrong here because a farmer buys the wrong bag. The name that governs a
    figure is the one immediately in front of it.

    The matched surface is returned as the document wrote it, so "DAP" stays
    "DAP"; only a bare letter is expanded, since "N" alone labels nothing.
    """
    text = fragment or ""
    matches = [m for m in _SUBSTANCE_NEAR.finditer(text)
               if before is None or m.start() < before]
    if matches:
        nearest = matches[-1]
        # NOTHING PRODUCT-LIKE MAY SIT BETWEEN THEM. "MOP 15-20 kg/acre, and
        # ZnSO? 10 kg/acre" puts an unrecognised product between MOP and the 10,
        # and without this the 10 was recorded as a MOP rate.
        if before is not None:
            unknown = _unknown_product_between(text[nearest.end():before])
            if unknown:
                return _canon_product(unknown)
        return _canon_product(nearest.group(0))
    letter = _LETTER_ONLY_RE.match(text)
    return _NUTRIENT_LETTER[letter.group(1).lower()] if letter else ""

def _substance_before(fragment: str, before: int) -> tuple:
    """(name, where it starts, where it ends) for the product in FRONT of `before`."""
    matches = [m for m in _SUBSTANCE_NEAR.finditer(fragment or "")
               if m.start() < before]
    if not matches:
        return "", -1, -1
    nearest = matches[-1]
    unknown = _unknown_product_between((fragment or "")[nearest.end():before])
    if unknown:
        return _canon_product(unknown), nearest.start(), nearest.end()
    return _canon_product(nearest.group(0)), nearest.start(), nearest.end()


def _already_claimed(clause: str, starts: int, ends: int, before: int) -> bool:
    """Is the product in front of this figure already another figure's?

    Two ways it can be. A rate BETWEEN the name and this figure took it on the
    way past. Or the name is itself a trailing one — a figure sits in front of
    it with no product of its own, so that figure is the one it names, and this
    figure has to look elsewhere: "apply 20-25 kg/ha MOP and 15-20 kg/ha urea"
    gave both figures to MOP, and the farmer was told to buy potash twice.
    """
    if starts < 0:
        return False
    if _RATE_RE.search(clause[ends:before]):
        return True
    head = clause[:starts]
    earlier = list(_RATE_RE.finditer(head))
    return bool(earlier and not _SUBSTANCE_NEAR.search(head[:earlier[-1].start()]))


def _substance_after(fragment: str, after: int) -> str:
    """The product named just AFTER a figure — "30-40 kg/ha urea at tillering".

    Half of a long-standing gap, and the safe half. Reading the trailing name in
    every case mislabels the commoner "DAP 35 kg/acre, MOP 15-20 kg/acre", where
    each figure follows its own product; that was tried and reverted. This runs
    only where the other reading has nothing to say — no product in front of the
    figure at all, or one that a nearer figure has already taken — so it can add
    a name where there was none but never move one.

    Another rate between the figure and the name means the name belongs to that
    rate instead, so nothing is claimed across a neighbour.
    """
    tail = (fragment or "")[after:]
    name = _SUBSTANCE_NEAR.search(tail)
    if not name:
        return ""
    figure = _RATE_RE.search(tail)
    if figure and figure.start() < name.start():
        return ""
    return _canon_product(name.group(0))


_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*[;|]\s*")
_MAX_RANGE_SPREAD = 3.0      # a 1-to-4 "range" is a misread table, not advice


def _clause_of(flat: str, at: int, end: int) -> str:
    head, tail = flat[:at], flat[end:]
    lo = max(head.rfind(". "), head.rfind("; "), head.rfind(" | "), head.rfind("\n"))
    hi = min([x for x in (tail.find(". "), tail.find("; "), tail.find("\n"))
              if x >= 0] or [len(tail)])
    return flat[(lo + 1 if lo >= 0 else 0): end + hi]


_NUTRIENT_LONG = {"N": "nitrogen (N)", "P": "phosphorus (P)", "K": "potassium (K)"}


def _long_name(name: str) -> str:
    """Spell a bare nutrient letter out — 'N' alone labels nothing."""
    return _NUTRIENT_LONG.get(name, name)


def _kind_word(name: str) -> str:
    """The quantity kind, so the grounding guard can label the computed figure."""
    return "seed rate" if name.lower() == "seed" else "fertiliser dose"


def _fmt(v: float) -> str:
    """One decimal place, trailing zero dropped.

    A farmer buying fertiliser weighs it on a shop scale; 48.56 kg implies a
    precision the recommendation never had. 48.6 is the honest figure.
    """
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return s or "0"


def _fmt_area(v: float) -> str:
    """Three decimals for the AREA, which is a conversion factor, not a dose."""
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


_SOURCE_MARKER_RE = re.compile(r"\[Source:\s*([^\]]*)\]")


def rates_in(context: str) -> list[dict]:
    """Per-area fertiliser / seed rates in the CONTEXT, ready to scale.

    READ ONE DOCUMENT AT A TIME, and each rate carries where it came from.
    Flattening the whole context lost the boundary between documents, and a
    per-acre question about direct-seeded rice was answered with a transplanted
    crop's per-HECTARE rates scaled down — "MOP 6.1 to 8.1 kg" beside the DSR
    document's own "MOP 15-20 kg/acre", from two documents at once, with nothing
    in the line to say so.

    Provenance recorded per rate:

        document    the [Source: ...] the rate was written in
        source      the rate exactly as the document wrote it
        area_unit   the area basis the document used
        stated      True when the document already gives it per the area asked
                    about, so the answer quotes it instead of deriving it
    """
    out: list[dict] = []
    for block in re.split(r"(?=\[Source:)", context or ""):
        if not block.strip():
            continue
        marker = _SOURCE_MARKER_RE.search(block)
        document = (marker.group(1).strip() if marker else "") or "unattributed"
        # The marker is OUR formatting; leaving it in the clause let its words be
        # read as part of the recommendation.
        body = _SOURCE_MARKER_RE.sub(" ", block)
        out.extend(_rates_in_block(body, document))
    return out


def _rates_in_block(context: str, document: str) -> list[dict]:
    flat = re.sub(r"\s+", " ", context or "")
    out: list[dict] = []

    for m in _NPK_RE.finditer(flat):
        clause = _clause_of(flat, m.start(), m.end())
        if _YIELD_NEAR.search(clause):
            continue
        unit, area = m.group(4).lower(), m.group(5).lower()
        for label, raw in zip(("N", "P", "K"), m.group(1, 2, 3)):
            out.append({
                "substance": label, "low": float(raw), "high": None,
                "kg_per_ha": float(raw) * _MASS_TO_KG[unit] / _AREA_TO_HA_FACTOR[area],
                "source": f"{m.group(1)}:{m.group(2)}:{m.group(3)} {unit}/{area} N:P:K",
                "clause": clause.strip(),
                "document": document, "area_unit": area, "mass_unit": unit,
            })

    for m in _RATE_RE.finditer(flat):
        # Skip the numbers already consumed as part of an N:P:K triple.
        if any(m.start() >= n.start() and m.end() <= n.end()
               for n in _NPK_RE.finditer(flat)):
            continue
        clause = _clause_of(flat, m.start(), m.end())
        if _YIELD_NEAR.search(clause) or not _SCALABLE_NEAR.search(clause):
            continue
        low = float(m.group("low"))
        high = float(m.group("high")) if m.group("high") else None
        if high and low and high / low > _MAX_RANGE_SPREAD:
            log.info("area_math: ignoring implausible range %s-%s", low, high)
            continue
        unit, area = m.group("unit").lower(), m.group("area").lower()
        factor = _MASS_TO_KG[unit] / _AREA_TO_HA_FACTOR[area]
        inside = _substance_in(m.group("mid") or "")
        # Where the figure sits inside its own clause, so the clause search can
        # take the name in FRONT of it rather than the first one in the line.
        at_in_clause = flat.find(clause.strip())
        offset = (m.start() - at_in_clause) if at_in_clause >= 0 else None
        # WHICH PRODUCT THIS FIGURE IS OF. The name in front of it, normally.
        # The name AFTER it in the two cases where the one in front cannot be
        # meant: there is none, or a nearer figure already took it. "Apply 30-40
        # kg/ha urea at tillering" has no name in front at all, so the rate was
        # recorded as belonging to no product — and an item with no product is
        # one completeness cannot require and selection cannot narrow to, which
        # is how a question naming urea and potash came back with the potash
        # alone and "I don't have a figure for urea".
        sub = inside
        if not sub and offset is not None:
            ahead = _substance_after(clause, offset + (m.end() - m.start()))
            behind, starts, ends = _substance_before(clause, offset)
            taken = bool(behind and ahead
                         and _already_claimed(clause, starts, ends, offset))
            sub = (ahead if (ahead and (not behind or taken)) else behind)
        elif not sub:
            sub = _substance_in(clause, before=offset)
        out.append({
            "substance": sub,
            "low": low, "high": high,
            "kg_per_ha": low * factor,
            "kg_per_ha_high": high * factor if high else None,
            "source": m.group(0).strip(),
            "clause": clause.strip(),
            "document": document, "area_unit": area, "mass_unit": unit,
            "offset_in_clause": offset,
        })
    return out


# Which family a rate belongs to, for deciding whether it answers the question at
# all. NUTRIENT and PRODUCT are one family here and one family only here: for
# SCALING they behave identically (both are "kg of something per hectare"), while
# answer_grounding keeps them strictly apart because converting one into the other
# needs a composition the documents must state. Multiplying by an area does not
# convert anything, so the distinction does not arise.
_FAMILY = {
    "seed_rate": "seed",
    "fertiliser_nutrient_rate": "fertiliser",
    "fertiliser_product_rate": "fertiliser",
    "pesticide_dose": "pesticide",
    "herbicide_dose": "herbicide",
    "irrigation_quantity": "irrigation",
}
_SEED_RE = re.compile(r"\bseeds?\b", re.IGNORECASE)


def _family_of(rate: dict) -> str:
    """The family a parsed rate belongs to, from its substance and its clause."""
    sub = (rate.get("substance") or "").lower()
    if sub == "seed" or (not sub and _SEED_RE.search(rate.get("clause", ""))):
        return "seed"
    if sub:
        return "fertiliser"
    return ""


def _asked_families(question: str) -> set:
    """Families the QUESTION is about. Imported late — area_math is the lower
    layer and must not depend on answer_grounding at import time."""
    try:
        from .answer_grounding import asked_kinds
    except Exception:                                    # pragma: no cover
        return set()
    return {_FAMILY[k] for k in asked_kinds(question) if k in _FAMILY}


_AREA_UNIT_ALIASES = {"ha": "hectare", "hectare": "hectare", "hectares": "hectare",
                      "acre": "acre", "acres": "acre",
                      "decimal": "decimal", "decimals": "decimal"}


def _same_area_basis(rate: dict, unit: str) -> bool:
    """Is this rate already expressed per the area the farmer asked about?

    Then there is nothing to work out. Converting it to hectares and back
    returns the number it started as, and calling that a computed total invites
    the answer to present a quoted recommendation as arithmetic — or, when the
    two differ by rounding, to state a figure the document does not contain.
    """
    return (_AREA_UNIT_ALIASES.get((rate.get("area_unit") or "").lower())
            == _AREA_UNIT_ALIASES.get((unit or "").lower()))


def _method_of(text: str) -> set:
    """The cultivation methods `text` names — direct seeded, transplanting…"""
    try:
        from .answer_grounding import conditions_in
    except Exception:                                    # pragma: no cover
        return set()
    return {phrase for kind, phrase in conditions_in(text or "")
            if kind == "method"}


def _for_this_system(question: str, rates: list) -> list:
    """Rates from documents about the cultivation method the farmer named.

    A direct-seeded question retrieves the transplanted manual too — the two
    share every product name — and with no boundary between them their rates
    went into one list. The answer then quoted a DSR figure and a transplanted
    figure side by side, both correct in their own document and not a
    recommendation anyone wrote.

    Only fires when the farmer named a method AND some rate belongs to it.
    Rates from a document that names no method at all are general and stay.
    """
    asked = _method_of(question)
    if not asked:
        return rates
    mine, general = [], []
    for r in rates:
        theirs = _method_of(r.get("clause", "")) | _method_of(r.get("document", ""))
        if theirs & asked:
            mine.append(r)
        elif not theirs:
            general.append(r)
    if not mine:
        return rates                       # nothing is method-specific; unchanged
    # ONE SET OF DOCUMENTS. A rate that names no method is only general WITHIN
    # the document that answers the question; the same words in another manual
    # are another crop's schedule. The transplanted manual's "40 kg/ha urea at 22
    # days after transplanting" names a STAGE rather than a method, so the method
    # test alone let it through — and it arrived in the answer as "16.2 kg",
    # beside the DSR document's own per-acre figures.
    documents = {r.get("document") for r in mine}
    general = [r for r in general if r.get("document") in documents]
    kept = mine + general
    dropped = len(rates) - len(kept)
    if dropped:
        log.info("area_math: dropped %d rate(s) not from the %s document(s) for %s",
                 dropped, ", ".join(sorted(documents))[:60], ", ".join(sorted(asked)))
    return kept


def _substance_gap(question: str, rate: dict) -> dict | None:
    """Is the farmer asking for a different SUBSTANCE from the one being scaled?

    Multiplying by an area never changes what a quantity is OF, so a rate given
    in a NUTRIENT stays a nutrient however many acres it is scaled to. Asked
    "how much urea for 1 acre?" against a source that says 80 kg N/ha, the
    arithmetic is still 32.4 — of nitrogen. Handing that over as the answer lets
    a farmer buy 32.4 kg of urea, which is roughly a third of the nitrogen the
    documents called for, because urea is about 46% N.

    Crossing that line needs a formulation the documents must state. This
    reports the gap so the caller can say so instead of scaling silently;
    rate_conversion.detect_conversion() makes the same check for a rate the
    farmer supplied themselves.
    """
    try:
        from .rate_conversion import _asked_substance, _kind_of_substance
    except Exception:                                    # pragma: no cover
        return None
    want = _asked_substance(question)
    have = (rate.get("substance") or "").lower()
    if not (want and have):
        return None
    want_kind, have_kind = _kind_of_substance(want), _kind_of_substance(have)
    if not (want_kind and have_kind) or want_kind == have_kind:
        return None
    return {"asked": want, "asked_kind": want_kind,
            "have": have, "have_kind": have_kind}


def _asks_one_stage(question: str) -> bool:
    """Did the farmer ask about ONE moment in the season?

    Then a rate for another stage really is inapplicable and must say so. Asked
    for the programme as a whole, the stage is the timetable, not a caveat.
    """
    return any(kind == "stage" for kind, _ in _conditions_of(question))


def _conditions_of(text: str) -> set:
    """conditions_in(), imported late — area_math is the lower layer."""
    try:
        from .answer_grounding import conditions_in
    except Exception:                                    # pragma: no cover
        return set()
    return conditions_in(text or "")


def _stage_slice(clause: str, offset) -> str:
    """The part of `clause` governed by the stage nearest BEFORE `offset`.

    One sentence of a schedule can carry two stages:

        "At the active tillering stage, apply Urea 30-35 kg/acre, and at panicle
         initiation, apply Urea 30-35 kg/acre along with MOP 15-20 kg/acre."

    Read whole, every figure in it inherits BOTH — so the MOP that belongs to
    panicle initiation was reported as "given only panicle initiation; tillering",
    a qualifier that is true of no recommendation in the document and reads as a
    contradiction. Each figure is governed by the stage introduced last before it.
    """
    if offset is None:
        return clause
    try:
        from .answer_grounding import _CONDITION_RX
    except Exception:                                    # pragma: no cover
        return clause
    starts = sorted({m.start() for rx in _CONDITION_RX.get("stage", ())
                     for m in rx.finditer(clause)})
    if len(starts) < 2:
        return clause                      # one stage or none — nothing to split
    begin = 0
    for pos in starts:
        if pos <= offset:
            begin = pos
        else:
            return clause[begin:pos]
    return clause[begin:]


def _unmet_conditions(question: str, clause: str, offset=None) -> list:
    """Conditions on this rate that the farmer has NOT stated.

    A rate given "at early tillering to promote tillering" is a real rate and a
    real recommendation — but it is not the answer to "how much fertiliser for my
    1.5 acres", and putting its total in front of the model alongside the general
    rate is how an unasked-for stage recommendation gets into the answer.
    """
    try:
        from .answer_grounding import conditions_in, purposes_in, _states_condition
    except Exception:                                    # pragma: no cover
        return []
    clause = _stage_slice(clause, offset)
    unmet = sorted({p for _, p in conditions_in(clause)
                    if not _states_condition(question, p)})
    q_low = (question or "").lower()
    unmet += sorted({f"to {p}" for p in purposes_in(clause)
                     if not re.search(rf"\b{re.escape(p.split()[0])}", q_low)})
    return unmet


# ── The same rate, per a different area ───────────────────────────────────
#
# scale_for_area() answers "how much for MY FIELD" and needs a field size to do
# it. "How much compost should I apply per acre?" gives no field size and is not
# asking for a total — it asks for the same recommendation with a different
# denominator, and with the documents saying "10-12 cartloads per hectare" the
# farmer was handed the hectare figure or refused.
#
# WHAT MAY AND MAY NOT CHANGE
# ---------------------------
# ONLY THE DENOMINATOR. The numerator is carried across untouched, whatever it
# is, which is what makes this safe for a unit with no defined size: 10 cartloads
# per hectare is 4 cartloads per acre because an acre is a fixed fraction of a
# hectare, and that arithmetic needs to know nothing whatever about how much a
# cartload weighs. A cartload therefore stays a cartload — it can never become a
# tonne here, because no code path converts the numerator at all.
#
# It also declines rather than guesses when the target area has no fixed size,
# for the same reason parse_area() does.

# A target area named with no quantity in front of it. "per acre", "/acre",
# "in one hectare" — a UNIT, not an amount.
_BARE_AREA_RE = re.compile(
    r"(?:per|/|in|for|to)\s+(?:one\s+|a\s+|1\s+)?("
    + "|".join(re.escape(w) for _, ws in _AREA_WORDS for w in ws)
    + r")\b", re.IGNORECASE)

# answer_grounding writes the denominator in its own short forms; _HA_PER is
# keyed by the long ones. Mapping them here rather than widening either keeps
# each module's vocabulary its own.
_AREA_CANON_LONG = {"ha": "hectare", "acre": "acre", "decimal": "decimal",
                    "cent": "cent", "gunta": "gunta", "m2": "square metre"}

# How precisely to state a rebased figure. A COUNT has no fractional form a
# farmer can act on — half a cartload is not a thing you load — so counts round
# to whole numbers and everything measured rounds to one decimal.
_COUNTABLE = re.compile(r"^(?:cart\s?load|bullock\s?cart\s?load|bag|trolley|"
                        r"basket|tractor\s?load)s?$", re.IGNORECASE)


def _rebase_value(value: float, countable: bool) -> str:
    return f"{round(value):g}" if countable else _fmt(value)


def _kind_phrase(figure: dict, substance: str) -> str:
    """What to call the converted figure, in the grounding layer's own words.

    Taken from the SOURCE figure's kinds so the label is the one the documents
    earned — a yield stays a yield. Only when the source carries no kind at all
    does it fall back to naming the family from the substance.
    """
    try:
        from .answer_grounding import kind_label
    except Exception:                                    # pragma: no cover
        return _kind_word(substance)
    kinds = figure.get("kinds") or ()
    return kind_label(kinds[0]) if kinds else _kind_word(substance)


def rebase_to_area(question: str, context: str) -> dict | None:
    """The CONTEXT's per-area figures restated per the area the QUESTION names.

    Returns None unless the question names a target area unit WITHOUT a size (a
    size means a total is wanted, which is scale_for_area's job) and the context
    holds a figure quoted per some OTHER fixed area unit.
    """
    # A FIELD SIZE OF EXACTLY ONE UNIT IS ALSO A REQUEST FOR THE RATE. "How much
    # N for 1 acre?" and "how much N per acre?" are the same question and have
    # the same answer, and only the second used to reach this function — so the
    # first was answered with a bare total ("32.4 kg"), the model naturally wrote
    # it as "32.4 kg per acre", and the guard rejected a figure differing from
    # the evidence only in carrying the unit the farmer had asked for.
    #
    # Any LARGER field is a different question: 2.5 acres wants 80.9 kg in total,
    # not a rate, and running both would put two numbers in front of the farmer.
    field = parse_area(question)
    if field and not (field.get("value") == 1 and not field.get("ambiguous")):
        return None
    m = _BARE_AREA_RE.search(question or "")
    if not m:
        return None
    target = _CANON_BY_WORD[m.group(1).lower()]
    target_ha = _HA_PER.get(target)
    if not target_ha:
        return None

    # Only figures of the kind that was asked about. A yield quoted per hectare
    # sitting in the same chunk is not an answer to "how much compost per acre",
    # and rebasing it would put a second, larger number in front of the farmer
    # wearing the units of the first.
    try:
        from .answer_grounding import (_SUBSTANCE_RE, _segment_around,
                                       asked_kinds, quantities_in)
    except Exception:                                    # pragma: no cover
        return None
    asked = set(asked_kinds(question))

    figures = quantities_in(context)
    # THE INPUT THEY NAMED, when they named one. A farmer asking about compost
    # gets the compost rate converted, not every fertiliser rate on the page —
    # each of which is a fertiliser rate of the kind asked for, so the kind
    # filter alone let seven conversions through where one was wanted.
    #
    # Two tiers, and the second is CLAUSE-scoped rather than window-scoped. A
    # document writes "farmyard manure (FYM) or compost", and the canonical
    # substance keeps only the first of those names, so a question about compost
    # finds nothing in the first tier and has to look at the wording. It may not
    # look far: a 90-character window reaches over a chunk boundary into the
    # neighbouring document, which made an unrelated urea rate match a question
    # about compost.
    named = {m.group(0).lower() for m in _SUBSTANCE_RE.finditer(question or "")}
    if named:
        exact = [q for q in figures if (q.get("substance") or "").lower() in named]
        nearby = [q for q in figures
                  if named & {m.group(0).lower() for m in _SUBSTANCE_RE.finditer(
                      _segment_around(q["flat"], q["at"], q["at"] + len(q["text"])))}]
        figures = exact or nearby or figures

    lines, evidence = [], []
    for q in figures:
        unit, _, area = (q.get("unit") or "").partition("/")
        if not area:
            continue                   # no denominator to change
        # Compared in ONE vocabulary. answer_grounding writes "ha" and the target
        # comes back as "hectare", so comparing the surfaces made a per-hectare
        # rate look like a rate for some other area: it was "converted" by a
        # factor of one and restated as "5.5-6 t/hectare", a second figure
        # identical to the first and phrased as though it were a finding.
        source = _AREA_CANON_LONG.get(area, area)
        if source == target:
            continue                   # already the area they asked about
        source_ha = _HA_PER.get(source)
        if not source_ha:
            continue                   # an area we do not convert (m2, gunta text)
        if asked and q["kinds"] and not (set(q["kinds"]) & asked):
            continue
        factor = target_ha / source_ha
        countable = bool(_COUNTABLE.match(unit))
        low = _rebase_value(float(q["value"]) * factor, countable)
        high = (_rebase_value(float(q["value_end"]) * factor, countable)
                if q.get("value_end") else None)
        # A range that collapses to a single figure after rounding is stated once
        # rather than as "4 to 4".
        amount = f"{low}-{high}" if high and high != low else low
        what = (q.get("substance") or "").strip()
        label = f"{what} " if what else ""
        lines.append(f"{label}{amount} {unit}/{target} "
                     f"(source rate {q['text']})")
        # THE EVIDENCE HAS TO SAY WHAT KIND OF FIGURE IT IS, for the same reason
        # the scaled totals do: the grounding guard labels a figure from the
        # words around it, and a converted rate written without one was read as
        # an unlabelled number — which grounds nothing that claims a kind, so the
        # answer stating it was reported as unsupported. The kind comes from the
        # SOURCE figure's own kinds, never from the substance, or a yield rebased
        # per acre would be written up as a fertiliser dose.
        kind_phrase = _kind_phrase(q, what)
        evidence.append(
            f"For one {target} the converted {kind_phrase} is: "
            f"{_long_name(what) if what else 'the recommended input'} = "
            f"{amount} {unit} per {target}, converted from the source rate "
            f"{q['text']} using 1 {target} = "
            f"{_fmt_area(target_ha / source_ha)} of the area it is given for. "
            f"The unit of the amount is unchanged: {unit} in the source, "
            f"{unit} in the converted figure.")

    if not lines:
        return None
    log.info("area_math: rebased %d rate(s) to per-%s", len(lines), target)
    return {"target": target, "lines": lines, "evidence": "\n".join(evidence)}


def _is_statement(question: str) -> bool:
    """Is the farmer TELLING us a rate rather than asking for one?"""
    try:
        from .completeness import STATEMENT_WITH_RATES, question_type
    except Exception:                                    # pragma: no cover
        return False
    return question_type(question or "") == STATEMENT_WITH_RATES


def question_scope(question: str) -> dict:
    """What the question narrows the answer to, in one place.

    system     the cultivation method named — direct seeded, transplanting
    stages     the crop stages named
    substances the products named
    area       the field size, when one was given
    """
    return {
        "system": sorted(_method_of(question)),
        "stages": sorted({p for kind, p in _conditions_of(question)
                          if kind == "stage"}),
        "substances": sorted({_canon_product(m.group(0))
                              for m in _SUBSTANCE_NEAR.finditer(question or "")}),
        "area": parse_area(question),
    }


def _preferred_by_scope(scope: dict, rates: list) -> tuple:
    """(kept, excluded) — the rates that answer the question at its own scope.

    Ordered preference, strongest first, and the first band that has anything in
    it is the answer:

        1. this system AND the stage that was named
        2. this system, no stage of its own (general within the system)
        3. anything left, which is the generic advice

    A GENERIC RATE NEVER OVERRIDES A SYSTEM-SPECIFIC ONE. Asked what to apply at
    panicle initiation in DSR, the answer came back "for general rice
    recommendations, apply 20-25 kg/ha MOP" while the DSR document's own
    "MOP 15-20 kg/acre at panicle initiation" sat in the same context — the two
    are different practices in different units, and the farmer asked for one of
    them by name.
    """
    stages = set(scope.get("stages") or ())
    if not stages:
        return rates, []

    at_stage, no_stage, elsewhere = [], [], []
    for r in rates:
        own = {p for kind, p in _conditions_of(
            _stage_slice(r.get("clause", ""), r.get("offset_in_clause")))
            if kind == "stage"}
        (at_stage if own & stages else no_stage if not own else elsewhere).append(r)

    if at_stage:
        return at_stage, no_stage + elsewhere
    if no_stage:
        return no_stage, elsewhere
    return rates, []


def select_rates(question: str, context: str) -> dict:
    """The rates that answer THIS question, whether or not a field size is given.

    Everything the answer stage is allowed to use, chosen once: the right family
    of figure, the right cultivation system, one set of documents, and the scope
    the question narrowed itself to.

    This used to live inside scale_for_area(), which returns early when the
    question names no field size — so "what fertilizer should I apply in DSR?"
    selected nothing at all. Nothing was filtered by system, nothing carried
    provenance, and the completeness check had no items to require, which is why
    a generic per-hectare rate could answer a question that named a cultivation
    system and why half a schedule counted as an answer.
    """
    rates = rates_in(context)
    if not rates:
        return {"rates": [], "excluded": [], "scope": question_scope(question)}

    # Only rates that answer the question that was asked. A seed rate retrieved
    # alongside a fertiliser rate used to be scaled too, so a farmer asking how
    # much fertiliser their 1.5 acres needs was handed a seed total as well.
    families = _asked_families(question)
    if families:
        kept = [r for r in rates if _family_of(r) in families or not _family_of(r)]
        if kept:
            rates = kept

    scope = question_scope(question)
    before = list(rates)
    rates = _for_this_system(question, rates)          # ONE CULTIVATION SYSTEM
    excluded = [(r, "written for another cultivation system or document")
                for r in before if r not in rates]

    if scope["substances"]:
        mine = [r for r in rates
                if _canon_product(r.get("substance") or "") in scope["substances"]]
        # A MESSAGE THAT STATES ITS OWN RATES IS SCOPED TO THEM, and to nothing
        # else. "Apply 20-25 kg/ha MOP and 15-20 kg/ha urea" names two products
        # and asks for none — answering it with every other product in the
        # documents replies to a question the farmer did not ask. Empty is a
        # legitimate result here: with no product of theirs in the documents,
        # there is nothing of theirs to say.
        if mine or _is_statement(question):
            excluded += [(r, "a different product from the one stated")
                         for r in rates if r not in mine]
            rates = mine

    rates, dropped = _preferred_by_scope(scope, rates)
    excluded += [(r, "written for another stage than the one asked about")
                 for r in dropped]

    for r, why in excluded:
        log.info("excluded rate: %s %s%s %s/%s | %s | excluded because %s",
                 r.get("substance") or "?", f"{r['low']:g}",
                 f"-{r['high']:g}" if r.get("high") else "",
                 r.get("mass_unit"), r.get("area_unit"), r.get("document"), why)
    return {"rates": rates, "excluded": excluded, "scope": scope}


def _item_of(rate: dict, area_unit: str, amount: str, stated: bool) -> dict:
    """One selected recommendation, with everything needed to check it later."""
    return {
        "substance": rate["substance"] or "the recommended input",
        "low": rate["low"], "high": rate.get("high"),
        "mass_unit": rate.get("mass_unit"), "area_unit": area_unit,
        "amount": amount,
        "stages": sorted({p for kind, p in _conditions_of(
            _stage_slice(rate.get("clause", ""), rate.get("offset_in_clause")))
            if kind == "stage"}),
        "system": sorted(_method_of(rate.get("clause", ""))
                         | _method_of(rate.get("document", ""))),
        "document": rate.get("document"), "source": rate.get("source"),
        "provenance": "stated" if stated else "derived",
    }


def selected_items(question: str, context: str) -> list:
    """The recommendations this question is owed, AS THE DOCUMENTS STATE THEM.

    No arithmetic: every item keeps the document's own figure and its own area
    basis. This is the path for a question that names a system or a stage but no
    field size — the commonest shape there is, and the one that used to select
    nothing at all because the whole selection layer sat inside the arithmetic.
    """
    selection = select_rates(question, context)
    items = []
    for r in selection["rates"]:
        amount = (f"{_fmt(r['low'])}-{_fmt(r['high'])}" if r.get("high")
                  else _fmt(r["low"]))
        item = _item_of(r, r.get("area_unit") or "acre", amount, stated=True)
        items.append(item)
        log.info("selected_rate: substance=%s value=%s unit=%s/%s system=%s "
                 "stage=%s provenance=stated document=%s",
                 item["substance"], amount, item["mass_unit"], item["area_unit"],
                 ", ".join(item["system"]) or "any",
                 ", ".join(item["stages"]) or "any", item["document"])
    return items


def scale_for_area(question: str, context: str) -> dict | None:
    """Work out what the retrieved rates mean for the farmer's own field.

    Returns None when there is nothing to compute — no area given, an area in an
    ambiguous unit, or no scalable rate in the documents. In those cases the
    caller must NOT invent a total; the existing rules already cover giving the
    per-area rate as it stands, or asking.
    """
    area = parse_area(question)
    if not area:
        return None
    if area.get("ambiguous"):
        log.info("area_math: %r has no fixed size — not converting", area["ambiguous"])
        return {"ambiguous": area["ambiguous"], "text": area["text"], "lines": []}

    # ONE SELECTION, shared with the no-area path. See select_rates().
    rates = select_rates(question, context)["rates"]
    if not rates:
        return None

    ha = area["hectares"]
    lines, evidence, conditional, gaps, items = [], [], [], [], []
    for r in rates:
        # ASKED FOR A DIFFERENT SUBSTANCE. Scaling never changes what a quantity
        # is OF, so this stays the nutrient the source gave and must be labelled
        # as one — see _substance_gap().
        gap = _substance_gap(question, r)
        if gap:
            gaps.append(gap)
        name = r["substance"] or "the recommended input"
        # ALREADY PER THE AREA THEY ASKED ABOUT. Quote it, do not derive it —
        # see _same_area_basis(). The whole recommendation is the document's,
        # including its range, and the answer should read as a quotation.
        stated = _same_area_basis(r, area["unit"]) and area["value"] == 1
        if stated:
            amount = (f"{_fmt(r['low'])} to {_fmt(r['high'])} kg"
                      if r.get("high") else f"{_fmt(r['low'])} kg")
        else:
            total_low = r["kg_per_ha"] * ha
            total_high = (r.get("kg_per_ha_high") or 0) * ha
            if total_high and total_high != total_low:
                amount = f"{_fmt(total_low)} to {_fmt(total_high)} kg"
            else:
                amount = f"{_fmt(total_low)} kg"
        unmet = _unmet_conditions(question, r.get("clause", ""),
                                  r.get("offset_in_clause"))
        # EVERY RATE THAT REACHES GENERATION, on the record. When an answer omits
        # one, this is what says whether it was ever offered.
        stages = sorted({p for kind, p in _conditions_of(
            _stage_slice(r.get("clause", ""), r.get("offset_in_clause")))
            if kind == "stage"})
        # THE SELECTED EVIDENCE, structured. Everything downstream that needs to
        # know what was offered — the completeness check most of all — reads this
        # rather than parsing the prose lines back apart.
        items.append({
            "substance": name, "low": r["low"], "high": r.get("high"),
            "mass_unit": r.get("mass_unit"),
            "area_unit": area["unit"] if stated else "ha",
            "amount": amount, "stages": stages,
            "document": r.get("document"), "source": r.get("source"),
            "provenance": "stated" if stated else "derived",
        })
        log.info("selected_rate: substance=%s value=%s unit=%s/%s system=%s "
                 "stage=%s provenance=%s document=%s",
                 name, f"{_fmt(r['low'])}-{_fmt(r['high'])}" if r.get("high")
                 else _fmt(r["low"]), r.get("mass_unit"), r.get("area_unit"),
                 ", ".join(sorted(_method_of(r.get("clause", "")))) or "any",
                 ", ".join(stages) or "any",
                 "stated" if stated else "derived", r.get("document"))
        provenance = f"{r['source']} — {r['document']}"
        line = ((f"{name}: {amount} per {area['unit']}, as stated in "
                 f"{r['document']} ({r['source']})")
                if stated else
                (f"{name}: {amount} for the whole {area['text']} "
                 f"(derived from {provenance})"))
        if unmet:
            # A STAGE IS NOT A DISQUALIFIER WHEN THE WHOLE PROGRAMME WAS ASKED
            # FOR. "but the documents give this only tillering" is the right
            # warning for a farmer who asked one question about one moment, and
            # for "what do I apply and how much" it labels every line of the
            # schedule as something not quite applicable — which is how the last
            # stages came to be dropped from the answer. Same figures, same
            # provenance, read as the timetable it is.
            stage_only = all(kind == "stage" for kind, _ in _conditions_of(
                _stage_slice(r.get("clause", ""), r.get("offset_in_clause"))))
            if stage_only and not _asks_one_stage(question):
                conditional.append(f"{line} — apply this at {'; '.join(unmet)}")
            else:
                conditional.append(f"{line} — but the documents give this only "
                                   f"{'; '.join(unmet)}")
        else:
            lines.append(line)
        # Written into the CONTEXT so the grounding guard treats these as
        # evidence — they are derived by this function, not by the model.
        #
        # The wording matters: the guard labels a figure from the words around it,
        # so the evidence has to SAY it is a dose and name the input. Written any
        # more tersely, the totals we just computed get rejected as unlabelled.
        # Opens with "For", not with a capitalised content word. A line beginning
        # "Computed fertiliser dose ..." was read by the variety guard as a
        # recommendation belonging to a variety called "Computed", which made
        # every derived total look variety-specific and switched the guard off.
        if stated:
            evidence.append(
                f"For {area['text']} the {_kind_word(name)} is stated directly by "
                f"the source: {_long_name(name)} = {amount} per {area['unit']}, "
                f"quoted from {r['document']} which gives it as {r['source']}. "
                f"Nothing was calculated.")
        else:
            evidence.append(
                f"For {area['text']} ({_fmt_area(ha)} ha) the computed "
                f"{_kind_word(name)} is: {_long_name(name)} = {amount} total, "
                f"derived from the source rate {r['source']} in {r['document']}.")

    working = (f"{_fmt(area['value'])} {area['unit']} x "
               f"{_HA_PER[area['unit']]} = {_fmt_area(ha)} ha")
    if area["unit"] in ("decimal", "cent"):
        working = (f"{_fmt(area['value'])} {area['unit']} = "
                   f"{_fmt(area['value'] / 100)} acre; {working}")

    # When EVERY rate carries a condition the farmer has not stated, there is no
    # general recommendation to prefer — so the conditioned ones are the answer,
    # with their condition attached. Refusing here would leave the farmer with
    # nothing when the documents do have a figure for them.
    if not lines and conditional:
        lines, conditional = conditional, []

    log.info("area_math: scaled %d rate(s) to %s (%s ha)%s",
             len(lines), area["text"], _fmt(ha),
             f" (+{len(conditional)} conditional)" if conditional else "")
    return {"area": area, "hectares": ha, "lines": lines,
            "conditional": conditional, "substance_gaps": gaps, "items": items,
            "working": working, "evidence": "\n".join(evidence)}
