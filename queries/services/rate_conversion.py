"""
rate_conversion.py
==================
The deterministic path for a question that is ALREADY a complete sum.

WHY THIS EXISTS
---------------
"If the fertilizer rate is 80 kg/ha, how much is required for 1 acre?" was being
answered "please tell me the variety, and whether you are transplanting or direct
seeding". Nothing about the variety changes 80 x 0.404686. The farmer supplied
both inputs; the only thing left to do is the multiplication.

Two separate faults produced that:

  1. The applicability guard treats every quantity question as an agricultural
     RECOMMENDATION, so it asked for the stage/method/variety that a retrieved
     recommendation would need. A supplied rate needs none of them.
  2. area_math.scale_for_area() reads rates ONLY from the retrieved context. Given
     the question above it ignored the farmer's 80 kg/ha entirely and scaled a
     seed rate out of an unrelated chunk — answering a fertiliser question with
     "8.1 to 10.1 kg of seed". A wrong number wearing the right units.

So this module answers from the QUESTION, and the pipeline routes to it before the
evidence and applicability guards run.

WHAT IT DOES NOT BECOME
-----------------------
A hole through which the model can invent a rate. The bypass requires the rate to
be written IN THE QUESTION — number, mass unit and per-area unit. No rate in the
question means no bypass, so "how much seed is required for BINA Dhan 11?" still
goes through retrieval and still refuses. The farmer's own number is an input to
arithmetic, never evidence for an agricultural recommendation.

It also declines, rather than guesses, when:

  * the target area is in a unit with no fixed size (bigha, katha)
  * the question reports a PROBLEM — lodging, disease, a deficiency. Then the
    answer the farmer needs is not the product of two numbers, and the normal
    evidence path must handle it.

PUBLIC API
----------
    convert_rate_to_total_quantity(rate, rate_unit, area, area_unit) -> dict
    convert_rate_to_rate(rate, rate_unit, target_area_unit)          -> dict
    detect_conversion(question)                                      -> dict | None
    validate_conversion_answer(answer, conv)                         -> list[str]
"""
from __future__ import annotations

import logging
import re

from .area_math import _HA_PER, _ODIA_DIGITS, parse_area

log = logging.getLogger("irri")

# Mass, and the two volume units a dose is written in. Everything is carried in
# the unit the farmer used — converting kg to g as well as ha to acre would be a
# second conversion nobody asked for.
_MASS_TO_BASE = {
    "kg": 1.0, "kgs": 1.0, "kilo": 1.0, "kilos": 1.0,
    "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gm": 0.001, "gms": 0.001, "gram": 0.001, "grams": 0.001,
    "t": 1000.0, "ton": 1000.0, "tons": 1000.0, "tonne": 1000.0, "tonnes": 1000.0,
    "q": 100.0, "quintal": 100.0, "quintals": 100.0,
    "l": 1.0, "litre": 1.0, "litres": 1.0, "liter": 1.0, "liters": 1.0,
    "ml": 0.001,
}

# The per-area part of the rate the farmer wrote. Deliberately narrower than the
# target-area vocabulary: a rate is quoted per hectare or per acre, and letting
# "per bigha" in here would put an unfixed unit on the left of the sum too.
_RATE_AREA_TO_HA = {
    "ha": 1.0, "hectare": 1.0, "hectares": 1.0, "hect": 1.0,
    "acre": 0.404686, "acres": 0.404686, "ac": 0.404686,
    "decimal": 0.00404686, "decimals": 0.00404686,
}

_MASS_RE = "|".join(sorted(_MASS_TO_BASE, key=len, reverse=True))
_RATE_AREA_RE = "|".join(sorted(_RATE_AREA_TO_HA, key=len, reverse=True))

# The rate as it is actually written. `mid` is what lets "80 kg nitrogen per
# hectare" parse — the same phrasing that, when it was unsupported, made every
# rate lose its per-area part. No digit or sentence break may sit inside it, so
# "20 kg urea and 30 kg per acre" cannot bind the acre to the 20.
_RATE_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_MASS_RE})\b"
    rf"(?P<mid>[^.;,:\d]{{0,25}}?)"
    rf"\s*(?:/|per\s+|\s+a\s+)\s*(?P<area>{_RATE_AREA_RE})\b",
    re.IGNORECASE)

# "80:40:40 kg/ha" — one rate written as three.
_NPK_RE = re.compile(
    rf"(?P<n>\d+(?:\.\d+)?)\s*:\s*(?P<p>\d+(?:\.\d+)?)\s*:\s*(?P<k>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_MASS_RE})\b[^.;\d]{{0,25}}?(?:/|per\s+)\s*(?P<area>{_RATE_AREA_RE})\b",
    re.IGNORECASE)

_SUBSTANCE_RE = re.compile(
    r"\b(urea|dap|mop|muriate of potash|ssp|npk|potash|nitrogen|phosphorus|"
    r"phosphate|potassium|zinc sulphate|zinc|sulphur|gypsum|lime|fym|"
    r"farmyard manure|vermicompost|compost|seed|fertili[sz]er)\b", re.IGNORECASE)

_NUTRIENT_LETTER = {"n": "nitrogen", "p": "phosphorus", "k": "potassium"}

# A NUTRIENT is an element; a PRODUCT is the bag it comes in. Urea is about 46%
# nitrogen, so 48.6 kg of nitrogen is roughly 106 kg of urea — answering one with
# the other is out by more than half, in the direction of under-feeding the crop.
# Multiplying by an area never crosses this line: the farmer's rate stays whatever
# substance it already was.
_NUTRIENTS = frozenset({"nitrogen", "phosphorus", "phosphate", "potassium",
                        "zinc", "sulphur", "boron", "iron", "npk"})
_PRODUCTS = frozenset({"urea", "dap", "mop", "muriate of potash", "ssp", "tsp",
                       "potash", "gypsum", "lime", "compost", "fym",
                       "farmyard manure", "vermicompost", "zinc sulphate"})


def _kind_of_substance(name: str) -> str:
    low = (name or "").lower().strip()
    if low in _NUTRIENTS:
        return "nutrient"
    if low in _PRODUCTS:
        return "product"
    return ""


def _asked_substance(question: str) -> str:
    """The substance the QUESTION wants the answer in."""
    for m in _SUBSTANCE_RE.finditer(question or ""):
        word = m.group(0).lower()
        if _kind_of_substance(word):
            return word
    return ""

# The question has to be ASKING FOR A NUMBER. Without this, any passing mention of
# a rate would divert an ordinary agronomy question into the arithmetic path.
#
# Deliberately excludes a bare "required" / "needed": "I applied 50 kg/ha urea on
# 1 acre, is more needed?" contains a rate and an area but is not a conversion —
# answering it with 20.2 kg would be answering a question nobody asked.
_ASKS_QUANTITY_RE = re.compile(
    r"\b(how much|how many|how\s+much\s+quantity|"
    r"what\s+(?:is|are|will\s+be|would\s+be)\s+the\s+"
    r"(?:equivalent|total|amount|quantity|rate|dose|requirement)|"
    r"equivalent\s+rate|convert|calculate|calculation|work\s+out|"
    r"total\s+(?:quantity|amount|requirement))\b", re.IGNORECASE)

# A reported problem takes the question out of arithmetic. "The rate is 80 kg/ha
# but my crop has lodged, how much should I use?" must NOT be answered with a
# multiplication — the evidence path has to say whether to apply it at all.
_PROBLEM_RE = re.compile(
    r"\b(lodg\w*|fall\w*en|disease[ds]?|infect\w*|pest\w*|insect\w*|blast|blight|"
    r"rot\w*|yellow\w*|drying|dried|wilt\w*|stunt\w*|droop\w*|deficien\w*|"
    r"damage[ds]?|infest\w*|toxicit\w*|dying|died)\b", re.IGNORECASE)

# "the equivalent rate per acre" — a target UNIT with no quantity in front of it.
# `parse_area` correctly finds nothing here, because there is no area to find.
_BARE_TARGET_RE = re.compile(
    rf"(?:per|/|in|to|for)\s+(?:one\s+|a\s+|1\s+)?({_RATE_AREA_RE}|gunta|guntha|"
    rf"cent|square\s+met(?:re|er)s?)\b", re.IGNORECASE)

_AMBIGUOUS_RE = re.compile(
    r"\b(bigha|katha|kattha|kanal|marla|ବିଘା|କଠା)\b", re.IGNORECASE)


def _round(v: float) -> float:
    """One decimal place — the precision a shop scale actually has."""
    return round(v + 0.0, 1)


def fmt(v: float) -> str:
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return s or "0"


def _ha_per_target(unit: str) -> float | None:
    return _HA_PER.get(unit) or _RATE_AREA_TO_HA.get(unit)


# ── The arithmetic ────────────────────────────────────────────────────────

def convert_rate_to_total_quantity(rate: float, rate_unit: str,
                                   area: float, area_unit: str) -> dict:
    """The total quantity for `area`, from a rate of `rate` `rate_unit`.

    `rate_unit` is a compound ("kg/ha", "kg per acre"); `area_unit` is a plain
    area word. Returns the figure plus the working, so the answer can show a
    farmer the same sum they would do by hand.
    """
    m = re.match(rf"\s*({_MASS_RE})\s*(?:/|per\s+)\s*({_RATE_AREA_RE})\s*$",
                 rate_unit or "", re.IGNORECASE)
    if not m:
        raise ValueError(f"unrecognised rate unit {rate_unit!r}")
    mass, src_area = m.group(1).lower(), m.group(2).lower()

    unit = (area_unit or "").lower().rstrip("s")
    ha_per = _ha_per_target(unit) or _ha_per_target(area_unit or "")
    if ha_per is None:
        raise ValueError(f"unrecognised area unit {area_unit!r}")

    src_ha = _RATE_AREA_TO_HA[src_area]
    target_ha = float(area) * ha_per
    # Per-hectare first, so a rate quoted per acre converts just as correctly.
    per_ha = float(rate) / src_ha
    total = per_ha * target_ha

    steps = []
    if ha_per != 1.0:
        steps.append(f"{fmt(float(area))} {unit} x {ha_per} = "
                     f"{target_ha:.4f}".rstrip("0").rstrip(".") + " ha")
    if src_ha != 1.0:
        steps.append(f"{fmt(float(rate))} {mass}/{src_area} = "
                     f"{fmt(per_ha)} {mass}/ha")
    steps.append(f"{fmt(per_ha)} x {target_ha:.6f}".rstrip("0").rstrip(".")
                 + f" = {fmt(total)} {mass}")

    return {
        "kind": "total",
        "total": _round(total),
        "exact": total,
        "mass_unit": mass,
        "source_value": float(rate),
        "source_unit": f"{mass}/{src_area}",
        "source_area_unit": src_area,
        "area_value": float(area),
        "area_unit": unit,
        "hectares": target_ha,
        "conversion_factor": ha_per / src_ha,
        "working": "; ".join(steps),
    }


def convert_rate_to_rate(rate: float, rate_unit: str, target_area_unit: str) -> dict:
    """The SAME rate expressed per a different area — not a total.

    "80 kg/ha is what per acre?" is 32.4 kg/ACRE. Keeping this separate from a
    total is the whole of rule 10: the two are numerically identical for one
    unit of area and mean entirely different things.
    """
    out = convert_rate_to_total_quantity(rate, rate_unit, 1.0, target_area_unit)
    out.update({
        "kind": "rate",
        "rate": out["total"],
        "rate_unit": f"{out['mass_unit']}/{out['area_unit']}",
        "working": (f"{fmt(float(rate))} {out['source_unit']} x "
                    f"{out['conversion_factor']} = {fmt(out['total'])} "
                    f"{out['mass_unit']}/{out['area_unit']}"),
    })
    return out


# ── Reading the question ──────────────────────────────────────────────────

def _substance_of(m: re.Match, question: str) -> str:
    """What the rate is OF, from inside the match first, then just before it."""
    inside = _SUBSTANCE_RE.search(m.group("mid") or "")
    if inside:
        return inside.group(0).lower()
    letter = re.fullmatch(r"\s*([NPK])\s*", m.group("mid") or "")
    if letter:
        return _NUTRIENT_LETTER[letter.group(1).lower()]
    before = question[max(0, m.start() - 30):m.start()]
    found = list(_SUBSTANCE_RE.finditer(before))
    if found:
        word = found[-1].group(0).lower()
        # "fertilizer rate is 80 kg/ha" names no substance worth repeating.
        return "" if word.startswith("fertili") else word
    return ""


def rates_in_question(question: str) -> list[dict]:
    """Rates the FARMER wrote. Not rates from any document."""
    flat = re.sub(r"\s+", " ", question or "").translate(_ODIA_DIGITS)
    out: list[dict] = []
    npk_spans = []

    for m in _NPK_RE.finditer(flat):
        npk_spans.append((m.start(), m.end()))
        mass, area = m.group("unit").lower(), m.group("area").lower()
        for label, raw in (("nitrogen", m.group("n")), ("phosphorus", m.group("p")),
                           ("potassium", m.group("k"))):
            out.append({"value": float(raw), "mass_unit": mass,
                        "area_unit": area, "substance": label,
                        "unit": f"{mass}/{area}", "text": m.group(0).strip()})

    for m in _RATE_RE.finditer(flat):
        if any(m.start() >= a and m.end() <= b for a, b in npk_spans):
            continue
        out.append({
            "value": float(m.group("value")),
            "mass_unit": m.group("unit").lower(),
            "area_unit": m.group("area").lower(),
            "substance": _substance_of(m, flat),
            "unit": f"{m.group('unit').lower()}/{m.group('area').lower()}",
            "text": m.group(0).strip(),
            "span": (m.start(), m.end()),
        })
    return out


def detect_conversion(question: str) -> dict | None:
    """Is this a deterministic conversion the farmer has fully specified?

    Returns None for everything else — every agricultural question keeps going
    through retrieval, the evidence check and the applicability check unchanged.
    """
    q = re.sub(r"\s+", " ", question or "").translate(_ODIA_DIGITS)
    if not q:
        return None

    rates = rates_in_question(q)
    if not rates:
        return None                     # no supplied rate -> not our business
    if not _ASKS_QUANTITY_RE.search(q):
        return None
    if _PROBLEM_RE.search(q):
        log.info("rate_conversion: %r reports a problem — leaving it to the "
                 "evidence path", q[:70])
        return None

    # The target area cannot be the rate's own denominator, so the rate spans
    # come out of the text before the area is looked for.
    stripped, last = [], 0
    for r in rates:
        a, b = r.get("span", (None, None))
        if a is None:
            continue
        stripped.append(q[last:a])
        last = b
    stripped.append(q[last:])
    remainder = " ".join(s for s in stripped if s) if len(stripped) > 1 else q
    for r in rates:
        if "span" not in r:
            remainder = remainder.replace(r["text"], " ")

    amb = _AMBIGUOUS_RE.search(remainder)
    area = parse_area(remainder)
    if area and area.get("ambiguous"):
        return {"kind": "ambiguous_area", "unit": area["ambiguous"],
                "text": area["text"], "rates": rates}
    if amb and not area:
        return {"kind": "ambiguous_area", "unit": amb.group(1).lower(),
                "text": amb.group(0), "rates": rates}

    target_unit, target_value, kind = None, None, None
    if area:
        target_unit, target_value, kind = area["unit"], area["value"], "total"
    else:
        bare = _BARE_TARGET_RE.search(remainder)
        if bare:
            unit = re.sub(r"\s+", " ", bare.group(1).lower()).rstrip("s")
            unit = {"square meter": "square metre", "guntha": "gunta"}.get(unit, unit)
            if _ha_per_target(unit) is not None:
                target_unit, target_value, kind = unit, 1.0, "rate"
    if not target_unit:
        return None

    results = []
    for r in rates:
        try:
            if kind == "rate":
                res = convert_rate_to_rate(r["value"], r["unit"], target_unit)
            else:
                res = convert_rate_to_total_quantity(
                    r["value"], r["unit"], target_value, target_unit)
        except ValueError as exc:
            log.info("rate_conversion: %s", exc)
            continue
        res["substance"] = r["substance"]
        res["source_text"] = r["text"]
        results.append(res)
    if not results:
        return None

    # Did they ask for the answer in a DIFFERENT substance from the rate they
    # gave? "The recommendation is 80 kg nitrogen/ha — how much urea for 1.5
    # acres?" is two questions: the area conversion, which is arithmetic, and
    # nitrogen-to-urea, which needs a composition the documents must state. The
    # first is answerable and the second is not, and saying "48.6 kg" without
    # saying which substance it is lets the farmer buy 48.6 kg of urea.
    gap = None
    # Searched in the REMAINDER, with the rate itself removed. Reading the whole
    # question finds the rate's own substance first — "the recommendation is 80 kg
    # NITROGEN per hectare, how much urea ...?" answered "nitrogen", so the
    # nitrogen-to-urea gap it exists to catch was never seen.
    want = _asked_substance(remainder)
    if want:
        for r in results:
            have = r.get("substance") or ""
            if have and _kind_of_substance(have) and _kind_of_substance(want) \
                    and _kind_of_substance(have) != _kind_of_substance(want):
                gap = {"asked": want, "asked_kind": _kind_of_substance(want),
                       "have": have, "have_kind": _kind_of_substance(have)}
                break

    log.info("rate_conversion: %d supplied rate(s) -> %s per %s %s%s",
             len(results), kind, fmt(target_value), target_unit,
             f" [asked for {gap['asked']}, rate is {gap['have']}]" if gap else "")
    return {"kind": kind, "area_value": target_value, "area_unit": target_unit,
            "area_text": (area or {}).get("text") or target_unit,
            "substance_gap": gap, "results": results}


# ── Checking what the model then wrote ────────────────────────────────────

_ANSWER_QTY_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_MASS_RE})\b"
    rf"(?P<per>\s*(?:/|per\s+)\s*(?:{_RATE_AREA_RE}|bigha|gunta|cent))?",
    re.IGNORECASE)


def _tolerance(expected: float) -> float:
    """Enough slack for honest rounding, not enough for a different number."""
    return max(0.11, abs(expected) * 0.005)


def validate_conversion_answer(answer: str, conv: dict) -> list[str]:
    """Problems with a generated conversion answer.

    Three failures matter. A quantity that is not the computed total and not the
    supplied rate is a number from nowhere. A total wearing a per-area unit
    ("48.6 kg/acre" for the total on 1.5 acres) silently turns an answer into a
    rate three times too small. And an answer that states NO figure at all is the
    original complaint — the sum was already done, so "tell me the variety and
    whether you are transplanting" is a wrong answer, not a cautious one.
    """
    flat = re.sub(r"\s+", " ", answer or "")
    problems: list[str] = []
    if conv.get("kind") == "ambiguous_area" or not conv.get("results"):
        return problems

    wants_rate = conv["kind"] == "rate"
    totals = [(r["exact"], r) for r in conv["results"]]
    sources = [(r["source_value"], r) for r in conv["results"]]
    stated = set()

    for m in _ANSWER_QTY_RE.finditer(flat):
        value = float(m.group("value"))
        per = (m.group("per") or "").strip()
        matched_total = next((r for v, r in totals if abs(value - v) <= _tolerance(v)), None)
        matched_src = next((r for v, r in sources if abs(value - v) <= _tolerance(v)), None)

        if matched_total is not None:
            stated.add(id(matched_total))
            if wants_rate:
                continue                     # a rate answer SHOULD carry a unit
            if per:
                area = re.sub(r"^(?:/|per)\s*", "", per, flags=re.IGNORECASE).lower()
                problems.append(
                    f"'{m.group(0).strip()}' — {fmt(value)} "
                    f"{matched_total['mass_unit']} is the TOTAL for "
                    f"{fmt(conv['area_value'])} {conv['area_unit']}, not a rate per "
                    f"{area}. Give it as a total.")
            continue
        if matched_src is not None:
            # The source rate may be repeated, but only AS the rate it is.
            if per:
                area = re.sub(r"^(?:/|per)\s*", "", per, flags=re.IGNORECASE).lower()
                if area.rstrip("s") not in (matched_src["source_area_unit"].rstrip("s"),
                                            "ha", "hectare"):
                    problems.append(
                        f"'{m.group(0).strip()}' — the rate given was "
                        f"{fmt(matched_src['source_value'])} "
                        f"{matched_src['source_unit']}; it must not be restated per "
                        f"{area}.")
                continue
            problems.append(
                f"'{m.group(0).strip()}' — that is the RATE "
                f"({fmt(matched_src['source_value'])} {matched_src['source_unit']}), "
                f"not the amount for {fmt(conv['area_value'])} {conv['area_unit']}. "
                f"Say it is a rate, or give the total instead.")
            continue

        expected = ", ".join(f"{fmt(r['exact'])} {r['mass_unit']}"
                             for _, r in totals)
        problems.append(f"'{m.group(0).strip()}' is not the computed figure "
                        f"(expected {expected}).")

    # The computed figure wearing the WRONG substance. The number is right and the
    # noun beside it is not, which is the most expensive kind of wrong here: the
    # farmer buys 48.6 kg of urea when the documents said 48.6 kg of nitrogen.
    gap = conv.get("substance_gap")
    if gap:
        for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*(?:{_MASS_RE})\b[^.;]{{0,20}}?"
                             rf"\b{re.escape(gap['asked'])}\b", flat, re.IGNORECASE):
            value = float(m.group(1))
            if any(abs(value - v) <= _tolerance(v) for v, _ in totals):
                problems.append(
                    f"'{m.group(0).strip()}' — that figure is {gap['have']}, not "
                    f"{gap['asked']}. You do not have the composition needed to "
                    f"convert {gap['have']} into {gap['asked']}, so no "
                    f"{gap['asked']} quantity may be stated at all.")

    missing = [r for _, r in totals if id(r) not in stated]
    if missing:
        problems.append(
            "the answer does not give " + ", ".join(
                f"{fmt(r['exact'])} {r['mass_unit']}"
                + (f" of {r['substance']}" if r.get("substance") else "")
                for r in missing)
            + ". The arithmetic is complete, so the figure must be stated — do not "
              "ask for the variety, stage or planting method instead.")
    return problems


def deterministic_answer(conv: dict) -> str:
    """The answer in plain words, without the model.

    Used when the model's wording fails validation twice. Plain, but every
    figure in it is arithmetic.
    """
    if conv.get("kind") == "ambiguous_area":
        return (f"A {conv['unit']} is not the same size in every district, so I "
                f"cannot work out the total. Tell me the area in acres, decimals "
                f"or hectares and I will calculate it.")
    bits = []
    for r in conv["results"]:
        what = f" of {r['substance']}" if r["substance"] else ""
        if conv["kind"] == "rate":
            bits.append(f"{fmt(r['rate'])} {r['rate_unit']}{what}")
        else:
            bits.append(f"{fmt(r['total'])} {r['mass_unit']}{what}")
    joined = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
    gap = conv.get("substance_gap")
    if gap and conv["kind"] != "rate":
        return (f"For {conv.get('area_text')} that works out to about {joined}. "
                f"That is {gap['have']}, not {gap['asked']} — I cannot tell you how "
                f"much {gap['asked']} that is, because I do not have the "
                f"{gap['asked']} composition needed to convert it.")
    if conv["kind"] == "rate":
        return (f"At {conv['results'][0]['source_value']:g} "
                f"{conv['results'][0]['source_unit']}, the equivalent rate is "
                f"about {joined}.")
    # The farmer's own wording for the area — "1.5 acres", not "1.5 acre".
    return (f"For {conv.get('area_text') or fmt(conv['area_value'])} you need about "
            f"{joined}, at the rate you gave "
            f"({conv['results'][0]['source_value']:g} "
            f"{conv['results'][0]['source_unit']}).")
