"""What was asked for, and how many of it.

THE FAULT THIS EXISTS TO FIX
----------------------------
"Which variety is suitable for upland?" and "Which varieties are suitable for
upland?" are the same question about the same catalogue rows, and the system
answered them differently: the singular wording retrieved a narrower pool and
was handed no candidate list, so a farmer who happened to say "variety" got one
name — sometimes not even the best-supported one — while a farmer who said
"varieties" got the list. The grammatical number of a noun is not evidence about
the world, and it must not decide what the retrieval layer is allowed to find.

THE RULE HERE
-------------
GRAMMATICAL NUMBER DECIDES NOTHING. "Which variety is suitable for medium land?"
and "Which varieties are suitable for medium land?" are the same request for
every option the documents support, and they get the same entity, the same
attributes, the same searches, the same pool and the same kind of answer. A
farmer says "which variety" the way anyone says it; they are not asking to be
told about one seed while two others they could buy go unmentioned.

ONE thing changes the shape of the answer, and it is not grammar: an EXPLICIT
request for a single option — "the best", "the top one", "only one". That, and
only that, is `quantity == "single_explicit"`, and even then the answer is
picked out of the same pool and may say plainly that the documents rank nothing.

THE SHAPE
---------
    question
      -> analyse()          entity + quantity + attributes (land type, traits)
      -> retrieval_variants()   both number-forms, so the searches are one set
      -> candidate_pool()   the entities the CONTEXT explicitly supports, ranked
      -> the caller formats: all supported options, or — only when the farmer
         asked for one — the strongest of them, with the evidence for it

This module holds no question-specific special cases and knows nothing about the
pipeline. It reads text and returns data; `pipeline.py` decides what to do with
it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger("irri")


# ── Entities ──────────────────────────────────────────────────────────────
#
# `singular` / `plural` are the surface forms a farmer uses, and BOTH are
# recognised for every entity — that is the singular/plural fold, done once here
# rather than in each caller. `names` says what a candidate answer looks like in
# the documents: a cultivar code, or a term from a fixed lexicon.
#
# `topic_only` marks the entities whose bare mention is enough to make a
# question about them ("tell me about drought tolerant varieties"). It is set
# for varieties alone, deliberately: "treat seed before sowing" is a question
# about seed treatment, not a request for seed names, and letting the bare noun
# decide would misread it.
@dataclass(frozen=True)
class EntityType:
    key: str
    singular: tuple
    plural: tuple
    names: str                    # "cultivar" | lexicon key
    topic_only: bool = False


ENTITY_TYPES = (
    # "varities"/"varity" and similar are frequent farmer misspellings of
    # "variety"/"varieties"; include them so a duration or land query still
    # routes to the variety path instead of being refused.
    EntityType("variety", ("variety", "cultivar", "kisam", "varity", "vareity"),
               ("varieties", "cultivars", "kisams",
                "varities", "varites", "vareties", "varieites"),
               names="cultivar", topic_only=True),
    # A farmer asking which SEED to sow is asking which variety to sow; the
    # documents answer with cultivar names either way.
    EntityType("seed", ("seed",), ("seeds",), names="cultivar"),
    EntityType("fertilizer",
               ("fertilizer", "fertiliser", "manure", "nutrient"),
               ("fertilizers", "fertilisers", "manures", "nutrients"),
               names="input"),
    EntityType("disease", ("disease", "infection"), ("diseases", "infections"),
               names="disease"),
    EntityType("pest", ("pest", "insect"), ("pests", "insects"), names="pest"),
    EntityType("document", ("document", "doc", "publication"),
               ("documents", "docs", "publications"), names="document"),
    EntityType("catalogue item",
               ("catalogue item", "catalog item", "catalogue entry"),
               ("catalogue items", "catalog items", "catalogue entries"),
               names="document"),
)
_BY_KEY = {e.key: e for e in ENTITY_TYPES}


def _forms_alternation(forms) -> str:
    """A regex alternation over surface forms, longest first so 'catalogue item'
    is taken before a shorter form inside it could be."""
    ordered = sorted(forms, key=len, reverse=True)
    return "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in ordered)


# ── How many the farmer wants ─────────────────────────────────────────────
#
# GRAMMATICAL SINGULAR DOES NOT MEAN THE FARMER WANTS ONLY ONE. "Which variety
# is suitable for medium land?" is how a farmer asks which varieties they could
# sow; answering with one name, when the sheets support three, hides two seeds
# they might actually be able to buy. English number is a habit of phrasing, not
# a statement about how many recommendations are wanted, and it decides nothing
# here.
#
# ONE thing decides it: whether the farmer explicitly asked for a single option.
# "The best", "the top one", "only one", "just one" — those are a request to
# pick, and the answer picks. Everything else returns every grounded option.
QUANTITY_MULTIPLE = "multiple"
QUANTITY_SINGLE_EXPLICIT = "single_explicit"

_EXPLICIT_SINGLE_RE = re.compile(
    r"\b(?:"
    r"best|top|most\s+suitable|highest[- ]yielding|"
    r"only\s+one|just\s+one|one\s+option|one\s+suitable|one\s+single|"
    r"a\s+single|single\s+(?:one|option|recommendation|variety|seed)|"
    r"which\s+one\b|"
    r"recommend\s+(?:me\s+)?(?:only\s+)?one|give\s+(?:me\s+)?(?:only\s+)?one|"
    r"suggest\s+(?:me\s+)?(?:only\s+)?one"
    r")\b", re.IGNORECASE)

# The verbs and adjectives a document uses when it VOUCHES for something.
SUPPORT_WORDS = ("suitable", "suited", "recommended", "recommend", "fit",
                 "adapted", "appropriate", "tolerant", "tolerance",
                 "resistant", "resistance", "effective", "advised", "preferred")
# The subset that is an explicit recommendation rather than a bare capability.
# A row that RECOMMENDS a variety for upland outranks one that merely says it
# tolerates something, which is what makes a singular answer's "best" defensible
# without any document having to rank anything.
RECOMMENDATION_WORDS = ("recommended", "recommend", "suitable", "suited", "fit",
                        "adapted", "appropriate", "preferred", "advised")
# Kept under the old name too: pipeline had this list and other rules read it.
TOLERANCE_WORDS = ("tolerant", "tolerance", "resistant", "resistance",
                   "suitable", "fit", "adapted", "recommended")

# THE SAME SENTENCE CAN SAY THE OPPOSITE. "Swarna Sub1 is submergence tolerant,
# NOT drought tolerant" carries every word the collector looks for, and reading
# it as support would put a variety on the list precisely because the document
# ruled it out.
NEGATED_SUPPORT_RE = re.compile(
    r"\b(?:not|non|never|neither|nor|without|lacks?|no)\s+(?:\w+\s+){0,2}"
    r"(?:tolerant|tolerance|resistant|resistance|suitable|fit|adapted|recommended)\b"
    r"|\bnon-(?:tolerant|resistant)\b|\bsusceptib\w*\b|\bnot\s+recommended\b",
    re.IGNORECASE)


# ── Attributes ────────────────────────────────────────────────────────────
#
# The canonical term on the left is the word the INDEXED DOCUMENTS use; the
# forms on the right are what farmers, transcripts and translations produce.
# This is the same canonical-term discipline as services/synonyms.py, applied to
# the attributes a listing question filters on — and it works in both
# directions: it recognises "high land" in the question, and it recognises
# "Fit for upland" in the document as an answer to it.
# WHY THE FACETS ARE SEPARATE FIELDS
# ----------------------------------
# "Medium" is five different attributes in these documents, and reading it as
# one is what produced the reported failure. A farmer asked, in Odia, for a
# variety for MEDIUM LAND (ମଧ୍ୟମ କିସମ ଜମି); it reached the pipeline as
# "medium-maturing variety", and from there nothing could tell whether the
# question was about a duration class, an ecosystem, a grain shape or a plant
# height. Each of these is now its own field, and NOTHING matches on the bare
# word "medium" — every phrase below names the facet it belongs to:
#
#   medium duration / medium maturity  -> maturity        = medium_duration
#   medium land / irrigated medium land-> land_type       = medium_land
#   medium slender                     -> grain_type      = medium_slender
#   medium height                      -> plant_height    = medium_height
#
# Two of these can be true at once ("medium duration variety for medium land"),
# which is precisely why they cannot share a field.
LAND_TYPES = {
    "upland": ("upland", "uplands", "up land", "high land", "highland",
               "high-land", "dry land", "dryland"),
    "medium_land": ("irrigated medium land", "rainfed medium land",
                    "irrigated/medium", "irrigated/med",
                    "medium land", "medium-land",
                    "medium lands", "mid land", "medium deep", "medium-deep"),
    # The lowland ecosystems the sheets distinguish. A variety released for
    # shallow favourable lowland is not thereby recommended for a deepwater
    # field, so the specific term wins over the general one — see
    # LAND_TYPE_PARENT, which keeps the general term searchable anyway.
    "shallow_lowland": ("shallow lowland", "shallow low land",
                        "shallow favourable lowland", "shallow favorable lowland",
                        "shallow favourable low land"),
    "irrigated_lowland": ("irrigated lowland", "irrigated low land"),
    "rainfed_lowland": ("rainfed lowland", "rainfed low land"),
    "deepwater": ("deepwater", "deep water land", "semi-deep water"),
    "lowland": ("lowland", "lowlands", "low land", "low-land", "waterlogged",
                "water logged", "water-logged", "submerged land"),
}
# A specific lowland is still a lowland when it comes to searching document
# lines: "shallow favourable lowland" should also be found by "lowland".
LAND_TYPE_PARENT = {"shallow_lowland": "lowland", "irrigated_lowland": "lowland",
                    "rainfed_lowland": "lowland", "deepwater": "lowland"}

# Duration class. NEVER a bare "medium"/"early"/"late": those words appear all
# over a varietal sheet attached to other facets.
MATURITIES = {
    "early_duration": ("early maturing", "early-maturing", "early maturity",
                       "early duration", "early-duration", "short duration",
                       "short-duration", "early duration variety"),
    "medium_duration": ("medium maturing", "medium-maturing", "medium maturity",
                        "medium duration", "medium-duration", "mid duration",
                        "mid-duration", "medium maturity group"),
    "late_duration": ("late maturing", "late-maturing", "late maturity",
                      "late duration", "late-duration", "long duration",
                      "long-duration"),
}
GRAIN_TYPES = {
    "medium_slender": ("medium slender", "medium-slender"),
    "long_slender": ("long slender", "long-slender", "long bold"),
    "short_bold": ("short bold", "short-bold"),
    "medium_bold": ("medium bold", "medium-bold"),
}
PLANT_HEIGHTS = {
    "medium_height": ("medium height", "medium-height", "medium tall",
                      "semi tall", "semi-tall"),
    "tall": ("tall plant", "tall variety", "tall statured", "tall type"),
    "dwarf": ("semi dwarf", "semi-dwarf", "dwarf"),
}
# The stresses and pathogens a suitability question filters on.
STRESS_CONDITIONS = {
    "drought": ("drought", "drought tolerant", "drought resistant", "water stress"),
    "submergence": ("submergence", "flood", "flooding", "flood prone", "sub1"),
    "salinity": ("salinity", "saline", "salt tolerant", "salt affected"),
    "blast": ("blast",),
    "bacterial_leaf_blight": ("bacterial leaf blight", "blb", "bacterial blight"),
}
SEASONS = {"kharif": ("kharif",), "rabi": ("rabi",), "zaid": ("zaid", "summer")}

# facet name -> its table. The order is the order they are reported in.
ATTRIBUTE_GROUPS = (
    ("land_type", LAND_TYPES),
    ("maturity", MATURITIES),
    ("grain_type", GRAIN_TYPES),
    ("plant_height", PLANT_HEIGHTS),
    ("stress_condition", STRESS_CONDITIONS),
    ("season", SEASONS),
)
_GROUP_TABLES = dict(ATTRIBUTE_GROUPS)

# A crop duration written as a number. The same bands duration_class() uses, and
# the same shape of guard: two or three digits immediately before "day(s)", and
# only inside a range a rice crop can plausibly take. That is what keeps "5 days
# after transplanting" (an interval) and "400 days" (not a crop) out of it.
_CROP_DAYS_RE = re.compile(r"\b(\d{2,3})\s*[-– ]?\s*day(?:s)?\b", re.I)
_CROP_DAYS_MIN, _CROP_DAYS_MAX = 70, 200
_DAY_BANDS = ((115, "early_duration"), (139, "medium_duration"))


def _maturity_from_days(text_low: str) -> tuple:
    """(maturity, the phrase that said it) for a day count, or ("", "")."""
    for match in _CROP_DAYS_RE.finditer(text_low or ""):
        days = int(match.group(1))
        if not (_CROP_DAYS_MIN <= days <= _CROP_DAYS_MAX):
            continue
        for ceiling, canonical in _DAY_BANDS:
            if days <= ceiling:
                return canonical, match.group(0)
        return "late_duration", match.group(0)
    return "", ""
# Facets that hold at most one value; stress_condition may hold several.
_SINGLE_VALUE_GROUPS = ("land_type", "maturity", "grain_type", "plant_height",
                        "season")


def family_for(group: str, canonical: str) -> tuple:
    """Every way the documents write `canonical`, including its parent's terms."""
    table = _GROUP_TABLES.get(group) or {}
    terms = tuple(table.get(canonical, ()))
    parent = LAND_TYPE_PARENT.get(canonical) if group == "land_type" else None
    if parent:
        terms += tuple(table.get(parent, ()))
    return terms


# ── Candidate names ───────────────────────────────────────────────────────

# Variety names as this corpus writes them: a code with digits (DRR 43,
# IR64-Drt1, MTU 1010), or a name built on "Dhan" (CR Dhan 100, Sahbhagi Dhan,
# DRR Dhan-48), or a word followed by a number (Hardinath 3), or a name whose
# second word carries the number without a space before it (Swarna Sub1) — the
# submergence-tolerant lines are written that way throughout, and without this
# alternative every Sub1 variety was invisible to the collector.
CULTIVAR_NAME_RE = re.compile(
    r"\b(?:"
    r"(?:[A-Z][A-Za-z]*[\s-]){0,2}Dhan(?:[\s-]?\d+)?|"
    r"[A-Z]{2,}\d*[\s-]?\d+(?:-[A-Za-z0-9]+)?|"
    r"[A-Z][a-z]+[\s-][A-Z][a-z]*\d+|"
    r"[A-Z][a-z]+[\s-]\d+"
    r")\b")
# THE NAMES THAT ARE JUST WORDS. A cultivar whose name carries no digit and no
# "Dhan" — Pooja, Lalat, Swarna — matches no shape, and the collector could not
# see it: "Pooja, CR Dhan 310, and Lalat are good yielding medium-duration
# varieties" yielded CR Dhan 310 alone, and the two the farmer is most likely to
# be offered at the seed shop went unmentioned.
#
# A LIST, NOT A LOOSER SHAPE. Widening the pattern to any capitalised word would
# put every proper noun on the page — districts, institutes, months — into the
# pool, which is a worse failure than missing a name: it would invent
# recommendations. These are released varieties named in this corpus's own
# documents, so each one is a fact about the corpus rather than a guess, and a
# new variety is one line here.
KNOWN_CULTIVARS = (
    "Pooja", "Lalat", "Swarna", "Sarala", "Manaswini", "Khandagiri", "Konark",
    "Gayatri", "Tapaswini", "Pratikshya", "Hasanta", "Indrani", "Kalachampa",
    "Sahbhagi", "Ranidhan", "Savitri", "Jagabandhu", "Mahanadi", "Naveen",
    "Ankit", "Sarunima", "Bina", "Veena", "Pyari", "Jyotirmayee", "Sonamani",
    "Chandrama", "Parijat", "Padmini", "Kanchan", "Sebati", "Durga",
)
_KNOWN_CULTIVAR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in sorted(KNOWN_CULTIVARS, key=len,
                                                     reverse=True)) + r")\b")

# KNOWN LIMIT: a cultivar that is neither shaped like a code nor in the list
# above is still not collected, and the answer is left to the CONTEXT and the
# grounding rules as before.
# Words that make a capitalised token part of the page, not a cultivar.
# Months included because CULTIVAR_NAME_RE's "word + number" branch reads a
# sowing-window date ("July 15", "Aug 30") as a cultivar code otherwise — the
# hazard the KNOWN_CULTIVARS note above warns about.
NOT_A_CULTIVAR_NAME = frozenset({
    "table", "figure", "page", "source", "chapter", "section", "annexure",
    "kharif", "rabi", "zone", "district", "state", "india", "odisha",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
})

# For entities whose names are vocabulary rather than codes, a lexicon is
# safer than a shape: nothing about "Urea" looks different from "Odisha".
_NAME_LEXICONS = {
    "input": ("urea", "dap", "mop", "ssp", "npk", "potash", "muriate of potash",
              "single super phosphate", "di-ammonium phosphate", "zinc sulphate",
              "zinc sulfate", "ammonium sulphate", "gypsum", "lime", "compost",
              "vermicompost", "farmyard manure", "fym", "green manure",
              "azolla", "rhizobium", "azotobacter", "azospirillum",
              "trichoderma", "boron", "sulphur", "sulfur", "micronutrient"),
    "disease": ("blast", "neck blast", "leaf blast", "bacterial leaf blight",
                "sheath blight", "sheath rot", "brown spot", "false smut",
                "tungro", "bakanae", "stem rot", "grain discolouration",
                "udbatta", "leaf streak"),
    "pest": ("stem borer", "yellow stem borer", "brown planthopper", "bph",
             "white backed planthopper", "leaf folder", "gall midge",
             "case worm", "rice hispa", "gundhi bug", "earhead bug",
             "thrips", "army worm", "cut worm", "root weevil", "nematode"),
}

# Words that describe the ASK rather than the thing asked for, so they are not
# evidence to look for in a document line.
CONDITION_STOPWORDS = frozenset({
    "please", "tell", "give", "name", "names", "some", "list", "kindly",
    "variety", "varieties", "cultivar", "cultivars", "seed", "seeds",
    "rice", "paddy", "crop", "crops", "farmer", "field", "which", "what",
    "that", "with", "have", "there", "about", "from", "this", "these",
    "type", "types", "kind", "kinds", "good", "best", "better",
    "should", "would", "could", "apply", "used", "using", "suitable",
    "recommend", "recommended", "grow", "grown", "sow", "sown", "plant",
    "document", "documents", "catalogue", "catalog", "item", "items",
    "fertilizer", "fertilizers", "fertiliser", "fertilisers", "disease",
    "diseases", "pest", "pests",
    # Quantity/application wording describes the QUESTION, not a condition a
    # candidate line must repeat. Without these, "What amount of fertiliser..."
    # leaves "amount" as the sole condition term, so valid dose/context lines
    # are all rejected for not containing that word.
    "amount", "amounts", "quantity", "quantities", "dose", "doses",
    "dosage", "rate", "rates", "application", "applications",
    # FACET NOUNS AND THEIR QUALIFIERS, WHICH MEAN NOTHING ALONE. Evidence is
    # matched by substring, so a bare "land" from "good to plant in my land"
    # matches up-LAND, low-LAND and medium LAND alike, and a bare "medium"
    # matches every medium-anything. Inside a facet phrase they are already
    # covered by that facet's family ("medium duration", "high land"); on their
    # own they are noise that lets an unrelated row into the pool.
    "land", "lands", "field", "fields", "soil", "area", "areas", "condition",
    "conditions", "ecosystem", "ecosystems", "duration", "maturity", "maturing",
    "grain", "grains", "height", "season", "seasons",
    "medium", "early", "late", "short", "long", "high", "deep", "shallow",
})


# ── The intent ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Attributes:
    """The facets a piece of text names, one field each.

    Read off a QUESTION to know what was asked, and off a DOCUMENT LINE to know
    what that line actually vouches for — the same code both ways, which is what
    keeps "the farmer asked about medium land" and "the source says medium
    duration" from ever being mistaken for each other.
    """
    land_type: str = ""
    maturity: str = ""
    grain_type: str = ""
    plant_height: str = ""
    stress_conditions: tuple = ()
    season: str = ""
    matched_phrases: tuple = ()

    def as_dict(self) -> dict:
        out = {}
        for name in _SINGLE_VALUE_GROUPS:
            value = getattr(self, name)
            if value:
                out[name] = value
        if self.stress_conditions:
            out["stress_condition"] = list(self.stress_conditions)
        return out

    @property
    def any_set(self) -> bool:
        return bool(self.as_dict())


@dataclass(frozen=True)
class QueryIntent:
    """What the question asks for. `quantity` never affects the candidate pool."""
    question: str
    entity: str                        # an ENTITY_TYPES key
    quantity: str                      # "multiple" | "single_explicit"
    attributes_found: Attributes = field(default_factory=Attributes)
    condition_terms: frozenset = field(default_factory=frozenset)

    @property
    def wants_list(self) -> bool:
        """True for every ordinary recommendation question, either wording."""
        return self.quantity == QUANTITY_MULTIPLE

    @property
    def wants_one(self) -> bool:
        """True only when the farmer asked for a single option in so many words."""
        return self.quantity == QUANTITY_SINGLE_EXPLICIT

    @property
    def name_kind(self) -> str:
        return _BY_KEY[self.entity].names

    # The facets, forwarded so callers read intent.land_type rather than
    # intent.attributes_found.land_type.
    @property
    def land_type(self) -> str:
        return self.attributes_found.land_type

    @property
    def maturity(self) -> str:
        return self.attributes_found.maturity

    @property
    def grain_type(self) -> str:
        return self.attributes_found.grain_type

    @property
    def plant_height(self) -> str:
        return self.attributes_found.plant_height

    @property
    def stress_conditions(self) -> tuple:
        return self.attributes_found.stress_conditions

    @property
    def season(self) -> str:
        return self.attributes_found.season

    @property
    def attributes(self) -> dict:
        """Every filter the question carries, as plain data (for tracing)."""
        return self.attributes_found.as_dict()

    @property
    def land_type_missing(self) -> bool:
        """A suitability question that never said which land it is about.

        The reported failure again, one step later: asked for a MEDIUM DURATION
        variety with no ecosystem given, the pipeline answered as though the
        variety suited the farmer's field. It cannot know that — the sheets
        recommend a variety FOR an ecosystem, and the farmer never said theirs.
        True here means the answer must either ask which land it is, or name the
        ecosystem its source actually states.
        """
        return self.name_kind == "cultivar" and not self.land_type


def _phrase_present(text_low: str, phrases) -> str:
    """The longest phrase of `phrases` present in `text_low`, or "".

    Longest wins so "irrigated medium land" is not read as plain "medium land",
    and "shallow favourable lowland" is not flattened to "lowland".
    """
    for phrase in sorted(phrases, key=len, reverse=True):
        # A trailing "s" is the same phrase. The sheets write "recommended for
        # shallow favourable lowlandS", and without this the specific ecosystem
        # fell back to the general "lowland" — which is exactly the widening
        # this module exists to prevent.
        if re.search(rf"(?<![A-Za-z]){re.escape(phrase)}s?(?![A-Za-z])", text_low):
            return phrase
    return ""


def detect_attributes(text: str) -> Attributes:
    """The facets `text` names, each in its own field.

    Matching is by MEANING, not spelling: "high land" and "dry land" both come
    back as the canonical "upland", which is the term the documents use. Every
    phrase is specific to one facet — there is no bare-"medium" rule, so a
    question about medium land can never be answered as one about medium
    duration.
    """
    low = (text or "").lower()
    values: dict = {}
    stresses: list = []
    matched: list = []
    for group_name, table in ATTRIBUTE_GROUPS:
        best_canonical, best_phrase = "", ""
        for canonical, phrases in table.items():
            hit = _phrase_present(low, phrases)
            if not hit:
                continue
            if group_name == "stress_condition":
                if canonical not in stresses:
                    stresses.append(canonical)
                matched.append(hit)
                continue
            # One value per facet: the most specific wording wins.
            if len(hit) > len(best_phrase):
                best_canonical, best_phrase = canonical, hit
        if best_canonical:
            values[group_name] = best_canonical
            matched.append(best_phrase)

    # A DAY COUNT IS A DURATION CLASS SPOKEN AS A NUMBER.
    #
    # MATURITIES is a phrase table, so "short duration" was recognised and
    # "105-day" was not — the farmer asking "ଶହେ ପାଞ୍ଚ ଦିନିଆ ଧାନ କିସମ ଗୁଡ଼ିକର
    # ନାମ କୁହନ୍ତୁ" ("name the 105-day varieties") produced no maturity, so no
    # condition terms, so candidate_pool() returned nothing and the whole
    # completeness machinery stayed dormant: the answer named three of four
    # varieties that were sitting in the context, with nothing to require the
    # fourth. The bands are the ones duration_class() already uses.
    #
    # Only when no phrase said it. A question that names BOTH ("105-day short
    # duration") keeps the words, which are the more specific evidence.
    if "maturity" not in values:
        derived, said = _maturity_from_days(low)
        if derived:
            values["maturity"] = derived
            matched.append(said)
    return Attributes(
        land_type=values.get("land_type", ""),
        maturity=values.get("maturity", ""),
        grain_type=values.get("grain_type", ""),
        plant_height=values.get("plant_height", ""),
        stress_conditions=tuple(stresses),
        season=values.get("season", ""),
        matched_phrases=tuple(matched),
    )


def _entity_patterns(entity: EntityType):
    """(plural_re, singular_re) for one entity, built from its surface forms."""
    plural = _forms_alternation(entity.plural)
    singular = _forms_alternation(entity.singular)
    both = _forms_alternation(tuple(entity.plural) + tuple(entity.singular))
    plural_re = re.compile(
        r"\b(?:"
        rf"(?:which|what)\s+(?:\w+[\s-]){{0,3}}(?:{plural})\b|"
        rf"(?:{plural})\s+(?:that|which|are|for|suitable|suited|tolerant|"
        r"resistant|recommended|available)\b|"
        rf"all\s+(?:the\s+)?(?:\w+[\s-]){{0,3}}(?:{plural})\b|"
        rf"(?:names|list)\s+of\s+(?:\w+[\s-]){{0,3}}(?:{both})\b|"
        rf"list\s+(?:the\s+|some\s+|all\s+)?(?:\w+[\s-]){{0,3}}(?:{both})\b"
        r")", re.IGNORECASE)
    singular_re = re.compile(
        r"\b(?:"
        rf"(?:which|what)\s+(?:\w+[\s-]){{0,3}}(?:{singular})\b|"
        rf"name\s+of\s+the\s+(?:\w+[\s-]){{0,3}}(?:{singular})\b|"
        rf"(?:best|suitable|recommended)\s+(?:\w+[\s-]){{0,3}}(?:{singular})\b|"
        rf"(?:{singular})\s+(?:is|that\s+is)\s+(?:\w+[\s-]){{0,3}}"
        r"(?:suitable|suited|recommended|fit|tolerant|resistant)\b"
        r")", re.IGNORECASE)
    return plural_re, singular_re


_PATTERNS = {e.key: _entity_patterns(e) for e in ENTITY_TYPES}
_TOPIC_RES = {
    e.key: re.compile(rf"\b(?:{_forms_alternation(tuple(e.plural) + tuple(e.singular))})\b",
                      re.IGNORECASE)
    for e in ENTITY_TYPES if e.topic_only
}


def condition_terms(question: str, attributes=None) -> frozenset:
    """The conditions a candidate must meet, as phrases to look for beside it.

    Content words from the question, minus the words that describe the asking
    rather than the thing asked for. Where a facet was recognised, its OWN words
    drop out and the canonical term's whole family goes in instead: a question
    about "high land" then looks for "upland" too, and — the part that matters —
    the bare word "medium" never becomes a term, so a medium-DURATION question
    cannot be answered from a medium-LAND row.
    """
    attributes = attributes if attributes is not None else detect_attributes(question)
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", question or "")}
    # SPLIT ON PUNCTUATION, NOT JUST SPACES. `phrase.split()` left
    # "medium-duration" as one token, so the bare words "medium" and "duration"
    # — which the word scan above extracts separately, because a hyphen is not a
    # letter — were never removed. A hyphenated facet therefore put "medium"
    # back into the term set, and since matching is by substring, "medium"
    # matched "medium bold", "medium height" and "medium land": a
    # medium-DURATION question collected medium-LAND varieties, which is the
    # separation this module exists to enforce.
    phrase_words = {w for phrase in attributes.matched_phrases
                    for w in re.split(r"[^A-Za-z]+", phrase.lower()) if w}
    terms = {w for w in words
             if w not in CONDITION_STOPWORDS
             and w not in TOLERANCE_WORDS
             and w not in phrase_words}
    for group, _table in ATTRIBUTE_GROUPS:
        if group == "stress_condition":
            for canonical in attributes.stress_conditions:
                terms |= set(family_for(group, canonical))
            continue
        canonical = getattr(attributes, group, "")
        if canonical:
            terms |= set(family_for(group, canonical))

    # THE DAY COUNT ITSELF, when the question gave one.
    #
    # The word scan above takes only [A-Za-z]{4,}, so a number can never become
    # a term — and the duration FAMILY does not help either, because a varietal
    # sheet writes the class as a bare "Early" in its own column while the
    # family holds phrases like "early duration". MATURITIES deliberately
    # refuses bare words (a bare "medium" matches medium LAND), so the two
    # vocabularies could not meet and candidate_pool() reported "nothing in the
    # context supports maturity: early_duration" for a question whose answers
    # were sitting in front of it.
    #
    # The number is the one term both sides spell the same way: the question
    # says 105 and the row says "| 105 |". Exact, unambiguous, and it needs no
    # relaxation of the bare-word rule.
    days = _CROP_DAYS_RE.search((question or "").lower())
    if days and _CROP_DAYS_MIN <= int(days.group(1)) <= _CROP_DAYS_MAX:
        terms.add(days.group(1))
    return frozenset(terms)


# A question about HOW MUCH is not a request for names, however it is worded.
# "What is the recommended seed rate?" reads as "recommended ... seed" to the
# shape patterns, and answering it with a list of cultivars would be a worse
# answer than the one the pipeline already gives.
_QUANTITY_QUESTION_RE = re.compile(
    r"\b(?:seed(?:ing)?\s+rate|how\s+much|how\s+many|what\s+(?:dose|quantity|rate)|"
    r"quantity\s+of|dose\s+of|rate\s+of)\b", re.IGNORECASE)


@lru_cache(maxsize=512)
def analyse(question: str):
    """The QueryIntent for `question`, or None if it asks for no entity.

    Cached because one question is analysed three times on the way through the
    pipeline — expansion, retrieval width, generation — and all three must get
    the same answer. The result is immutable, so sharing it is safe.

    Detection is question-SHAPED, not noun-shaped: "which/what X", "names of X",
    "list X", "X that are ...". A sentence that merely contains the word "seed"
    ("how should I treat seed before sowing?") is not a request for seed names
    and does not come back as one.
    """
    text = (question or "").strip()
    if not text:
        return None
    if _QUANTITY_QUESTION_RE.search(text):
        return None

    # WHICH ENTITY. Both numbers are recognised by the same patterns and mean
    # the same thing; which of the two the farmer happened to use is not
    # recorded, because nothing downstream is allowed to depend on it.
    entity_key = ""
    for entity in ENTITY_TYPES:
        plural_re, singular_re = _PATTERNS[entity.key]
        if plural_re.search(text) or singular_re.search(text):
            entity_key = entity.key
            break
    if not entity_key:
        for key, topic_re in _TOPIC_RES.items():
            if topic_re.search(text):
                entity_key = key
                break
    if not entity_key:
        return None

    # HOW MANY. Explicit wording only — see the note on _EXPLICIT_SINGLE_RE.
    quantity = (QUANTITY_SINGLE_EXPLICIT if _EXPLICIT_SINGLE_RE.search(text)
                else QUANTITY_MULTIPLE)

    attributes = detect_attributes(text)
    intent = QueryIntent(
        question=text, entity=entity_key, quantity=quantity,
        attributes_found=attributes,
        condition_terms=condition_terms(text, attributes),
    )
    log.info("intent | entity=%s quantity=%s attributes=%s%s",
             intent.entity, intent.quantity, intent.attributes or "-",
             " (land type NOT given)" if intent.land_type_missing else "")
    return intent


# ── Retrieval ─────────────────────────────────────────────────────────────

def _swap_number(question: str, entity: EntityType) -> str:
    """The same question with the entity noun in its other number.

    Crude on purpose — the result is a SEARCH STRING, never shown to anyone. Its
    only job is to put the documents' own wording in front of the index, so
    "which variety" and "which varieties" reach the same rows.

    The verb is carried across with the noun, and that is not cosmetic: it is
    what makes the swap an INVOLUTION. "Which variety is suitable" becomes
    "Which varieties are suitable", whose own twin is the original again — so
    the SET of strings sent to the index is the same set whichever way the
    farmer phrased it. Without the verb the two sets would differ by an "is",
    and the two wordings would search differently after all.
    """
    twin, _went_plural = _number_twin(question, entity)
    return twin


def _number_twin(question: str, entity: EntityType):
    """(twin, went_plural). went_plural is None when there was nothing to swap."""
    out, went_plural = question, None
    for singular, plural in zip(entity.singular, entity.plural):
        if re.search(rf"\b{re.escape(plural)}\b", out, re.I):
            out = re.sub(rf"\b{re.escape(plural)}\b", singular, out, flags=re.I)
            went_plural = False
        elif re.search(rf"\b{re.escape(singular)}\b", out, re.I):
            out = re.sub(rf"\b{re.escape(singular)}\b", plural, out, flags=re.I)
            went_plural = True
    if went_plural is None:
        return question, None
    pairs = ((r"\bis\b", "are"), (r"\bhas\b", "have"), (r"\bdoes\b", "do")) \
        if went_plural else ((r"\bare\b", "is"), (r"\bhave\b", "has"),
                             (r"\bdo\b", "does"))
    for pattern, replacement in pairs:
        out = re.sub(pattern, replacement, out, flags=re.I)
    return out, went_plural


def canonical_term(group: str, canonical: str) -> str:
    """The plain document wording for a canonical value, for search strings.

    The family is ordered most-specific-first for MATCHING; a search string
    wants the ordinary form, so "medium_land" goes to the index as "medium
    land" and not as "irrigated medium land", which would narrow the very
    search it was added to widen.
    """
    plain = canonical.replace("_", " ")
    family = family_for(group, canonical)
    if plain in family:
        return plain
    return family[0] if family else plain


def retrieval_variants(question: str, intent=None) -> list:
    """Extra search phrasings so number and wording cannot change the pool.

    Two kinds: the entity noun in its other number, and the canonical document
    term for any attribute the farmer phrased differently ("high land" ->
    "upland"). Both are additions — the original question is always searched.
    """
    intent = intent or analyse(question)
    if intent is None:
        return []
    entity = _BY_KEY[intent.entity]
    variants, low = [], question.lower()

    # THE SEARCHED SET MUST NOT DEPEND ON THE WORDING. Both number-forms are
    # searched, and every canonical variant is built from the SAME pair of
    # forms rather than from whatever the farmer typed — so "which variety
    # ...?" and "which varieties ...?" send the index the same set of strings,
    # not merely overlapping ones.
    twin, went_plural = _number_twin(question, entity)
    if twin and twin.lower() != low:
        variants.append(twin)
    plural_form = twin if went_plural else question
    singular_form = question if went_plural else twin

    # THE DOCUMENTS' OWN WORD, IN THE QUESTION'S OWN SENTENCE. The farmer says
    # "high land"; the sheets say "upland", and a vector or BM25 search for the
    # farmer's phrase can miss the row that answers it outright. Appending the
    # canonical term to the question was not enough — the farmer's phrase still
    # dominated the embedding — so the phrase is SUBSTITUTED here, giving the
    # index a sentence written the way the corpus writes it. The original
    # wording is never dropped: it is the first thing searched, and these are
    # additions to it.
    canonical_pair = [_with_canonical_terms(form, intent.attributes_found)
                      for form in (singular_form, plural_form) if form]
    for rewritten in canonical_pair:
        if rewritten and rewritten.lower() != low:
            variants.append(rewritten)

    # A bare statement of the requirement, for the rows that are not written as
    # questions at all ("Fit for upland").
    if intent.land_type and entity.names == "cultivar":
        term = canonical_term("land_type", intent.land_type)
        variants.append(f"rice varieties recommended for {term}")
        variants.append(f"rice variety suitable for {term}")
    for canonical in intent.stress_conditions:
        term = canonical_term("stress_condition", canonical)
        if term.lower() not in low:
            variants.append(f"{plural_form} {term}")

    # Dedupe, keeping order and never repeating the question itself.
    seen, out = {low}, []
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            out.append(variant)
    return out[:6]


def _with_canonical_terms(text: str, attributes) -> str:
    """`text` with every recognised facet phrase replaced by the document's word.

    "Which rice variety is suitable for high land?" becomes "... for upland?".
    Only phrases the detector actually matched are touched, so nothing else in
    the farmer's sentence moves.
    """
    out = text or ""
    for group, _table in ATTRIBUTE_GROUPS:
        canonicals = (attributes.stress_conditions if group == "stress_condition"
                      else [getattr(attributes, group, "")])
        for canonical in [c for c in canonicals if c]:
            term = canonical_term(group, canonical)
            for phrase in family_for(group, canonical):
                if phrase == term:
                    continue
                out = re.sub(rf"(?<![A-Za-z]){re.escape(phrase)}s?(?![A-Za-z])",
                             term, out, flags=re.IGNORECASE)
    return out


# ── The candidate pool ────────────────────────────────────────────────────

def _names_in(line: str, name_kind: str) -> list:
    if name_kind == "cultivar":
        found, seen = [], set()
        for match in CULTIVAR_NAME_RE.finditer(line):
            name = re.sub(r"\s+", " ", match.group(0)).strip(" -")
            if name.split()[0].lower() in NOT_A_CULTIVAR_NAME:
                continue
            found.append(name)
            seen.add(name.lower())
        # The word-shaped names the pattern cannot see. Skipped when the coded
        # name already found contains it ("Sahbhagi Dhan" must not also yield
        # "Sahbhagi", which would put the same variety in the pool twice).
        for match in _KNOWN_CULTIVAR_RE.finditer(line):
            name = match.group(0)
            if any(name.lower() in already for already in seen):
                continue
            found.append(name)
            seen.add(name.lower())
        return found
    if name_kind == "document":
        return []                      # supplied by the caller from metadata
    lexicon = _NAME_LEXICONS.get(name_kind, ())
    low = line.lower()
    return [term for term in lexicon
            if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", low)]


def _lines(hits):
    """(chunk_rank, line, metadata) over the retrieved text.

    Line-scoped on purpose. A varietal sheet is one row per variety, and a
    document titled "drought tolerant varieties" does not make every name
    printed in it drought tolerant — the row has to say so.
    """
    for rank, hit in enumerate(hits or [], start=1):
        for line in re.split(r"[\n;]|(?<=[.!?])\s+", hit.get("chunk") or ""):
            if line.strip():
                yield rank, line, (hit.get("metadata") or {})


# How many lines a record header vouches for before it lapses. A varietal
# block is a handful of attribute lines; a bigger span is prose that happens to
# follow a name, and attributing it would be guessing.
_RECORD_SPAN = 12
# Letters left over after the names are removed. A line that is JUST the name —
# "CR Dhan 100", "Variety: CR Dhan 100", "CR Dhan 100 (2015)" — opens a record.
# A line that says something about it — "CR Dhan 100 is fit for upland." — does
# not, because it already speaks for itself.
_RECORD_HEADER_SLACK = 12


# A finite verb makes a line a sentence with its own subject, and a sentence
# that merely sits under a variety's name is not that variety's property —
# "Upland areas of Odisha grow rice" says nothing about the cultivar above it.
_SENTENCE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|should|must|may|will|"
    r"grow|grows|grown|give|gives|produce|produces|mature|matures|yield|"
    r"yields|require|requires|need|needs|include|includes|show|shows)\b",
    re.IGNORECASE)
_FIELD_VALUE_WORDS = 8


def _is_field_value(line: str) -> bool:
    """True for a table cell, false for a sentence.

    Inside a record the header has already said whose properties these are, so
    a bare value — "Medium duration", "cultivation in upland ecosystem" — is the
    document asserting it, with no "is suitable for" needed. A full sentence is
    not: it has its own subject and may be about something else entirely.
    """
    body = re.sub(r"^\s*[A-Za-z][A-Za-z /]{0,24}:\s*", "", line).strip()
    if not body or len(body.split()) > _FIELD_VALUE_WORDS:
        return False
    return not _SENTENCE_VERB_RE.search(body)


def _record_header_name(line: str, name_kind: str) -> str:
    """The cultivar this line is the heading for, or "".

    A PDF table flattened to text puts the variety in one cell and its
    properties in the next, and they come out as separate lines:

        CR Dhan 100
        Fit for upland

    Requiring name and evidence on the SAME line — which is what this collector
    did — found nothing here, and the answer came back "the information
    available does not specify a variety", with the evidence sitting in the
    context. A name-only line opens a record; the lines under it are read as
    being about that name.
    """
    names = _names_in(line, name_kind)
    if not names:
        return ""
    remainder = line
    for name in names:
        remainder = remainder.replace(name, " ")
    if len(re.sub(r"[^A-Za-z]", "", remainder)) > _RECORD_HEADER_SLACK:
        return ""                      # a sentence about the name, not a header
    return names[0]


def _requested_facets(intent) -> dict:
    """The facets the question asked for, as {group: canonical or [canonicals]}."""
    return intent.attributes if intent is not None else {}


def _states_requested_facet(stated, intent) -> bool:
    """True when this line names one of the facets the farmer asked about.

    Used as evidence in its own right: a row that says "medium-duration" has
    asserted the property, whether or not it also uses the word "suitable".
    """
    for group, value in _requested_facets(intent).items():
        found = stated.as_dict().get(group)
        if not found:
            continue
        if isinstance(value, list):
            if set(value) & set(found if isinstance(found, list) else [found]):
                return True
        elif found == value:
            return True
    return False


def candidate_pool(intent, hits, report=None) -> list:
    """Everything the CONTEXT explicitly supports for what was asked, ranked.

    THE SAME POOL FOR BOTH NUMBERS. `intent.quantity` is not read here — it
    decides presentation, not membership, and a singular question that reached a
    smaller pool is the bug this module was written for.

    Each entry: {name, score, evidence, recommended, mentions, rank}.
    Ranking is evidence-strength, not a claim the documents do not make: a row
    that RECOMMENDS a candidate for the asked condition outranks one that merely
    records a tolerance, more supporting rows outrank fewer, and an earlier
    retrieved chunk breaks the remaining ties.

    Pass a dict as `report` to have the accepted names, the rejected lines with
    the reason each was dropped, and the condition terms written into it — what
    a reviewer needs when an answer says the documents specify nothing and the
    evidence is visibly there.
    """
    if intent is None:
        return []
    wanted = intent.condition_terms
    if not wanted:
        return []
    name_kind = intent.name_kind

    found: dict = {}
    rejected: list = []

    def reject(what, reason):
        if len(rejected) < 40 and what:
            rejected.append({"line": what[:120], "reason": reason})

    record_name, record_age, record_rank = "", 0, 0
    for rank, line, _meta in _lines(hits):
        low = line.lower()

        # Record scope. A name-only line opens one and the lines under it are
        # read as being about that name; a new header, a new chunk, or too many
        # lines closes it.
        header = _record_header_name(line, name_kind)
        if header or rank != record_rank:
            record_name = header
            record_age, record_rank = 0, rank
        else:
            record_age += 1
            if record_age > _RECORD_SPAN:
                record_name = ""

        has_term = any(term in low for term in wanted)
        # WHAT THE LINE ITSELF SAYS THIS ROW IS FOR. Read with the same facet
        # detector the question went through, so an answer can state the
        # ecosystem the document actually gives — necessary when the farmer
        # never said which land they have — and so the requested facet can be
        # required below.
        stated = detect_attributes(line)
        # A support word is required in prose, where nearness is not assertion.
        # Inside a record it is not: the header said whose sheet this is, so the
        # field value IS the claim — which is how varietal tables are written.
        #
        # A line that NAMES a candidate and STATES a requested facet is the
        # third case. "BRRI Dhan 69 is a medium-duration variety" carries no
        # "suitable" or "fit" — those words belong to ecosystems and
        # tolerances, not to a duration class — and was being dropped for it.
        #
        # It must name the candidate ITSELF, not sit under one: "Upland areas of
        # Odisha grow rice" states a land type and is about the district, not
        # about the variety heading the record above it. Requiring the name on
        # the same line is what keeps that sentence from vouching for it.
        has_support = (any(word in low for word in SUPPORT_WORDS)
                       or bool(record_name and _is_field_value(line))
                       or (_states_requested_facet(stated, intent)
                           and bool(_names_in(line, name_kind))))
        if not has_term or not has_support:
            # A record header carries no evidence by design; reporting it as a
            # rejection would bury the real reasons in noise.
            if _names_in(line, name_kind) and not header:
                reject(line, "no condition term on the line" if not has_term
                       else "no suitability or tolerance word on the line")
            continue
        # ...unless the line is denying it. A denial never supports a candidate,
        # whatever was asked.
        #
        # KNOWN LIMIT: this collector answers "which X HAVE the property". A
        # question asking which are SUSCEPTIBLE gets no list from here and is
        # answered from the context as before, rather than being handed a list
        # assembled by the opposite rule.
        if NEGATED_SUPPORT_RE.search(low):
            reject(line, "the line denies the condition")
            continue
        recommended = any(word in low for word in RECOMMENDATION_WORDS)
        # The line's own names if it has any; otherwise the record it sits in.
        names = _names_in(line, name_kind) or ([record_name] if record_name else [])
        if not names:
            reject(line, "evidence with no candidate name on the line or above it")
        for name in names:
            key = name.lower().replace("-", " ")
            entry = found.get(key)
            if entry is None:
                found[key] = {"name": name, "evidence": line.strip(),
                              "recommended": recommended, "mentions": 1,
                              "rank": rank, "score": 0.0,
                              "conditions": stated.as_dict(),
                              "stated_phrases": list(stated.matched_phrases)}
            else:
                entry["mentions"] += 1
                entry["rank"] = min(entry["rank"], rank)
                for phrase in stated.matched_phrases:
                    if phrase not in entry["stated_phrases"]:
                        entry["stated_phrases"].append(phrase)
                entry["conditions"] = {**stated.as_dict(), **entry["conditions"]}
                if recommended and not entry["recommended"]:
                    entry["recommended"] = True
                    entry["evidence"] = line.strip()

    # THE ASKED-FOR FACET MUST BE STATED, NOT MERELY NEARBY.
    #
    # A term matching somewhere on a line is what gets a candidate LOOKED at; it
    # is not evidence that the candidate has the property. Asked for a
    # medium-duration variety, this collector returned CR Dhan 100 on the
    # strength of "Fit for upland" — and its own `conditions` said land_type
    # upland and no maturity at all. It knew, and did not check.
    #
    # `conditions` is read off the candidate's OWN evidence lines by the same
    # facet detector the question went through, so requiring the facet here
    # cannot be satisfied by a substring, by a neighbouring row, or by evidence
    # belonging to a different variety. Only facets the farmer actually asked
    # for are required: a question with no facet keeps the behaviour it had.
    requested = intent.attributes
    if requested:
        kept = {}
        for key, entry in found.items():
            stated = entry.get("conditions") or {}
            missing = [
                f"{group}={value}" for group, value in requested.items()
                if (not set(value) & set(stated.get(group) or [])
                    if isinstance(value, list)
                    else stated.get(group) != value)
            ]
            if missing:
                reject(f"{entry['name']}: {entry['evidence']}",
                       "no evidence for the requested "
                       + ", ".join(missing)
                       + (f" (its row states {stated})" if stated else ""))
                continue
            kept[key] = entry
        found = kept

    candidates = list(found.values())
    for index, entry in enumerate(candidates):
        entry["score"] = round(
            (2.0 if entry["recommended"] else 0.0)
            + 0.5 * min(entry["mentions"], 4)
            + 1.0 / entry["rank"], 3)
        entry["_seen"] = index
    candidates.sort(key=lambda c: (-c["score"], c["_seen"]))
    for entry in candidates:
        entry.pop("_seen", None)

    # DIAGNOSTICS. When an answer says "the information available does not
    # specify a variety" and the evidence was plainly there, this is the line
    # that says which step dropped it.
    log.info("candidates | %s | asked for %s | %d accepted: %s",
             intent.entity, sorted(intent.attributes.items()) or "-",
             len(candidates),
             ", ".join(f"{c['name']}({c['score']})" for c in candidates[:10]) or "none")
    for item in rejected[:10]:
        log.info("candidates | rejected: %s -- %s", item["reason"], item["line"])
    if report is not None:
        report["accepted"] = [
            {"name": c["name"], "score": c["score"], "evidence": c["evidence"][:200],
             "conditions": c["conditions"]} for c in candidates]
        report["rejected"] = rejected
        report["condition_terms"] = sorted(wanted)
        report["evidence_found"] = bool(candidates)
    return candidates


# ── Variety -> ecosystem, for a question about something else ─────────────
#
# "Which fertiliser will be applied to Pooja Dhan?" is not a question about
# fertiliser in general and not a question about varieties. It is a question
# about a RELATIONSHIP the documents hold in two halves: one page says which
# land Pooja is for, another says what to apply on that land. Answered without
# the first half, the fertiliser figures come from whichever ecosystem the
# retrieval happened to return — which is a dose for someone else's field.
#
# These two helpers are read-only and reuse the machinery already here: the
# same cultivar shape, the same record scope, the same facet detector. They
# establish the first half; the caller decides what to do with it.


def named_varieties(text: str) -> list:
    """The cultivars the QUESTION itself names, in order, without repeats."""
    found, seen = [], set()
    for name in _names_in(text or "", "cultivar"):
        key = name.lower().replace("-", " ")
        if key not in seen:
            seen.add(key)
            found.append(name)
    return found


def variety_conditions(name: str, hits) -> dict:
    """What the CONTEXT states this one variety is for: {facet: canonical}.

    Only lines that are about THAT variety count — its own row, or a line under
    its own record header. A neighbouring variety's ecosystem never leaks in,
    which is the whole point: Pooja's fertiliser must not be chosen from the row
    above it.

    Returns {} when the documents do not establish it, and the caller must then
    say so rather than guess.
    """
    wanted_key = (name or "").lower().replace("-", " ")
    if not wanted_key:
        return {}
    conditions: dict = {}
    phrases: list = []
    record_name, record_age, record_rank = "", 0, 0
    for rank, line, _meta in _lines(hits):
        header = _record_header_name(line, "cultivar")
        if header or rank != record_rank:
            record_name = header
            record_age, record_rank = 0, rank
        else:
            record_age += 1
            if record_age > _RECORD_SPAN:
                record_name = ""

        on_this_line = [n.lower().replace("-", " ")
                        for n in _names_in(line, "cultivar")]
        # The line is about this variety when it names it, or when it sits in
        # this variety's record and names no other.
        about_it = (wanted_key in on_this_line
                    or (record_name.lower().replace("-", " ") == wanted_key
                        and not on_this_line))
        if not about_it:
            continue
        if NEGATED_SUPPORT_RE.search(line.lower()):
            continue                        # "not suitable for upland" is not a for
        stated = detect_attributes(line)
        for facet, value in stated.as_dict().items():
            conditions.setdefault(facet, value)
        for phrase in stated.matched_phrases:
            if phrase not in phrases:
                phrases.append(phrase)
    if conditions:
        conditions["_phrases"] = phrases
        log.info("variety conditions | %s -> %s", name,
                 {k: v for k, v in conditions.items() if k != "_phrases"})
    return conditions


def supported_names(question: str, hits) -> list:
    """The names the CONTEXT supports for `question`, strongest first.

    The convenience form of candidate_pool() for callers that hold a question
    rather than an intent. Falls back to a cultivar reading when the question is
    not shaped like a listing request, because a caller asking this has already
    decided the question is about varieties.
    """
    intent = analyse(question)
    if intent is None:
        attributes = detect_attributes(question)
        intent = QueryIntent(
            question=question or "", entity="variety",
            quantity=QUANTITY_MULTIPLE,
            attributes_found=attributes,
            condition_terms=condition_terms(question, attributes))
    return [c["name"] for c in candidate_pool(intent, hits)]


def variety_request(question: str):
    """"multiple", "single_explicit" or None, for varieties and seeds.

    The narrow reading the pipeline has always used, now answered from the
    general intent layer so the two cannot disagree. Note that "single_explicit"
    is returned only for a question that ASKED for one option: the grammatical
    singular of "which variety" comes back as "multiple", which is the whole
    point of this layer.
    """
    intent = analyse(question)
    if intent is None or _BY_KEY[intent.entity].names != "cultivar":
        return None
    return intent.quantity


# ── Source-faithful agronomic claims ──────────────────────────────────────
#
# A rating is the document's word and it travels whole. Two ways an answer
# breaks that, both seen in real answers and neither caught by the dose check:
#
#   * PROMOTION. The sheet says "moderately resistant to blast"; the answer says
#     "resistant to blast". A farmer reading the second buys that seed for a
#     blast-prone field and gets the losses the qualifier was there to warn
#     about.
#   * GENERALISATION. The sheet lists two diseases; the answer says "resistant
#     to major pests and diseases", which covers everything the sheet never
#     claimed — including the one that will actually hit the crop.
#
# Detection is deliberately conservative: a claim is only reported when the
# CONTEXT contains the topic AND every statement of it there is qualified. An
# unhedged source is left alone, and a topic the context never mentions is the
# existing grounding rules' business, not this one's.
HEDGE_WORDS = ("moderate", "moderately", "mild", "mildly", "partial",
               "partially", "slightly", "somewhat", "medium", "fairly")
_CLAIM_WORDS = ("resistant", "resistance", "tolerant", "tolerance", "immune")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")
# "resistant to major pests and diseases" and its family. The blanket itself is
# the fault: no varietal sheet makes this claim.
OVERGENERAL_CLAIM_RE = re.compile(
    r"\b(?:resistant|resistance|tolerant|tolerance|immune)\b[^.\n]{0,40}?\b"
    r"(?:most|major|many|all|common|various|several|wide\s+range\s+of|"
    r"a\s+wide\s+range\s+of)\s+"
    # Longest first: "pests and diseases" must be taken whole, or the report
    # names half the claim it found.
    r"(?:pests?\s+and\s+diseases?|diseases?\s+and\s+pests?|"
    r"pests?|diseases?|insects?)\b", re.IGNORECASE)

# The topics a strength claim can be made about: the pathogens and pests the
# lexicons already name, plus the stresses.
_CLAIM_TOPICS = tuple(sorted(
    set(_NAME_LEXICONS["disease"]) | set(_NAME_LEXICONS["pest"])
    | {phrase for phrases in STRESS_CONDITIONS.values() for phrase in phrases},
    key=len, reverse=True))


def _claim_sentences(text: str):
    """Sentences that assert a resistance or tolerance, with their hedging."""
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        low = sentence.lower()
        if any(word in low for word in _CLAIM_WORDS):
            yield low, any(hedge in low for hedge in HEDGE_WORDS)


def overstated_claims(answer: str, context: str) -> list:
    """Claims the ANSWER makes more strongly than the CONTEXT supports.

    Returns human-readable strings for the retry prompt; empty when the answer
    is faithful. Never raises and never guesses: it reports only what it can
    point at in both texts.
    """
    problems: list = []

    blanket = OVERGENERAL_CLAIM_RE.search(answer or "")
    if blanket and not OVERGENERAL_CLAIM_RE.search(context or ""):
        problems.append(
            f"\"{blanket.group(0).strip()}\" — the CONTEXT names specific pests "
            "or diseases and makes no blanket claim")

    answer_claims = list(_claim_sentences(answer))
    context_claims = list(_claim_sentences(context))
    for topic in _CLAIM_TOPICS:
        said_plainly = [s for s, hedged in answer_claims
                        if topic in s and not hedged]
        if not said_plainly:
            continue
        about_topic = [(s, hedged) for s, hedged in context_claims if topic in s]
        if not about_topic:
            continue                    # the grounding rules' problem, not this
        if all(hedged for _s, hedged in about_topic):
            hedge = next((h for h in HEDGE_WORDS if h in about_topic[0][0]),
                         "moderate")
            problems.append(
                f"\"{topic}\" — the CONTEXT says {hedge} tolerance/resistance, "
                "the answer states it without that qualifier")
    if problems:
        log.warning("answer overstated the source: %s", problems)
    return problems


# ── Timeline intent classification ───────────────────────────────────────

_TIMELINE_SEEDLING_AGE = re.compile(
    r"(?:how\s+old|seedling\s+age|age\s+of\s+seedling|days?\s+old\s+.{0,20}transplant"
    r"|seedling.{0,15}(?:transplant|plant))", re.I,
)
_TIMELINE_DAT = re.compile(
    r"(?:days?\s+after\s+transplant|DAT\b|after\s+transplant(?:ing|ed)?)", re.I,
)
_TIMELINE_DAS = re.compile(
    r"(?:days?\s+after\s+sow|DAS\b|after\s+sowing)", re.I,
)
_TIMELINE_METHOD = re.compile(
    r"(?:\bSRI\b|system\s+of\s+rice\s+intensification|mat\s+(?:type\s+)?(?:nursery|transplant)"
    r"|mechanical\s+(?:mat\s+)?transplant|conventional\s+(?:nursery|transplant))", re.I,
)

TIMELINE_SEEDLING_AGE = "seedling_age_for_transplanting"
TIMELINE_DAT = "days_after_transplanting"
TIMELINE_DAS = "days_after_sowing"
TIMELINE_METHOD = "transplanting_method"
TIMELINE_AMBIGUOUS = "unknown_or_ambiguous"


def classify_timeline(question: str) -> str:
    """Classify the agricultural timeline intent of a question."""
    q = question.strip()
    if _TIMELINE_SEEDLING_AGE.search(q):
        return TIMELINE_SEEDLING_AGE
    if _TIMELINE_DAS.search(q):
        return TIMELINE_DAS
    if _TIMELINE_DAT.search(q):
        return TIMELINE_DAT
    if _TIMELINE_METHOD.search(q):
        return TIMELINE_METHOD
    has_transplant = re.search(r"transplant", q, re.I)
    has_days = re.search(r"\d+\s*days?", q, re.I)
    if has_transplant and has_days:
        return TIMELINE_AMBIGUOUS
    return ""
