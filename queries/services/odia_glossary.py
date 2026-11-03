"""
odia_glossary.py
================
Single source of truth for English → Odia agricultural term translations.

HOW IT WORKS
------------
1. Loads  odia_glossary.json  (sits next to this file) at first import.
2. Exposes two public helpers:

      apply_glossary(text)          → replace English terms in a string
      build_prompt_glossary()       → formatted string for LLM system prompts

HOW TO UPDATE THE GLOSSARY
---------------------------
- Run  extract_agri_terms.py  to auto-generate  odia_glossary_draft.json
  from your ChromaDB documents.
- Review/correct the draft with an Odia speaker.
- Save the final file as  odia_glossary.json  next to this file.
- No code changes needed — this module hot-reloads on next Django restart.

PIPELINE INTEGRATION (pipeline.py)
------------------------------------
Replace your current hardcoded glossary block in the Odia sys prompt with:

    from .odia_glossary import build_prompt_glossary

    sys = (
        "...your existing rules...\n"
        "12. Common agricultural terms and their Odia equivalents:\n"
        + build_prompt_glossary()
        + "\nTEXT TO TRANSLATE:"
    )

And after the _chat() call in translate(), use apply_glossary() as a
fast pre-pass before the LLM re-prompt guard:

    result = _chat(...)
    result = apply_glossary(result)          # fast local replacements
    # ... then your Latin-script guard ...
    result = _correct_odia(result)
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

# Guards concurrent writes to odia_glossary.json (auto-learning is called from
# request threads).
_WRITE_LOCK = threading.Lock()

# ── Path to the JSON file ─────────────────────────────────────────────────
_GLOSSARY_PATH = Path(__file__).parent / "odia_glossary.json"

# Fallback built-in terms (used if odia_glossary.json is missing).
# These cover the most common terms seen in rice-farming documents.
# Add more here only as a last resort — prefer updating the JSON file.
_BUILTIN_FALLBACK: dict[str, str] = {
    # Soil & organic matter
    "green manure": "ସବୁଜ ସାର",
    "organic matter": "ଜୈବ ପଦାର୍ଥ",
    "organic manure": "ଜୈବ ସାର",
    "compost": "କମ୍ପୋଷ୍ଟ",
    "vermicompost": "ଭର୍ମି କମ୍ପୋଷ୍ଟ",
    "soil health": "ମାଟିର ସ୍ୱାସ୍ଥ୍ୟ",
    "soil fertility": "ମାଟିର ଉର୍ବରତା",

    # People vs the activity. ଚାଷ is farming, ଚାଷୀ is the farmer — the -ୀ is a
    # suffix, not a typo. Recorded here because the glossary had no plain
    # "farmer" entry at all, which left the word to the translator.
    "farmer": "ଚାଷୀ",
    "farmers": "ଚାଷୀ",
    "farming": "ଚାଷ",

    # Crops & cultivation
    "rice": "ଧାନ",
    "paddy": "ଧାନ",
    "crop": "ଫସଲ",
    "cultivation": "ଚାଷ",
    "transplanting": "ରୁଆ",
    "transplantation": "ରୁଆ",
    "nursery": "ବିହନ ତଳ",
    "seedling": "ଚାରା",
    "tillage": "ହଳ",
    "harvesting": "ଅମଳ",
    "yield": "ଅମଳ",
    "variety": "ଜାତ",
    "seed": "ବିହନ",
    "germination": "ଗଜା",
    "kharif": "ଖରିଫ",
    "rabi": "ରବି",

    # Nutrients & fertilizers
    "nitrogen": "ନାଇଟ୍ରୋଜେନ",
    "phosphorus": "ଫସଫରସ",
    "potassium": "ପୋଟାସିୟମ",
    "nitrogen fixation": "ନାଇଟ୍ରୋଜେନ ସ୍ଥିରୀକରଣ",
    "basal dose": "ମୂଳ ମାତ୍ରା",
    "top dressing": "ଉପରି ପ୍ରୟୋଗ",
    "foliar spray": "ପତ୍ର ଉପରେ ସ୍ପ୍ରେ",
    "micronutrient": "ଅଣୁ ପୋଷକ",
    "zinc": "ଜିଙ୍କ",
    "iron": "ଲୌହ",
    # Muriate of Potash. Corrected from field feedback — the spellings recorded
    # in odia_glossary.json (ମୁରିଆଟ ଅଫ୍ ପୋଟାସ୍ / ମ୍ୟୁରେଟ ଅଫ ପଟାସ) are wrong.
    # odia_translit._FUZZY_FIXES normalises the wrong ones wherever they appear,
    # so the answer is right even against a glossary file that still has them.
    "muriate of potash": "ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ",
    "murate of potash": "ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ",
    "muriate": "ମ୍ୟୁରିଏଟ୍",
    "potash": "ପୋଟାଶ",

    # Diseases & pests
    "blast disease": "ବ୍ଲାଷ୍ଟ ରୋଗ",
    "blast": "ବ୍ଲାଷ୍ଟ",
    "sheath blight": "ଶୀଥ ବ୍ଲାଇଟ",
    "brown spot": "ବାଦାମୀ ଦାଗ",
    "bacterial blight": "ଜୀବାଣୁ ବ୍ଲାଇଟ",
    "tungro": "ଟୁଙ୍ଗ୍ରୋ",
    "stem borer": "ଡାଣ୍ଡ ବିନ୍ଧା",
    "leaf folder": "ପତ୍ର ମୋଡ଼ା",
    "gall midge": "ଗଲ ମିଜ",
    "whitefly": "ଧଳା ମାଛି",
    "aphid": "ଆଫିଡ",
    "weed": "ଘାସ",
    "weed management": "ଘାସ ନିୟନ୍ତ୍ରଣ",

    # Pest control
    "pesticide": "କୀଟନାଶକ",
    "insecticide": "କୀଟନାଶକ",
    "fungicide": "ଛତୁ ନାଶକ",
    "herbicide": "ଘାସ ନାଶକ",
    "bio-pesticide": "ଜୈବ କୀଟନାଶକ",
    "spray": "ସ୍ପ୍ରେ",

    # Water management
    "irrigation": "ଜଳସେଚନ",
    "drainage": "ଜଳ ନିଷ୍କାସନ",
    "flood": "ବନ୍ୟା",
    "drought": "ଖରା",
    "waterlogging": "ଜଳ ଜମା",

    # Specific plants
    "dhaincha": "ଢ଼େଞ୍ଚା",
    "sunhemp": "ସନ ହେମ୍ପ",
    "azolla": "ଆଜୋଲା",

    # Crop stages
    "tillering": "କୁଆଁରା ବାହାରିବା",
    "flowering": "ଫୁଲ ଫୁଟା",
    "grain filling": "ଶସ୍ୟ ପୂରଣ",
    "maturity": "ପ୍ରଡ",
    "heading": "ଥୋଡ଼ ବାହାର",
    # Reverse-map so translations of these terms use farmer wording, matching
    # the synonym canonical forms added on the retrieval side.
    "internode elongation stage": "ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା",
    "jointing stage": "ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା",
    "stem elongation stage": "ଗଣ୍ଠି ଛାଡିବା ଅବସ୍ଥା",

    # Variety-duration classes (see synonyms.py canonicalisation)
    "late-maturing variety": "ବିଳମ୍ବ କିସମ",
    "late maturing variety": "ବିଳମ୍ବ କିସମ",
    "long duration variety": "ବିଳମ୍ବ କିସମ",
    "medium-maturing variety": "ମଧ୍ୟମ କିସମ",
    "medium duration variety": "ମଧ୍ୟମ କିସମ",
    "early-maturing variety": "ଅଳ୍ପ କାଳୀନ କିସମ",
    "short duration variety": "ଅଳ୍ପ କାଳୀନ କିସମ",

    # Soil terms
    "soil": "ମାଟି",
    "clay": "ଚିକଣ ମାଟି",
    "loam": "ଦୋ-ଆଁଶ ମାଟି",
    "sandy soil": "ବାଲୁଆ ମାଟି",
    "ph": "pH",
}


# ── Core loader ───────────────────────────────────────────────────────────

def _as_variants(value) -> list[str]:
    """Normalise a glossary value to a list of Odia terms, PRIMARY FIRST.

    An entry may be written either way, and both mean the same thing:

        "tillering": "କୁଆଁରା ବାହାରିବା"                       one term
        "tillering": ["କୁଆଁରା ବାହାରିବା", "ପିଲ ବାହାରିବା"]      several

    The FIRST term is the one written into answers. The rest are alternatives
    that are still RECOGNISED — they matter for the reverse direction, where a
    farmer may say any of them and the question must still be understood. That
    is why a second word is worth recording rather than choosing between them.
    """
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        seen, out = set(), []
        for item in value:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return []


# ── Odia words a glossary VALUE may never be ──────────────────────────────
#
# An ordinary Odia function word cannot be the translation of a technical term,
# so if one turns up as a glossary value it is a data error — and a costly one,
# because the glossary is TRUSTED in both directions:
#
#   forward  the answer uses the wrong word
#   reverse  the translator is TOLD the farmer's function word is a technical
#            term, so the glossary CAUSES the error instead of preventing it
#
# This is not hypothetical. odia_glossary.json carried
#
#     "micronutrients": "ଅନୁସାର"        (ନ)   -- "according to"
#     correct:          "ଅଣୁସାର"        (ଣ)   -- micro-nutrient
#
# a one-letter typo. ଅନୁସାରେ ("according to") is one of the commonest words in
# a farmer's question, so "ମାଟି ପରୀକ୍ଷା ଅନୁସାରେ ସାର କିପରି ଦେବି?" ("how do I
# apply fertiliser according to the soil test?") was translated as "... according
# to soil testing AND MICRONUTRIENT". The answer then advised a micronutrient
# spray the farmer had never asked about.
#
# The list is deliberately only postpositions, conjunctions, pronouns and
# question words — none of which can ever be a legitimate agricultural term.
# Entries are matched whole and normalised, so an inflected form is unaffected;
# only the bare function word is rejected.
_VALUE_STOPWORDS: frozenset = frozenset({
    # postpositions / relators
    "ଅନୁସାର", "ଅନୁସାରେ", "ପାଇଁ", "ସହିତ", "ସହ", "ଠାରୁ", "ପରେ", "ଆଗେ", "ଆଗରୁ",
    "ମଧ୍ୟରେ", "ଭିତରେ", "ବାହାରେ", "ଉପରେ", "ତଳେ", "ପାଖରେ", "ଦ୍ୱାରା", "ବିନା",
    # conjunctions / particles
    "ଏବଂ", "ଓ", "କିମ୍ବା", "ଅଥବା", "ଯଦି", "ତେବେ", "ତେଣୁ", "କିନ୍ତୁ", "ମାତ୍ର",
    "ମଧ୍ୟ", "ହିଁ", "ତଥା", "ନିମନ୍ତେ",
    # pronouns / determiners / question words
    "ଏହା", "ସେହି", "ଏହି", "ତାହା", "ଆପଣ", "ଆପଣଙ୍କ", "ମୁଁ", "ଆମେ", "ସେ",
    "କେତେ", "କିପରି", "କେମିତି", "କେଉଁ", "କେବେ", "କାହିଁକି", "କଣ", "କ'ଣ",
    # bare copulas / negators
    "ଅଛି", "ନାହିଁ", "ନୁହେଁ", "ହେବ", "ଥିଲା", "ଅଟେ",
})


@lru_cache(maxsize=1)
def _value_stopwords_normalised() -> frozenset:
    return frozenset(normalize_odia(w) for w in _VALUE_STOPWORDS)


# ── English words a glossary KEY may never be ─────────────────────────────
#
# TWO TIERS, because the damage is not the same for every ordinary word.
#
# TIER 1 — DESTRUCTIVE (_DESTRUCTIVE_KEYS, below).
# Function words and common verbs. `apply_glossary()` rewrites its key wherever
# it appears, so an entry for one of these is substituted into the middle of
# every sentence that contains it and the English is wrecked before the
# translator ever sees it:
#
#     "for the tillering stage, apply 30 kg"
#      -> "ପାଇଁ ଟି କୁଆଁରା ବାହାରିବା ଅବସ୍ଥା, apply 30 kg"      ("for"->ପାଇଁ, "the"->ଟି)
#
# Verbs belong here for the reason pipeline._GLOSSARY_STOPWORDS gives: frozen as
# a bare Odia noun in the English verb slot they break the clause, whereas left
# alone the translator inflects them correctly. NOTHING may create these — not
# the auto-learner, not a human, not the admin.
#
# TIER 2 — ORDINARY (_ORDINARY_KEYS).
# Everyday adjectives, adverbs and generic nouns: "strong", "high", "usually".
# These are single lexical items, so substituting one costs nothing structurally
# — "strong" -> ଶକ୍ତିଶାଳୀ puts one adjective where an adjective was. They are
# blocked for the AUTO-LEARNER only, which has no judgement and would otherwise
# record every English word a translation happened to leave behind. A human
# curating the glossary deliberately may add them, and that is the difference
# that matters: provenance, not the word itself.
#
# Agricultural nouns that happen to be everyday words (water, soil, seed, plant,
# leaf, grain, spray, yield, field, root, stem) are in NEITHER list.
_DESTRUCTIVE_KEYS: frozenset = frozenset({
    # articles, pronouns, determiners
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "our", "us", "you", "your", "yours", "i", "my", "me",
    "he", "she", "him", "her", "his", "who", "whom", "whose", "which", "what",
    "all", "any", "some", "each", "every", "both", "either", "neither", "other",
    "another", "such", "same", "own", "few", "several", "enough", "none",
    # prepositions, conjunctions, particles
    "and", "or", "but", "if", "then", "else", "so", "than", "as", "because",
    "to", "of", "in", "on", "at", "for", "with", "without", "from", "by",
    "into", "onto", "upon", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "over", "under", "up", "down", "out",
    "off", "again", "further", "while", "until", "till", "when", "where", "why",
    "how", "here", "there", "also", "too", "very", "just", "only", "even",
    "not", "no", "yes", "nor", "per",
    # copulas / auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "done", "have", "has", "had", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    # everyday verbs — see the note above
    "use", "used", "using", "make", "makes", "made", "get", "gets", "got",
    "give", "gives", "given", "take", "takes", "taken", "keep", "keeps",
    "put", "puts", "help", "helps", "helped", "need", "needs", "needed",
    "want", "wants", "start", "starts", "started", "stop", "stops", "become",
    "becomes", "grow", "grows", "growing", "check", "checks", "see", "look",
    "know", "find", "add", "adds", "apply", "applies", "applied", "applying",
    "control", "controls", "controlling", "avoid", "avoids", "ensure", "allow",
    "cause", "causes", "improve", "improves", "increase", "reduce", "follow",
    "release", "mature", "matures", "suit", "contain",
    "contains", "containing",
})

# TIER 2 — ordinary but harmless to substitute. Blocked for the auto-learner,
# allowed for a human curating the glossary on purpose. "strong" is here rather
# than above because ଶକ୍ତିଶାଳୀ in an adjective slot is still an adjective; it does
# not fragment the clause the way "the" -> ଟି does.
_ORDINARY_KEYS: frozenset = frozenset({
    "good", "better", "best", "bad", "worse", "worst", "big", "small", "large",
    "little", "long", "short", "high", "higher", "low", "lower", "new", "old",
    "young", "first", "second", "third", "last", "next", "early", "late",
    "easy", "hard", "strong", "stronger", "weak", "fast", "slow", "quick",
    "right", "wrong", "proper", "properly", "correct", "important", "useful",
    "helpful", "healthy", "ready", "full", "empty", "clean", "dry", "wet",
    "warm", "cold", "hot", "usual", "usually", "often", "always", "never",
    "sometimes", "shortly", "soon", "quickly", "slowly", "well", "much",
    "many", "more", "most", "less", "least", "different", "possible",
    "necessary", "available", "suitable", "effective", "efficient", "released",
    # bare time / quantity nouns with no agricultural sense of their own
    "time", "times", "day", "days", "week", "month", "year", "amount",
    "quantity", "number", "part", "parts", "content", "type", "kind", "way",
    "thing", "things", "level", "area", "place", "point", "result", "results",
})


def _all_words_in(english: str, vocabulary) -> bool:
    """True if EVERY word of `english` is in `vocabulary`.

    A phrase is only ordinary when nothing in it is a real term. One real word is
    enough — "grain filling", "top dressing", "long duration" and "standing
    water" all contain an everyday word and all belong in the glossary. "that
    time" contains nothing else, and does not.
    """
    key = (english or "").strip().lower()
    if not key:
        return False
    words = [w for w in re.split(r"[\s\-/]+", key) if w]
    return bool(words) and all(w in vocabulary for w in words)


def is_destructive_key(english: str) -> bool:
    """True if an entry for `english` would corrupt sentences. Never allowed."""
    return _all_words_in(english, _DESTRUCTIVE_KEYS)


def is_auto_learn_blocked(english: str) -> bool:
    """True if the AUTO-LEARNER must not record `english`.

    Wider than `is_destructive_key`: the learner has no judgement and would
    otherwise record every English word a translation happened to leave behind.
    A human may still add anything this rejects, as long as it is not destructive.
    """
    return _all_words_in(english, _DESTRUCTIVE_KEYS | _ORDINARY_KEYS)


# ── Action verbs: entries that must INFLECT, so may never be frozen ───────
#
# THE ONE AUTHORITATIVE TEST. Both outbound selection paths — the placeholder
# markers in pipeline._force_glossary_terms and the "Output EXACTLY" list in the
# LLM-fallback prompt — call is_action_verb_entry(), so they cannot disagree.
#
# Freezing a term means inserting it verbatim. That is right for a noun and
# wrong for a verb: an Odia verb carries tense and mood in its ending, and a
# frozen unit can never change it. Production:
#
#   'apply' -> ପ୍ରୟୋଗ କରିବା, frozen as ZQX5XQZ
#   "For Veena rice, ZQX5XQZ 70 kg ..."  ->  "ପ୍ରୟୋଗ କରିବା ପାଇଁ ... ବ୍ୟବହାର କରନ୍ତୁ"
#
# WHY THE ENGLISH KEY IS NOT ENOUGH. The hazard lives in the ODIA VALUE: a noun
# followed by a LIGHT verb (କରିବା / ଦେବା / ହେବା / କରେ / ହୁଏ ...). The glossary
# holds these under NOUN keys too — puddling -> କାଦୁଅ କରିବା, harrowing -> ମଇ ଦେବା,
# application -> ପ୍ରୟୋଗ କରିବା — and "Do puddling" frozen becomes
# "କାଦୁଅ କରିବା କରନ୍ତୁ". So the value is read, not only the key.
#
# WHY "ENDS IN -ବା" IS WRONG. About forty stage and process nouns end in a
# CONTENT verb and are terminology, not advice: tillering -> ପିଲ ଧରିବା,
# heading -> କେଣ୍ଡା ବାହାରିବା, lodging -> ଫସଲ ପଡ଼ିଯିବା. They stay frozen. Only
# the light verbs, which exist to be conjugated, mark an action entry.
_LIGHT_VERBS: frozenset = frozenset({
    "କରିବା", "କରିବ", "କରନ୍ତୁ", "କରେ", "କରନ୍ତି", "କରାଯାଏ", "କରାଯିବ", "କରାଯାଇଛି",
    "ଦେବା", "ଦେବ", "ଦିଅନ୍ତୁ", "ଦିଏ", "ଦିଆଯାଏ", "ଦିଆଯିବ",
    "ନେବା", "ନିଅନ୍ତୁ", "ନିଏ",
    "ହେବା", "ହେବ", "ହୁଏ", "ହୁଅନ୍ତି",
    "ରଖିବା", "ରଖନ୍ତୁ",
})
_LIGHT_VERBS_NORMALISED: frozenset = frozenset(normalize_odia(v) for v in _LIGHT_VERBS)


def is_action_verb_entry(english: str, odia: str) -> bool:
    """True if this glossary entry must inflect and so must never be frozen.

    Either side can decide it: an English key made only of function words and
    everyday verbs (is_destructive_key), or an Odia value whose last word is a
    light verb. Nothing is removed from the glossary — the entry is simply not
    eligible to be inserted verbatim.
    """
    if is_destructive_key(english):
        return True
    words = normalize_odia(odia or "").split()
    return bool(words) and words[-1] in _LIGHT_VERBS_NORMALISED


def action_verb_stem(odia: str) -> str:
    """The approved word an action entry carries, with its light verb removed.

    ପ୍ରୟୋଗ କରିବା -> ପ୍ରୟୋଗ; ଜମି ସମତଳ କରିବା -> ଜମି ସମତଳ; a bare କରିବା -> "".
    This is what stays mandatory: the terminology, free of the ending.
    """
    words = (odia or "").split()
    while words and normalize_odia(words[-1]) in _LIGHT_VERBS_NORMALISED:
        words.pop()
    return " ".join(words)


def glossary_action_stems(text: str) -> dict[str, str]:
    """{english: approved word} for every light-verb entry present in `text`.

    Unfrozen is not unprotected. These words cannot be placeholders, but they are
    still the glossary's choice (ପ୍ରୟୋଗ for "apply", not ବ୍ୟବହାର), so they are
    handed to the fluency pass to be REQUIRED — inflected, but present. An entry
    refused only for its English key (then -> ବେଳେ) has no light verb to strip
    and imposes nothing.
    """
    found = {}
    for english, odia in relevant_glossary(text).items():
        if not is_action_verb_entry(english, odia):
            continue
        stem = action_verb_stem(odia)
        if stem and normalize_odia(stem) != normalize_odia(odia):
            found[english] = stem
    return found


# ── Contextual phrases: guidance, not frozen terminology ──────────────────
#
# Being in the glossary is not, on its own, a reason to insert a phrase
# verbatim. Some entries are RELATIONS, not things: their meaning depends on
# words outside them, so a sealed placeholder breaks the sentence. Production:
#
#   'days after transplanting' -> ରୁଆ ପରର ଦିନ, frozen as ZQX0
#   "within 5 ZQX0" -> "5 ରୁଆ ପରର ଦିନ ମଧ୍ୟରେ"   the number cannot reach ଦିନ
#
# Odia is head-final, so the test reads the ENDS of the value:
#
#   ends in a relator        ଜଳମଗ୍ନ ହେବା ପରେ, ଏହି ସମୟରେ, ଅଳ୍ପ ସମୟ ପରେ
#   starts with a relator    ପାଇଁ ଉପଯୁକ୍ତ  (a postposition waiting for its noun)
#   ends in a counted noun   ରୁଆ ପରର ଦିନ   (its number lives outside the term)
#
# A relator INSIDE a noun-headed term does not count, and that distinction is
# the whole reason this reads the ends rather than asking "contains a relator":
# foliar spray -> ପତ୍ର ଉପରେ ସିଞ୍ଚନ, pre-emergence application ->
# ଗଜା ବାହାରିବା ପୂର୍ବରୁ ପ୍ରୟୋଗ and climate change adaptation ->
# ଜଳବାୟୁ ପରିବର୍ତ୍ତନ ସହ ଅନୁକୂଳନ are all terminology and stay frozen.
_RELATORS: frozenset = frozenset({
    "ପରେ", "ପରର", "ପରଠାରୁ", "ପୂର୍ବରୁ", "ପୂର୍ବର", "ଆଗରୁ", "ଆଗେ",
    "ସମୟରେ", "ମଧ୍ୟରେ", "ଭିତରେ", "ପାଇଁ", "ନିମନ୍ତେ", "ଦ୍ୱାରା", "ସହିତ", "ସହ",
    "ଉପରେ", "ତଳେ", "ଠାରୁ", "ପର୍ଯ୍ୟନ୍ତ", "ଅନୁସାରେ", "ବେଳେ", "ବିନା",
})
_RELATORS_NORMALISED: frozenset = frozenset(normalize_odia(w) for w in _RELATORS)
_COUNTED_TIME_NOUNS: frozenset = frozenset(
    normalize_odia(w) for w in ("ଦିନ", "ସପ୍ତାହ", "ମାସ", "ଘଣ୍ଟା", "ବର୍ଷ"))
_COUNTED_TIME_ENGLISH: frozenset = frozenset({
    "day", "days", "week", "weeks", "month", "months", "hour", "hours",
    "year", "years",
})


def is_contextual_entry(english: str, odia: str) -> bool:
    """True if this entry is a relational/temporal phrase to translate
    compositionally, rather than terminology to insert verbatim."""
    words = normalize_odia(odia or "").split()
    if len(words) < 2:
        return False                     # a single word is a term, not a relation
    if words[-1] in _RELATORS_NORMALISED or words[0] in _RELATORS_NORMALISED:
        return True
    if words[-1] in _COUNTED_TIME_NOUNS:
        return True
    key = (english or "").lower().split()
    return len(key) > 1 and key[0] in _COUNTED_TIME_ENGLISH


def glossary_term_class(english: str, odia: str) -> str:
    """How an outbound glossary entry may be used. THE authoritative split.

        "action"      its verb must inflect           -> required word, inflected
        "contextual"  a relational/temporal phrase    -> guidance, translated freely
        "protected"   noun-headed terminology         -> frozen exactly

    Action is checked first, so the action-verb classifier keeps precedence
    and is not altered by this one.
    """
    if is_action_verb_entry(english, odia):
        return "action"
    if is_contextual_entry(english, odia):
        return "contextual"
    return "protected"


def freezable_sub_terms(english_key: str) -> dict[str, str]:
    """PROTECTED glossary terms nested inside a longer key.

    The glossary scan is leftmost-longest, so "days after transplanting"
    CONSUMES "transplanting". When the longer key turns out to be contextual and
    is not frozen, the terminology inside it must still be — otherwise
    ଧାନ ରୁଆ is lost the moment its surrounding phrase is freed. Whole-word
    sub-phrases only, longest first.
    """
    _refresh_if_changed()
    glossary = _load_glossary()
    words = (english_key or "").lower().split()
    found: dict[str, str] = {}
    for size in range(len(words) - 1, 0, -1):
        for start in range(len(words) - size + 1):
            sub = " ".join(words[start:start + size])
            odia = glossary.get(sub)
            if (odia and glossary_term_class(sub, odia) == "protected"
                    and not any(sub in longer.split() or f" {sub} " in f" {longer} "
                                for longer in found)):
                found[sub] = odia
    return found


def _usable_variants(value) -> list[str]:
    """`_as_variants` minus anything that is a bare Odia function word."""
    kept, dropped = [], []
    for v in _as_variants(value):
        (dropped if normalize_odia(v) in _value_stopwords_normalised() else kept).append(v)
    if dropped:
        log.warning("odia_glossary: ignoring function-word value(s) %s — almost "
                    "certainly a typo in the glossary entry", dropped)
    return kept


@lru_cache(maxsize=1)
def _load_variants() -> dict[str, list[str]]:
    """{english: [odia, ...]}, primary first — the full picture.

    Everything else in this module is derived from here. `_load_glossary()` keeps
    returning {english: primary} so the ~20 existing call sites are untouched.
    """
    # NOTHING is filtered by KEY at load any more. The glossary is the operator's
    # authority over the translation, and an entry they put there has to be the
    # one that is used — the whole point of having it is not to be at the mercy
    # of whatever Google Translate returns.
    #
    # The earlier load-time block was justified with an example that turned out
    # to be wrong: apply_glossary() is only ever called on ODIA text (the answer
    # after translation, and the TTS string), so an English key can only match a
    # word that survived INTO an Odia answer — which is exactly the case it is
    # there to clean up. The English input side never goes through this function;
    # it goes through _force_glossary_terms(), which has its own guard.
    #
    # Values are still filtered: an Odia function word as a VALUE is a data error
    # in any direction (see _VALUE_STOPWORDS).
    out = {}
    for k, v in _load_raw().items():
        usable = _usable_variants(v)      # called ONCE per entry: it logs
        if usable:
            out[k] = usable
    return out


def get_variants(english: str) -> list[str]:
    """All Odia terms recorded for `english`, primary first."""
    _refresh_if_changed()
    return list(_load_variants().get((english or "").strip().lower(), []))


def get_all_variants() -> dict[str, list[str]]:
    """Read-only copy of the whole {english: [odia, ...]} map."""
    _refresh_if_changed()
    return {k: list(v) for k, v in _load_variants().items()}


def _load_raw() -> dict:
    """Load glossary from JSON file, falling back to built-in dict.

    Values may be a single Odia string or a LIST of them — see `_as_variants`.
    """
    if _GLOSSARY_PATH.exists():
        try:
            data = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                log.info(
                    "odia_glossary: loaded %d terms from %s",
                    len(data), _GLOSSARY_PATH,
                )
                # Merge: JSON file wins over built-in fallback
                merged = {**_BUILTIN_FALLBACK, **{k.lower(): v for k, v in data.items()}}
                return merged
        except Exception as exc:
            log.error("odia_glossary: JSON parse FAILED (%s). Trying backup...", exc)
            try:
                bpath = _GLOSSARY_PATH.with_name("odia_glossary.backup.json")
                if bpath.exists():
                    bdata = json.loads(bpath.read_text(encoding="utf-8"))
                    if isinstance(bdata, dict) and bdata:
                        log.warning("odia_glossary: loaded %d terms from BACKUP.", len(bdata))
                        return {**_BUILTIN_FALLBACK, **{k.lower(): v for k, v in bdata.items()}}
            except Exception as bexc:
                log.error("odia_glossary: backup also failed (%s).", bexc)
    else:
        log.warning(
            "odia_glossary: %s not found; using built-in fallback only. "
            "Run extract_agri_terms.py to generate it.",
            _GLOSSARY_PATH,
        )
    return dict(_BUILTIN_FALLBACK)


@lru_cache(maxsize=1)
def _load_glossary() -> dict[str, str]:
    """{english: PRIMARY odia} — the contract every existing caller expects.

    A multi-term entry collapses to its first term here, so translation, the
    prompt block and forced terminology all use one consistent word. The
    alternatives stay available through `get_variants()` and are wired into the
    reverse map, where recognising them actually matters.
    """
    return {k: v[0] for k, v in _load_variants().items()}


def reload_glossary() -> None:
    """Force reload from disk (call after updating odia_glossary.json at runtime).

    EVERY cache in this module is cleared, found by asking for `cache_clear`
    rather than by listing them. The list was maintained by hand and had gone
    out of date: `_reverse_scanner()` — the Odia->English alternation, added
    after this function was written — was never cleared, so a term added through
    the web editor was written to disk and still not recognised in a farmer's
    question until the process was restarted. A cache added tomorrow is covered
    without anyone remembering to come back here.
    """
    global _LOADED_STAMP
    stamp = _disk_stamp()
    for obj in list(globals().values()):
        clear = getattr(obj, "cache_clear", None)
        if callable(clear):
            clear()
    _LOADED_STAMP = stamp
    _load_glossary()


# The glossary file is shared mutable state: the web editor writes it, and under
# any multi-process server (gunicorn/uwsgi workers, a second runserver, a
# management command) the process that serves the NEXT request is usually not the
# one that saved. reload_glossary() only clears the caches of the process that
# calls it, so the saving worker saw the new term and every other worker kept
# serving its old lru_cache — the term was on disk and invisible in the page.
#
# So the read paths below check the file's stamp first: any process notices any
# other process's write, and editing the JSON by hand no longer needs a restart.
# One stat() per call, against an lru_cache hit — not worth optimising.
_LOADED_STAMP: object = None


def _disk_stamp():
    """Cheap change-detector for the glossary file: (mtime_ns, size)."""
    try:
        st = _GLOSSARY_PATH.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _refresh_if_changed() -> None:
    """Reload if the file on disk is not the one the caches were built from."""
    if _disk_stamp() != _LOADED_STAMP:
        reload_glossary()


def add_terms(mapping: dict[str, str]) -> dict[str, str]:
    """
    Persist new English→Odia term mappings into odia_glossary.json, then hot-reload
    so they take effect immediately (no Django restart needed).

    - Keys are stored lower-cased to match the loader.
    - Existing terms are NOT overwritten (first translation wins; refine later with
      verify_odia_glossary.py).
    - Values containing Latin letters are rejected (a bad translation would just put
      English back into the TTS text).
    - Ordinary English keys are rejected. This is the auto-learner's entry point
      (glossary_learn), and it offers up every Latin word left over in the TTS
      text — so without this the glossary silently accumulates "strong", "you",
      "is" and "to", each of which apply_glossary() then rewrites mid-sentence.
    - Returns the dict of terms that were actually newly added.
    """
    clean = {
        str(k).strip().lower(): str(v).strip()
        for k, v in mapping.items()
        if str(k).strip() and str(v).strip() and not re.search(r"[A-Za-z]", str(v))
    }
    refused = [k for k in clean if is_auto_learn_blocked(k)]
    if refused:
        log.info("add_terms: refused %d ordinary word(s), not glossary terms: %s",
                 len(refused), sorted(refused))
        clean = {k: v for k, v in clean.items() if k not in refused}
    if not clean:
        return {}

    with _WRITE_LOCK:
        # Read the on-disk file only (not the merged built-in fallback).
        existing: dict[str, str] = {}
        if _GLOSSARY_PATH.exists():
            try:
                loaded = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception as exc:
                log.warning("add_terms: could not read existing glossary (%s)", exc)

        added = {k: v for k, v in clean.items() if k not in existing}
        if not added:
            return {}

        existing.update(added)

        # Atomic write: write to a temp file in the same dir, then replace, so a
        # concurrent reader never sees a half-written file.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(_GLOSSARY_PATH.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, _GLOSSARY_PATH)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        log.info("odia_glossary: learned %d new term(s): %s", len(added), list(added))

    reload_glossary()   # refresh in-memory caches so new terms apply right away
    return added


# ── Compiled regex for fast replacement ───────────────────────────────────

@lru_cache(maxsize=1)
def _compiled_pattern() -> re.Pattern:
    """
    Build a single compiled regex that matches any glossary term.

    - Longest terms first so 'green manure' is matched before 'manure'.
    - Multi-word terms tolerate extra whitespace, so 'green  manure' (double
      space) still matches (the plain '\\b...\\b' pattern used to match only the
      trailing word and leave 'green' in English).
    - Terms of 3+ chars use LATIN-aware boundaries instead of '\\b', so a term
      glued to Odia script — e.g. 'ଏକnitrogen' — is still matched (Odia letters
      count as word chars, which broke '\\b'), while a term embedded in a longer
      ENGLISH word ('nitrogenous') is still NOT matched.
    - 1-2 char terms keep strict '\\b' boundaries so they never fire on a stray
      letter sitting next to Odia script.
    """
    glossary = _load_glossary()
    # Sort by length descending so longer phrases match first
    sorted_terms = sorted(glossary.keys(), key=len, reverse=True)

    def _esc(term: str) -> str:
        # Flexible whitespace between words; everything else stays literal.
        return r"\s+".join(re.escape(tok) for tok in term.split())

    long_terms = [_esc(t) for t in sorted_terms if len(t) >= 3]
    short_terms = [re.escape(t) for t in sorted_terms if len(t) < 3]

    alts = []
    if long_terms:
        alts.append(r"(?<![A-Za-z0-9])(?:" + "|".join(long_terms) + r")(?![A-Za-z0-9])")
    if short_terms:
        alts.append(r"\b(?:" + "|".join(short_terms) + r")\b")
    return re.compile("|".join(alts), re.IGNORECASE)


def _norm_key(matched: str) -> str:
    """Normalise a matched span back to its glossary key form (collapse the
    flexible whitespace so 'green  manure' looks up 'green manure')."""
    return re.sub(r"\s+", " ", matched.lower())


# ── Public API ────────────────────────────────────────────────────────────

def apply_glossary(text: str) -> str:
    """
    Fast pre-pass: replace known English agricultural terms in `text`
    with their Odia equivalents using a single compiled regex.

    Call this BEFORE your LLM-based Latin-script guard in translate().
    It handles the common cases cheaply without an extra API call.

    Example:
        apply_glossary("ଧାନ ଚାଷ ପାଇଁ green manure ବ୍ୟବହାର କରନ୍ତୁ")
        → "ଧାନ ଚାଷ ପାଇଁ ସବୁଜ ସାର ବ୍ୟବହାର କରନ୍ତୁ"
    """
    _refresh_if_changed()
    glossary = _load_glossary()
    pattern = _compiled_pattern()

    def _replace(match: re.Match) -> str:
        return glossary.get(_norm_key(match.group(0)), match.group(0))

    return pattern.sub(_replace, text)


def build_prompt_glossary(max_terms: int = 60) -> str:
    """
    Return a formatted string suitable for inserting into an LLM system prompt.
    Lists up to `max_terms` most important terms (prioritises multi-word phrases).

    Usage in pipeline.py:
        from .odia_glossary import build_prompt_glossary
        sys_prompt += "\\n12. Common terms:\\n" + build_prompt_glossary()
    """
    glossary = _load_glossary()

    # Prioritise multi-word phrases, then alphabetical
    sorted_items = sorted(
        glossary.items(),
        key=lambda kv: (-len(kv[0].split()), kv[0]),
    )[:max_terms]

    lines = [f"    - {en} → {od}" for en, od in sorted_items]
    return "\n".join(lines)


def relevant_glossary(text: str, max_terms: int = 150) -> dict[str, str]:
    """
    Return ONLY the glossary entries whose English term actually appears in `text`
    (case-insensitive, whole-word). Used to give the translator the exact Odia for
    every known term in THIS request, so it never transliterates a translatable word.
    """
    _refresh_if_changed()
    glossary = _load_glossary()
    pattern = _compiled_pattern()
    found: dict[str, str] = {}
    for m in pattern.finditer(text or ""):
        key = _norm_key(m.group(0))
        od = glossary.get(key)
        if od and key not in found:
            found[key] = od
            if len(found) >= max_terms:
                break
    return found


# ── Single-term edit helpers (used by the web glossary editor) ─────────────

def _atomic_write_glossary(data: dict) -> None:
    """Atomically write the glossary dict to disk (sorted, UTF-8)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(_GLOSSARY_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, _GLOSSARY_PATH)
    except Exception:
        try: os.remove(tmp_path)
        except OSError: pass
        raise


def _read_disk_glossary() -> dict:
    if _GLOSSARY_PATH.exists():
        try:
            d = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception as exc:
            log.warning("read glossary failed: %s", exc)
    return {}


def upsert_term(english: str, odia: str) -> bool:
    """Set the PRIMARY Odia term (human edit wins). Returns True if written.

    Any alternatives already recorded are kept, and the new value becomes the
    first — so correcting the word a farmer hears never quietly discards the
    other spellings the reverse direction relies on.
    """
    en = (english or "").strip().lower()
    od = (odia or "").strip()
    if not en or not od:
        return False
    # A human edit is AUTHORITATIVE — nothing is refused here. If the operator
    # decides a word needs a fixed Odia translation, that is the answer; the
    # glossary exists so the system is not at the mercy of the translator.
    if is_destructive_key(en):
        log.info("upsert_term: %r is a structural word — recorded because a human "
                 "asked for it, but it will be substituted wherever it appears", en)
    with _WRITE_LOCK:
        data = _read_disk_glossary()
        existing = [v for v in _as_variants(data.get(en)) if v != od]
        data[en] = [od, *existing] if existing else od
        _atomic_write_glossary(data)
    reload_glossary()
    return True


def add_variant(english: str, odia: str) -> bool:
    """Record an ADDITIONAL Odia term for `english`, keeping the primary first."""
    en = (english or "").strip().lower()
    od = (odia or "").strip()
    if not en or not od:
        return False
    if is_destructive_key(en):       # allowed, but worth a line in the log
        log.info("add_variant: %r is a structural word — recorded on request", en)
    with _WRITE_LOCK:
        data = _read_disk_glossary()
        variants = _as_variants(data.get(en))
        if not variants:
            variants = _as_variants(_BUILTIN_FALLBACK.get(en))
        if od in variants:
            return False
        data[en] = [*variants, od] if variants else od
        _atomic_write_glossary(data)
    reload_glossary()
    return True


def delete_variant(english: str, odia: str) -> bool:
    """Remove ONE Odia term. The last remaining term cannot be removed — use
    delete_term() to drop the whole entry, so a term never ends up with none."""
    en = (english or "").strip().lower()
    od = (odia or "").strip()
    with _WRITE_LOCK:
        data = _read_disk_glossary()
        variants = _as_variants(data.get(en)) or _as_variants(_BUILTIN_FALLBACK.get(en))
        if od not in variants or len(variants) <= 1:
            return False
        variants.remove(od)
        data[en] = variants[0] if len(variants) == 1 else variants
        _atomic_write_glossary(data)
    reload_glossary()
    return True


def delete_term(english: str) -> bool:
    """Remove ONE term. Returns True if it existed and was removed."""
    en = (english or "").strip().lower()
    with _WRITE_LOCK:
        data = _read_disk_glossary()
        if en not in data:
            return False
        data.pop(en)
        _atomic_write_glossary(data)
    reload_glossary()
    return True


def get_glossary() -> dict[str, str]:
    """Return a copy of the full (merged) glossary dict for read-only use."""
    _refresh_if_changed()
    return dict(_load_glossary())


# Native Odia agricultural terms -> their standard English meaning (reverse only).
#
# These are the words FARMERS SPEAK. The main glossary was built by translating
# the documents' ENGLISH terms into Odia (see glossary_builder/), so it holds the
# Odia a translator produces — ଅଣୁ ପୋଷକ for micronutrient, ବର୍ମିକମ୍ପୋଷ୍ଟ for
# vermicompost — not the Odia a farmer actually says (ଅଣୁସାର, ଜିଆ ଖତ). The
# reverse direction reads farmer speech, so the spoken form must be entered here
# or it cannot be matched at all.
#
# IMPORTANT: enter the WHOLE spoken term, not a component. Matching is
# whole-word and longest-first, so a compound entry (ଜିଆ ଖତ) is matched and
# masked before its parts (ଜିଆ) are considered.
_ODIA_SYNONYMS: dict[str, str] = {
    "ଯବକ୍ଷାରଜାନ": "nitrogen",      # native word for nitrogen (NOT a herbicide!)
    "ଯବକ୍ଷାରଜାନ୍": "nitrogen",
    "ଗନ୍ଧକ": "sulphur",            # native word for sulphur
    "ନୀଚା ଜମି": "lowland",         # nicha jami = lowland
    "ଖାଲ ଜମି": "lowland",          # khala jami = lowland (NOT dry/fallow land)
    "ଖାଲୁଆ ଜମି": "lowland",
    "ପତିତ ଜମି": "fallow land",
    "ମରୁଡ଼ି": "drought",
    "ଗୋବର ଖତ": "farmyard manure",
    "ଗୋବର ସାର": "farmyard manure",

    # Vermicompost. The farmer says "ଜିଆ ଖତ" / "ଜିଆ ସାର" (worm manure), written
    # spaced or unspaced. ଜିଆ ALONE means the animal (earthworm) and is kept
    # last on purpose: as a bare component it used to fire inside every compound
    # above and turn "how much vermicompost" into "how many earthworms".
    "ଜିଆ ଖତ": "vermicompost",
    "ଜିଆ ସାର": "vermicompost",
    "ଜିଆ ଖାଦ": "vermicompost",
    "କେଞ୍ଚୁଆ ଖତ": "vermicompost",
    "ଜିଆ": "earthworm",

    # Micronutrient. Farmers say ଅଣୁସାର; the glossary only has the translated
    # forms (ଅଣୁ ପୋଷକ / ଅଣୁପୁଷ୍ଟିକର ଉପାଦାନ), and ଅଣୁସାର appears in it only buried
    # inside a longer phrase, so the spoken word had no entry of its own.
    "ଅଣୁସାର": "micronutrient",
    "ଅଣୁ ସାର": "micronutrient",
    "ଅଣୁ ପୋଷକ": "micronutrient",
    "ଅଣୁପୋଷକ": "micronutrient",

    # Tillering. ପିଲ ବାହାରିବା is the spoken form; the glossary has କୁଆଁରା.
    "ପିଲ ବାହାରିବା": "tillering",
    "ପିଲ": "tiller",

    # LODGING — the crop falling over. The glossary only has the formal noun
    # (lodging -> ଫସଲ ପଡ଼ିଯିବା); a farmer reports it as a conjugated VERB
    # ("ଧାନ ପଡ଼ିଗଲା" = the paddy fell). With no entry the verb was dropped and
    # "my crop has lodged, can I fertilise?" became "when is fertiliser applied?"
    # — inverted advice, since the corpus says excess fertiliser CAUSES lodging.
    #
    # Only CROP-QUALIFIED phrases are listed. The bare verb ପଡ଼ିଗଲା means "fell"
    # generally (ବର୍ଷା ପଡ଼ିଗଲା = it rained), so mapping it alone would inject a
    # wrong hint into unrelated sentences. A separated subject (ପୂଜାଧାନ ୫୫
    # ଦିନରେ ପଡ଼ିଗଲା) is handled by the prompt rule in translate() instead.
    "ଫସଲ ପଡ଼ିଗଲା": "lodging",
    "ଫସଲ ପଡ଼ିଛି": "lodging",
    "ଫସଲ ପଡ଼ିଯାଇଛି": "lodging",
    "ଫସଲ ପଡ଼ିଯିବା": "lodging",
    "ଧାନ ପଡ଼ିଗଲା": "lodging",
    "ଧାନ ପଡ଼ିଛି": "lodging",
    "ଗଛ ପଡ଼ିଗଲା": "lodging",
    "ନଇଁପଡ଼ିଛି": "lodging",
    "ନଇଁ ପଡ଼ିଛି": "lodging",
    "ଶୋଇପଡ଼ିଛି": "lodging",

    # The glossary maps ସାର ପ୍ରୟୋଗ -> "manuring". The corpus uses "fertilizer
    # application" in more chunks (11 vs 7) and "manuring" reads as organic-only.
    "ସାର ପ୍ରୟୋଗ": "fertilizer application",

    # TRANSPLANTING. The glossary holds only ରୁଆ (the seedling/the act, as the
    # document translator wrote it); a farmer says the VERB — ରୋଇବା, ରୋଇଲା. With
    # no entry for the spoken form, "ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?"
    # ("how many days before TRANSPLANTING rice should ploughing be done?") was
    # translated as "how many days before HARVESTING should rice be cut?" — the
    # model guessed the opposite end of the season, and having decided the
    # sentence was about harvesting it then overrode the correctly-matched
    # ହଳ କରିବା -> ploughing and wrote "cut". One missing word took a matched one
    # down with it, which is why the whole family is listed here.
    # THE STEM, not just the finished forms. ରୋଇବା / ରୋଇଲା / ରୋଇବ are three of
    # the endings this verb takes; the farmer in production used a fourth —
    # "ଧାନ ରୋଇ ସାରିବା ପରେ" (after finishing transplanting the rice) — and the
    # bare stem matched nothing, so Google was free to read it as "after
    # harvesting rice" and the question changed season.
    #
    # Listing the stem covers every construction built on it, because the
    # scanner's trailing boundary only permits a NOUN case ending after a term:
    # ରୋଇ cannot match inside ରୋଇବା (ବା is not a case ending), so the longer
    # entries still win where they apply and the stem catches the rest.
    #
    # A GENERAL "allow any verb ending on any term" RULE WAS CONSIDERED AND
    # REJECTED. It would make ସାର (fertilizer) match inside ସାରିବା ("to
    # finish") — a word in this very sentence — and turn a timing clause into a
    # fertiliser one. Verb morphology is listed per term, deliberately.
    "ରୋଇ": "transplanting",
    "ରୋଇବା": "transplanting",
    "ରୋଇବ": "transplanting",
    "ରୋଇଲା": "transplanting",
    "ରୋଇବାର": "transplanting",
    "ରୋଇଥିଲି": "transplanting",
    "ରୋଇଦେଲି": "transplanting",
    "ରୋପଣ": "transplanting",
    "ରୋପଣ କରିବା": "transplanting",

    # BASAL, not "basic". ମୂଳସାର is the first dose, worked in before or at
    # sowing; Google renders it "basic fertilizer", which is not a term the
    # corpus uses and which retrieval therefore cannot match to the basal rows.
    "ମୂଳସାର": "basal fertilizer",
    "ମୂଳ ସାର": "basal fertilizer",

    # The word that fixes the SHAPE of that question. Without it the model was
    # free to read "X ଆଗରୁ Y" as "Y before X" and invert which operation comes
    # first — advice a farmer would act on in the wrong order.
    "ଆଗରୁ": "before",

    # PESTS AND DISEASES. ରୋଗପୋକ is the compound a farmer uses for the whole
    # problem (ରୋଗ disease + ପୋକ insect) and had no entry anywhere. Left to
    # guess, and with ବାତ୍ୟା (cyclone) in the same sentence, the translator
    # rendered it as "cyclone" — so a question about what pests to expect after
    # a storm became a question about the storm, and the pest half vanished.
    #
    # The compound is listed before its parts; matching is leftmost-longest, so
    # ରୋଗପୋକ is taken whole and never as ରୋଗ + ପୋକ.
    "ରୋଗପୋକ": "pest and disease",
    "ରୋଗ ପୋକ": "pest and disease",
    "ରୋଗପୋକର": "pest and disease",
    "ପୋକମାଛି": "pest",
    "ପୋକ": "pest",
    "ରୋଗ": "disease",
    "ପ୍ରତିରୋଧକ": "control",

    # THE WEATHER STAYS THE WEATHER. These are here so the storm half of the
    # same sentence is equally stable — NOT to be confused with the pest terms
    # above. A question can be about both at once, and usually is.
    "ବାତ୍ୟା": "cyclone",
    "ବର୍ଷା": "rainfall",
}

# ── Reverse direction: Odia -> English (for translating the farmer's question) ──

@lru_cache(maxsize=1)
def _reverse_map() -> dict[str, str]:
    """
    Build an Odia-text -> English-term map (the inverse of the glossary).
    Several English terms can share one Odia value (rice/paddy -> ଧାନ); the
    shortest English key wins so we get the simplest standard term.

    Keys are NFD-normalised. Odia nukta letters have two valid encodings and NFC
    does NOT unify them (ଡ଼ / ଢ଼ / ୟ are Unicode composition exclusions), so a raw
    substring test matched for one STT backend and silently failed for another —
    ~27% of reverse terms were unreachable depending on which backend ran.
    """
    rev: dict[str, str] = {}
    # COLLISIONS ARE SILENT AND THEY LOSE REAL ENTRIES. Two English terms can
    # record the same Odia word, and only one survives — ଅମଳ is recorded for
    # both "yield" and "harvesting", and the shortest-key rule below means it
    # reverses to "yield", so a question about harvesting is hinted as a
    # question about yield. Nothing said so. The rule is kept exactly as it was
    # (changing it would move 108 existing entries at once), but every discarded
    # entry is now logged, so the next one is found by reading the log instead of
    # by a farmer getting the wrong answer.
    displaced: list[tuple] = []
    # EVERY variant maps back, not just the primary. A farmer may say any of the
    # words recorded for a term, and the question has to be understood whichever
    # one they use — that is the main reason to record more than one.
    for en, variants in _load_variants().items():
        for od in variants:
            od = normalize_odia((od or "").strip())
            if not od:
                continue
            if od not in rev:
                rev[od] = en
            elif len(en) < len(rev[od]):
                displaced.append((od, rev[od], en))         # kept: the new one
                rev[od] = en
            elif en != rev[od]:
                displaced.append((od, en, rev[od]))         # kept: the old one
    if displaced:
        log.info("glossary reverse map: %d collision(s); one English term per "
                 "Odia word survives: %s", len(displaced),
                 "; ".join(f"{od!r} -> {kept!r} (dropped {lost!r})"
                           for od, lost, kept in displaced[:12]))
    # Native Odia synonyms the forward glossary does not use as its value but
    # which farmers / documents DO use. These must map back to the standard
    # English term (e.g. so ଯବକ୍ଷାରଜାନ is never mistaken for "herbicide").
    # Normalised too: these were authored in the DECOMPOSED form while most
    # glossary values are precomposed, so the two halves of this map previously
    # disagreed with each other (e.g. ମରୁଡ଼ି "drought" never matched).
    rev.update({normalize_odia(k): v for k, v in _ODIA_SYNONYMS.items()})
    return rev


# Odia block. Used as a "letter" class so a term glued to more Odia script is
# treated as part of a longer word, not as a match.
_ODIA_RANGE = r"଀-୿"

# Odia is agglutinative: case and number suffixes attach directly to the noun, so
# ପିଲ ବାହାରିବା ("tillering") is spoken as ପିଲ ବାହାରିବାରେ ("in tillering"). A strict
# trailing boundary rejects every inflected form, so one optional suffix is
# allowed after a term. Longest-first inside the group so ଙ୍କୁ beats ଙ୍କ.
#
# Only the TRAILING boundary is relaxed. The LEADING boundary stays strict, and
# that is the one preventing fragment matches — ସାର inside ଅଣୁସାର is blocked by
# the character before it being Odia, regardless of what follows.
_ODIA_SUFFIX = (
    r"(?:ମାନଙ୍କୁ|ମାନଙ୍କର|ମାନଙ୍କ|ଗୁଡ଼ିକୁ|ଗୁଡ଼ିକର|ଗୁଡ଼ିକ|ଗୁଡ଼ାକ|ମାନେ"
    r"|ଙ୍କୁ|ଙ୍କର|ଙ୍କ|ରେ|ରୁ|କୁ|ଟି|ଟା|ର|େ)?"
)


@lru_cache(maxsize=1)
def _reverse_scanner() -> tuple[re.Pattern, dict[str, str]]:
    """One combined alternation for LEFTMOST-LONGEST scanning, plus its lookup.

    Mirrors `_compiled_pattern()` on the English side. Three properties this buys
    over the previous `od in text` test:

    * WORD BOUNDARIES. Odia compounds embed shorter glossary terms, and a bare
      substring test matched the fragment and then instructed the translator to
      use the fragment's meaning. ଅଣୁସାର (micronutrient) contains ସାର
      (fertilizer), so the prompt was told "ସାର -> fertiliser" and the model duly
      answered "fertiliser" — the glossary CAUSED the error rather than failing
      to prevent it. The lookarounds require the term to stand alone.

    * FLEXIBLE WHITESPACE (`\\s*`, not `\\s+`). Odia compounds are written both
      spaced and unspaced by speakers and by STT — ଜିଆ ଖତ and ଜିଆଖତ are one word
      — so a single entry must match both.

    * LEFTMOST-LONGEST. Overlapping terms must be resolved by position, not by
      global length. In "ଜିଆ ସାର ପ୍ରୟୋଗ", plain longest-first picks the longer
      ସାର ପ୍ରୟୋଗ ("manuring"), consuming the ସାର that belongs to the compound
      ଜିଆ ସାର ("vermicompost") and leaving a bare ଜିଆ ("earthworm") behind. A
      single left-to-right scan with alternatives ordered longest-first takes the
      longest term that starts EARLIEST, which is the compound.
    """
    rev = _reverse_map()
    terms = sorted((od for od in rev if len(od) >= 3), key=len, reverse=True)

    alts: list[str] = []
    # Whitespace-squashed key -> English, so a spaced entry still resolves when
    # the farmer said it unspaced (and vice versa).
    squashed: dict[str, str] = {}
    for od in terms:
        alts.append(r"\s*".join(re.escape(w) for w in od.split()))
        squashed.setdefault(re.sub(r"\s+", "", od), rev[od])

    # Group 1 is the term WITHOUT any inflectional suffix, so the lookup key is
    # unaffected by how the farmer inflected it.
    #
    # The suffix list is NFD-normalised like everything else here. Its ଗୁଡ଼ିକ /
    # ଗୁଡ଼ାକ entries carry a nukta and were written precomposed, so against the
    # decomposed haystack those plural forms could never match — the same
    # encoding mismatch this module's docstring warns about, one line below it.
    pattern = re.compile(
        rf"(?<![{_ODIA_RANGE}])((?:" + "|".join(alts) + r"))"
        + normalize_odia(_ODIA_SUFFIX)
        + rf"(?![{_ODIA_RANGE}])"
    )
    return pattern, squashed


def relevant_reverse_glossary(text: str, max_terms: int = 150) -> dict[str, str]:
    """
    Return {odia_term: english_term} for every glossary Odia value that appears in
    `text`. Used to translate a farmer's Odia question into the EXACT standard
    English terms the documents use, so retrieval matches well.

    Matching is whole-word and leftmost-longest in a single pass, so overlapping
    terms cannot both land in the prompt (see `_reverse_scanner`). A term the
    translator is told to use must be the one the farmer actually said.
    """
    # The farmer-facing direction: a term added by the operator (in another
    # worker, or by hand in the file) has to be recognised on the very next
    # question, not after a restart.
    _refresh_if_changed()
    pattern, squashed = _reverse_scanner()
    # NFD both sides: the map keys are normalised, so the haystack must be too or
    # every nukta-bearing term misses (see _reverse_map).
    haystack = normalize_odia(text or "")

    found: dict[str, str] = {}
    for m in pattern.finditer(haystack):
        matched = m.group(1).strip()      # group 1 = term without inflection
        en = squashed.get(re.sub(r"\s+", "", matched))
        if not en or matched in found:
            continue
        found[matched] = en
        if len(found) >= max_terms:
            break
    return found
