"""
odia_register.py
================
Literary ("ଶୁଦ୍ଧ") Odia -> everyday SPOKEN Odia.

WHY THIS EXISTS
---------------
The translator produces correct but bookish Odia — the register of a textbook,
not of two people talking beside a field:

    ଆପଣଙ୍କର କ୍ଷେତ୍ରକୁ ଜଳ ମିଶାଇ …      ("apply water to your domain")
    ଆପଣଙ୍କର କ୍ଷେତକୁ ପାଣି ମିଶାଇ …      (what a farmer actually says)

The translation prompt already tells the model to avoid Sanskritised words and
it does not hold — the same lesson as the numbers, the acronyms and the filler:
when the outcome depends on the model picking particular words, pick them in
code instead. The prompt still carries the rule; this layer is what enforces it.

It matters more here than it looks. The answer is READ ALOUD. A farmer with
little schooling can parse a bookish sentence on paper, but spoken once through
a phone it just sounds like officialese and the advice is lost.

HOW REPLACEMENT IS SAFE IN AN AGGLUTINATIVE LANGUAGE
----------------------------------------------------
Odia glues case endings onto the noun (କ୍ଷେତ୍ର + କୁ -> କ୍ଷେତ୍ରକୁ). Each entry
therefore matches the stem and leaves the inflection alone, so it rides along
untouched:

    କ୍ଷେତ୍ରକୁ  -> କ୍ଷେତକୁ        କ୍ଷେତ୍ରରେ -> କ୍ଷେତରେ

BOTH boundaries are anchored, with a list of inflectional endings allowed on the
right. Anchoring only the left — as this did originally — also matches a stem
that is merely the first morpheme of a COMPOUND, and Odia builds its technical
vocabulary by compounding, so ଜଳସେଚନ (irrigation) became ପାଣିସେଚନ, which is not
a word. See `_compiled()`.

SCOPE — DELIBERATELY CONSERVATIVE
---------------------------------
Only pairs where the spoken word means exactly the same thing in an agricultural
sentence. No verb-phrase rewrites (ପ୍ରୟୋଗ କରନ୍ତୁ -> ଦିଅନ୍ତୁ): those change the
grammar around them and can break the sentence, which is the other half of what
was reported. Grow this list from farmer feedback, not from a dictionary.

NOT REVIEWED BY A NATIVE SPEAKER — the seed pairs below are the common,
well-established ones, but register is a judgement call. Have an Odia speaker
check `synonyms`-style before trusting it broadly.

PUBLIC API
----------
    to_spoken(text)            -> text with everyday words
    upsert_register(a, b)      -> add/overwrite one pair
    delete_register(a)         -> remove one pair
    get_register()             -> read-only copy
    reload_register()          -> re-read after an external edit
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

from .technical_terms import compose_odia, normalize_odia

log = logging.getLogger("irri")

_WRITE_LOCK = threading.Lock()
_PATH = Path(__file__).parent / "odia_register.json"

_ODIA_RANGE = r"଀-୿"

# literary / bookish  ->  what a farmer says
_BUILTIN: dict[str, str] = {
    # ── nouns a farmer uses every day ──
    "ଜଳ": "ପାଣି",              # water
    "ମୃତ୍ତିକା": "ମାଟି",          # soil
    "କ୍ଷେତ୍ର": "କ୍ଷେତ",          # field (କ୍ଷେତ୍ର also = a domain/area — wrong sense here)
    "ଉଦ୍ଭିଦ": "ଗଛ",             # plant
    "ବୃକ୍ଷ": "ଗଛ",              # tree
    "ଶସ୍ୟ": "ଫସଲ",              # crop
    "ବାୟୁ": "ପବନ",              # air / wind
    "ଅଗ୍ନି": "ନିଆଁ",             # fire
    "ଗୃହ": "ଘର",                # house
    "ମାର୍ଗ": "ବାଟ",             # way / path

    # ── scientific coinages farmers do not use ──
    # ଯବକ୍ଷାରଜାନ is the textbook Sanskrit word for nitrogen. A farmer has never
    # heard it — field feedback records it being misread as a herbicide.
    "ଯବକ୍ଷାରଜାନ": "ନାଇଟ୍ରୋଜେନ",   # nitrogen
    "ଅମ୍ଳଜାନ": "ଅମ୍ଳଜାନ",         # oxygen (kept — no common spoken alternative)

    # ── everyday function words ──
    "ନିମନ୍ତେ": "ପାଇଁ",           # for
    "ଯଥା": "ଯେମିତି",             # such as
    "ତଥା": "ଏବଂ",               # and
    "ଯଦ୍ୟପି": "ଯଦିଓ",           # although
    "ପରନ୍ତୁ": "କିନ୍ତୁ",          # but
    "ଅଥବା": "କିମ୍ବା",           # or
    "ପ୍ରାୟତଃ": "ପ୍ରାୟ",          # mostly

    # ── crop words the translator reaches for, but a farmer does not ──
    # ଶିର୍ଷ is "the top / the head" in the abstract; the paddy ear is ଶିଷ. Two
    # answers to the same farmer used one word each, so the same crop stage was
    # named two different ways in one session.
    "ଶିର୍ଷ": "ଶିଷ",             # panicle / ear of paddy
    "ପର୍ଯ୍ୟାୟ": "ଅବସ୍ଥା",       # stage (ପର୍ଯ୍ୟାୟ is the bookish half of the pair)

    # ── qualifiers ──
    "ପର୍ଯ୍ୟାପ୍ତ": "ଯଥେଷ୍ଟ",       # sufficient
    "ସମ୍ପୂର୍ଣ୍ଣ": "ପୁରା",         # complete
    "ପ୍ରାରମ୍ଭ": "ଆରମ୍ଭ",         # beginning
    "ଅତ୍ୟଧିକ": "ବେଶୀ",          # excessive
    "ନ୍ୟୂନ": "କମ୍",              # low / less
    "ତତ୍‌କ୍ଷଣାତ୍": "ସଙ୍ଗେ ସଙ୍ଗେ",  # immediately
}
# DELIBERATELY EXCLUDED — the spoken word does not fit every slot:
#   ଆବଶ୍ୟକ -> ଦରକାର   works as a predicate ("ଏହା ଦରକାର") but not before a noun:
#                      "ଆବଶ୍ୟକ ପରିମାଣ" would become the ungrammatical
#                      "ଦରକାର ପରିମାଣ".
#   ପ୍ରୟୋଗ କରନ୍ତୁ -> ଦିଅନ୍ତୁ  changes the grammar of the whole clause.
# Anything whose replacement depends on grammatical position belongs in the
# prompt, not in this table.


# ── Storage (mirrors odia_glossary / synonyms) ────────────────────────────

def _read_disk() -> dict:
    if _PATH.exists():
        try:
            d = json.loads(_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception as exc:
            log.warning("odia_register: read failed (%s)", exc)
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
def _load() -> dict[str, str]:
    return {**_BUILTIN, **{str(k): str(v) for k, v in _read_disk().items()}}


def reload_register() -> None:
    _load.cache_clear()
    _compiled.cache_clear()
    try:
        _compiled_imperative.cache_clear()
    except NameError:
        pass
    _load()


# Inflectional endings that may follow a word without making it a DIFFERENT word.
# Same list the glossary uses (odia_glossary._ODIA_SUFFIX) — the two layers must
# agree on where a word stops, or one will rewrite what the other matched.
ODIA_INFLECTION = (
    r"(?:ମାନଙ୍କୁ|ମାନଙ୍କର|ମାନଙ୍କ|ଗୁଡ଼ିକୁ|ଗୁଡ଼ିକର|ଗୁଡ଼ିକ|ଗୁଡ଼ାକ|ମାନେ"
    r"|ଙ୍କୁ|ଙ୍କର|ଙ୍କ|ରେ|ରୁ|କୁ|ଟି|ଟା|ର|େ)?"
)


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[re.Pattern, str]]:
    """Longest-first, whole-word (plus one inflection) so COMPOUNDS survive.

    THE RIGHT BOUNDARY IS NOT OPTIONAL — this was the bug it fixed.
    Anchoring only the left, as this did originally, means a stem also matches
    when it is merely the FIRST MORPHEME of a longer compound. Odia builds
    technical vocabulary by compounding, so ଜଳ -> ପାଣି fired inside words that
    have nothing to do with plain water, and turned them into non-words:

        ଜଳସେଚନ    irrigation   -> ପାଣିସେଚନ     (not a word)
        ଜଳବାୟୁ    climate      -> ପାଣିବାୟୁ     (not a word)
        ବାୟୁଜୀବୀ  aerobic      -> ପବନଜୀବୀ      (not a word)
        ଜଳମଗ୍ନ    submerged    -> ପାଣିମଗ୍ନ     (not a word)

    168 of 1525 glossary values were being rewritten this way — the glossary
    chose the right technical term and this layer silently overwrote it.

    The suffix is matched in a LOOKAHEAD, not consumed, so `sub()` replaces only
    the stem and whatever case ending the farmer used rides along untouched:
    ଜଳରେ -> ପାଣିରେ still works, which is what the left-only anchor was protecting.
    """
    out = []
    for lit in sorted(_load(), key=len, reverse=True):
        nl = normalize_odia(lit)
        rx = re.compile(
            rf"(?<![{_ODIA_RANGE}]){re.escape(nl)}"
            rf"(?={normalize_odia(ODIA_INFLECTION)}(?![{_ODIA_RANGE}]))"
        )
        out.append((rx, _load()[lit]))
    return out


def to_spoken(text: str) -> str:
    """Rewrite bookish words as the everyday spoken ones.

    Matching happens on NFD (so both nukta encodings are caught), but the RESULT
    is composed back to NFC. This function used to return the decomposed text and
    it is the last pass over the answer, so NFD was reaching the screen and the
    TTS engine — which is what made every ୋ sound like େ (କିଲୋଗ୍ରାମ heard as
    କିଲେଗ୍ରାମ). See compose_odia() in technical_terms.py.
    """
    if not (text or "").strip():
        return text
    out = normalize_odia(text)
    changed = []
    for rx, spoken in _compiled():
        if rx.search(out):
            out = rx.sub(spoken, out)
            changed.append(spoken)
    if changed:
        log.info("odia_register: %d word(s) made conversational", len(changed))
    return compose_odia(out)


# ── Verb-phrase register: "should X" -> direct imperative "please X" ──────
#
# WHY THIS EXISTS
# The translator keeps producing `... କରିବା ଉଚିତ` ("... should be applied")
# for advice sentences. That reads as an academic hedge, not as instruction —
# the farmer wants "ପ୍ରୟୋଗ କରନ୍ତୁ" ("apply it"), the same tone a friendly
# extension worker uses. Field feedback: "ଉଚିତ" makes people unsure whether
# the advice is real or optional.
#
# The general docstring above warns against verb-phrase rewrites because they
# can change the grammar around them. The compound `କରିବା ଉଚିତ` → `କରନ୍ତୁ` is
# the SAFE exception: both forms are the clause-final finite verb and take
# the same subject/object slots. Nothing to the left or right has to move.
#
# Kept as a SEPARATE function (not folded into `to_spoken`) so the intent is
# obvious in call sites and it can be disabled independently if a specific
# response needs the softer register (e.g. warnings, disclaimers).

# Auxiliaries that turn "V-ବା AUX" into "should V" and its close cousins.
# Include politeness qualifiers so we do not leave stranded fillers behind:
#   "ପ୍ରୟୋଗ କରିବା ଉଚିତ।"          -> "ପ୍ରୟୋଗ କରନ୍ତୁ।"
#   "ପ୍ରୟୋଗ କରିବା ଉଚିତ ଅଟେ।"      -> "ପ୍ରୟୋଗ କରନ୍ତୁ।"
#   "ପ୍ରୟୋଗ କରିବା ଆବଶ୍ୟକ।"        -> "ପ୍ରୟୋଗ କରନ୍ତୁ।"
#   "ପ୍ରୟୋଗ କରିବା ଦରକାର।"         -> "ପ୍ରୟୋଗ କରନ୍ତୁ।"
# The optional trailing halant matters: the translator writes ଉଚିତ୍ as often as
# ଉଚିତ, and without it half the hedges survived.
_SHOULD_AUX_PLAIN = r"(?:ଉଚିତ୍?|ଆବଶ୍ୟକ୍?|ଦରକାର୍?)(?:\s+(?:ଅଟେ|ହେବ|ଅଛି))?"

# "... କରିବା ଭଲ" ("it is best to do ...") is the same hedge wearing a different
# word, and the translator reaches for it whenever the English says "it's best
# to". But ଭଲ is ALSO the everyday adjective "good", so it is only the hedge
# when it CLOSES the clause. Without that guard,
#     "ଏହା କରିବା ଭଲ ଫଳ ଦେବ"   (doing this will give a good result)
# would lose its verb and become "ଏହା କରନ୍ତୁ ଫଳ ଦେବ".
#
# The clause end is `$` as often as it is a danda: to_imperative() splits the
# text on its terminators first, so a sentence arrives here without one.
_SHOULD_AUX_FINAL = r"(?:ଭଲ|ଉତ୍ତମ)(?=\s*(?:[।.!?,;]|$))"

_SHOULD_AUX = rf"(?:{_SHOULD_AUX_PLAIN}|{_SHOULD_AUX_FINAL})"

# Negation AFTER the auxiliary: "... କରିବା ଉଚିତ ନୁହେଁ" = "should NOT do".
# This MUST be matched before the positive rule. Substituting the positive rule
# there produced "ପ୍ରୟୋଗ କରନ୍ତୁ ନୁହେଁ" — a stranded negator that reads as an
# instruction to APPLY, i.e. the advice inverted. The negative imperative in
# spoken Odia is "... ନ୍ତୁ ନାହିଁ".
#
# Note this never fires for the ଭଲ hedge: that one requires the clause to END at
# the auxiliary, so "କରିବା ଭଲ ନୁହେଁ" matches NEITHER rule and is left alone.
# Leaving a prohibition untouched is the safe failure; inverting it is not.
_NEG_AFTER_AUX = r"\s+(?:ନୁହେଁ|ନୁହେ|ନୁହଁ|ନାହିଁ)"

# Compound-verb stems whose imperative form is IRREGULAR — the generic rule
# below would derive the wrong word for these, so they are spelled out and
# matched first. Everything to the left (the object noun) is untouched.
_IMPERATIVE: dict[str, str] = {
    "ଦେବା": "ଦିଅନ୍ତୁ",         # give   (ends in bare ବା, not ିବା)
    "ନେବା": "ନିଅନ୍ତୁ",         # take
    "ପାଇବା": "ପାଆନ୍ତୁ",        # get
    "ଖାଇବା": "ଖାଆନ୍ତୁ",        # eat
    "ପିଇବା": "ପିଅନ୍ତୁ",        # drink
    "ଯିବା": "ଯାଆନ୍ତୁ",         # go
    # Passive "is to be applied" -> the same direct instruction. The farmer's own
    # question is usually passive (ପ୍ରୟୋଗ କରାଯିବ), and the translator echoes it.
    "କରାଯିବା": "କରନ୍ତୁ",
    "କରାଯିବ": "କରନ୍ତୁ",
    "ଦିଆଯିବା": "ଦିଅନ୍ତୁ",
    "ଦିଆଯିବ": "ଦିଅନ୍ତୁ",
    "ପ୍ରୟୋଗ କରାଯିବା": "ପ୍ରୟୋଗ କରନ୍ତୁ",
    "ପ୍ରୟୋଗ କରାଯିବ":  "ପ୍ରୟୋଗ କରନ୍ତୁ",
}

# Verbal nouns the generic rule must NOT touch. In "X ହେବା ଆବଶ୍ୟକ" / "ମାଟି ଓଦା
# ରହିବା ଦରକାର" the subject is the FIELD or the CROP, not the farmer, so an
# imperative would tell the farmer to become wet. These describe a condition,
# not an action a person performs, and they keep the softer form on purpose.
_NO_IMPERATIVE: frozenset = frozenset({
    "ହେବା", "ହୋଇବା", "ହୋଇଥିବା", "ଥିବା", "ରହିବା", "ରହିଥିବା",
    "ପଡ଼ିବା", "ପଡ଼ିଯିବା", "ଲାଗିବା", "ମିଳିବା", "ଆସିବା", "ବଢ଼ିବା",
    "ଶୁଖିବା", "ମରିବା", "ପଚିବା", "ଚାଲିବା", "ଘଟିବା", "ହୋଇଯିବା",
})

# GENERIC RULE — the regular Odia verbal noun.
# An Odia verbal noun is STEM + ିବା and its polite imperative is STEM + ନ୍ତୁ:
#     କରିବା -> କରନ୍ତୁ    ରଖିବା -> ରଖନ୍ତୁ    ବୁଣିବା -> ବୁଣନ୍ତୁ    ଯାଞ୍ଚ କରିବା -> ଯାଞ୍ଚ କରନ୍ତୁ
# Causatives end STEM + ାଇବା and drop to STEM + ାନ୍ତୁ:
#     ମିଶାଇବା -> ମିଶାନ୍ତୁ   ଲଗାଇବା -> ଲଗାନ୍ତୁ   ପକାଇବା -> ପକାନ୍ତୁ   ଛିଞ୍ଚାଇବା -> ଛିଞ୍ଚାନ୍ତୁ
# This is why the fix now holds for verbs nobody has listed yet. The old code
# carried a hand-written table of ten stems, so any answer that used an
# eleventh verb kept the "ଉଚିତ" hedge — which is what was reported.
#
# At least two Odia letters must precede the ending, so one-syllable verbs
# (whose imperatives are irregular) fall through to the table above instead.
# ‌ / ‍ are the zero-width non-joiner and joiner: they sit INSIDE Odia
# conjuncts and would otherwise cut a stem in two.
_GENERIC_STEM = rf"(?<![{_ODIA_RANGE}])([{_ODIA_RANGE}‌‍]{{2,}}?)(ାଇବା|ିବା)"


@lru_cache(maxsize=1)
def _no_imperative_normalised() -> frozenset:
    """`_NO_IMPERATIVE` in the same normal form the matcher works in, so a
    nukta-bearing entry (ପଡ଼ିବା) is recognised whichever encoding arrived."""
    return frozenset(normalize_odia(v) for v in _NO_IMPERATIVE)


@lru_cache(maxsize=1)
def _compiled_imperative() -> list[tuple[re.Pattern, object]]:
    """(pattern, replacement) pairs, applied in order.

    Order is load-bearing twice over: NEGATIVE forms before positive ones (so a
    prohibition is never turned into an instruction), and explicit stems before
    the generic rule (so an irregular verb never gets the regular ending).
    """
    out: list[tuple[re.Pattern, object]] = []

    # Patterns are built from NFD so they match whichever encoding the text
    # arrives in — the same reason every comparison in technical_terms.py is
    # normalised. Replacements are stored composed; to_imperative() composes the
    # whole string on the way out.
    def _stem_rx(stem: str, negated: bool) -> re.Pattern:
        tail = _NEG_AFTER_AUX if negated else ""
        return re.compile(
            rf"(?<![{_ODIA_RANGE}]){re.escape(normalize_odia(stem))}\s+{_SHOULD_AUX}{tail}"
        )

    for negated in (True, False):
        # Longest stem first so "ପ୍ରୟୋଗ କରାଯିବା" is matched before "କରାଯିବା" alone.
        for stem in sorted(_IMPERATIVE, key=len, reverse=True):
            imp = _IMPERATIVE[stem]
            out.append((_stem_rx(stem, negated), imp + " ନାହିଁ" if negated else imp))

        rx = re.compile(
            normalize_odia(_GENERIC_STEM) + rf"\s+{_SHOULD_AUX}"
            + (_NEG_AFTER_AUX if negated else "")
        )
        out.append((rx, _generic_repl(negated)))
    return out


def _generic_repl(negated: bool):
    """Build the replacement callback for the generic verbal-noun rule."""
    def repl(m: re.Match) -> str:
        if normalize_odia(m.group(1) + m.group(2)) in _no_imperative_normalised():
            return m.group(0)                       # condition, not an instruction
        ending = "ାନ୍ତୁ" if m.group(2) == "ାଇବା" else "ନ୍ତୁ"
        return m.group(1) + ending + (" ନାହିଁ" if negated else "")
    return repl


# ── The subject the imperative leaves behind ──────────────────────────────
#
# Odia licenses a DATIVE subject with "ଉଚିତ": "ଆପଣଙ୍କୁ ପ୍ରୟୋଗ କରିବା ଉଚିତ" is
# correct and means "you ought to apply it". An IMPERATIVE does not — it takes
# the plain nominative. So rewriting only the verb strands the subject in the
# wrong case:
#
#     ଆପଣଙ୍କୁ ... ପ୍ରୟୋଗ କରିବା ଉଚିତ।     correct
#     ଆପଣଙ୍କୁ ... ପ୍ରୟୋଗ କରନ୍ତୁ।         WRONG — reads as "apply ... TO you"
#     ଆପଣ    ... ପ୍ରୟୋଗ କରନ୍ତୁ।         correct
#
# Reported from the field, and caused by the verb rewrite above: the two have to
# move together, which is why this runs per SENTENCE and only where a rewrite
# actually fired.
_DATIVE_SUBJECT: dict[str, str] = {
    "ଆପଣଙ୍କୁ": "ଆପଣ",
    "ଆପଣମାନଙ୍କୁ": "ଆପଣମାନେ",
    "ତୁମକୁ": "ତୁମେ",
    "ତୁମମାନଙ୍କୁ": "ତୁମେମାନେ",
}

# Openers a sentence may start with before the pronoun. The pronoun must be the
# SUBJECT slot, and only the sentence-initial position is a safe test for that:
# elsewhere ଆପଣଙ୍କୁ is nearly always a genuine object ("ଏହା ଆପଣଙ୍କୁ ସାହାଯ୍ୟ
# କରିବ" — this will help YOU), which must not be touched.
_SENTENCE_OPENERS = r"(?:ତେବେ|ତେଣୁ|ଏଣୁ|ପ୍ରଥମେ|ତା'ପରେ|ତାପରେ|ପରେ|ଏହାପରେ)"


@lru_cache(maxsize=1)
def _compiled_dative() -> list[tuple[re.Pattern, str]]:
    out = []
    for dative, nominative in _DATIVE_SUBJECT.items():
        rx = re.compile(
            rf"^(\s*(?:{normalize_odia(_SENTENCE_OPENERS)}\s*,?\s*)?)"
            rf"{re.escape(normalize_odia(dative))}(?![{_ODIA_RANGE}])"
        )
        out.append((rx, nominative))
    return out


# Sentence terminator. A full stop BETWEEN DIGITS is a decimal point, not the end
# of a sentence — splitting "2.5" would put the dose in one sentence and the verb
# in the next.
_SENT_END = r"(?:[।!?]|(?<!\d)\.(?!\d))"
_SENTENCE_SPLIT = re.compile(rf"({_SENT_END}\s*)")


def _imperative_sentence(sentence: str) -> str:
    """Rewrite one sentence's hedges, and its subject if any hedge changed."""
    out = sentence
    hits = 0
    for rx, imp in _compiled_imperative():
        out, n = rx.subn(imp, out)
        hits += n
    if not hits:
        return out
    for rx, nominative in _compiled_dative():
        out, n = rx.subn(lambda m: m.group(1) + nominative, out, count=1)
        if n:
            log.info("odia_register: dative subject made nominative for the imperative")
            break
    return out


def to_imperative(text: str) -> str:
    """Rewrite "X କରିବା ଉଚିତ" → "X କରନ୍ତୁ" and equivalents. Idempotent.

    Also handles the negative ("X କରିବା ଉଚିତ ନୁହେଁ" → "X କରନ୍ତୁ ନାହିଁ") and puts
    the sentence's subject back into the case an imperative needs.

    Works sentence by sentence: the subject fix must only apply where the verb in
    the SAME sentence actually became an imperative.
    """
    if not (text or "").strip():
        return text
    # Split on the terminator, KEEPING it (odd indices), so nothing is lost.
    parts = _SENTENCE_SPLIT.split(normalize_odia(text))
    rewritten = [p if i % 2 else _imperative_sentence(p) for i, p in enumerate(parts)]
    out = "".join(rewritten)
    return compose_odia(out)


# ── Editing ───────────────────────────────────────────────────────────────

def upsert_register(literary: str, spoken: str) -> bool:
    a, b = (literary or "").strip(), (spoken or "").strip()
    if not a or not b or a == b or re.search(r"[A-Za-z]", a + b):
        return False
    with _WRITE_LOCK:
        data = _read_disk() or dict(_BUILTIN)
        data[a] = b
        _atomic_write(data)
    reload_register()
    return True


def delete_register(literary: str) -> bool:
    a = (literary or "").strip()
    with _WRITE_LOCK:
        data = _read_disk() or dict(_BUILTIN)
        if a not in data:
            return False
        data.pop(a)
        _atomic_write(data)
    reload_register()
    return True


def get_register() -> dict[str, str]:
    return dict(_load())
