"""Action verbs from the glossary must inflect; technical nouns stay frozen.

THE PRODUCTION FAILURE
----------------------
    1 english  "For Veena rice, apply 70 kg of urea per acre. ..."
    2 forced   {... 'ZQX5XQZ': 'ପ୍ରୟୋଗ କରିବା', ...}
    3 google   "For ZQX4XQZ ZQX6XQZ, ZQX5XQZ 70 kg of ZQX7XQZ per acre. ..."
    5 restored "ବୀଣା ଧାନ, ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରତି ଏକର ପାଇଁ 70 କିଲୋଗ୍ରାମ ୟୁରିଆ
                ବ୍ୟବହାର କରନ୍ତୁ।"

A placeholder cannot be conjugated. Put one where a verb goes and Google
demotes it to a noun ("for ZQX5XQZ") and supplies a verb of its own; the
restored sentence then says "for applying ... use".

WHY THE ENGLISH-KEY CHECK WAS NOT ENOUGH
---------------------------------------
The first repair refused English keys that are everyday verbs. But the hazard
is in the ODIA VALUE: noun + LIGHT verb (କରିବା / ଦେବା / ହେବା / କରେ / ହୁଏ). The
light verb carries tense and mood, so it must inflect — and a frozen unit never
can. The live glossary has these under NOUN keys the English check passes:

    puddling      -> କାଦୁଅ କରିବା       "Do puddling" -> "କାଦୁଅ କରିବା କରନ୍ତୁ"
    harrowing     -> ମଇ ଦେବା
    land leveling -> ଜମି ସମତଳ କରିବା

WHAT MUST STAY FROZEN
---------------------
Stage and process nouns that end in a CONTENT verb are terminology, not advice:
tillering -> ପିଲ ଧରିବା, heading -> କେଣ୍ଡା ବାହାରିବା. So the rule is the light
verb, never "ends in -ବା". And freezing is not the only protection: an action
entry's noun head (ପ୍ରୟୋଗ, କାଦୁଅ) is still required in the answer, so the
terminology is kept while the verb inflects.
"""
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import odia_glossary as g
from queries.services import pipeline
from queries.services.technical_terms import normalize_odia

PRODUCTION = ("For Veena rice, apply 70 kg of urea per acre. At the basal stage "
              "or after transplanting, use 10-15 kg of urea. Then, at the active "
              "tillering stage and at panicle initiation stage, apply 30-35 kg "
              "of urea per acre each time.")

APPLY_ODIA = "ପ୍ରୟୋଗ କରିବା"

# Verbatim from the live glossary.
GLOSSARY = {
    # action verbs / light-verb compounds — must NOT be frozen
    "apply": APPLY_ODIA,
    "application": APPLY_ODIA,          # a NOUN key with the same verb value
    "use": "ବ୍ୟବହାର କରିବା",
    "puddling": "କାଦୁଅ କରିବା",
    "harrowing": "ମଇ ଦେବା",
    "land leveling": "ଜମି ସମତଳ କରିବା",
    "helps": "ସାହାଯ୍ୟ କରେ",
    # technical nouns, entities and stages — MUST stay frozen
    "veena": "ବୀଣା",
    "rice": "ଧାନ",
    "urea": "ୟୁରିଆ",
    "basal": "ମୂଳ",
    "transplanting": "ଧାନ ରୁଆ",
    "active tillering": "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା",
    "panicle initiation stage": "କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥା",
    "tillering": "ପିଲ ଧରିବା",           # ends in -ବା, but a CONTENT verb: a stage
    "heading": "କେଣ୍ଡା ବାହାରିବା",
    "weed control": "ଘାସ ନିୟନ୍ତ୍ରଣ",
}

BROKEN = re.compile(r"ପ୍ରୟୋଗ କରିବା ପାଇଁ.*ବ୍ୟବହାର କରନ୍ତୁ")


def n(text):
    return normalize_odia(text or "")


class GlossaryCase(SimpleTestCase):
    def setUp(self):
        self.real = g._GLOSSARY_PATH
        tmp = Path(tempfile.mkdtemp()) / "odia_glossary.json"
        tmp.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")
        g._GLOSSARY_PATH = tmp
        g.reload_glossary()
        self.addCleanup(self._restore)

    def _restore(self):
        g._GLOSSARY_PATH = self.real
        g.reload_glossary()

    def forced(self, text):
        marked, marks = pipeline._force_glossary_terms(text)
        return marked, [n(v) for v in marks.values()]


# ── Unit level: the selection layer ────────────────────────────────────────

class ActionVerbsAreNotFrozenTests(GlossaryCase):
    """Requirements 11a-d and 11f, at the selection layer."""

    def test_a_apply_70_kg_of_urea_per_acre(self):
        marked, forced = self.forced("apply 70 kg of urea per acre")

        self.assertIn("apply", marked)
        self.assertNotIn(n(APPLY_ODIA), forced)
        self.assertIn(n("ୟୁରିଆ"), forced)

    def test_b_apply_30_35_kg_of_urea(self):
        marked, forced = self.forced("apply 30-35 kg of urea")

        self.assertIn("apply", marked)
        self.assertNotIn(n(APPLY_ODIA), forced)

    def test_c_use_10_15_kg_of_urea(self):
        marked, forced = self.forced("use 10-15 kg of urea")

        self.assertIn("use", marked)
        self.assertNotIn(n("ବ୍ୟବହାର କରିବା"), forced)

    def test_d_a_sentence_with_both_apply_and_use(self):
        marked, forced = self.forced("Apply DAP at sowing and use urea later.")

        self.assertIn("Apply", marked)
        self.assertIn("use", marked)
        self.assertNotIn(n(APPLY_ODIA), forced)
        self.assertNotIn(n("ବ୍ୟବହାର କରିବା"), forced)

    def test_f_prayoga_karibaa_is_never_a_placeholder_under_any_key(self):
        # The value decides, not the key: "application" is a NOUN in English and
        # passed the old verb check, but its value still cannot inflect frozen.
        for sentence in (PRODUCTION,
                         "Fertilizer application is important.",
                         "Timely application of urea helps.",
                         "Apply urea. Then apply MOP."):
            with self.subTest(sentence=sentence):
                _marked, forced = self.forced(sentence)
                self.assertNotIn(n(APPLY_ODIA), forced)

    def test_light_verb_compounds_under_noun_keys_are_not_frozen(self):
        # The gap the English-key check left open.
        for sentence, value in (("Do puddling before transplanting.", "କାଦୁଅ କରିବା"),
                                ("Complete harrowing before sowing.", "ମଇ ଦେବା"),
                                ("Apply urea after land leveling.", "ଜମି ସମତଳ କରିବା")):
            with self.subTest(sentence=sentence):
                _marked, forced = self.forced(sentence)
                self.assertNotIn(n(value), forced)


class TechnicalTermsStayFrozenTests(GlossaryCase):
    """Requirement 11e: nothing the glossary exists for is loosened."""

    def test_e_every_noun_and_stage_in_the_production_sentence_is_frozen(self):
        _marked, forced = self.forced(PRODUCTION)

        for term in ("ବୀଣା", "ଧାନ", "ୟୁରିଆ", "ମୂଳ", "ଧାନ ରୁଆ",
                     "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା",
                     "କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥା"):
            self.assertIn(n(term), forced)

    def test_stage_nouns_ending_in_a_content_verb_stay_frozen(self):
        # "Ends in -ବା" is NOT the rule — these are terminology.
        for sentence, value in (("Apply urea at tillering.", "ପିଲ ଧରିବା"),
                                ("Drain the field before heading.", "କେଣ୍ଡା ବାହାରିବା")):
            with self.subTest(sentence=sentence):
                _marked, forced = self.forced(sentence)
                self.assertIn(n(value), forced)

    def test_a_multi_word_term_containing_a_verb_word_stays_frozen(self):
        _marked, forced = self.forced("Practise weed control early.")

        self.assertIn(n("ଘାସ ନିୟନ୍ତ୍ରଣ"), forced)

    def test_the_numbers_are_untouched(self):
        marked, _forced = self.forced(PRODUCTION)

        for figure in ("70 kg", "10-15 kg", "30-35 kg"):
            self.assertIn(figure, marked)


class TheAuthoritativeClassifierTests(SimpleTestCase):
    """One mechanism, in odia_glossary, that both selection paths call."""

    def test_the_classifier_reads_the_odia_value(self):
        from queries.services.odia_glossary import is_action_verb_entry

        for english, odia in (("apply", APPLY_ODIA), ("application", APPLY_ODIA),
                              ("use", "ବ୍ୟବହାର କରିବା"), ("puddling", "କାଦୁଅ କରିବା"),
                              ("harrowing", "ମଇ ଦେବା"), ("helps", "ସାହାଯ୍ୟ କରେ"),
                              ("do", "କରିବା"), ("give", "ଦେବା")):
            with self.subTest(english=english):
                self.assertTrue(is_action_verb_entry(english, odia))

    def test_the_classifier_keeps_nouns_and_stages_freezable(self):
        from queries.services.odia_glossary import is_action_verb_entry

        for english, odia in (("veena", "ବୀଣା"), ("rice", "ଧାନ"), ("urea", "ୟୁରିଆ"),
                              ("tillering", "ପିଲ ଧରିବା"),
                              ("heading", "କେଣ୍ଡା ବାହାରିବା"),
                              ("active tillering", "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା"),
                              ("weed control", "ଘାସ ନିୟନ୍ତ୍ରଣ"),
                              ("irrigation", "ଜଳସେଚନ")):
            with self.subTest(english=english):
                self.assertFalse(is_action_verb_entry(english, odia))

    def test_the_noun_head_is_what_remains_required(self):
        from queries.services.odia_glossary import action_verb_stem

        self.assertEqual(n("ପ୍ରୟୋଗ"), n(action_verb_stem(APPLY_ODIA)))
        self.assertEqual(n("ଜମି ସମତଳ"), n(action_verb_stem("ଜମି ସମତଳ କରିବା")))
        self.assertEqual("", action_verb_stem("କରିବା"))       # nothing left to keep


# ── Second selection path: the LLM-fallback prompt ─────────────────────────

class TheFallbackPromptTests(GlossaryCase):
    """translate()'s LLM path told the model to 'Output EXACTLY' every value —
    including ପ୍ରୟୋଗ କରିବା — which freezes the verb in the prompt instead."""

    def fallback_prompt(self, text=PRODUCTION):
        seen = {}

        def chat(messages, **kwargs):
            seen.setdefault("sys", messages[0]["content"])
            return "ଅନୁବାଦ"

        with patch.object(pipeline, "_google_translate",
                          side_effect=RuntimeError("google down")), \
                patch.object(pipeline, "_chat", side_effect=chat), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t):
            pipeline.translate(text, "English", "Odia")
        return seen.get("sys", "")

    def test_the_verb_is_not_in_the_output_exactly_list(self):
        prompt = self.fallback_prompt()
        exact = prompt.split("Output EXACTLY")[1].split("\n\n")[0] if "Output EXACTLY" in prompt else ""

        self.assertNotIn(f"apply → {APPLY_ODIA}", exact)

    def test_the_nouns_are_still_in_the_output_exactly_list(self):
        prompt = self.fallback_prompt()

        self.assertIn("urea → ୟୁରିଆ", prompt)
        self.assertIn("veena → ବୀଣା", prompt)


# ── Full English -> Odia pipeline ──────────────────────────────────────────

def google_like(text, source, target):
    """A deterministic stand-in with Google's real behaviour on placeholders.

    A marker sitting where a verb belongs (immediately before a quantity) is
    treated as a noun and given a verb of Google's own — exactly what was
    logged. A real English verb is conjugated.
    """
    out = []
    for sentence in re.split(r"(?<=\.)\s+", text):
        m = re.search(r"(ZQX\d+XQZ) (\d[\d-]*) kg", sentence)
        if m:
            out.append(f"{m.group(1)} ପାଇଁ {m.group(2)} କିଲୋଗ୍ରାମ ZQX9XQZ ବ୍ୟବହାର କରନ୍ତୁ।")
            continue
        figure = re.search(r"(\d[\d-]*) kg", sentence)
        qty = f"{figure.group(1)} କିଲୋଗ୍ରାମ " if figure else ""
        verb = ("ପ୍ରୟୋଗ କରନ୍ତୁ" if re.search(r"\bapply\b", sentence, re.I)
                else "ବ୍ୟବହାର କରନ୍ତୁ" if re.search(r"\buse\b", sentence, re.I)
                else "କରନ୍ତୁ")
        markers = " ".join(re.findall(r"ZQX\d+XQZ", sentence))
        out.append(f"{markers} {qty}{verb}।".strip())
    return " ".join(out)


class TheFullPipelineTests(GlossaryCase):
    """Requirement 12: English -> Odia end to end."""

    def translate(self, text=PRODUCTION):
        with patch.object(pipeline, "_google_translate", side_effect=google_like), \
                patch.object(pipeline, "_fix_odia_fluency",
                             side_effect=lambda t, s="", **k: t), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            return pipeline.translate(text, "English", "Odia")

    def test_the_production_answer_has_no_broken_verb_construction(self):
        out = n(self.translate())

        self.assertIsNone(BROKEN.search(out), out)
        self.assertNotIn(n("ପ୍ରୟୋଗ କରିବା ପାଇଁ"), out)

    def test_the_production_answer_keeps_its_frozen_terms(self):
        out = n(self.translate())

        for term in ("ବୀଣା", "ୟୁରିଆ", "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା",
                     "କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥା"):
            self.assertIn(n(term), out)

    def test_apply_is_conjugated_not_restored_as_an_infinitive(self):
        out = n(self.translate("apply 70 kg of urea per acre."))

        self.assertIn(n("ପ୍ରୟୋଗ କରନ୍ତୁ"), out)
        self.assertNotIn(n(APPLY_ODIA), out)

    def test_apply_and_use_together(self):
        out = n(self.translate("Apply 20 kg of urea. Use 10-15 kg of urea later."))

        self.assertIsNone(BROKEN.search(out), out)


class TheNounHeadIsStillEnforcedTests(GlossaryCase):
    """Unfrozen is not unprotected: ପ୍ରୟୋଗ must still reach the answer."""

    def test_the_fluency_pass_is_told_the_noun_head(self):
        from queries.services.odia_glossary import glossary_action_stems

        stems = glossary_action_stems(PRODUCTION)

        self.assertIn(n("ପ୍ରୟୋଗ"), [n(s) for s in stems.values()])

    def test_a_rewrite_that_swaps_the_head_for_a_synonym_is_retried(self):
        prompts = []
        replies = ["20 କିଲୋଗ୍ରାମ ୟୁରିଆ ବ୍ୟବହାର କରନ୍ତୁ।",      # lost ପ୍ରୟୋଗ
                   "20 କିଲୋଗ୍ରାମ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।"]      # kept it, inflected

        def gemini(prompt, text):
            prompts.append(prompt)
            return replies[min(len(prompts) - 1, 1)]

        with patch.object(pipeline, "_gemini_text", gemini):
            out = pipeline._fix_odia_fluency(
                "ପ୍ରୟୋଗ କରିବା ପାଇଁ 20 କିଲୋଗ୍ରାମ ୟୁରିଆ ବ୍ୟବହାର କରନ୍ତୁ।",
                "Apply 20 kg of urea.", glossary_stems=["ପ୍ରୟୋଗ"])

        self.assertEqual(2, len(prompts))
        self.assertIn("ପ୍ରୟୋଗ କରନ୍ତୁ", out)
