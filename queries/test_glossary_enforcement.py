"""An approved glossary term is not a wording choice.

THE CONFLICT THIS RESOLVES
--------------------------
The fluency pass is told to rewrite machine-translated Odia into what a farmer
would actually say. The glossary is told to fix the exact term for a concept.
Those two collide on every glossary term, and the fluency pass used to win —
it runs AFTER _restore_glossary_terms, so by the time it sees the text the
approved terms look like ordinary wording it should improve:

    glossary : tillers -> ପିଲ
    draft    : ପିଲ ବିକଶିତ ହେବା ପରେ ...
    rewrite  : କୁଶି ବିକାଶ ସମୟରେ ...        <- approved term silently replaced

The glossary outranks fluency, style, localisation and synonym choice. So the
terms the glossary forced into the text are now handed to the pass, stated as
mandatory in the prompt, AND CHECKED afterwards — a rewrite that dropped one is
discarded whole, the same way a rewrite that changed a dose is. Asking a model
nicely is not enforcement.

The Odia -> English direction was already enforced this way (the term is
substituted into the question and the translation is rejected if it does not
survive); it is asserted here too, so both directions are covered by one file.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import odia_glossary as g
from queries.services import pipeline

TILLER = "ପିଲ"            # the approved term for "tiller"
IRRIGATION = "ଜଳସେଚନ"     # the approved term for "irrigation"

DRAFT = "ପିଲ ବିକଶିତ ହେବା ପରେ ମାଟିର ଉପର ସ୍ତରକୁ ପାଖାପାଖି ସଂତୃପ୍ତ ରଖନ୍ତୁ।"
KEEPS_IT = "ପିଲ ବାହାରିବା ସମୟରେ ମାଟି ଭିଜା ରହିବାକୁ ଧ୍ୟାନ ଦିଅନ୍ତୁ।"
REPLACES_IT = "କୁଶି ବିକାଶ ସମୟରେ ମାଟି ଭିଜା ରହିବାକୁ ଧ୍ୟାନ ଦିଅନ୍ତୁ।"


def rewrite(draft, reply, terms=(), source_en="Keep the topsoil moist."):
    """Run the fluency pass with the model stubbed. Returns (output, prompt)."""
    seen = {}

    def gemini(prompt, text):
        seen["prompt"] = prompt
        return reply

    with patch.object(pipeline, "_gemini_text", gemini):
        out = pipeline._fix_odia_fluency(draft, source_en, glossary_terms=terms)
    return out, seen.get("prompt", "")


class TheTermSurvivesTheRewriteTests(SimpleTestCase):
    """The reported failure, both outcomes."""

    def test_a_rewrite_that_keeps_the_term_is_accepted(self):
        out, _prompt = rewrite(DRAFT, KEEPS_IT, terms=[TILLER])

        self.assertEqual(KEEPS_IT, out)

    def test_a_rewrite_that_swaps_it_for_a_synonym_is_discarded(self):
        out, _prompt = rewrite(DRAFT, REPLACES_IT, terms=[TILLER])

        self.assertEqual(DRAFT, out)

    def test_the_more_common_word_does_not_win_either(self):
        # ଜଳସେଚନ is approved for irrigation even though ପାଣି ଦେବା is commoner.
        draft = "ମାଟି ପୃଷ୍ଠରୁ ୧୫ ସେମି ତଳକୁ ଖସିଗଲେ ପୁନଃ ଜଳସେଚନ କରନ୍ତୁ।"
        out, _prompt = rewrite(
            draft, "ପାଣି ସ୍ତର ୧୫ ସେ.ମି. ତଳକୁ ଗଲେ ପୁଣି ଥରେ ପାଣି ଦିଅନ୍ତୁ।",
            terms=[IRRIGATION])

        self.assertEqual(draft, out)

    def test_several_terms_must_all_survive(self):
        draft = f"{TILLER} ଅବସ୍ଥାରେ {IRRIGATION} କରନ୍ତୁ।"
        half = f"{TILLER} ଅବସ୍ଥାରେ ପାଣି ଦିଅନ୍ତୁ।"
        out, _prompt = rewrite(draft, half, terms=[TILLER, IRRIGATION])

        self.assertEqual(draft, out)

    def test_a_term_that_is_not_in_the_text_is_not_demanded(self):
        # Otherwise a term forced elsewhere in a long answer would veto every
        # rewrite of a paragraph that never contained it.
        out, _prompt = rewrite(DRAFT, KEEPS_IT, terms=[TILLER, "ଆଜୋଲା"])

        self.assertEqual(KEEPS_IT, out)

    def test_the_check_is_normalisation_proof(self):
        # The glossary value and the model's output can be different valid
        # encodings of the same Odia word; that must not read as a dropped term.
        import unicodedata
        decomposed = unicodedata.normalize("NFD", KEEPS_IT)
        out, _prompt = rewrite(DRAFT, decomposed, terms=[unicodedata.normalize("NFC", TILLER)])

        self.assertEqual(decomposed, out)


class ThePromptStatesThePriorityTests(SimpleTestCase):
    """It is stated before the rules it overrides, and lists the real terms."""

    def prompt(self, terms=(TILLER,)):
        _out, prompt = rewrite(DRAFT, KEEPS_IT, terms=terms)
        return prompt

    def test_the_enforcement_block_comes_first(self):
        prompt = self.prompt()

        self.assertTrue(prompt.startswith("GLOSSARY ENFORCEMENT (HIGHEST PRIORITY)"))
        self.assertLess(prompt.index("GLOSSARY ENFORCEMENT"),
                        prompt.index("You are an expert Odia"))

    def test_it_says_what_the_glossary_outranks(self):
        prompt = self.prompt()

        self.assertIn("outranks translation, fluency, localisation, style, "
                      "paraphrasing and synonym choice", prompt)

    def test_the_terms_themselves_are_listed(self):
        prompt = self.prompt(terms=(TILLER, IRRIGATION))

        self.assertIn(f"  - {TILLER}", prompt)
        # Only terms present in the text are listed.
        self.assertNotIn(f"  - {IRRIGATION}", prompt)

    def test_it_forbids_the_three_ways_a_term_gets_lost(self):
        prompt = self.prompt()

        self.assertIn("Do NOT replace one with a synonym", prompt)
        self.assertIn("more natural expression", prompt)
        self.assertIn("Do NOT infer a different meaning", prompt)

    def test_it_says_to_rebuild_the_sentence_instead(self):
        prompt = self.prompt()

        self.assertIn("Improve only the sentence structure AROUND them", prompt)
        self.assertIn("rebuild the rest of the sentence — never the term", prompt)

    def test_the_simplify_rule_is_subordinated_to_the_glossary(self):
        # Requirement 14 (ପାଣି not ଜଳ) would otherwise strip ଜଳ out of
        # ଜଳସେଚନ — and 83 glossary values contain ଜଳ.
        prompt = self.prompt()

        self.assertIn("THIS RULE STOPS AT A GLOSSARY TERM", prompt)
        self.assertIn("ଜଳସେଚନ is the approved term for irrigation", prompt)

    def test_the_examples_show_the_term_staying_put(self):
        prompt = self.prompt()

        self.assertIn("rebuild the sentence, never the approved term", prompt)
        self.assertIn("ପିଲ is the approved term for tiller", prompt)
        # And the violation is shown as WRONG, not as the goal.
        self.assertIn("WRONG: କୁଶି ବିକାଶ", prompt)

    def test_no_block_when_the_glossary_forced_nothing(self):
        prompt = self.prompt(terms=())

        self.assertNotIn("GLOSSARY ENFORCEMENT", prompt)
        self.assertTrue(prompt.startswith("You are an expert Odia"))


class TheForcedTermsReachThePassTests(SimpleTestCase):
    """translate() has to hand them over — the pass cannot guess them."""

    def test_the_english_to_odia_path_passes_the_forced_terms(self):
        seen = {}

        def fluency(text, source_en="", glossary_terms=()):
            seen["terms"] = list(glossary_terms)
            return text

        with patch.object(pipeline, "_force_glossary_terms",
                          return_value=("ZQX0XQZ development is important.",
                                        {"ZQX0XQZ": TILLER})), \
                patch.object(pipeline, "_google_translate",
                             side_effect=lambda t, s, d: f"{TILLER} ବିକାଶ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ।"), \
                patch.object(pipeline, "_fix_odia_fluency", side_effect=fluency), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            pipeline.translate("Tiller development is important.", "English", "Odia")

        self.assertEqual([TILLER], seen.get("terms"))


class TheOdiaToEnglishDirectionTests(SimpleTestCase):
    """Already enforced; asserted so it stays that way.

    Uses a temporary glossary holding the spec's own example mapping, so the
    test does not depend on what the deployed glossary happens to contain.
    """

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        self.real = g._GLOSSARY_PATH
        tmp = Path(tempfile.mkdtemp()) / "odia_glossary.json"
        tmp.write_text(json.dumps({"tiller": TILLER, "tillers": TILLER,
                                   "rice": "ଧାନ"}, ensure_ascii=False),
                       encoding="utf-8")
        g._GLOSSARY_PATH = tmp
        g.reload_glossary()
        self.addCleanup(self._restore)

    def _restore(self):
        g._GLOSSARY_PATH = self.real
        g.reload_glossary()

    def test_the_approved_english_term_is_forced_into_the_question(self):
        marked, applied = pipeline._force_reverse_glossary_terms("ଧାନରେ ପିଲ କମ ହେଉଛି")

        self.assertIn("tiller", marked)
        self.assertIn("tiller", applied.values())

    def test_a_translation_that_says_shoots_instead_is_rejected(self):
        _marked, applied = pipeline._force_reverse_glossary_terms("ଧାନରେ ପିଲ କମ ହେଉଛି")

        missing = pipeline._missing_glossary_terms(
            "The rice crop is producing fewer shoots.", applied)

        self.assertIn("tiller", missing)

    def test_the_approved_translation_passes(self):
        _marked, applied = pipeline._force_reverse_glossary_terms("ଧାନରେ ପିଲ କମ ହେଉଛି")

        missing = pipeline._missing_glossary_terms(
            "The rice crop is producing fewer tillers.", applied)

        self.assertEqual([], missing)
