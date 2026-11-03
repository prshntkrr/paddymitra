"""The last gate: broken Odia does not leave the pipeline unchallenged.

Asking the model for natural Odia is a request. This is the check — the faults
that can be detected mechanically ARE detected, and an answer still carrying one
goes back to the model once with the fault named. Three kinds, all heard in real
answers:

    broken verb phrase   ପ୍ରୟୋଗ କରିବାରେ, ପ୍ରୟୋଗ କରିବା କରନ୍ତୁ, ଦେଖୁପାରିବା
    duplicated word      ଆରମ୍ଭ ଆରମ୍ଭ ହୁଏ
    a symbol read aloud  ୟୁରିଆ + ଏମଓପି, DAP/MOP

The exemptions matter as much as the faults: 20 kg/ha is a dose, 20+10 is
arithmetic and K2O is a formula. Rewriting those would corrupt the advice, so
they are asserted clean.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.pipeline import (_bare_symbols, _naturalisation_faults,
                                       _fix_odia_fluency)

DRAFT = "ୟୁରିଆ + ଏମଓପି ପ୍ରୟୋଗ କରିବାରେ 20 କିଲୋ ଦିଅନ୍ତୁ।"
CLEAN = "20 କିଲୋ ୟୁରିଆ ଓ ଏମଓପି ପ୍ରୟୋଗ କରନ୍ତୁ।"


class TheFaultsAreDetectedTests(SimpleTestCase):
    """Each named construction is found."""

    def test_every_banned_phrase_is_caught(self):
        for phrase in ("ପ୍ରୟୋଗ କରିବାରେ", "ପ୍ରୟୋଗ କରିବା କରନ୍ତୁ",
                       "ପ୍ରୟୋଗ କରିବା ପିଲ ସାର", "ଦେଖୁପାରିବା", "ଦେଖୁବାକୁ"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, _naturalisation_faults(f"ସାର {phrase} ଦିଅନ୍ତୁ।"))

    def test_the_textbook_hedges_are_faults_too(self):
        for hedge in ("କରିବା ଉଚିତ", "କରିବା ଆବଶ୍ୟକ", "କରିବା ଦରକାର"):
            with self.subTest(hedge=hedge):
                self.assertTrue(_naturalisation_faults(f"ସାର ଦେବା {hedge}।"))

    def test_a_duplicated_word_is_caught(self):
        faults = _naturalisation_faults("ପିଲ ଆରମ୍ଭ ଆରମ୍ଭ ହୁଏ।")

        self.assertIn("duplicated word: ଆରମ୍ଭ", faults)

    def test_deliberate_reduplication_is_not_a_fault(self):
        # Odia really does say these; flagging them would send a correct
        # sentence back to be "fixed".
        for good in ("ଧୀରେ ଧୀରେ ପାଣି ଦିଅନ୍ତୁ।", "ଟିକେ ଟିକେ ସାର ଦିଅନ୍ତୁ।"):
            with self.subTest(text=good):
                self.assertEqual([], _naturalisation_faults(good))

    def test_a_clean_sentence_has_no_faults(self):
        self.assertEqual([], _naturalisation_faults(CLEAN))


class GappedConstructionTests(SimpleTestCase):
    """Broken patterns with words in the middle — and the good ones they resemble.

    "X କରିବା ପାଇଁ Y କରନ୍ତୁ" was reported as an unacceptable pattern, but as a
    general shape it is correct, idiomatic Odia — it is the construction the
    glossary rule in this very module recommends. Only the degenerate form,
    where the SAME verb appears on both sides, is broken. Banning the general
    shape would send correct sentences back to be "fixed", so both halves are
    asserted here: the broken one is caught, the good ones stay clean.
    """

    def test_the_same_verb_twice_across_a_purpose_clause_is_a_fault(self):
        faults = _naturalisation_faults("ସାର ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରୟୋଗ କରନ୍ତୁ।")

        self.assertTrue(faults)
        self.assertIn("same verb", faults[0])

    def test_a_purpose_clause_with_two_different_verbs_is_correct(self):
        for good in ("ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ ମାନକୋଜେବ ସ୍ପ୍ରେ କରନ୍ତୁ।",
                     "ଅମଳ ବଢ଼ାଇବା ପାଇଁ ସମୟରେ ସାର ପ୍ରୟୋଗ କରନ୍ତୁ।",
                     "ପୋକ ମାରିବା ପାଇଁ ଔଷଧ ବ୍ୟବହାର କରନ୍ତୁ।"):
            with self.subTest(text=good):
                self.assertEqual([], _naturalisation_faults(good))

    def test_the_karibare_construction_is_a_fault(self):
        self.assertTrue(_naturalisation_faults("ସାର ପ୍ରୟୋଗ କରିବାରେ ଧ୍ୟାନ ଦିଅନ୍ତୁ କରନ୍ତୁ।"))

    def test_hebaa_paain_hue_is_a_fault(self):
        faults = _naturalisation_faults("ଅମଳ ଭଲ ହେବା ପାଇଁ ହୁଏ।")

        self.assertTrue(faults)

    def test_the_pattern_does_not_reach_across_a_sentence_boundary(self):
        # Two sound sentences, each with its own verb, are not one broken one.
        text = "ସାର ପ୍ରୟୋଗ କରନ୍ତୁ। ପାଣି ଦେବା ପାଇଁ ନାଳ ଖୋଲନ୍ତୁ।"

        self.assertEqual([], _naturalisation_faults(text))


class SymbolsTests(SimpleTestCase):
    """A symbol in a sentence is a fault; a symbol in a dose is not."""

    def test_a_bare_plus_is_a_fault(self):
        self.assertTrue(_bare_symbols("ୟୁରିଆ + ଏମଓପି ଦିଅନ୍ତୁ।"))

    def test_a_bare_slash_between_products_is_a_fault(self):
        self.assertTrue(_bare_symbols("DAP/MOP ପ୍ରୟୋଗ କରନ୍ତୁ।"))

    def test_an_ampersand_is_a_fault(self):
        self.assertTrue(_bare_symbols("ଜିଙ୍କ & ୟୁରିଆ ମିଶାନ୍ତୁ।"))

    def test_dosage_notation_is_left_alone(self):
        for dose in ("20 kg/ha ଦିଅନ୍ତୁ।", "2 ml/l ସ୍ପ୍ରେ କରନ୍ତୁ।",
                     "5 g/l ମିଶାନ୍ତୁ।", "40 କିଲୋ/ଏକର ଦିଅନ୍ତୁ।"):
            with self.subTest(dose=dose):
                self.assertEqual([], _bare_symbols(dose))

    def test_arithmetic_between_numbers_is_left_alone(self):
        for sums in ("20+10 କିଲୋ ମିଶାନ୍ତୁ।", "3/4 ଭାଗ ଦିଅନ୍ତୁ।"):
            with self.subTest(text=sums):
                self.assertEqual([], _bare_symbols(sums))

    def test_a_chemical_formula_is_left_alone(self):
        self.assertEqual([], _bare_symbols("K2O ଏବଂ P2O5 ପ୍ରୟୋଗ କରନ୍ତୁ।"))

    def test_the_odia_conjunctions_are_not_faults(self):
        self.assertEqual([], _bare_symbols("ୟୁରିଆ ଓ ଏମଓପି, ଜିଙ୍କ ସହିତ ୟୁରିଆ।"))


def run(replies, draft=DRAFT, source_en="Apply 20 kg urea and MOP.", terms=()):
    """Run the pass, handing out `replies` in order. Returns (out, prompts)."""
    prompts = []

    def gemini(prompt, text):
        prompts.append(prompt)
        return replies[min(len(prompts) - 1, len(replies) - 1)]

    with patch.object(pipeline, "_gemini_text", gemini):
        out = _fix_odia_fluency(draft, source_en, glossary_terms=terms)
    return out, prompts


class TheRetryTests(SimpleTestCase):
    """One more attempt, with the faults named."""

    def test_a_faulty_rewrite_is_sent_back_once(self):
        _out, prompts = run([DRAFT, CLEAN])

        self.assertEqual(2, len(prompts))

    def test_the_retry_names_the_actual_faults(self):
        _out, prompts = run([DRAFT, CLEAN])

        self.assertIn("YOUR PREVIOUS ANSWER STILL CONTAINED THESE FAULTS", prompts[1])
        self.assertIn("ପ୍ରୟୋଗ କରିବାରେ", prompts[1])
        self.assertIn("symbol:", prompts[1])

    def test_the_retry_forbids_changing_anything_else(self):
        _out, prompts = run([DRAFT, CLEAN])

        self.assertIn("Change nothing else", prompts[1])
        self.assertIn("approved glossary term stays exactly as it is", prompts[1])

    def test_a_cleaner_retry_is_used(self):
        out, _prompts = run([DRAFT, CLEAN])

        self.assertEqual(CLEAN, out)

    def test_a_retry_that_is_no_cleaner_is_not_used(self):
        out, prompts = run([DRAFT, DRAFT])

        self.assertEqual(2, len(prompts))
        self.assertEqual(DRAFT, out)

    def test_a_clean_first_answer_is_not_retried(self):
        out, prompts = run([CLEAN])

        self.assertEqual(1, len(prompts))
        self.assertEqual(CLEAN, out)

    def test_a_retry_that_changed_a_number_is_refused(self):
        out, _prompts = run([DRAFT, "25 କିଲୋ ୟୁରିଆ ଓ ଏମଓପି ପ୍ରୟୋଗ କରନ୍ତୁ।"])

        self.assertEqual(DRAFT, out)

    def test_a_retry_that_dropped_a_glossary_term_is_refused(self):
        draft = "ପିଲ ଅବସ୍ଥାରେ ୟୁରିଆ + ଏମଓପି ପ୍ରୟୋଗ କରିବାରେ ଧ୍ୟାନ ଦିଅନ୍ତୁ।"
        out, _prompts = run([draft, "କୁଶି ଅବସ୍ଥାରେ ୟୁରିଆ ଓ ଏମଓପି ପ୍ରୟୋଗ କରନ୍ତୁ।"],
                            draft=draft, terms=["ପିଲ"])

        self.assertEqual(draft, out)


class ThePromptTests(SimpleTestCase):
    """The rules are stated as well as checked."""

    def prompt(self):
        _out, prompts = run([CLEAN])
        return prompts[0]

    def test_the_banned_constructions_are_listed(self):
        prompt = self.prompt()

        for phrase in ("ପ୍ରୟୋଗ କରିବାରେ", "ପ୍ରୟୋଗ କରିବା କରନ୍ତୁ",
                       "ଆରମ୍ଭ ଆରମ୍ଭ ହୁଏ", "ଦେଖୁପାରିବା", "ଜିଙ୍କ ଦେଖୁବାକୁ"):
            self.assertIn(phrase, prompt)

    def test_the_expected_verbs_are_listed(self):
        prompt = self.prompt()

        for verb in ("ପ୍ରୟୋଗ କରନ୍ତୁ", "ଦିଅନ୍ତୁ", "ବ୍ୟବହାର କରନ୍ତୁ",
                     "ଧ୍ୟାନ ଦିଅନ୍ତୁ", "ଯାଞ୍ଚ କରନ୍ତୁ", "ଅପେକ୍ଷା କରନ୍ତୁ", "ରଖନ୍ତୁ"):
            self.assertIn(verb, prompt)

    def test_the_symbol_conversions_are_shown(self):
        prompt = self.prompt()

        self.assertIn("SYMBOLS ARE NOT WORDS", prompt)
        self.assertIn("ୟୁରିଆ ଓ ଏମଓପି", prompt)
        self.assertIn("DAP ଏବଂ MOP", prompt)
        self.assertIn("ଜିଙ୍କ ସହିତ ୟୁରିଆ", prompt)

    def test_the_symbol_exemptions_are_stated(self):
        prompt = self.prompt()

        self.assertIn("dosage notation", prompt)
        self.assertIn("chemical formula", prompt)

    def test_the_purpose_clause_rule_says_which_form_is_wrong(self):
        # Stated with BOTH sides, so the model does not conclude that every
        # 'କରିବା ପାଇଁ' clause is suspect.
        prompt = self.prompt()

        self.assertIn("Never repeat the SAME verb on both sides of a purpose "
                      "clause", prompt)
        self.assertIn("Two DIFFERENT verbs there are correct and welcome", prompt)
        self.assertIn("ମାନକୋଜେବ ସ୍ପ୍ରେ କରନ୍ତୁ", prompt)

    def test_natural_reduplication_is_protected_in_the_prompt(self):
        prompt = self.prompt()

        for keep in ("ଧୀରେ ଧୀରେ", "ଟିକେ ଟିକେ", "ବାରମ୍ବାର"):
            self.assertIn(keep, prompt)
        self.assertIn("repetition created BY the translation", prompt)

    def test_the_fluency_review_question_is_asked(self):
        prompt = self.prompt()

        self.assertIn("Would a native Odia speaker naturally say this?", prompt)

    def test_the_glossary_still_outranks_all_of_it(self):
        _out, prompts = run([CLEAN], draft="ପିଲ ଅବସ୍ଥାରେ ସାର ଦିଅନ୍ତୁ।",
                            terms=["ପିଲ"])

        self.assertTrue(prompts[0].startswith("GLOSSARY ENFORCEMENT"))
