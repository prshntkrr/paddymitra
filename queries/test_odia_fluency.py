"""The Odia answer has to sound like a person, not like a dictionary.

Google Translate is the base for English -> Odia, and it maps word to word:
"Apply 20 kg urea to control this disease" came back as
"... ସାହାଯ୍ୟ କରିବା ପାଇଁ ନିୟନ୍ତ୍ରଣ ଏହି ରୋଗ ପାଇଁ" — English word order wearing Odia
letters. _fix_odia_fluency used to PROOF-READ that draft ("rewrite only what is
needed", "a sentence that already reads naturally must be returned UNCHANGED"),
which is why the answers stayed literal: a stiff-but-parseable sentence was
left alone. It now re-says the whole thing in spoken Odia.

What must NOT move is the facts, so the pass is bounded: a rewrite that changes
any number is thrown away and the literal draft kept, because a wrong dose read
aloud to a farmer is worse than clumsy word order.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline

# The literal Google draft, and what a person would actually say.
LITERAL = "ପ୍ରୟୋଗ କରନ୍ତୁ 20 କିଲୋ ୟୁରିଆ ନିୟନ୍ତ୍ରଣ ଏହି ରୋଗ ପାଇଁ।"
SPOKEN = "ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ 20 କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।"
SOURCE_EN = "Apply 20 kg urea to control this disease."


def rewrite(draft, reply, source_en=SOURCE_EN):
    """Run the pass with the model's answer stubbed. Returns (output, prompt)."""
    seen = {}

    def gemini(prompt, text):
        seen["prompt"] = prompt
        return reply

    with patch.object(pipeline, "_gemini_text", gemini):
        out = pipeline._fix_odia_fluency(draft, source_en)
    return out, seen.get("prompt", "")


class TheRewriteIsAcceptedTests(SimpleTestCase):
    """A fluent re-saying of the same facts replaces the literal draft."""

    def test_the_spoken_version_is_returned(self):
        out, _prompt = rewrite(LITERAL, SPOKEN)

        self.assertEqual(SPOKEN, out)

    def test_a_longer_rewrite_is_allowed(self):
        # Sentence boundaries may move; the old prompt discouraged this.
        two = ("ଏହି ରୋଗ ଦେଖାଦେଲେ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ। "
               "20 କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।")
        out, _prompt = rewrite(LITERAL, two)

        self.assertEqual(two, out)

    def test_odia_numerals_are_not_treated_as_a_changed_number(self):
        # The guard compares numbers across scripts, so ୨୦ still counts as 20.
        odia_digits = "ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ ୨୦ କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।"
        out, _prompt = rewrite(LITERAL, odia_digits)

        self.assertEqual(odia_digits, out)


class TheFactsAreFixedTests(SimpleTestCase):
    """Fluency never buys itself a changed fact."""

    def test_a_changed_dose_is_rejected_and_the_draft_kept(self):
        wrong = "ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ 25 କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।"
        out, _prompt = rewrite(LITERAL, wrong)

        self.assertEqual(LITERAL, out)

    def test_a_dropped_number_is_rejected(self):
        out, _prompt = rewrite(LITERAL, "ଏହି ରୋଗକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।")

        self.assertEqual(LITERAL, out)

    def test_an_empty_gemini_reply_falls_through_to_openai(self):
        # Pre-existing design: a blank primary reply is not an answer, so the
        # second engine is asked rather than the literal draft being kept.
        with patch.object(pipeline, "_gemini_text", side_effect=lambda p, t: "  "), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: SPOKEN) as chat:
            out = pipeline._fix_odia_fluency(LITERAL, SOURCE_EN)

        chat.assert_called()
        self.assertEqual(SPOKEN, out)

    def test_a_blank_reply_from_both_engines_keeps_the_draft(self):
        with patch.object(pipeline, "_gemini_text", side_effect=lambda p, t: "  "), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "  "):
            out = pipeline._fix_odia_fluency(LITERAL, SOURCE_EN)

        self.assertEqual(LITERAL, out)

    def test_a_model_failure_keeps_the_draft(self):
        def boom(prompt, text):
            raise RuntimeError("gemini down")

        with patch.object(pipeline, "_gemini_text", boom), \
                patch.object(pipeline, "_chat", side_effect=RuntimeError("openai down")):
            out = pipeline._fix_odia_fluency(LITERAL, SOURCE_EN)

        self.assertEqual(LITERAL, out)

    def test_empty_input_is_returned_untouched(self):
        self.assertEqual("", pipeline._fix_odia_fluency("", SOURCE_EN))


class ThePromptTests(SimpleTestCase):
    """It asks for a re-saying, and no longer for a proof-read."""

    def prompt(self):
        _out, prompt = rewrite(LITERAL, SPOKEN)
        return prompt

    def test_it_asks_for_a_rewrite_not_a_translation(self):
        prompt = self.prompt()

        self.assertIn("NOT to translate the text below word-for-word", prompt)
        self.assertIn("conversational, farmer-friendly Odia", prompt)
        self.assertIn("originally written in Odia", prompt)

    def test_it_carries_the_worked_examples(self):
        # The bad/good pairs are what teach the register: କୁଶି not ପିଲ,
        # ମାଟି ଭିଜା not ସଂତୃପ୍ତ, ପାଣି ଦିଅନ୍ତୁ not ଜଳସେଚନ କରନ୍ତୁ.
        prompt = self.prompt()

        self.assertIn("କୁଶି ବିକାଶ ସମୟରେ ମାଟି ଭିଜା", prompt)
        self.assertIn("ପୁଣି ଥରେ ପାଣି ଦିଅନ୍ତୁ", prompt)

    def test_it_names_the_simpler_word_for_each_sanskritised_one(self):
        prompt = self.prompt()

        for simple, sanskritised in (("ପାଣି", "ଜଳ"), ("ଜମି", "କ୍ଷେତ୍ର"),
                                     ("ମାଟି", "ମୃତ୍ତିକା"), ("ସାର", "ଉର୍ବରକ")):
            self.assertIn(f"{simple} not {sanskritised}", prompt)

    def test_it_asks_for_the_extension_officers_register(self):
        prompt = self.prompt()

        self.assertIn("agricultural extension officer", prompt)
        self.assertIn("ଆପଣ", prompt)
        self.assertIn("NEVER copy English sentence structure", prompt)

    def test_it_asks_the_model_to_check_itself_before_answering(self):
        prompt = self.prompt()

        self.assertIn("does this sound natural to a native Odia speaker", prompt)
        self.assertIn("Would a farmer immediately understand it", prompt)

    def test_it_asks_for_odia_only_with_no_commentary(self):
        prompt = self.prompt()

        self.assertIn("Output ONLY the final Odia response", prompt)
        self.assertIn("Do not explain your translation choices", prompt)

    def test_the_final_check_is_the_last_instruction_before_the_output_rule(self):
        # Recency matters in a prompt this long: the self-check has to be the
        # thing the model reads immediately before it writes, not a rule buried
        # in the middle of fifteen others.
        prompt = self.prompt()

        self.assertIn("FINAL CHECK", prompt)
        self.assertIn("verify that every sentence sounds natural to a native "
                      "Odia-speaking farmer", prompt)
        self.assertIn("as if written originally in Odia", prompt)
        self.assertGreater(prompt.index("FINAL CHECK"), prompt.index("15. Before"))

    def test_the_final_check_names_what_must_be_rewritten(self):
        prompt = self.prompt()

        for fault in ("literal translation", "unnatural phrase", "English word",
                      "grammatically awkward construction"):
            self.assertIn(fault, prompt)

    def test_the_final_check_does_not_licence_translating_product_names(self):
        # "Rewrite any English word" is the last thing read, so the exemption
        # has to travel with it or DAP starts coming back as Odia.
        prompt = self.prompt()
        final = prompt[prompt.index("FINAL CHECK"):]

        self.assertIn("they stay exactly as written", final)

    def test_the_proof_read_framing_is_gone(self):
        # These two lines are why the answers stayed literal.
        prompt = self.prompt()

        self.assertNotIn("Rewrite ONLY what is needed", prompt)
        self.assertNotIn("must be returned UNCHANGED", prompt)

    def test_the_facts_are_still_constrained_in_the_prompt(self):
        # Requirement 5 — a fluent rewrite must not buy itself a changed fact.
        prompt = self.prompt()

        self.assertIn("Preserve the original meaning", prompt)
        self.assertIn("Add no advice, drop no advice, invent no detail", prompt)
        self.assertIn("EXACTLY as", prompt)
        self.assertIn("DAP", prompt)
        self.assertIn("variety name", prompt)

    def test_product_names_are_exempt_from_the_simplify_jargon_rule(self):
        # "Avoid technical jargon" must not be read as licence to translate DAP.
        prompt = self.prompt()

        self.assertIn("not jargon to be simplified", prompt)

    def test_the_english_source_is_given_as_the_authority(self):
        prompt = self.prompt()

        self.assertIn("AUTHORITY", prompt)
        self.assertIn(SOURCE_EN, prompt)

    def test_the_textbook_hedges_are_still_banned(self):
        prompt = self.prompt()

        for hedge in ("କରିବା ଉଚିତ", "କରିବା ଆବଶ୍ୟକ", "କରିବା ଦରକାର"):
            self.assertIn(hedge, prompt)

    def test_the_bookish_words_are_named(self):
        prompt = self.prompt()

        for spoken, bookish in (("ପାଣି", "ଜଳ"), ("ଜମି", "କ୍ଷେତ୍ର"), ("ମାଟି", "ମୃତ୍ତିକା")):
            self.assertIn(spoken, prompt)
            self.assertIn(bookish, prompt)


class TheAnswerPathTests(SimpleTestCase):
    """The pass is actually reached on the English -> Odia answer path."""

    def test_translate_runs_the_fluency_pass_on_the_google_draft(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: LITERAL), \
                patch.object(pipeline, "_fix_odia_fluency",
                             side_effect=lambda text, src="": SPOKEN) as fluency, \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda text, src="": text), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: SPOKEN):
            out = pipeline.translate(SOURCE_EN, "English", "Odia")

        fluency.assert_called()
        # The English source is handed over, so the rewrite has the meaning to
        # work from rather than only the literal draft.
        self.assertEqual(SOURCE_EN, fluency.call_args[0][1])
        self.assertIn("ୟୁରିଆ", out)

    def test_the_odia_to_english_direction_never_calls_it(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "How much urea?"), \
                patch.object(pipeline, "_fix_odia_fluency") as fluency, \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "How much urea?"):
            pipeline.translate("କେତେ ୟୁରିଆ ଦେବି?", "Odia", "English")

        fluency.assert_not_called()
