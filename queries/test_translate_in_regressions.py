"""A spoken range became a sum, and a pest question became a weather question.

THE PRODUCTION CASE
-------------------
    ରୋଗ ବାର ଦଶ ପନ୍ଦର ଦିନ ହେଇଗଲାଣି। … ଆମର ରୋଗପୋକ କଣ ଏବେ ବାତ୍ୟା ଚାଲିଛି …

    log:  odia_numbers: converted 'ବାର ଦଶ ପନ୍ଦର'->37
    out:  "The disease has been going on for 37 days. … we will face a cyclone,
           what can we do to prevent it?"

Two independent faults:

  37   — _evaluate() adds a run of cardinals, which is right for a composite
         (ଏକ ଶହ କୋଡ଼ିଏ = 120) but wrong for a farmer speaking approximately.
         12 + 10 + 15 = 37, and the farmer was told their disease had lasted
         37 days.

  pest — ରୋଗପୋକ had no entry in the glossary, the reverse table, or anywhere
         else. With ବାତ୍ୟା (cyclone) in the same sentence the translator
         guessed, and the pest half of the question disappeared into the storm.

Both halves of the sentence are real: it IS about a cyclone AND about pests.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.odia_glossary import (_reverse_map, normalize_odia,
                                            relevant_reverse_glossary)
from queries.services.odia_numbers import convert_odia_numbers

PRODUCTION_Q = (
    "ରୋଗ ବାର ଦଶ ପନ୍ଦର ଦିନ ହେଇଗଲାଣି। ତା ଭିତରେ ଆମର ରୋଗପୋକ କଣ ଏବେ ବାତ୍ୟା ଚାଲିଛି, "
    "ବର୍ଷା ବାତ୍ୟା ଚାଲିଛି। ଆଉ ସେଇ ଅନୁସାରେ ଆମକୁ ରୋଗପୋକ ଆସିବ କଣ ରୋଗପୋକର "
    "ପ୍ରତିରୋଧକ କରିବା ପାଇଁ ସେଇଟା କଣ କୁହନ୍ତୁ।"
)
# What Google returns for the protected sentence (captured live).
PRODUCTION_EN = (
    "It has been twelve to fifteen days since the disease. During that time, we "
    "have a cyclone, a rainfall cyclone. And according to that, we will have a "
    "pest and disease, what should we do to control the pest and disease?"
)


class SpokenRangeTests(SimpleTestCase):
    """A run of bare cardinals is a list, not a sum."""

    def test_the_reported_phrase_never_becomes_37(self):
        out, conversions = convert_odia_numbers("ରୋଗ ବାର ଦଶ ପନ୍ଦର ଦିନ ହେଇଗଲାଣି।")

        self.assertNotIn("37", out)
        self.assertEqual([], conversions)

    def test_the_words_survive_for_the_translator_to_read(self):
        out, _ = convert_odia_numbers("ବାର ଦଶ ପନ୍ଦର ଦିନ")

        for word in ("ବାର", "ଦଶ", "ପନ୍ଦର"):
            self.assertIn(normalize_odia(word), normalize_odia(out))

    def test_two_cardinals_are_a_range_too(self):
        out, conversions = convert_odia_numbers("ଦଶ ପନ୍ଦର ଦିନ")

        self.assertEqual([], conversions)
        self.assertNotIn("25", out)

    def test_a_single_cardinal_still_converts(self):
        _out, conversions = convert_odia_numbers("ଗୋଟେ ଏକରରେ")

        # The reported phrase comes back NFD-normalised, so compare in that form.
        self.assertEqual(1, len(conversions))
        self.assertEqual(normalize_odia("ଗୋଟେ"), normalize_odia(conversions[0][0]))
        self.assertEqual("1", conversions[0][1])

    def test_a_composite_with_a_multiplier_still_converts(self):
        # The additive rule is correct HERE — a multiplier joins the parts.
        for phrase, expected in (("ସାତ ହଜାର ଏକଷଠି", "7061"),
                                 ("ହଜାରେ ନଅ", "1009"),
                                 ("ଏକ ଶହ କୋଡ଼ିଏ", "120")):
            with self.subTest(phrase=phrase):
                out, conversions = convert_odia_numbers(phrase)

                self.assertEqual(expected, conversions[0][1], out)

    def test_a_fraction_still_converts(self):
        for phrase, expected in (("ଅଢ଼େଇ", "2.5"), ("ସାଢ଼େ ତିନି", "3.5")):
            with self.subTest(phrase=phrase):
                _out, conversions = convert_odia_numbers(f"{phrase} ଏକରରେ")

                self.assertEqual(expected, conversions[0][1])

    def test_a_cardinal_with_a_modifier_still_converts(self):
        _out, conversions = convert_odia_numbers("ସାଢ଼େ ତିନି ଏକର")

        self.assertEqual("3.5", conversions[0][1])


class PestTerminologyTests(SimpleTestCase):
    """ରୋଗପୋକ has a meaning now, and it is not the weather."""

    def test_the_compound_resolves_to_pest_and_disease(self):
        self.assertEqual("pest and disease",
                         _reverse_map()[normalize_odia("ରୋଗପୋକ")])

    def test_the_compound_is_matched_whole_not_as_its_parts(self):
        found = {normalize_odia(k): v
                 for k, v in relevant_reverse_glossary("ଆମର ରୋଗପୋକ କଣ").items()}

        self.assertEqual("pest and disease", found[normalize_odia("ରୋଗପୋକ")])
        self.assertNotIn(normalize_odia("ରୋଗ"), found)
        self.assertNotIn(normalize_odia("ପୋକ"), found)

    def test_the_inflected_form_resolves_too(self):
        found = {normalize_odia(k): v
                 for k, v in relevant_reverse_glossary("ରୋଗପୋକର ପ୍ରତିରୋଧକ").items()}

        self.assertIn("pest and disease", found.values())

    def test_pest_is_never_a_weather_word(self):
        for odia in ("ରୋଗପୋକ", "ରୋଗ", "ପୋକ"):
            with self.subTest(odia=odia):
                english = _reverse_map()[normalize_odia(odia)].lower()

                self.assertNotIn("cyclone", english)
                self.assertNotIn("storm", english)
                self.assertNotIn("rain", english)

    def test_the_weather_words_stay_weather(self):
        self.assertEqual("cyclone", _reverse_map()[normalize_odia("ବାତ୍ୟା")])
        self.assertEqual("rainfall", _reverse_map()[normalize_odia("ବର୍ଷା")])

    def test_weather_is_never_mapped_to_pest(self):
        for odia in ("ବାତ୍ୟା", "ବର୍ଷା"):
            with self.subTest(odia=odia):
                english = _reverse_map()[normalize_odia(odia)].lower()

                self.assertNotIn("pest", english)
                self.assertNotIn("disease", english)


class ProductionSentenceTests(SimpleTestCase):
    """Both halves of the real question survive to the English."""

    def translate(self, reply=PRODUCTION_EN):
        sent = {}

        def google(text, source, target):
            sent["input"] = text
            return reply

        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate(PRODUCTION_Q, "Odia", "English")
        return out, sent.get("input", ""), chat

    def test_the_pest_half_and_the_weather_half_both_survive(self):
        out, _sent, _chat = self.translate()
        low = out.lower()

        self.assertIn("pest", low)
        self.assertIn("disease", low)
        self.assertIn("cyclone", low)
        self.assertIn("rainfall", low)

    def test_the_answer_is_not_about_37_days(self):
        out, sent, _chat = self.translate()

        self.assertNotIn("37", sent)
        self.assertNotIn("37", out)

    def test_google_is_given_both_concepts_separately(self):
        _out, sent, _chat = self.translate()

        self.assertIn("pest and disease", sent)
        self.assertIn("cyclone", sent)
        self.assertIn("rainfall", sent)

    def test_the_spoken_range_reaches_google_unconverted(self):
        _out, sent, _chat = self.translate()

        for word in ("ବାର", "ଦଶ", "ପନ୍ଦର"):
            self.assertIn(normalize_odia(word), normalize_odia(sent))

    def test_google_answers_and_openai_is_not_needed(self):
        _out, _sent, chat = self.translate()

        chat.assert_not_called()

    def test_a_reply_that_loses_the_pest_term_falls_back(self):
        # The original production output: the pest half turned into weather.
        losing = ("The disease has been going on for 37 days. During that time, "
                  "we have been facing a cyclone, a rainstorm.")
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: losing), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: "fallback") as chat:
            out = pipeline.translate(PRODUCTION_Q, "Odia", "English")

        chat.assert_called()
        self.assertEqual("fallback", out)


class PreviousRegressionsStillPassTests(SimpleTestCase):
    """Yesterday's two cases must not have moved."""

    def test_the_transplanting_ploughing_question(self):
        reply = "How many days before rice transplanting should ploughing be done?"
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: reply), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate("ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?",
                                     "Odia", "English")

        chat.assert_not_called()
        for term in ("rice", "transplanting", "before", "ploughing"):
            self.assertIn(term, out.lower())
        self.assertNotIn("harvest", out.lower())

    def test_the_dap_question(self):
        reply = "How much DAP to plant in 1 acre?"
        sent = {}

        def google(text, source, target):
            sent["input"] = text
            return reply

        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"):
            out = pipeline.translate("ଡିଏପି ଗୋଟେ ଏକରରେ କେତେ ପକେଇବା?",
                                     "Odia", "English")

        self.assertIn("DAP", sent["input"])
        self.assertIn("1", sent["input"])          # ଗୋଟେ still converts
        self.assertNotIn("dap 1", out.lower())
        self.assertNotIn("harvest", out.lower())


class DebugLoggingTests(SimpleTestCase):
    """Every stage is visible, so the next one of these is read, not deduced."""

    def test_each_stage_is_logged_in_order(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: PRODUCTION_EN), \
                patch.object(pipeline, "_chat"):
            with self.assertLogs("irri", level="INFO") as logs:
                pipeline.translate(PRODUCTION_Q, "Odia", "English")
        written = "\n".join(logs.output)

        for stage in ("1 original odia", "2 after protect_input_terms",
                      "3 after canonicalise", "4 after convert_odia_numbers",
                      "5 glossary matches", "6 protected input to google",
                      "7 raw google output", "8 glossary-restored output",
                      "9 final output"):
            self.assertIn(stage, written)

    def test_the_range_decision_is_logged(self):
        with self.assertLogs("irri", level="INFO") as logs:
            convert_odia_numbers("ବାର ଦଶ ପନ୍ଦର ଦିନ")

        self.assertIn("spoken list/range, not a composite",
                      "\n".join(logs.output))
