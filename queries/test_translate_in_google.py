"""Odia -> English goes through Google first, with the glossary made to bind.

WHY THE ENGINE CHANGED
----------------------
Each engine was good at the half the other got wrong.

  ଡିଏପି ଗୋଟେ ଏକରରେ କେତେ ପକେଇବା?
    OpenAI:  "How much can be harvested per acre in DAP 1?"   <- invented an
             operation, and glued a quantity onto the input's name
    Google:  "How much DAP to plant in 1 acre?"

  ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?
    Google:  "How many days before planting should the rice be plowed?"
             <- "planting" and "plowed"; the glossary says transplanting and
                ploughing, and Google has never heard of this glossary

So Google runs first, with the glossary's ENGLISH substituted into the Odia
before it is sent — Google keeps a Latin word it cannot translate and builds the
sentence around it, which is the same property protect_input_terms() already
relies on for acronyms. The result is accepted only if every substituted term
survived; anything else falls back to OpenAI, whose prompt carries the same
glossary as authoritative constraints.

Every test here stubs the network. The live behaviour of both engines is quoted
above and in the final report, not asserted here.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.odia_glossary import normalize_odia

PLOUGH_Q = "ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?"
DAP_Q = "ଡିଏପି ଗୋଟେ ଏକରରେ କେତେ ପକେଇବା?"
# What Google actually returns for the substituted sentences (captured live).
PLOUGH_EN = "How many days before rice transplanting should ploughing be done?"
DAP_EN = "How much DAP to plant in 1 acre?"


def google_returning(*replies):
    """A stub for _google_translate that records what it was given."""
    calls = {"n": 0, "inputs": []}

    def fake(text, source, target):
        calls["inputs"].append(text)
        calls["n"] += 1
        return replies[min(calls["n"] - 1, len(replies) - 1)]

    fake.calls = calls
    return fake


class SubstitutionTests(SimpleTestCase):
    """What is handed to Google, and what is deliberately left alone."""

    def test_content_terms_are_replaced_by_their_glossary_english(self):
        marked, applied = pipeline._force_reverse_glossary_terms(PLOUGH_Q)

        self.assertIn("rice", marked)
        self.assertIn("transplanting", marked)
        self.assertIn("ploughing", marked)
        # One key may carry two concepts ('transplanting rice'), so assert the
        # concepts rather than the exact set of terms.
        substituted = " ".join(applied.values())
        for concept in ("rice", "transplanting", "ploughing"):
            self.assertIn(concept, substituted)

    def test_a_function_word_is_left_in_odia(self):
        # Substituting ଆଗରୁ -> "before" made Google invert which operation comes
        # first ("before PLOUGHING should rice be TRANSPLANTED"), so function
        # words are hinted to the LLM but never inlined for Google.
        marked, applied = pipeline._force_reverse_glossary_terms(PLOUGH_Q)

        self.assertIn(normalize_odia("ଆଗରୁ"), normalize_odia(marked))
        self.assertNotIn("before", applied.values())

    def test_the_longer_phrase_is_substituted_not_its_parts(self):
        marked, applied = pipeline._force_reverse_glossary_terms("ହଳ କରିବା ଉଚିତ")

        self.assertIn("ploughing", marked)
        self.assertNotIn("tillage", marked)
        self.assertEqual(["ploughing"], list(applied.values()))

    def test_the_other_ploughing_phrasing_substitutes_too(self):
        marked, _applied = pipeline._force_reverse_glossary_terms("ହଳ ଧରିବା ଉଚିତ")

        self.assertIn("ploughing", marked)

    def test_an_inflected_term_is_substituted_with_its_ending(self):
        # ଜିଆ ଖତ + a case ending must still be replaced whole.
        marked, applied = pipeline._force_reverse_glossary_terms("ଜିଆ ଖତରେ କେତେ?")

        self.assertIn("vermicompost", marked)
        self.assertEqual(["vermicompost"], list(applied.values()))

    def test_a_sentence_with_no_glossary_terms_is_untouched(self):
        marked, applied = pipeline._force_reverse_glossary_terms(DAP_Q)

        self.assertEqual({}, applied)
        self.assertEqual(normalize_odia(DAP_Q), normalize_odia(marked))

    def test_missing_terms_are_detected_case_insensitively(self):
        applied = {"ରୋଇବା": "transplanting", "ହଳ କରିବା": "ploughing"}

        self.assertEqual([], pipeline._missing_glossary_terms(
            "Before TRANSPLANTING, do the ploughing.", applied))
        self.assertEqual(["transplanting"], pipeline._missing_glossary_terms(
            "Before planting, do the ploughing.", applied))

    def test_a_prefix_does_not_count_as_the_term(self):
        # "planting" must not satisfy "transplanting".
        self.assertEqual([], pipeline._missing_glossary_terms(
            "transplanting time", {"x": "transplanting"}))
        self.assertEqual(["transplanting"], pipeline._missing_glossary_terms(
            "planting time", {"x": "transplanting"}))


class GooglePrimaryTests(SimpleTestCase):
    """Google answers, and OpenAI is not called."""

    def test_the_ploughing_question_keeps_all_four_meanings(self):
        google = google_returning(PLOUGH_EN)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate(PLOUGH_Q, "Odia", "English")

        chat.assert_not_called()
        for term in ("rice", "transplanting", "before", "ploughing"):
            self.assertIn(term, out.lower())
        for wrong in ("harvest", "cutting"):
            self.assertNotIn(wrong, out.lower())
        # "planting" only as its own word — "transplanting" legitimately
        # contains it.
        import re as _re
        self.assertIsNone(_re.search(r"(?<![a-z])planting", out.lower()))

    def test_the_dap_question_is_about_application_not_harvest(self):
        google = google_returning(DAP_EN)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate(DAP_Q, "Odia", "English")

        chat.assert_not_called()
        self.assertIn("dap", out.lower())
        self.assertIn("acre", out.lower())
        for wrong in ("harvested", "harvesting", "harvest yield", "dap 1"):
            self.assertNotIn(wrong, out.lower())

    def test_google_receives_the_substituted_sentence(self):
        google = google_returning(PLOUGH_EN)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"):
            pipeline.translate(PLOUGH_Q, "Odia", "English")

        sent = google.calls["inputs"][0]
        self.assertIn("transplanting", sent)
        self.assertIn("ploughing", sent)

    def test_several_glossary_terms_survive_together(self):
        reply = "How much vermicompost and micronutrient for lowland?"
        google = google_returning(reply)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate("ଜିଆ ଖତ ଓ ଅଣୁସାର ଖାଲ ଜମି ପାଇଁ କେତେ?",
                                     "Odia", "English")

        chat.assert_not_called()
        for term in ("vermicompost", "micronutrient", "lowland"):
            self.assertIn(term, out.lower())


class NumberConversionTests(SimpleTestCase):
    """convert_odia_numbers() still runs, and before the engine sees the text."""

    def test_a_fraction_word_still_becomes_digits(self):
        google = google_returning("How much for 2.5 acres?")
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"):
            out = pipeline.translate("ଅଢ଼େଇ ଏକରରେ କେତେ ଦେବି?", "Odia", "English")

        self.assertIn("2.5", google.calls["inputs"][0])
        self.assertIn("2.5", out)

    def test_the_quantity_stays_with_its_unit_not_the_input_name(self):
        # ଗୋଟେ belongs to the acre phrase. The engine must be given the digit
        # next to the measurement, and the answer must never read "DAP 1".
        google = google_returning(DAP_EN)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"):
            out = pipeline.translate(DAP_Q, "Odia", "English")

        self.assertIn("1", google.calls["inputs"][0])
        self.assertNotIn("dap 1", out.lower())

    def test_the_acronym_is_still_pinned_before_translation(self):
        google = google_returning(DAP_EN)
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"):
            pipeline.translate(DAP_Q, "Odia", "English")

        self.assertIn("DAP", google.calls["inputs"][0])


class FallbackTests(SimpleTestCase):
    """Google fails or loses a term -> OpenAI, with the glossary still binding."""

    def openai_prompt(self, google, reply="fallback answer"):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return reply

        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat", side_effect=capture):
            out = pipeline.translate(PLOUGH_Q, "Odia", "English")
        return out, seen.get("sys", "")

    def test_a_google_exception_falls_back_to_openai(self):
        def boom(text, source, target):
            raise RuntimeError("google is down")

        out, prompt = self.openai_prompt(boom)

        self.assertEqual("fallback answer", out)
        self.assertIn("AUTHORITATIVE TERMINOLOGY CONSTRAINTS", prompt)

    def test_a_dropped_glossary_term_falls_back_to_openai(self):
        # Google's own plain output: "planting"/"plowed" instead of the
        # glossary's terms. That must not be shipped.
        google = google_returning(
            "How many days before planting should the rice be plowed?")
        out, prompt = self.openai_prompt(google)

        self.assertEqual("fallback answer", out)
        self.assertIn("transplanting", prompt)
        self.assertIn("ploughing", prompt)

    def test_an_empty_google_reply_falls_back_to_openai(self):
        out, prompt = self.openai_prompt(google_returning("   "))

        self.assertEqual("fallback answer", out)
        self.assertIn("AUTHORITATIVE TERMINOLOGY CONSTRAINTS", prompt)

    def test_the_fallback_prompt_still_forbids_swapping_operations(self):
        _out, prompt = self.openai_prompt(google_returning(""))

        self.assertIn("NEVER turn one field operation into another", prompt)
        self.assertIn("Transplanting is not harvesting", prompt)

    def test_both_engines_failing_returns_the_original_text(self):
        # The pre-existing last-resort behaviour, unchanged.
        def boom(text, source, target):
            raise RuntimeError("down")

        with patch.object(pipeline, "_google_translate", boom), \
                patch.object(pipeline, "_chat", side_effect=RuntimeError("down")):
            out = pipeline.translate(PLOUGH_Q, "Odia", "English")

        # The passthrough returns the PREPROCESSED text, which canonicalise()
        # has NFD-normalised; compare in the same encoding.
        self.assertEqual(normalize_odia(PLOUGH_Q), normalize_odia(out))


class LoggingTests(SimpleTestCase):
    """The backend and the glossary decisions are visible in the log."""

    def test_a_successful_google_translation_names_the_backend(self):
        with patch.object(pipeline, "_google_translate", google_returning(PLOUGH_EN)), \
                patch.object(pipeline, "_chat"):
            with self.assertLogs("irri", level="INFO") as logs:
                pipeline.translate(PLOUGH_Q, "Odia", "English")
        written = "\n".join(logs.output)

        self.assertIn("translate_in | backend = google", written)
        self.assertIn("glossary matches", written)
        self.assertIn("protected input", written)

    def test_a_fallback_says_why(self):
        google = google_returning("How many days before planting?")
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            with self.assertLogs("irri", level="WARNING") as logs:
                pipeline.translate(PLOUGH_Q, "Odia", "English")
        written = "\n".join(logs.output)

        self.assertIn("falling back to OpenAI", written)
        self.assertIn("transplanting", written)


class ForwardDirectionUnchangedTests(SimpleTestCase):
    """English -> Odia must not have moved."""

    def test_the_outbound_path_still_uses_google_with_odia_markers(self):
        google = google_returning("ଅନୁବାଦ")
        # The Odia output passes call Gemini; stubbed so the suite stays offline.
        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat"), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, source_en="": t), \
                patch.object(pipeline, "_fix_odia_fluency",
                             side_effect=lambda t, source_en="": t), \
                patch.object(pipeline, "_force_reverse_glossary_terms") as reverse:
            pipeline.translate("Apply urea at tillering.", "English", "Odia")

        reverse.assert_not_called()

    def test_same_language_is_returned_untouched(self):
        self.assertEqual("hello", pipeline.translate("hello", "English", "English"))
