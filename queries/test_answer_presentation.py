"""The farmer never hears the machinery.

THE REPORTED FAILURE
--------------------
Answers were coming out as

    CR Dhan 100 - the CONTEXT says: Fit for upland
    Swarna Sub 1 - the CONTEXT says: submergence tolerance up to 2 weeks, lowland

"The CONTEXT says" is a label from this module's own prompt. It was being
copied straight through into text that is translated into Odia and read aloud to
someone standing in a field, where it means nothing at all.

Nothing about WHICH varieties are named changes here, or on what evidence. Only
the wording of the final answer does — and the qualifiers inside that wording
("moderately resistant", "up to 2 weeks") must survive the rewrite untouched,
which is most of what this file checks.
"""
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.pipeline import _strip_internal_wording

# Every phrase that belongs to the machinery. An answer containing any of these
# has leaked, whatever else it got right.
BANNED = ("CONTEXT says", "the CONTEXT", "the context", "context says",
          "source says", "the sources say", "the documents say",
          "according to the context", "based on the context",
          "retrieved", "chunk", "candidate pool", "grounding rules",
          "[its row states")

SHEET = (
    "CR Dhan 100\nFit for upland\n"
    "Sahbhagi Dhan\nSuitable for upland and drought prone areas\n"
)


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks or (SHEET,)]


def assert_clean(case, answer):
    low = answer.lower()
    for phrase in BANNED:
        case.assertNotIn(phrase.lower(), low,
                         f"answer leaked internal wording {phrase!r}: {answer!r}")


class StripInternalWordingTests(SimpleTestCase):
    """The scrub itself: labels off, meaning and qualifiers intact."""

    def test_the_reported_leak_is_removed(self):
        cleaned = _strip_internal_wording(
            "CR Dhan 100 - the CONTEXT says: Fit for upland.")

        assert_clean(self, cleaned)
        self.assertIn("CR Dhan 100", cleaned)
        self.assertIn("Fit for upland", cleaned)

    def test_the_second_reported_leak_keeps_its_duration(self):
        cleaned = _strip_internal_wording(
            "Swarna Sub 1 - the CONTEXT says: submergence tolerance up to "
            "2 weeks, lowland.")

        assert_clean(self, cleaned)
        self.assertIn("Swarna Sub 1", cleaned)
        self.assertIn("up to 2 weeks", cleaned)
        self.assertIn("lowland", cleaned)

    def test_provenance_openers_are_dropped_without_losing_the_sentence(self):
        for opener in ("According to the context, ", "Based on the context, ",
                       "As per the retrieved documents, ",
                       "The context says that "):
            with self.subTest(opener=opener):
                cleaned = _strip_internal_wording(
                    opener + "CR Dhan 310 is suitable for medium land.")

                assert_clean(self, cleaned)
                self.assertIn("CR Dhan 310 is suitable for medium land", cleaned)

    def test_a_refusal_still_says_that_nothing_is_known(self):
        cleaned = _strip_internal_wording(
            "The context does not specify any variety suitable for high land.")

        assert_clean(self, cleaned)
        self.assertIn("does not specify", cleaned)
        self.assertIn("high land", cleaned)

    def test_qualifiers_and_figures_are_never_touched(self):
        original = ("CR Dhan 910 is moderately resistant to blast and tolerant "
                    "of submergence for up to 2 weeks. Apply 40 kg urea per acre.")

        self.assertEqual(original, _strip_internal_wording(original))

    def test_moderately_resistant_is_not_promoted(self):
        cleaned = _strip_internal_wording(
            "The context says that CR Dhan 910 is moderately resistant to blast.")

        self.assertIn("moderately resistant", cleaned)
        self.assertNotIn("is resistant to blast", cleaned)

    def test_tolerant_does_not_become_resistant(self):
        cleaned = _strip_internal_wording(
            "Swarna Sub1 - the CONTEXT says: tolerant of submergence.")

        self.assertIn("tolerant", cleaned)
        self.assertNotIn("resistant", cleaned)

    def test_an_ordinary_farming_sentence_is_left_alone(self):
        # "source" has a farming sense and must survive; only provenance
        # wording is removed.
        for sentence in ("Use a good source of nitrogen such as urea.",
                         "Apply 40 kg urea per acre at tillering.",
                         "Keep only a little water in the field.",
                         "Your field needs a nearby water source."):
            with self.subTest(sentence=sentence):
                self.assertEqual(sentence, _strip_internal_wording(sentence))

    def test_an_echoed_bracket_note_is_removed(self):
        cleaned = _strip_internal_wording(
            "CR Dhan 100  [its row states: Fit for upland] suits your field.")

        assert_clean(self, cleaned)
        self.assertIn("CR Dhan 100", cleaned)
        self.assertIn("Fit for upland", cleaned)

    def test_empty_and_clean_answers_pass_through(self):
        self.assertEqual("", _strip_internal_wording(""))
        self.assertEqual("", _strip_internal_wording(None))
        self.assertEqual("CR Dhan 100 is suitable for upland land.",
                         _strip_internal_wording(
                             "CR Dhan 100 is suitable for upland land."))


class GeneratedAnswerTests(SimpleTestCase):
    """Whatever the model writes, what leaves generate_answer() is clean."""

    def answer_for(self, question, reply, *chunks):
        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: reply):
            return pipeline.generate_answer(question, hits(*chunks))

    def test_a_leaking_model_reply_is_cleaned_before_it_is_returned(self):
        answer = self.answer_for(
            "Which rice variety is suitable for high land?",
            "CR Dhan 100 - the CONTEXT says: Fit for upland. Sahbhagi Dhan - "
            "the CONTEXT says: Suitable for upland and drought prone areas.")

        assert_clean(self, answer)
        self.assertIn("CR Dhan 100", answer)
        self.assertIn("Sahbhagi Dhan", answer)

    def test_the_varieties_and_their_evidence_survive_the_cleaning(self):
        answer = self.answer_for(
            "Which varieties are suitable for high land?",
            "According to the context, CR Dhan 100 is fit for upland and "
            "Sahbhagi Dhan is suitable for upland and drought prone areas.")

        assert_clean(self, answer)
        for name in ("CR Dhan 100", "Sahbhagi Dhan"):
            self.assertIn(name, answer)
        self.assertIn("drought prone", answer)

    def test_an_explicit_single_answer_is_cleaned_the_same_way(self):
        answer = self.answer_for(
            "Recommend only one variety for upland.",
            "CR Dhan 100 - the CONTEXT says: Fit for upland.")

        assert_clean(self, answer)
        self.assertIn("CR Dhan 100", answer)

    def test_a_refusal_is_cleaned_and_still_refuses(self):
        answer = self.answer_for(
            "Which rice variety is suitable for high land?",
            "The context does not specify any variety suitable for high land.",
            "Rice is grown across Odisha in three seasons.")

        assert_clean(self, answer)
        self.assertIn("does not specify", answer)

    def test_a_clean_reply_is_returned_unchanged(self):
        reply = "For upland land, CR Dhan 100 and Sahbhagi Dhan are suitable."
        answer = self.answer_for("Which variety is suitable for high land?", reply)

        self.assertEqual(reply, answer)


class StoredFaqAnswerTests(SimpleTestCase):
    """An approved answer is served verbatim — so it is cleaned on the way out.

    FAQ entries are promoted from queries whose feedback was Good. One promoted
    before this fix holds the leak, and that path never reaches generate_answer:
    without cleaning here, every farmer who asked that question again would keep
    hearing it. Nothing about matching, storage or promotion is touched — only
    the text handed to the farmer.
    """

    def serve(self, answer_en, answer_local="", farmer_language="English"):
        from queries.services import pipeline

        faq = MagicMock(pk=7, answer_en=answer_en, answer_local=answer_local,
                        question_en="which variety for high land",
                        source_query_id=None)
        query = MagicMock(pk=1)
        query.ai_response_voice_file = MagicMock()
        with patch.object(pipeline, "faq_record_hit"), \
                patch.object(pipeline, "translate", side_effect=lambda t, *a: t):
            pipeline._answer_from_faq(
                query, faq, 0.95, {"steps": []}, 0.0,
                farmer_language=farmer_language, generate_tts=False,
                hook=None, whatsapp_number="")
        return query

    def test_a_stored_answer_that_leaks_is_cleaned_before_serving(self):
        served = self.serve("CR Dhan 100 - the CONTEXT says: Fit for upland.")

        assert_clean(self, served.ai_response_text)
        self.assertIn("CR Dhan 100", served.ai_response_text)
        self.assertIn("Fit for upland", served.ai_response_text)

    def test_the_stored_local_answer_is_cleaned_too(self):
        # A non-English asker gets the entry's OWN stored local answer, which is
        # served without re-translation so the approved wording cannot drift.
        # That copy has to be cleaned as well.
        served = self.serve("clean english",
                            "Swarna Sub 1 - the CONTEXT says: lowland.",
                            farmer_language="Odia")

        assert_clean(self, served.ai_response_text_local)
        self.assertIn("Swarna Sub 1", served.ai_response_text_local)

    def test_the_cleaned_english_is_what_gets_translated(self):
        served = self.serve("CR Dhan 100 - the CONTEXT says: Fit for upland.",
                            farmer_language="Odia")

        assert_clean(self, served.ai_response_text_local)

    def test_an_approved_answer_that_is_already_clean_is_untouched(self):
        approved = "Apply 40 kg urea per acre when the tillers come out."
        served = self.serve(approved)

        self.assertEqual(approved, served.ai_response_text)


class PromptWordingTests(SimpleTestCase):
    """The prompt asks for natural wording, and says the note is not for saying."""

    def prompt_for(self, question, *chunks):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(question, hits(*chunks))
        return seen["sys"]

    def test_the_candidate_note_is_bracketed_and_marked_as_internal(self):
        prompt = self.prompt_for("Which varieties are suitable for high land?")

        # The note carries the conditions the candidate's own row states — the
        # canonical facets, unchanged by this work — and is now bracketed and
        # addressed to the model rather than phrased as something to repeat.
        self.assertIn("CR Dhan 100  [its row states: upland]", prompt)
        self.assertIn("Sahbhagi Dhan  [its row states: upland, drought]", prompt)
        self.assertIn("FOR YOU, not for the farmer", prompt)
        self.assertIn("NEVER copy the brackets", prompt)
        self.assertNotIn("the CONTEXT says:", prompt)

    def test_the_prompt_forbids_naming_the_machinery(self):
        prompt = self.prompt_for("Which varieties are suitable for high land?")

        self.assertIn("NEVER NAME THE MACHINERY", prompt)

    def test_the_prompt_asks_for_the_qualifiers_to_be_kept(self):
        prompt = self.prompt_for("Which varieties are suitable for high land?")

        self.assertIn("moderately resistant", prompt)
        self.assertIn("up to 2 weeks", prompt)

    def test_the_single_option_note_carries_the_same_instruction(self):
        prompt = self.prompt_for("Recommend only one variety for high land.")

        self.assertIn("FOR YOU, not for the farmer", prompt)
        self.assertIn("The farmer asked for ONE option", prompt)
