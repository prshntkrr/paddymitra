"""Two ways a NEAR match was treated as an EXACT one, from queries 1029 and 1030.

Both ended the same way: an approved answer returned verbatim, retrieval and
generation skipped, and the farmer told something confident and wrong.

QUERY 1030 — the probe answered a question nobody asked.
    "ଶହେ ଦିନିଆ ଧାନ କିସମ କଣ କଣ ଅଛି?" ("what are the 100-day varieties?") scored
    0.0 against every entry on its own words. duration_probe rewrote it as "For
    short duration rice varieties, how much fertilizer to be applied and when?",
    which matched FAQ-137 at 0.891 and was SERVED. The farmer got a fertiliser
    schedule; the top-ranked chunk — "Sahabhagidhan and CR Dhan 207 are good
    100-day varieties" — was never read, because chunks=0.

QUERY 1029 — the right answer was blocked by a spelling difference.
    "If there is NO urea…" against FAQ-136/142 "If Urea is NOT available…":
    blocked with `polarity ['no'] != ['not']` at 0.881. Two ways of saying the
    same thing read as opposite meanings, so the farmer got a generic answer
    with no doses in it.

The fixes are different in kind and that matters. Gating the probe stops THIS
question being hijacked; forbidding the probe to serve bounds EVERY future
mismatch to "slightly noisy context" instead of "wrong answer with authority".
"""
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from catalogues.models import Catalogue
from queries.services import pipeline
from queries.services.faq import _decisive_conflict, _decisive_tokens


class TheProbeOnlyFiresForApplicationQuestionsTests(SimpleTestCase):
    """A day count is not a reason to ask about fertiliser.

    entity_intent cannot separate these — "what to apply for a 100-day variety?"
    and "what are the 100-day varieties?" are BOTH ('variety', cultivar, list).
    Whether anything is being APPLIED is what separates them.
    """

    def test_a_variety_listing_question_gets_no_probe(self):
        for question in ("What are the 100-day rice varieties?",
                         "Talk about the 100-day rice variety.",
                         "Name some 100 day rice varieties"):
            with self.subTest(question=question):
                self.assertEqual("", pipeline.duration_probe(question))

    def test_give_does_not_count_as_applying(self):
        # "which varieties GIVE good yield" is a listing question. Putting
        # "give" in the application vocabulary put it straight back.
        self.assertEqual("", pipeline.duration_probe(
            "Which 150-day varieties give good yield?"))

    def test_a_named_nutrient_is_itself_the_intent(self):
        self.assertIn("urea", pipeline.duration_probe(
            "How many kg of urea for the 100-day variety?"))

    def test_an_application_verb_with_no_nutrient_still_probes(self):
        # The case the "fertilizer" default exists for, and it must survive.
        self.assertIn("fertilizer", pipeline.duration_probe(
            "What to apply for a 100-day variety?"))

    def test_the_word_fertilizer_alone_is_enough(self):
        self.assertIn("short duration", pipeline.duration_probe(
            "How much fertilizer for a 100 day variety?"))


class TheProbeMayNotServeTests(TestCase):
    """It may add context. It may not return an approved answer verbatim.

    duration_probe's docstring already claimed this, on the strength of one
    measurement (0.787, under the 0.80 serve threshold). That was an
    observation, not a guarantee: FAQ-137 scores 0.891 on the probe. The
    guarantee lives in code now, so this asserts the ENFORCEMENT rather than
    the arithmetic that used to stand in for it.
    """

    def setUp(self):
        from queries.models import FAQ
        self.catalogue = Catalogue.objects.create(name="KB", slug="kb-probe")
        self.entry = FAQ.objects.create(
            question_en="For short duration rice varieties like Sahabhagi Dhan, "
                        "how much fertilizer to be applied and at what time?",
            answer_en="Apply 22-24 kg DAP and 5-8 kg Urea per acre.",
            is_active=True)

    def run_pipeline(self, question, lookup):
        seen = {}
        with patch.object(pipeline, "faq_lookup", side_effect=lookup), \
                patch.object(pipeline, "faq_nearest",
                             side_effect=lambda *a, **k: (None, 0.0)), \
                patch.object(pipeline, "retrieve", side_effect=lambda *a, **k: []), \
                patch.object(pipeline, "generate_answer",
                             side_effect=lambda *a, **k: seen.setdefault(
                                 "generated", "a generated answer")), \
                patch.object(pipeline, "translate", side_effect=lambda t, s, d: t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            query = pipeline.run_pipeline_for_text(
                catalogue=self.catalogue, farmer_language="English",
                query_text=question)
        return query, seen

    def probe_only(self, question, catalogue=None):
        """Matches what the PROBE asks; never what the farmer asked."""
        if question.startswith("For short duration"):
            return self.entry, 0.891
        return None, 0.0

    def test_a_probe_match_does_not_short_circuit_the_pipeline(self):
        query, seen = self.run_pipeline(
            "How much urea for a 100-day variety?", self.probe_only)

        self.assertIn("generated", seen,
                      "the probe served an approved answer verbatim")
        self.assertNotEqual(self.entry.answer_en, query.ai_response_text)

    def test_a_direct_match_is_still_served_verbatim(self):
        # The fix must not disarm the FAQ itself. A question the farmer really
        # asked, matching an approved entry, still skips the pipeline.
        query, seen = self.run_pipeline(
            "For short duration rice varieties like Sahabhagi Dhan, how much "
            "fertilizer to be applied and at what time?",
            lambda q, catalogue=None: (self.entry, 0.95))

        self.assertNotIn("generated", seen, "an exact FAQ match stopped serving")
        self.assertEqual(self.entry.answer_en, query.ai_response_text)

    def test_the_probe_is_recorded_in_the_trace(self):
        # Without this the trace shows a confident FAQ match and no sign the
        # question was rewritten — query 1030 was unreviewable for that reason.
        query, _seen = self.run_pipeline(
            "How much urea for a 100-day variety?",
            lambda q, catalogue=None: (None, 0.0))

        steps = query.pipeline_trace.get("steps") or []
        probe = next((s.get("duration_probe") for s in steps
                      if s.get("stage") == "faq_lookup"), None)
        self.assertIsNotNone(probe, "the rewritten question is not in the trace")
        self.assertIn("short duration", probe)

    def test_a_question_with_no_probe_records_none(self):
        query, _seen = self.run_pipeline(
            "What are the 100-day rice varieties?",
            lambda q, catalogue=None: (None, 0.0))

        steps = query.pipeline_trace.get("steps") or []
        faq_step = next(s for s in steps if s.get("stage") == "faq_lookup")
        self.assertNotIn("duration_probe", faq_step)


class PolarityIsComparedByMeaningTests(SimpleTestCase):
    """"no urea" and "not available" are the same polarity.

    _POLARITY used to be a flat SET, so the token kept was whichever word the
    farmer happened to use — and a set of words is not a polarity. Every
    sibling group (_STAGES, _SEASONS, _PRODUCTS) already canonicalises.
    """

    def test_no_and_not_are_the_same_polarity(self):
        self.assertEqual("", _decisive_conflict(
            "If there is no urea, then what should be applied in rice farming?",
            "If Urea is not available, what should I apply in rice?"))

    def test_a_negative_question_still_conflicts_with_a_positive_one(self):
        # The guard's whole purpose, which must survive the fix.
        self.assertIn("polarity", _decisive_conflict(
            "If there is no urea, then what should be applied in rice farming?",
            "How much urea I have to apply in my rice crop?"))

    def test_absent_is_not_prohibited_and_neither_is_excessive(self):
        # Genuinely different instructions, kept in separate classes.
        for a, b in (("no urea", "stop urea"),
                     ("no urea", "excess urea"),
                     ("stop applying urea", "excess urea")):
            with self.subTest(pair=(a, b)):
                self.assertIn("polarity", _decisive_conflict(a, b))

    def test_synonyms_within_a_class_collapse(self):
        for word in ("not", "no", "never", "cannot", "without"):
            with self.subTest(word=word):
                self.assertEqual(
                    frozenset({"neg"}),
                    _decisive_tokens("urea is " + word + " there")["polarity"])

    def test_the_stage_guard_is_untouched(self):
        # FAQ-135 is the same question AT BASAL STAGE. The farmer named no
        # stage, so it must still be kept out — the polarity fix must not make
        # the guard permissive in general.
        self.assertIn("stages", _decisive_conflict(
            "If there is no urea, then what should be applied in rice farming?",
            "If Urea is not available, what should I apply in rice at basal stage?"))
