""""Medium" is five different attributes, and they are not interchangeable.

The reported failure, end to end: a farmer asked in Odia for a variety for
MEDIUM LAND (ମଧ୍ୟମ କିସମ ଜମି). It reached the pipeline as "medium-maturing
variety", the land type was lost, and the answer recommended a variety as though
it suited the farmer's field — a claim no document made, because the sheets
recommend a variety FOR a stated ecosystem.

Three things are asserted here. That each facet is detected separately and never
from a bare "medium"; that a suitability question with no land type is answered
with the source's own ecosystem and a question back, rather than a false general
claim; and that a rating never gets promoted past what the source wrote.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline
from queries.services.synonyms import canonicalise

# A varietal sheet in which "medium" appears in four different roles, which is
# exactly how these documents are written. The names all carry a code or "Dhan"
# because that is the shape the collector recognises (see the KNOWN LIMIT on
# entity_intent.CULTIVAR_NAME_RE).
SHEET = (
    "CR Dhan 910 is a medium duration variety, recommended for shallow "
    "favourable lowlands and irrigated late conditions.\n"
    "MTU 1010 is suitable for medium land and matures in 125 days.\n"
    "Swarna Sub1 is a long duration variety with medium slender grain, "
    "suitable for irrigated lowland.\n"
    "MTU 1001 is medium height and suitable for medium land.\n"
    "Sahbhagi Dhan is an early duration variety suitable for upland.\n"
)
BLAST = (
    "CR Dhan 910 is moderately resistant to blast and moderately tolerant of "
    "stem borer.\n"
    "It is recommended for shallow favourable lowlands.\n"
)


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks]


class MediumIsNotOneAttributeTests(SimpleTestCase):
    """Each facet is read into its own field, and never off a bare "medium"."""

    def test_medium_maturity_is_a_duration_not_a_land_type(self):
        for wording in ("medium maturity", "medium maturing", "medium-maturing",
                        "medium duration", "medium-duration"):
            with self.subTest(wording=wording):
                found = entity_intent.detect_attributes(
                    f"Which variety is {wording}?")

                self.assertEqual("medium_duration", found.maturity)
                self.assertEqual("", found.land_type)

    def test_medium_land_is_a_land_type_not_a_duration(self):
        for wording in ("medium land", "irrigated medium land",
                        "rainfed medium land"):
            with self.subTest(wording=wording):
                found = entity_intent.detect_attributes(
                    f"Which variety is suitable for {wording}?")

                self.assertEqual("medium_land", found.land_type)
                self.assertEqual("", found.maturity)

    def test_medium_slender_is_a_grain_type_not_a_duration(self):
        found = entity_intent.detect_attributes(
            "Which variety has medium slender grain?")

        self.assertEqual("medium_slender", found.grain_type)
        self.assertEqual("", found.maturity)
        self.assertEqual("", found.land_type)

    def test_medium_height_is_a_plant_height(self):
        found = entity_intent.detect_attributes("Which variety is medium height?")

        self.assertEqual("medium_height", found.plant_height)
        self.assertEqual("", found.maturity)

    def test_a_bare_medium_matches_no_facet_at_all(self):
        for question in ("Which variety is medium?", "Is this medium?",
                         "Which medium variety should I sow?"):
            with self.subTest(question=question):
                self.assertEqual({}, entity_intent.detect_attributes(question).as_dict())

    def test_two_facets_can_be_true_at_once(self):
        found = entity_intent.detect_attributes(
            "Which medium duration variety is suitable for medium land?")

        self.assertEqual("medium_duration", found.maturity)
        self.assertEqual("medium_land", found.land_type)

    def test_the_specific_lowland_wins_over_the_general_one(self):
        self.assertEqual("shallow_lowland", entity_intent.detect_attributes(
            "Which variety suits shallow favourable lowland?").land_type)
        self.assertEqual("rainfed_lowland", entity_intent.detect_attributes(
            "Which variety suits rainfed lowland?").land_type)
        self.assertEqual("irrigated_lowland", entity_intent.detect_attributes(
            "Which variety suits irrigated lowland?").land_type)
        self.assertEqual("lowland", entity_intent.detect_attributes(
            "Which variety suits lowland?").land_type)


class TheOdiaQuestionTests(SimpleTestCase):
    """The reported query, at the point where it went wrong."""

    QUESTION = "ମଧ୍ୟମ କିସମ ଜମି ପାଇଁ କି କେଉଁ ଧାନ ଭେରାଇଟି ଉପଯୁକ୍ତ ହେବ?"

    def test_medium_type_land_canonicalises_to_a_land_type(self):
        # ଜମି is land. The old entry read "ମଧ୍ୟମ କିସମ" as a duration class and
        # the land type was lost before retrieval ever ran.
        text, terms = canonicalise(self.QUESTION)

        self.assertIn("medium land", text)
        self.assertIn("medium land", terms)
        self.assertNotIn("medium-maturing variety", text)

    def test_a_genuine_duration_question_is_still_a_duration_question(self):
        text, terms = canonicalise("ମଧ୍ୟମ କିସମ ଧାନ କେଉଁଟି ଭଲ?")

        self.assertIn("medium-maturing variety", text)

    def test_the_translated_question_is_read_as_a_land_question(self):
        intent = entity_intent.analyse(
            "Which rice variety is suitable for medium land?")

        self.assertEqual("variety", intent.entity)
        self.assertEqual("medium_land", intent.land_type)
        self.assertEqual("", intent.maturity)
        self.assertFalse(intent.land_type_missing)


class CandidatePoolSeparatesFacetsTests(SimpleTestCase):
    """A facet only matches evidence about the SAME facet."""

    def pool(self, question, *chunks):
        return entity_intent.candidate_pool(entity_intent.analyse(question),
                                            hits(*chunks or (SHEET,)))

    def test_a_medium_land_question_does_not_collect_medium_duration_rows(self):
        names = [c["name"] for c in self.pool(
            "Which variety is suitable for medium land?")]

        self.assertCountEqual(["MTU 1010", "MTU 1001"], names)
        self.assertNotIn("CR Dhan 910", names)      # medium DURATION, not land

    def test_a_medium_duration_question_does_not_collect_medium_land_rows(self):
        names = [c["name"] for c in self.pool(
            "Which varieties are medium duration?")]

        self.assertIn("CR Dhan 910", names)
        self.assertNotIn("MTU 1010", names)
        self.assertNotIn("MTU 1001", names)

    def test_a_grain_question_does_not_collect_duration_or_land_rows(self):
        names = [c["name"] for c in self.pool(
            "Which variety has medium slender grain?")]

        self.assertEqual(["Swarna Sub1"], names)

    def test_the_candidate_carries_the_condition_its_source_states(self):
        pool = self.pool("Which varieties are medium duration?")
        cr = next(c for c in pool if c["name"] == "CR Dhan 910")

        self.assertEqual("shallow_lowland", cr["conditions"]["land_type"])
        self.assertEqual("medium_duration", cr["conditions"]["maturity"])
        self.assertIn("shallow favourable lowland", cr["stated_phrases"])


class MissingLandTypeTests(SimpleTestCase):
    """No land type given, no claim that a variety suits the farmer's land."""

    def prompt_for(self, question, *chunks):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(question, hits(*chunks or (SHEET,)))
        return seen["sys"]

    def test_a_duration_only_question_is_flagged_as_missing_the_land_type(self):
        intent = entity_intent.analyse(
            "Which rice variety will be suitable for medium duration?")

        self.assertTrue(intent.land_type_missing)

    def test_a_question_that_gives_the_land_type_is_not_flagged(self):
        intent = entity_intent.analyse(
            "Which rice variety is suitable for upland?")

        self.assertFalse(intent.land_type_missing)

    def test_the_answer_is_told_to_state_the_ecosystem_and_ask(self):
        prompt = self.prompt_for("Which variety is medium duration?")

        self.assertIn("DID NOT SAY WHICH LAND TYPE", prompt)
        self.assertIn("suitable for their land", prompt)
        self.assertIn("which land type they have", prompt)

    def test_the_sources_own_ecosystem_travels_with_the_name(self):
        prompt = self.prompt_for("Which varieties are medium duration?")

        self.assertIn("CR Dhan 910  [its row states:", prompt)
        self.assertIn("shallow favourable lowland", prompt)

    def test_a_question_with_a_land_type_gets_no_such_instruction(self):
        prompt = self.prompt_for("Which variety is suitable for medium land?")

        self.assertNotIn("DID NOT SAY WHICH LAND TYPE", prompt)

    def test_the_grounding_rules_forbid_inventing_a_condition(self):
        prompt = self.prompt_for("Which variety is medium duration?")

        self.assertIn("DO NOT INVENT A SUITABILITY CONDITION", prompt)


class ClaimStrengthTests(SimpleTestCase):
    """A rating is the document's word, and it travels whole."""

    def test_moderate_tolerance_must_not_become_resistant(self):
        problems = entity_intent.overstated_claims(
            "CR Dhan 910 is resistant to blast.", BLAST)

        self.assertTrue(problems)
        self.assertIn("blast", problems[0])

    def test_keeping_the_qualifier_is_not_a_problem(self):
        self.assertEqual([], entity_intent.overstated_claims(
            "CR Dhan 910 is moderately resistant to blast.", BLAST))

    def test_an_unhedged_source_may_be_stated_plainly(self):
        context = "Swarna Sub1 is resistant to blast.\n"

        self.assertEqual([], entity_intent.overstated_claims(
            "Swarna Sub1 is resistant to blast.", context))

    def test_a_named_list_must_not_become_a_blanket(self):
        problems = entity_intent.overstated_claims(
            "CR Dhan 910 is resistant to major pests and diseases.", BLAST)

        self.assertTrue(problems)
        self.assertIn("major pests and diseases", problems[0])

    def test_a_topic_the_context_never_mentions_is_left_to_the_other_rules(self):
        # Not this check's business: the grounding rules and the scope check
        # already deal with a claim about something absent from the CONTEXT.
        self.assertEqual([], entity_intent.overstated_claims(
            "MTU 1001 is tolerant of salinity.", BLAST))

    def test_an_overstated_answer_is_retried_and_the_fix_kept(self):
        replies = ["CR Dhan 910 is resistant to blast.",
                   "CR Dhan 910 is moderately resistant to blast."]

        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: replies.pop(0)):
            answer = pipeline.generate_answer(
                "Which variety is suitable for shallow lowland?", hits(BLAST))

        self.assertIn("moderately resistant", answer)
        self.assertEqual([], replies)          # the retry did happen

    def test_an_unfixed_retry_is_rejected_and_the_original_kept(self):
        replies = ["CR Dhan 910 is resistant to blast.",
                   "CR Dhan 910 is resistant to blast, definitely."]

        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: replies.pop(0)):
            answer = pipeline.generate_answer(
                "Which variety is suitable for shallow lowland?", hits(BLAST))

        self.assertNotIn("definitely", answer)

    def test_the_prompt_carries_both_rules(self):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer("Which variety resists blast?", hits(BLAST))

        self.assertIn("A HEDGE IS PART OF THE RATING", seen["sys"])
        self.assertIn("NEVER GENERALISE A LIST INTO A BLANKET", seen["sys"])


class SamePoolBothNumbersTests(SimpleTestCase):
    """Still true with the facets separated: quantity only affects presentation."""

    def pool(self, question):
        return [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(question), hits(SHEET))]

    def test_singular_and_plural_reach_identical_pools(self):
        for singular, plural in (
                ("Which variety is suitable for medium land?",
                 "Which varieties are suitable for medium land?"),
                ("Which variety is medium duration?",
                 "Which varieties are medium duration?"),
                ("Which seed is suitable for upland?",
                 "Which seeds are suitable for upland?"),
                ("Which variety has medium slender grain?",
                 "Which varieties have medium slender grain?")):
            with self.subTest(question=singular):
                one, many = self.pool(singular), self.pool(plural)

                self.assertEqual(many, one)
                self.assertEqual(
                    entity_intent.analyse(singular).condition_terms,
                    entity_intent.analyse(plural).condition_terms)

    def test_nothing_at_all_differs_between_the_two_wordings(self):
        one = entity_intent.analyse("Which variety is suitable for medium land?")
        many = entity_intent.analyse("Which varieties are suitable for medium land?")

        self.assertEqual("multiple", one.quantity)
        self.assertEqual("multiple", many.quantity)
        self.assertEqual(many.attributes, one.attributes)
        self.assertEqual(many.entity, one.entity)
        self.assertEqual(many.condition_terms, one.condition_terms)
