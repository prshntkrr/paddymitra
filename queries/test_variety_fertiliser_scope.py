"""Two relationships the documents hold, and neither may be improvised.

PROBLEM 1 — a facet with no support is not an invitation to answer anyway.

    "Which medium-duration rice variety is good to plant in my land?"

came back as "Suitable varieties for medium land are: Pooja, CR Dhan 310,
CR Dhan 317, CR Dhan 318. Each of these varieties is good for medium-duration
planting." The candidate pool was already right — it returned nothing, because
no row says any of them is medium duration. What was missing was anything
TELLING the model that. It held the raw context (all medium-LAND rows), the rule
that a which-variety question is answered with names, and the land-type note
asking it to state each variety's ecosystem — so it listed the medium-land
varieties and bolted the duration claim on the end.

PROBLEM 2 — a fertiliser question that names a variety is a relationship
question, and the documents hold it in two halves:

    Pooja  ->  medium land            (the varietal sheet)
    medium land  ->  Urea 40 kg/acre  (the fertiliser table)

Answered without the first half, the dose comes from whichever ecosystem the
retrieval happened to return: an upland rate handed to a farmer with a lowland
field. The variety's own land type is now established first, from its own rows
only, and the answer is scoped to it — or refused when the documents do not
establish it.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

MEDIUM_LAND_SHEET = (
    "Pooja\nSuitable for medium land\n"
    "CR Dhan 310\nSuitable for medium land\n"
    "CR Dhan 317\nSuitable for medium land\n"
    "CR Dhan 318\nSuitable for medium land\n"
)
# The two halves of the fertiliser relationship, as separate rows.
FERTILISER_SHEET = (
    "Pooja Dhan\nSuitable for medium land\n"
    "For medium land apply Urea 40 kg per acre at tillering.\n"
    "For upland apply DAP 25 kg per acre as basal.\n"
    "For lowland apply MOP 20 kg per acre.\n"
)
DURATION_QUESTION = "Which medium-duration rice variety is good to plant in my land?"
FERTILISER_QUESTION = "Which fertiliser will be applied to Pooja Dhan?"


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "sheet", "document_id": i}}
            for i, c in enumerate(chunks)]


def prompt_for(question, *chunks):
    seen = {}

    def capture(messages, **kwargs):
        seen["sys"] = messages[0]["content"]
        return "ok"

    with patch.object(pipeline, "_chat", side_effect=capture):
        pipeline.generate_answer(question, hits(*chunks))
    return seen["sys"]


class MediumDurationIsNotMediumLandTests(SimpleTestCase):
    """The reported answer, and why it can no longer be produced."""

    def test_my_land_is_not_read_as_medium_land(self):
        intent = entity_intent.analyse(DURATION_QUESTION)

        self.assertEqual("medium_duration", intent.maturity)
        self.assertEqual("", intent.land_type)
        self.assertEqual({"maturity": "medium_duration"}, intent.attributes)

    def test_medium_land_varieties_are_not_in_the_pool(self):
        pool = entity_intent.candidate_pool(
            entity_intent.analyse(DURATION_QUESTION), hits(MEDIUM_LAND_SHEET))

        self.assertEqual([], pool)

    def test_the_model_is_told_that_nothing_supports_the_facet(self):
        prompt = prompt_for(DURATION_QUESTION, MEDIUM_LAND_SHEET)

        self.assertIn("NO VARIETY IN THE CONTEXT IS SUPPORTED", prompt)
        self.assertIn("maturity: medium_duration", prompt)
        self.assertIn("Do NOT name a variety from the CONTEXT", prompt)

    def test_the_medium_land_names_are_never_offered_as_candidates(self):
        prompt = prompt_for(DURATION_QUESTION, MEDIUM_LAND_SHEET)
        note = prompt.split("NO VARIETY IN THE CONTEXT")[1].split("\nCONTEXT:")[0]

        for name in ("Pooja", "CR Dhan 310", "CR Dhan 317", "CR Dhan 318"):
            self.assertNotIn(name, note)

    def test_a_variety_with_real_duration_evidence_is_still_returned(self):
        sheet = MEDIUM_LAND_SHEET + "BRRI Dhan 69\nMedium-duration\n"
        prompt = prompt_for(DURATION_QUESTION, sheet)

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split("\nCONTEXT:")[0]
        self.assertIn("BRRI Dhan 69", listed)
        for name in ("Pooja", "CR Dhan 310", "CR Dhan 317", "CR Dhan 318"):
            self.assertNotIn(name, listed)

    def test_other_attributes_never_satisfy_a_duration_request(self):
        for row in ("Pooja\nSuitable for medium land\n",
                    "Pooja\nFit for upland\n",
                    "Pooja\nSuitable for lowland\n",
                    "Pooja\nSuitable for irrigated land\n",
                    "Pooja\nMedium-bold grain\n",
                    "Pooja\nMedium height\n"):
            with self.subTest(row=row.replace("\n", " | ")):
                self.assertEqual([], entity_intent.candidate_pool(
                    entity_intent.analyse(DURATION_QUESTION), hits(row)))


class VarietyConditionsTests(SimpleTestCase):
    """Step 2: what the documents say THIS variety is for."""

    def test_the_variety_own_land_type_is_found(self):
        found = entity_intent.variety_conditions("Pooja Dhan", hits(FERTILISER_SHEET))

        self.assertEqual("medium_land", found["land_type"])
        self.assertIn("medium land", found["_phrases"])

    def test_it_works_from_a_sentence_as_well_as_a_record(self):
        found = entity_intent.variety_conditions(
            "Pooja Dhan", hits("Pooja Dhan is suitable for medium land.\n"))

        self.assertEqual("medium_land", found["land_type"])

    def test_a_neighbouring_variety_land_type_does_not_leak_in(self):
        sheet = ("CR Dhan 100\nFit for upland\n"
                 "Pooja Dhan\nSuitable for medium land\n"
                 "Swarna Sub1\nSuitable for lowland\n")
        found = entity_intent.variety_conditions("Pooja Dhan", hits(sheet))

        self.assertEqual("medium_land", found["land_type"])

    def test_an_unmapped_variety_returns_nothing(self):
        found = entity_intent.variety_conditions(
            "Pooja Dhan", hits("For medium land apply Urea 40 kg per acre.\n"))

        self.assertEqual({}, found)

    def test_a_denial_is_not_a_mapping(self):
        found = entity_intent.variety_conditions(
            "Pooja Dhan", hits("Pooja Dhan is not suitable for upland.\n"))

        self.assertEqual({}, found)

    def test_the_question_own_variety_name_is_extracted(self):
        self.assertEqual(["Pooja Dhan"],
                         entity_intent.named_varieties(FERTILISER_QUESTION))
        self.assertEqual(["CR Dhan 310"],
                         entity_intent.named_varieties("Which fertilizer for CR Dhan 310?"))
        self.assertEqual([], entity_intent.named_varieties("Which fertilizer should I apply?"))


class FertiliserThroughTheVarietyTests(SimpleTestCase):
    """Steps 3 and 4: scoped to the variety's own ecosystem, or refused."""

    def test_the_answer_is_scoped_to_the_variety_own_land(self):
        prompt = prompt_for(FERTILISER_QUESTION, FERTILISER_SHEET)

        self.assertIn("ANSWERED THROUGH THE LAND IT IS GROWN ON", prompt)
        self.assertIn("Pooja Dhan is for: medium land", prompt)
        self.assertIn("Do NOT use a recommendation the CONTEXT gives for a "
                      "DIFFERENT land type", prompt.replace("\n", " "))

    def test_an_unestablished_mapping_is_refused_not_guessed(self):
        # The fertiliser table is present; the varietal row is not.
        sheet = ("For medium land apply Urea 40 kg per acre.\n"
                 "For upland apply DAP 25 kg per acre.\n")
        prompt = prompt_for(FERTILISER_QUESTION, sheet)

        self.assertIn("does not say which land type", prompt)
        self.assertIn("Do NOT guess the land type", prompt)
        self.assertNotIn("ANSWERED THROUGH THE LAND IT IS GROWN ON", prompt)

    def test_the_mapping_comes_from_the_named_variety_not_a_neighbour(self):
        sheet = ("CR Dhan 100\nFit for upland\n"
                 "Pooja Dhan\nSuitable for medium land\n"
                 "For upland apply DAP 25 kg per acre.\n"
                 "For medium land apply Urea 40 kg per acre.\n")
        prompt = prompt_for(FERTILISER_QUESTION, sheet)

        self.assertIn("Pooja Dhan is for: medium land", prompt)
        self.assertNotIn("Pooja Dhan is for: upland", prompt)

    def test_a_general_fertiliser_question_is_completely_unchanged(self):
        # No variety named, so none of this applies and the existing schedule
        # rules answer it exactly as before.
        prompt = prompt_for("Which fertilizer should I apply?", FERTILISER_SHEET)

        self.assertNotIn("ANSWERED THROUGH THE LAND IT IS GROWN ON", prompt)
        self.assertNotIn("does not establish which land", prompt)
        self.assertIn("COMPLETE schedule", prompt)

    def test_a_plural_fertiliser_question_naming_a_variety_is_scoped_too(self):
        prompt = prompt_for("Which fertilisers are applied to Pooja Dhan?",
                            FERTILISER_SHEET)

        self.assertIn("Pooja Dhan is for: medium land", prompt)


class ExistingBehaviourUnchangedTests(SimpleTestCase):
    """The queries that already worked must answer exactly as they did."""

    def test_the_upland_variety_question_is_untouched(self):
        prompt = prompt_for("Which rice variety is suitable for high land?",
                            "CR Dhan 100\nFit for upland\n")

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        self.assertIn("CR Dhan 100", prompt)
        self.assertNotIn("NO VARIETY IN THE CONTEXT IS SUPPORTED", prompt)

    def test_a_supported_medium_land_question_still_answers(self):
        prompt = prompt_for("Which variety is suitable for medium land?",
                            MEDIUM_LAND_SHEET)

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        self.assertIn("CR Dhan 310", prompt)
        self.assertNotIn("NO VARIETY IN THE CONTEXT IS SUPPORTED", prompt)

    def test_an_explicit_single_request_still_gives_one(self):
        prompt = prompt_for("Recommend only one variety for medium land.",
                            MEDIUM_LAND_SHEET)

        self.assertIn("The farmer asked for ONE option", prompt)

    def test_a_question_with_no_facet_gets_no_unsupported_note(self):
        # Nothing structured was asked for, so there is no facet to be
        # unsupported and the old behaviour stands.
        prompt = prompt_for("Which variety is good for puffed rice?",
                            "Rice is grown across Odisha in three seasons.\n")

        self.assertNotIn("NO VARIETY IN THE CONTEXT IS SUPPORTED", prompt)

    def test_the_answer_never_leaks_internal_wording(self):
        for question, chunk in ((DURATION_QUESTION, MEDIUM_LAND_SHEET),
                                (FERTILISER_QUESTION, FERTILISER_SHEET)):
            with self.subTest(question=question):
                with patch.object(pipeline, "_chat",
                                  side_effect=lambda *a, **k:
                                  "The CONTEXT says: nothing is supported."):
                    answer = pipeline.generate_answer(question, hits(chunk))

                self.assertNotIn("CONTEXT", answer)
