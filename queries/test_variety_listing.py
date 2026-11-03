"""Asking for NAMES gets the list, not a sample of it.

The reported failure: nine drought-tolerant varieties in the retrieved text and
three in the answer, with nothing saying the rest existed. Two faults produced
it — the variety-question detector missed "names of the DROUGHT-RESISTANT
varieties" because it required those words to be adjacent, so the proper-noun
guard stripped the names it did not recognise; and nothing told generation how
many varieties the evidence actually supported.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline

DROUGHT = (
    "Drought tolerant rice varieties released in India\n"
    "IR64-Drt1 / DRR 42 is drought tolerant and suited to rainfed uplands.\n"
    "DRR 43 is drought tolerant with a duration of 115 days.\n"
    "DRR 44 / Hardinath 3 is drought tolerant.\n"
    "DRR Dhan-48 is tolerant to drought stress.\n"
    "DRR Dhan-45 shows drought tolerance at flowering.\n"
    "Sahbhagi Dhan is drought tolerant and matures in 105 days.\n"
    "CR Dhan 802 is drought tolerant, suitable for rainfed lowland.\n"
    "Swarna Shakti Dhan is drought tolerant.\n"
    "Sukha Dhan 5 is drought tolerant in Nepal and eastern India.\n"
)
UPLAND = "CR Dhan 100 - Fit for upland, suitable for irrigated upland ecosystem.\n"


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks]


class RequestKindTests(SimpleTestCase):
    """Every variety question is a request for the options, unless one was asked
    for in so many words. Grammatical number decides nothing."""

    def test_plural_names_is_a_multiple_request(self):
        self.assertEqual("multiple", pipeline._variety_request(
            "Please tell me the names of the drought-resistant varieties of paddy crop."))

    def test_plural_names_with_a_singular_noun_is_still_multiple(self):
        # The reported question. "Names" is the word that decides it.
        self.assertEqual("multiple", pipeline._variety_request(
            "Please tell me the names of the drought-resistant variety of paddy crop."))

    def test_which_varieties_is_multiple(self):
        self.assertEqual("multiple",
                         pipeline._variety_request("Which varieties tolerate drought?"))

    def test_a_singular_noun_is_still_a_request_for_the_options(self):
        # Both of these USED to come back as "single", and that is the bug: a
        # farmer who says "the variety" is not asking to be told about one seed
        # while the others go unmentioned.
        self.assertEqual("multiple", pipeline._variety_request(
            "Please tell me the name of the drought-resistant variety of paddy crop."))
        self.assertEqual("multiple",
                         pipeline._variety_request("Which variety is drought tolerant?"))

    def test_only_an_explicit_ask_for_one_is_a_single_request(self):
        for question in ("Which is the best variety for drought?",
                         "Recommend only one variety for upland.",
                         "Give me the top variety.",
                         "Suggest just one variety for lowland."):
            with self.subTest(question=question):
                self.assertEqual("single_explicit",
                                 pipeline._variety_request(question))

    def test_a_non_variety_question_is_neither(self):
        self.assertIsNone(pipeline._variety_request("How should I treat seed before sowing?"))
        self.assertIsNone(pipeline._variety_request("How much urea at tillering?"))

    def test_the_old_boolean_helper_still_answers_its_old_question(self):
        # Existing callers ask "is this about varieties at all?" and still get a
        # straight bool.
        self.assertTrue(pipeline._is_variety_question("Which variety is drought tolerant?"))
        self.assertTrue(pipeline._is_variety_question(
            "Please tell me the names of the drought-resistant varieties of paddy crop."))
        self.assertFalse(pipeline._is_variety_question("How much urea at tillering?"))


class SupportedVarietyTests(SimpleTestCase):
    """Which varieties the CONTEXT actually ties to the condition."""

    def ask(self, question, *chunks):
        return pipeline._varieties_supported_by(question, hits(*chunks))

    def test_every_supported_variety_is_collected(self):
        found = self.ask(
            "Please tell me the names of the drought-resistant varieties of paddy crop.",
            DROUGHT)

        for name in ("IR64-Drt1", "DRR 43", "DRR 44", "DRR Dhan-48", "DRR Dhan-45",
                     "Sahbhagi Dhan", "CR Dhan 802", "Swarna Shakti Dhan",
                     "Sukha Dhan 5"):
            self.assertIn(name, found)

    def test_the_same_variety_in_two_chunks_is_listed_once(self):
        found = self.ask("Which varieties are drought tolerant?",
                         "Sahbhagi Dhan is drought tolerant.",
                         "Sahbhagi Dhan is drought tolerant and short duration.")

        self.assertEqual(["Sahbhagi Dhan"], found)

    def test_a_variety_the_context_does_not_tie_to_the_condition_is_left_out(self):
        # Appearing in a drought-varieties document is not evidence; the row has
        # to say so.
        found = self.ask("Which varieties are drought tolerant?",
                         "Drought tolerant varieties of rice\n"
                         "Sahbhagi Dhan is drought tolerant.\n"
                         "CR Dhan 100 is fit for upland.\n")

        self.assertIn("Sahbhagi Dhan", found)
        self.assertNotIn("CR Dhan 100", found)

    def test_a_denial_is_not_support(self):
        found = self.ask("Which varieties are drought tolerant?",
                         "Sahbhagi Dhan is drought tolerant.\n"
                         "CR Dhan 907 is submergence tolerant, not drought tolerant.\n"
                         "Swarna Sub1 is not tolerant to drought.\n")

        self.assertEqual(["Sahbhagi Dhan"], found)

    def test_another_condition_selects_another_variety(self):
        found = self.ask("Which varieties are suitable for upland?", DROUGHT + UPLAND)

        self.assertIn("CR Dhan 100", found)
        self.assertNotIn("Sahbhagi Dhan", found)


class GenerationTests(SimpleTestCase):
    """What the model is handed, and what it is not."""

    def prompt_for(self, question, *chunks):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(question, hits(*chunks))
        return seen["sys"]

    def test_a_multiple_request_is_given_every_supported_variety(self):
        prompt = self.prompt_for(
            "Please tell me the names of the drought-resistant varieties of paddy crop.",
            DROUGHT)

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        for name in ("IR64-Drt1", "DRR 43", "Sahbhagi Dhan", "Swarna Shakti Dhan",
                     "Sukha Dhan 5"):
            self.assertIn(name, prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1])

    def test_the_list_forbids_ranking_and_additions(self):
        prompt = self.prompt_for(
            "Which varieties are drought tolerant?", DROUGHT)
        note = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1]

        self.assertIn("do not rank them", note)
        self.assertIn("not in this list", note.replace("\n", " "))

    def test_a_singular_question_gets_the_same_list(self):
        # It used to get a lead recommendation instead. Same pool, same
        # instruction: the wording is not a request to narrow anything.
        singular = self.prompt_for("Which variety is drought tolerant?", DROUGHT)
        plural = self.prompt_for("Which varieties are drought tolerant?", DROUGHT)

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", singular)
        self.assertEqual(plural, singular)

    def test_an_explicit_ask_for_one_gets_one(self):
        prompt = self.prompt_for("Which is the best variety for drought?", DROUGHT)

        self.assertIn("The farmer asked for ONE option", prompt)
        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)

    def test_a_non_variety_question_is_unchanged(self):
        prompt = self.prompt_for("How should I treat seed before sowing?",
                                 "Treat seed with fungicide before sowing.")

        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)

    def test_the_high_land_question_still_reaches_its_variety(self):
        # The evidence is in the prompt, and the question is read as a request
        # for the suitable seeds — singular wording and all. "High land"
        # reaching "upland" is the land-type equivalence in
        # entity_intent.LAND_TYPES, the same one the answer rules rely on.
        question = ("What type of seed should be used for high land to get "
                    "good rice yield?")
        prompt = self.prompt_for(question, UPLAND)

        self.assertIn("CR Dhan 100", prompt)
        self.assertEqual("multiple", pipeline._variety_request(question))
        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
