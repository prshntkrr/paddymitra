"""Singular and plural are the same question.

The reported failure: "Which variety is suitable for upland?" came back with one
name while "Which varieties are suitable for upland?" came back with several,
from the same documents. The number of a noun is not evidence about the world,
so everything here is written as a PAIR — the singular and plural wording of the
same question, asserted to reach the same candidate pool and to differ only in
how that pool is presented.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

# One varietal sheet, the shape the corpus actually holds: one row per variety,
# each row saying what it is fit for.
SHEET = (
    "Rice varieties recommended for Odisha ecosystems\n"
    "CR Dhan 100 is recommended for upland, suitable for direct seeded upland.\n"
    "Sahbhagi Dhan is suitable for upland and drought prone areas.\n"
    "DRR 43 is fit for upland conditions with a duration of 115 days.\n"
    "Swarna Sub1 is recommended for lowland and submergence prone lowland.\n"
    "CR Dhan 505 is suitable for lowland areas.\n"
    "Pooja is not suitable for upland.\n"
)
FERTILISER = (
    "Apply Urea in three splits; Urea is recommended at tillering.\n"
    "DAP is recommended as a basal application for rice.\n"
    "MOP is suitable for potassium supply at panicle initiation.\n"
)


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks]


# The eight questions the fix has to cover, as four pairs.
PAIRS = [
    ("Which variety is suitable for upland?",
     "Which varieties are suitable for upland?", "variety", "upland"),
    ("Which variety is suitable for lowland?",
     "Which varieties are suitable for lowland?", "variety", "lowland"),
    ("Which seed is suitable for upland?",
     "Which seeds are suitable for upland?", "seed", "upland"),
    ("Which fertilizer should I apply?",
     "Which fertilizers should I apply?", "fertilizer", ""),
]


class IntentTests(SimpleTestCase):
    """Both numbers are read as the same request, quantity included."""

    def test_every_pair_agrees_on_everything(self):
        for singular, plural, entity, land in PAIRS:
            with self.subTest(question=singular):
                one, many = entity_intent.analyse(singular), entity_intent.analyse(plural)

                self.assertIsNotNone(one)
                self.assertIsNotNone(many)
                self.assertEqual(entity, one.entity)
                self.assertEqual(entity, many.entity)
                # The wording does not change the request: both are asking
                # which options are suitable.
                self.assertEqual("multiple", one.quantity)
                self.assertEqual("multiple", many.quantity)
                self.assertEqual(land, one.land_type)
                self.assertEqual(land, many.land_type)
                # The pool is built from these, so they must not differ.
                self.assertEqual(one.condition_terms, many.condition_terms)

    def test_the_land_type_is_canonical_not_the_farmers_wording(self):
        for wording in ("high land", "highland", "dry land", "upland"):
            with self.subTest(wording=wording):
                intent = entity_intent.analyse(f"Which variety is suitable for {wording}?")
                self.assertEqual("upland", intent.land_type)
        for wording in ("low land", "lowland", "waterlogged"):
            with self.subTest(wording=wording):
                intent = entity_intent.analyse(f"Which variety is suitable for {wording}?")
                self.assertEqual("lowland", intent.land_type)

    def test_a_question_that_merely_contains_the_noun_is_not_a_request(self):
        self.assertIsNone(entity_intent.analyse("How should I treat seed before sowing?"))
        self.assertIsNone(entity_intent.analyse("How much urea at tillering?"))
        # A quantity question stays a quantity question, however it is worded.
        self.assertIsNone(entity_intent.analyse("What is the recommended seed rate?"))

    def test_the_other_entities_are_recognised_in_both_numbers(self):
        for singular, plural, entity in (
                ("Which disease affects rice?", "Which diseases affect rice?", "disease"),
                ("Which pest is common in kharif?", "Which pests are common in kharif?", "pest"),
                ("Which document covers this?", "Which documents cover this?", "document"),
                ("Which catalogue item covers this?",
                 "Which catalogue items cover this?", "catalogue item")):
            with self.subTest(entity=entity):
                self.assertEqual(entity, entity_intent.analyse(singular).entity)
                self.assertEqual("multiple", entity_intent.analyse(singular).quantity)
                self.assertEqual(entity, entity_intent.analyse(plural).entity)
                self.assertEqual("multiple", entity_intent.analyse(plural).quantity)


class CandidatePoolTests(SimpleTestCase):
    """One pool per question, whatever number it was asked in."""

    def pool(self, question, *chunks):
        return entity_intent.candidate_pool(entity_intent.analyse(question),
                                            hits(*chunks or (SHEET,)))

    def test_singular_and_plural_reach_the_same_pool(self):
        for singular, plural, _entity, _land in PAIRS[:3]:
            with self.subTest(question=singular):
                one = [c["name"] for c in self.pool(singular)]
                many = [c["name"] for c in self.pool(plural)]

                self.assertEqual(many, one)
                self.assertGreater(len(one), 1, "the singular pool was narrowed")

    def test_the_upland_pool_holds_every_upland_variety_and_no_others(self):
        names = [c["name"] for c in self.pool("Which variety is suitable for upland?")]

        self.assertCountEqual(["CR Dhan 100", "Sahbhagi Dhan", "DRR 43"], names)

    def test_lowland_selects_the_lowland_varieties(self):
        names = [c["name"] for c in self.pool("Which varieties are suitable for lowland?")]

        self.assertCountEqual(["Swarna Sub1", "CR Dhan 505"], names)

    def test_a_denial_never_puts_a_variety_in_the_pool(self):
        names = [c["name"] for c in self.pool("Which variety is suitable for upland?")]

        self.assertNotIn("Pooja", names)

    def test_the_farmers_wording_does_not_change_the_pool(self):
        # "high land" and "upland" are the same question to the documents.
        farmer = [c["name"] for c in self.pool("Which variety is suitable for high land?")]
        document = [c["name"] for c in self.pool("Which variety is suitable for upland?")]

        self.assertEqual(document, farmer)

    def test_the_strongest_evidence_is_first(self):
        # "recommended for upland" outranks "fit for upland conditions": both are
        # support, one is an explicit recommendation. Nothing here claims the
        # documents rank anything.
        pool = self.pool("Which variety is suitable for upland?")

        self.assertEqual("CR Dhan 100", pool[0]["name"])
        self.assertTrue(pool[0]["recommended"])
        self.assertGreaterEqual(pool[0]["score"], pool[-1]["score"])

    def test_an_input_question_collects_inputs_not_cultivars(self):
        pool = entity_intent.candidate_pool(
            entity_intent.analyse("Which fertilizers are recommended at tillering?"),
            hits(FERTILISER))

        self.assertIn("urea", [c["name"] for c in pool])


class RetrievalVariantTests(SimpleTestCase):
    """Both wordings must SEARCH alike, not merely be answered alike."""

    def test_the_number_twin_is_searched_too(self):
        variants = entity_intent.retrieval_variants(
            "Which variety is suitable for upland?")

        self.assertTrue(any("varieties" in v for v in variants), variants)

    def test_the_documents_own_land_term_is_searched(self):
        variants = entity_intent.retrieval_variants(
            "Which variety is suitable for high land?")

        self.assertTrue(any("upland" in v for v in variants), variants)

    def test_a_non_entity_question_gets_no_variants(self):
        self.assertEqual([], entity_intent.retrieval_variants("How much urea at tillering?"))

    def test_the_pipeline_expands_both_wordings_into_the_same_search_set(self):
        # _expand_queries also calls the LLM; stub it so this asserts the
        # deterministic half, which is the half that must not depend on number.
        with patch.object(pipeline, "_chat", side_effect=RuntimeError("no LLM in tests")):
            one = pipeline._expand_queries("Which variety is suitable for upland?")
            many = pipeline._expand_queries("Which varieties are suitable for upland?")

        self.assertTrue(any("varieties" in q for q in one))
        self.assertTrue(any("variety" in q for q in many))


class GenerationTests(SimpleTestCase):
    """Same pool in the prompt; different instruction about presenting it."""

    def prompt_for(self, question, *chunks):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(question, hits(*chunks or (SHEET,)))
        return seen["sys"]

    def test_a_singular_question_is_given_the_whole_pool_as_options(self):
        prompt = self.prompt_for("Which variety is suitable for upland?")

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        for name in ("CR Dhan 100", "Sahbhagi Dhan", "DRR 43"):
            self.assertIn(name, prompt)
        # No lead recommendation and no "other options": nothing in the
        # question asked anyone to choose.
        self.assertNotIn("The farmer asked for ONE option", prompt)
        self.assertIn("do NOT introduce the rest as", prompt)

    def test_a_plural_question_is_told_to_give_them_all(self):
        prompt = self.prompt_for("Which varieties are suitable for upland?")

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1]
        for name in ("CR Dhan 100", "Sahbhagi Dhan", "DRR 43"):
            self.assertIn(name, listed)

    def test_both_numbers_name_the_same_varieties(self):
        singular = self.prompt_for("Which seed is suitable for upland?")
        plural = self.prompt_for("Which seeds are suitable for upland?")

        for name in ("CR Dhan 100", "Sahbhagi Dhan", "DRR 43"):
            self.assertIn(name, singular)
            self.assertIn(name, plural)

    def test_nothing_is_recommended_that_the_context_does_not_support(self):
        for question in ("Which variety is suitable for upland?",
                         "Which varieties are suitable for upland?"):
            with self.subTest(question=question):
                prompt = self.prompt_for(question)

                # A lowland-only variety is never offered for an upland question,
                # and the one the sheet rules out is never offered at all.
                note = prompt.split("CONTEXT SUPPORTS")[1].split("\nCONTEXT:")[0]
                self.assertNotIn("Swarna Sub1", note)
                self.assertNotIn("Pooja", note)

    def test_an_unsupported_question_gets_no_invented_list(self):
        prompt = self.prompt_for("Which variety is suitable for upland?",
                                 "Rice is grown across Odisha in three seasons.")

        self.assertNotIn("LEAD WITH", prompt)
        self.assertNotIn("CONTEXT SUPPORTS FOR WHAT WAS ASKED", prompt)

    def test_a_fertiliser_question_is_left_to_the_existing_rules(self):
        # Both numbers, unchanged: the schedule rules in the grounding prompt
        # already answer these, and a candidate list would fight them.
        for question in ("Which fertilizer should I apply?",
                         "Which fertilizers should I apply?"):
            with self.subTest(question=question):
                prompt = self.prompt_for(question, FERTILISER)

                self.assertNotIn("LEAD WITH", prompt)
                self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
                self.assertIn("COMPLETE schedule", prompt)


class RetrievalWidthTests(SimpleTestCase):
    """The width of the pool is decided by the ask, never by the noun's number."""

    def test_both_numbers_ask_the_index_for_the_same_number_of_chunks(self):
        for singular, plural, _entity, _land in PAIRS:
            with self.subTest(question=singular):
                self.assertIsNotNone(entity_intent.analyse(singular))
                self.assertIsNotNone(entity_intent.analyse(plural))

    def test_an_ordinary_question_is_not_widened(self):
        self.assertIsNone(entity_intent.analyse("How much urea should I apply at tillering?"))
