"""Grammatical singular does not mean the farmer wants one recommendation.

"Which rice variety is suitable for medium land?" is how a farmer asks which
varieties they could sow. Answering with one name, when the documents support
three, hides two seeds they might have been able to buy — so the singular and
plural wordings are read as the same request, search the same strings, build the
same pool and are answered the same way.

The one exception is a farmer who asks for a single option in so many words —
"the best", "only one", "the top one". That gets one, chosen out of the same
pool, and says plainly when the documents rank nothing.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

# Three varieties for medium land, so "gave only one" is visible when it
# happens; one upland and one lowland row to keep the facets honest.
SHEET = (
    "CR Dhan 310 is suitable for medium land and matures in 125 days.\n"
    "CR Dhan 317 is recommended for medium land.\n"
    "MTU-1140 is suitable for medium land in Odisha.\n"
    "CR Dhan 100 is recommended for upland conditions.\n"
    "Sahbhagi Dhan is suitable for upland and drought prone areas.\n"
    "Swarna Sub1 is suitable for lowland and submergence prone lowland.\n"
    "CR Dhan 505 is recommended for lowland.\n"
)
MEDIUM_LAND = ["CR Dhan 310", "CR Dhan 317", "MTU-1140"]
UPLAND = ["CR Dhan 100", "Sahbhagi Dhan"]
LOWLAND = ["Swarna Sub1", "CR Dhan 505"]

# The eight questions from the brief, as four pairs, with what each should find.
PAIRS = [
    ("Which rice variety is suitable for medium land?",
     "Which rice varieties are suitable for medium land?", MEDIUM_LAND),
    ("Which seed should I plant in upland?",
     "Which seeds should I plant in upland?", UPLAND),
    ("Which rice variety is suitable for high land?",
     "Which rice varieties are suitable for high land?", UPLAND),
    ("Which rice variety is suitable for lowland?",
     "Which rice varieties are suitable for lowland?", LOWLAND),
]


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks or (SHEET,)]


def prompt_for(question, *chunks):
    seen = {}

    def capture(messages, **kwargs):
        seen["sys"] = messages[0]["content"]
        return "ok"

    with patch.object(pipeline, "_chat", side_effect=capture):
        pipeline.generate_answer(question, hits(*chunks))
    return seen["sys"]


class BothWordingsAreOneRequestTests(SimpleTestCase):
    """Every pair from the brief: same reading, same searches, same pool."""

    def test_the_intent_is_identical_apart_from_the_question_text(self):
        for singular, plural, _expected in PAIRS:
            with self.subTest(question=singular):
                one = entity_intent.analyse(singular)
                many = entity_intent.analyse(plural)

                self.assertEqual(many.entity, one.entity)
                self.assertEqual(many.quantity, one.quantity)
                self.assertEqual("multiple", one.quantity)
                self.assertEqual(many.land_type, one.land_type)
                self.assertEqual(many.attributes, one.attributes)
                self.assertEqual(many.condition_terms, one.condition_terms)
                self.assertEqual(many.land_type_missing, one.land_type_missing)

    def test_the_index_is_sent_the_same_set_of_strings(self):
        # Not "overlapping" — the same set. Every attribute variant is built on
        # the plural form precisely so this holds.
        for singular, plural, _expected in PAIRS:
            with self.subTest(question=singular):
                one = {singular, *entity_intent.retrieval_variants(singular)}
                many = {plural, *entity_intent.retrieval_variants(plural)}

                self.assertEqual(many, one)

    def test_retrieval_width_is_the_same(self):
        # The width follows "is this an entity question at all", which both
        # wordings answer the same way.
        for singular, plural, _expected in PAIRS:
            with self.subTest(question=singular):
                self.assertIsNotNone(entity_intent.analyse(singular))
                self.assertIsNotNone(entity_intent.analyse(plural))

    def test_the_candidate_pool_is_identical_and_complete(self):
        for singular, plural, expected in PAIRS:
            with self.subTest(question=singular):
                one = [c["name"] for c in entity_intent.candidate_pool(
                    entity_intent.analyse(singular), hits())]
                many = [c["name"] for c in entity_intent.candidate_pool(
                    entity_intent.analyse(plural), hits())]

                self.assertEqual(many, one)
                self.assertCountEqual(expected, one)

    def test_both_answers_are_given_all_the_supported_options(self):
        for singular, plural, expected in PAIRS:
            with self.subTest(question=singular):
                one, many = prompt_for(singular), prompt_for(plural)

                self.assertEqual(many, one)
                self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", one)
                for name in expected:
                    self.assertIn(name, one)

    def test_the_singular_answer_is_not_told_to_pick_one(self):
        prompt = prompt_for("Which rice variety is suitable for medium land?")

        self.assertNotIn("The farmer asked for ONE option", prompt)
        self.assertIn("do NOT introduce the rest as", prompt)
        self.assertIn("Do NOT single one", prompt)

    def test_other_recommendation_phrasings_read_the_same_way(self):
        for question in ("Suggest a variety for lowland.",
                         "What rice variety should I grow in lowland?",
                         "Which variety is recommended for lowland?",
                         "Which varieties are recommended for lowland?"):
            with self.subTest(question=question):
                intent = entity_intent.analyse(question)

                self.assertEqual("multiple", intent.quantity)
                self.assertCountEqual(LOWLAND, [
                    c["name"] for c in entity_intent.candidate_pool(intent, hits())])


class ExplicitSingleRequestTests(SimpleTestCase):
    """One option, only when one was asked for."""

    def test_an_explicit_ask_is_the_only_thing_that_narrows_the_answer(self):
        for question in ("Which is the best variety for medium land?",
                         "Recommend only one variety for medium land.",
                         "Give me the top variety for medium land.",
                         "Suggest just one variety for medium land.",
                         "Give me one suitable variety for medium land."):
            with self.subTest(question=question):
                self.assertEqual("single_explicit",
                                 entity_intent.analyse(question).quantity)

    def test_the_pool_behind_the_pick_is_still_the_whole_pool(self):
        explicit = entity_intent.analyse("Recommend only one variety for medium land.")
        ordinary = entity_intent.analyse("Which variety is suitable for medium land?")

        self.assertCountEqual(
            [c["name"] for c in entity_intent.candidate_pool(ordinary, hits())],
            [c["name"] for c in entity_intent.candidate_pool(explicit, hits())])

    def test_one_variety_is_given_and_the_evidence_with_it(self):
        prompt = prompt_for("Recommend only one variety for medium land.")

        self.assertIn("The farmer asked for ONE option", prompt)
        self.assertIn("that evidence is why", prompt)
        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)

    def test_a_best_question_does_not_invent_a_best(self):
        # The sheet says three varieties suit medium land and ranks none of
        # them. The answer must say so rather than crown one.
        prompt = prompt_for("Which is the best variety for medium land?")

        self.assertIn("does NOT rank these varieties", prompt)
        self.assertIn("do not identify a single best one", prompt)
        self.assertIn("never call it 'the best'", prompt.replace("\n", " "))

    def test_a_clearly_stronger_candidate_is_led_with_and_not_hedged(self):
        # Here the CONTEXT does separate them: one row recommends, the other
        # only records a tolerance.
        sheet = ("CR Dhan 317 is recommended for medium land.\n"
                 "MTU-1140 is tolerant of drought in medium land.\n")
        prompt = prompt_for("Which is the best variety for medium land?", sheet)

        self.assertIn("The farmer asked for ONE option", prompt)
        self.assertIn("CR Dhan 317", prompt)
        self.assertNotIn("does NOT rank these varieties", prompt)


class StrictAttributesStillHoldTests(SimpleTestCase):
    """The facet separation must survive the quantity change."""

    def test_medium_land_and_medium_duration_stay_apart(self):
        sheet = ("CR Dhan 910 is a medium duration variety for shallow lowland.\n"
                 "CR Dhan 310 is suitable for medium land.\n")
        for wording in ("Which variety is suitable for medium land?",
                        "Which varieties are suitable for medium land?"):
            with self.subTest(question=wording):
                names = [c["name"] for c in entity_intent.candidate_pool(
                    entity_intent.analyse(wording), hits(sheet))]

                self.assertEqual(["CR Dhan 310"], names)

    def test_the_odia_land_phrase_still_wins_over_the_duration_phrase(self):
        from queries.services.synonyms import canonicalise

        text, terms = canonicalise("ମଧ୍ୟମ କିସମ ଜମି ପାଇଁ କି କେଉଁ ଧାନ ଭେରାଇଟି ଉପଯୁକ୍ତ ହେବ?")

        self.assertIn("medium land", terms)
        self.assertNotIn("medium-maturing variety", text)

    def test_land_synonyms_still_reach_the_canonical_term(self):
        for wording, canonical in (("high land", "upland"), ("highland", "upland"),
                                   ("dry land", "upland"), ("low land", "lowland"),
                                   ("waterlogged", "lowland"),
                                   ("medium land", "medium_land")):
            with self.subTest(wording=wording):
                for question in (f"Which variety is suitable for {wording}?",
                                 f"Which varieties are suitable for {wording}?"):
                    self.assertEqual(canonical,
                                     entity_intent.analyse(question).land_type)


class MissingLandTypeStillAsksTests(SimpleTestCase):
    """A recommendation question with no land type still does not guess."""

    def test_a_question_with_no_land_type_is_flagged_in_both_wordings(self):
        for question in ("Which rice variety should I plant?",
                         "Which rice varieties should I plant?"):
            with self.subTest(question=question):
                self.assertTrue(entity_intent.analyse(question).land_type_missing)

    def test_a_question_with_nothing_to_filter_on_asks_rather_than_guesses(self):
        # "Which rice variety should I plant?" names no condition at all, so
        # there is nothing to build a grounded pool from. The answer is the
        # question back — not a list assembled from whatever the retrieval
        # happened to return.
        prompt = prompt_for("Which rice variety should I plant?")

        self.assertIn("DID NOT SAY WHICH LAND TYPE", prompt)
        self.assertIn("which land type they have", prompt)
        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)

    def test_a_condition_without_a_land_type_gets_the_options_and_the_question(self):
        # Here there IS something to filter on, so the supported options are
        # given — each carrying the ecosystem its own source states — and the
        # land type is asked for on top.
        sheet = ("CR Dhan 910 is a medium duration variety, recommended for "
                 "shallow favourable lowlands.\n")
        prompt = prompt_for("Which variety is medium duration?", sheet)

        self.assertIn("DID NOT SAY WHICH LAND TYPE", prompt)
        # The note is bracketed and addressed to the model, never to the
        # farmer; _strip_internal_wording() is what guarantees the answer.
        self.assertIn("CR Dhan 910  [its row states:", prompt)
        self.assertIn("shallow favourable lowland", prompt)
