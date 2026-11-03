"""The answer is framed on the facet that was asked for.

THE REPORTED FAILURE
--------------------
    "Which medium-duration rice variety is good to plant in my land?"

    -> "Suitable varieties for medium land are: Pooja, CR Dhan 310,
        CR Dhan 317, and CR Dhan 318. Each of these varieties is good for
        medium-duration planting."

The right names under the wrong heading. Tracing it, the intent was correct
(maturity=medium_duration, land_type empty) and the pool was correct — the
heading came out of the PROMPT, which carried a hardcoded example:

    Present them as the supported options, plainly - for example
    'Suitable varieties for medium land are: ...'

The model copied that example whatever the question had been, and nothing
anywhere stated the requested facet as the criterion the answer must be framed
on. The example is now built from the question, and the criterion is spelled
out.

The same trace showed two of the expected varieties missing from the pool:
"Pooja" and "Lalat" carry no digit and no "Dhan", so the cultivar shape could
not see them — "Pooja, CR Dhan 310, and Lalat are good yielding medium-duration
varieties" yielded CR Dhan 310 alone.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

QUESTION = "Which medium-duration rice variety is good to plant in my land?"
LAND_QUESTION = "Which variety is suitable for medium land?"

# The context as reported: medium-duration evidence and medium-land evidence,
# in separate rows, for overlapping sets of varieties.
DURATION_ROWS = (
    "Pooja, CR Dhan 310, and Lalat are good yielding medium-duration varieties.\n"
    "CR Dhan 317 and CR Dhan 318 are new medium-duration paddy varieties.\n"
    "MTU-1140 is a medium-duration variety (115-120 days) suitable for medium land.\n"
)
LAND_ROWS = (
    "Pooja is suitable for medium land.\n"
    "CR Dhan 310 and CR Dhan 317 are suitable for medium land.\n"
)
EXPECTED = ["Pooja", "CR Dhan 310", "Lalat", "CR Dhan 317", "CR Dhan 318",
            "MTU-1140"]


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "sheet", "document_id": i}}
            for i, c in enumerate(chunks or (DURATION_ROWS, LAND_ROWS))]


def prompt_for(question, *chunks):
    seen = {}

    def capture(messages, **kwargs):
        seen["sys"] = messages[0]["content"]
        return "ok"

    with patch.object(pipeline, "_chat", side_effect=capture):
        pipeline.generate_answer(question, hits(*chunks))
    return seen["sys"]


class IntentTests(SimpleTestCase):
    """1 and 2 — the question is read as a duration question, not a land one."""

    def test_the_requested_facet_is_maturity(self):
        intent = entity_intent.analyse(QUESTION)

        self.assertEqual("medium_duration", intent.attributes["maturity"])

    def test_my_land_does_not_become_medium_land(self):
        intent = entity_intent.analyse(QUESTION)

        self.assertEqual("", intent.land_type)
        self.assertNotIn("land_type", intent.attributes)

    def test_the_criterion_handed_to_generation_is_the_duration(self):
        criterion = pipeline._requested_criterion(entity_intent.analyse(QUESTION))

        self.assertEqual("medium duration", criterion)
        self.assertNotIn("land", criterion)


class CandidatePoolTests(SimpleTestCase):
    """3 and 4 — every variety with duration evidence, and only those."""

    def pool(self, question, *chunks):
        return [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(question), hits(*chunks))]

    def test_every_variety_with_medium_duration_evidence_is_included(self):
        self.assertCountEqual(EXPECTED, self.pool(QUESTION))

    def test_the_word_shaped_names_are_seen_too(self):
        # Pooja and Lalat carry no digit and no "Dhan"; they used to be
        # invisible to the collector and silently dropped from the answer.
        names = self.pool(QUESTION)

        self.assertIn("Pooja", names)
        self.assertIn("Lalat", names)

    def test_a_medium_land_only_variety_is_excluded(self):
        self.assertEqual([], self.pool(QUESTION,
                                       "Sarala is suitable for medium land.\n"))

    def test_medium_land_evidence_alone_never_qualifies(self):
        # Pooja appears in both kinds of row. Given ONLY the land row, it is out.
        self.assertEqual([], self.pool(QUESTION, LAND_ROWS))

    def test_a_variety_stating_both_facts_is_included_once(self):
        names = self.pool(QUESTION)

        self.assertEqual(1, names.count("MTU-1140"))

    def test_a_coded_name_is_not_also_counted_as_a_word_name(self):
        # "Sahbhagi Dhan" must not also yield a separate "Sahbhagi".
        names = self.pool("Which varieties are drought tolerant?",
                          "Sahbhagi Dhan is drought tolerant.\n")

        self.assertEqual(["Sahbhagi Dhan"], names)


class Mtu1140Tests(SimpleTestCase):
    """MTU-1140 states BOTH facts, and the duration one is why it qualifies.

    It was reported missing from the answer. Traced end to end, it is not
    dropped by extraction, by the pool, by the facet requirement or by the
    prompt — its row says "medium-duration variety" and that is what admits it.
    The two forms at the bottom are the ones that genuinely do not qualify, and
    both are deliberate: a duration given only in days states no duration CLASS,
    and a heading does not vouch for the rows under it.
    """

    ROW = ("MTU-1140 is a medium-duration variety (115-120 days) "
           "suitable for medium land.")

    def pool(self, *chunks):
        return [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(QUESTION), hits(*chunks))]

    def test_it_is_extracted_as_a_cultivar_name(self):
        self.assertIn("MTU-1140", entity_intent._names_in(self.ROW, "cultivar"))

    def test_its_row_states_the_duration_facet(self):
        stated = entity_intent.detect_attributes(self.ROW).as_dict()

        self.assertEqual("medium_duration", stated["maturity"])
        self.assertEqual("medium_land", stated["land_type"])

    def test_it_is_in_the_pool_on_its_own(self):
        self.assertEqual(["MTU-1140"], self.pool(self.ROW))

    def test_it_is_in_the_pool_alongside_the_other_sources(self):
        self.assertIn("MTU-1140", self.pool())

    def test_it_survives_every_way_the_row_is_written(self):
        for text in (
                "MTU-1140 is a medium-duration variety (115-120 days).",
                "MTU-1140 is a medium duration variety suitable for medium land.",
                "MTU-1140 is a medium-duration variety (115–120 days).",   # en dash
                "3. MTU-1140 is a medium-duration variety suitable for medium land.",
                "3.\xa0MTU-1140 is a medium-duration variety.",                 # nbsp
                "A good medium-duration variety is MTU-1140.",
                "MTU 1140 is a medium-duration variety.",
                "MTU-1140\nMedium-duration\nSuitable for medium land\n"):
            with self.subTest(text=text.replace("\n", " | ")):
                found = self.pool(text)

                self.assertEqual(1, len(found), found)
                self.assertTrue(found[0].lower().startswith("mtu"), found)

    def test_the_duration_evidence_is_what_qualifies_it_not_the_land(self):
        # Strip the duration and it drops out; strip the land and it stays.
        self.assertEqual([], self.pool("MTU-1140 is suitable for medium land."))
        self.assertEqual(["MTU-1140"],
                         self.pool("MTU-1140 is a medium-duration variety."))

    def test_it_reaches_the_generation_prompt_with_its_own_conditions(self):
        prompt = prompt_for(QUESTION)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split(
            "Present them")[0]

        self.assertIn("MTU-1140", listed)
        self.assertIn("medium-duration", listed)

    def test_it_also_answers_the_medium_land_question(self):
        # The same row states both, so it belongs to both pools — the facts are
        # kept separate, not merged.
        names = [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(LAND_QUESTION), hits(self.ROW))]

        self.assertEqual(["MTU-1140"], names)

    def test_a_duration_given_only_in_days_states_no_duration_class(self):
        # DELIBERATE, and the limit worth knowing: "115-120 days" is a figure,
        # not a class. Reading it as medium duration would be an agronomic
        # judgement the document has not made — the same range is early in one
        # sheet and medium in another.
        self.assertEqual([], self.pool(
            "MTU-1140 (Sarala) - Duration: 115-120 days. Suitable for medium land."))

    def test_a_heading_naming_the_class_does_not_vouch_for_the_rows(self):
        # DELIBERATE, and the existing guard: a document titled "medium duration
        # varieties" does not make every name printed under it one.
        self.assertEqual([], self.pool(
            "Medium duration varieties (115-120 days)\nMTU-1140\n"
            "Suitable for medium land\n"))


class FramingTests(SimpleTestCase):
    """5, 6 and 7 — the prompt names the criterion and forbids the reframing."""

    def test_the_prompt_states_the_requested_facet(self):
        prompt = prompt_for(QUESTION)

        self.assertIn("PRIMARY ANSWER CRITERION", prompt)
        self.assertIn("MEDIUM DURATION", prompt)

    def test_the_prompt_forbids_reframing_it_as_medium_land(self):
        prompt = prompt_for(QUESTION)

        self.assertIn("must NOT become the heading", prompt)
        self.assertIn("Never write 'suitable varieties for medium land'", prompt)

    def test_the_worked_example_is_built_from_the_question(self):
        prompt = prompt_for(QUESTION)

        self.assertIn("'Suitable medium duration varieties are: ...'", prompt)
        self.assertNotIn("'Suitable varieties for medium land are: ...'", prompt)

    def test_the_example_follows_a_different_facet(self):
        prompt = prompt_for("Which varieties are suitable for high land?",
                            "CR Dhan 100\nFit for upland\n")

        self.assertIn("'Suitable upland varieties are: ...'", prompt)

    def test_every_expected_variety_reaches_the_model(self):
        prompt = prompt_for(QUESTION)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split(
            "\nCONTEXT:")[0]

        for name in EXPECTED:
            self.assertIn(name, listed)

    def test_a_reframed_answer_is_not_what_the_model_is_asked_for(self):
        # The heading from the report must not appear anywhere in the
        # instructions the model is given, in any casing.
        prompt = prompt_for(QUESTION)
        instructions = prompt.split("\nCONTEXT:")[0].lower()

        self.assertNotIn("suitable varieties for medium land are:", instructions)


class ExistingBehaviourUnchangedTests(SimpleTestCase):
    """8 — the land question still answers exactly as it did."""

    def test_the_medium_land_question_still_returns_medium_land_varieties(self):
        names = [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(LAND_QUESTION), hits())]

        self.assertIn("Pooja", names)
        self.assertIn("CR Dhan 310", names)
        self.assertIn("CR Dhan 317", names)

    def test_the_land_question_is_framed_on_the_land(self):
        prompt = prompt_for(LAND_QUESTION)

        self.assertIn("MEDIUM LAND", prompt)
        self.assertIn("'Suitable medium land varieties are: ...'", prompt)

    def test_the_upland_question_is_untouched(self):
        prompt = prompt_for("Which rice variety is suitable for high land?",
                            "CR Dhan 100\nFit for upland\n")

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        self.assertIn("CR Dhan 100", prompt)

    def test_an_explicit_single_request_still_gives_one_and_states_the_facet(self):
        prompt = prompt_for("Recommend only one medium-duration variety.",
                            DURATION_ROWS)

        self.assertIn("The farmer asked for ONE option", prompt)
        self.assertIn("PRIMARY ANSWER CRITERION", prompt)

    def test_a_question_with_no_facet_gets_no_criterion_note(self):
        prompt = prompt_for("Which variety is good for puffed rice?",
                            "CR Dhan 907 is suitable for puffed rice.\n")

        self.assertNotIn("PRIMARY ANSWER CRITERION", prompt)
        self.assertIn("'Suitable varieties are: ...'", prompt)

    def test_the_answer_still_never_leaks_internal_wording(self):
        with patch.object(pipeline, "_chat",
                          side_effect=lambda *a, **k:
                          "The CONTEXT says: Pooja is medium duration."):
            answer = pipeline.generate_answer(QUESTION, hits())

        self.assertNotIn("CONTEXT", answer)
        self.assertIn("Pooja", answer)
