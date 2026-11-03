"""A variety that was supported must not fall out of the answer.

THE OBSERVED FAILURE
--------------------
Everything upstream was right, and the answer was still wrong. From the stored
trace of a real local run (Q-976):

    accepted pool : MTU-1140, CR Dhan 310, Pooja, Lalat, CR Dhan 317, CR Dhan 318
    answer        : "Suitable medium-duration varieties are: Pooja, CR Dhan 310,
                     Lalat, CR Dhan 317, and CR Dhan 318."

Six handed over with "give ALL of these, not a selection"; five came back. The
farmer has no way to tell a seed was left out, and MTU-1140 may be the only one
their dealer stocks.

This check is the same shape as the dose and claim-strength checks that already
follow generation: detect, retry ONCE naming exactly what was dropped, accept
the retry only if it is complete and has invented nothing, otherwise keep the
original. It never decides WHICH varieties are supported — it only compares the
answer against the list generation was already given.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.pipeline import (_missing_candidates, _name_pattern,
                                       _unsupported_varieties)

SHEET = (
    "Pooja, CR Dhan 310, and Lalat are good yielding medium-duration varieties.\n"
    "CR Dhan 317 and CR Dhan 318 are new medium-duration paddy varieties.\n"
    "MTU-1140 is a medium-duration variety (115-120 days) suitable for medium land.\n"
)
QUESTION = "Which medium-duration rice variety is good to plant in my land?"
ALL_SIX = ["MTU-1140", "CR Dhan 310", "Pooja", "Lalat", "CR Dhan 317",
           "CR Dhan 318"]
COMPLETE = ("Suitable medium-duration varieties are: Pooja, CR Dhan 310, Lalat, "
            "CR Dhan 317, CR Dhan 318, and MTU-1140.")
# The real answer from Q-976: MTU-1140 dropped.
MISSING_ONE = ("Suitable medium-duration varieties are: Pooja, CR Dhan 310, "
               "Lalat, CR Dhan 317, and CR Dhan 318.")


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "sheet", "document_id": i}}
            for i, c in enumerate(chunks or (SHEET,))]


class Conversation:
    """Scripted model replies, counting how many times it was called."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def __call__(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def answer_with(*replies, question=QUESTION, chunks=None):
    chat = Conversation(*replies)
    with patch.object(pipeline, "_chat", side_effect=chat):
        answer = pipeline.generate_answer(question, hits(*(chunks or ())))
    return answer, chat


class NameMatchingTests(SimpleTestCase):
    """Punctuation and case must not fool it; a longer name must not fool it either."""

    def test_the_same_name_written_differently_counts_as_present(self):
        for written in ("MTU-1140", "MTU 1140", "MTU1140", "mtu-1140", "Mtu 1140"):
            with self.subTest(written=written):
                self.assertEqual([], _missing_candidates(
                    f"Try {written} this season.", ["MTU-1140"]))

    def test_a_name_that_is_merely_a_prefix_is_not_a_match(self):
        # Plain substring matching would call this present.
        self.assertEqual(["CR Dhan 31"],
                         _missing_candidates("CR Dhan 310 is suitable.", ["CR Dhan 31"]))

    def test_a_longer_name_is_not_matched_by_a_shorter_one(self):
        self.assertEqual(["MTU-11400"],
                         _missing_candidates("MTU-1140 is good.", ["MTU-11400"]))

    def test_a_word_name_inside_another_word_is_not_a_match(self):
        self.assertEqual(["Pooja"],
                         _missing_candidates("The Poojan festival.", ["Pooja"]))

    def test_the_pattern_is_case_insensitive_and_anchored(self):
        self.assertTrue(_name_pattern("Pooja").search("pooja is suitable"))
        self.assertIsNone(_name_pattern("Pooja").search("Poojas"))


class CompletenessRetryTests(SimpleTestCase):
    """Detect, retry once, and only keep a retry that is actually better."""

    def test_a_complete_answer_is_not_retried(self):
        answer, chat = answer_with(COMPLETE)

        self.assertEqual(1, chat.calls)
        self.assertEqual(COMPLETE, answer)

    def test_one_missing_variety_triggers_exactly_one_retry(self):
        answer, chat = answer_with(MISSING_ONE, COMPLETE)

        self.assertEqual(2, chat.calls)
        self.assertIn("MTU-1140", answer)

    def test_the_retry_names_the_omission_and_the_whole_list(self):
        _answer, chat = answer_with(MISSING_ONE, COMPLETE)
        instruction = chat.prompts[-1]

        self.assertIn("left out MTU-1140", instruction)
        for name in ALL_SIX:
            self.assertIn(name, instruction)
        self.assertIn("Do not add any other variety", instruction)

    def test_several_missing_varieties_are_all_named_in_the_retry(self):
        thin = "Suitable medium-duration varieties are: Pooja and Lalat."
        _answer, chat = answer_with(thin, COMPLETE)
        instruction = chat.prompts[-1]

        self.assertEqual(2, chat.calls)
        for dropped in ("MTU-1140", "CR Dhan 310", "CR Dhan 317", "CR Dhan 318"):
            self.assertIn(dropped, instruction)

    def test_a_successful_retry_is_the_answer_that_is_used(self):
        answer, _chat = answer_with(MISSING_ONE, COMPLETE)

        self.assertEqual(COMPLETE, answer)

    def test_a_retry_that_is_still_incomplete_is_rejected(self):
        answer, chat = answer_with(MISSING_ONE, MISSING_ONE)

        self.assertEqual(2, chat.calls)          # no second retry
        self.assertEqual(MISSING_ONE, answer)    # the original is kept

    def test_it_never_loops(self):
        # Ten identical incomplete replies must still cost exactly two calls.
        answer, chat = answer_with(*([MISSING_ONE] * 10))

        self.assertEqual(2, chat.calls)
        self.assertEqual(MISSING_ONE, answer)

    def test_a_retry_that_invents_a_variety_is_rejected(self):
        invented = (COMPLETE[:-1] + ", and Swarna Sub1.")
        answer, chat = answer_with(MISSING_ONE, invented)

        self.assertEqual(2, chat.calls)
        self.assertEqual(MISSING_ONE, answer)
        self.assertNotIn("Swarna Sub1", answer)

    def test_an_empty_retry_is_rejected(self):
        answer, _chat = answer_with(MISSING_ONE, "   ")

        self.assertEqual(MISSING_ONE, answer)

    def test_the_unsupported_check_allows_a_variety_the_farmer_named(self):
        # A name from the question is not an invention.
        self.assertEqual([], _unsupported_varieties(
            "CR Dhan 100 is not in the list.", ["Pooja"],
            question="Is CR Dhan 100 any good?"))
        self.assertEqual(["CR Dhan 100"], _unsupported_varieties(
            "CR Dhan 100 is not in the list.", ["Pooja"], question=""))


class ScopeTests(SimpleTestCase):
    """It applies to multiple-option variety questions and nothing else."""

    def test_an_explicit_single_request_is_never_forced_to_list_them_all(self):
        for question in ("Which is the best medium-duration variety?",
                         "Recommend only one medium-duration variety.",
                         "Give me the top medium-duration variety."):
            with self.subTest(question=question):
                one = "CR Dhan 310 is a medium-duration variety."
                answer, chat = answer_with(one, COMPLETE, question=question)

                self.assertEqual(1, chat.calls)
                self.assertEqual(one, answer)

    def test_a_single_supported_candidate_is_not_checked(self):
        # Nothing to be incomplete about; no retry is worth an LLM call.
        sheet = "MTU-1140 is a medium-duration variety.\n"
        answer, chat = answer_with("Something else entirely.", COMPLETE,
                                   chunks=(sheet,))

        self.assertEqual(1, chat.calls)
        self.assertEqual("Something else entirely.", answer)

    def test_a_non_variety_question_is_unaffected(self):
        reply = "Apply 40 kg urea per acre at tillering."
        answer, chat = answer_with(reply, COMPLETE,
                                   question="How much urea should I apply at tillering?",
                                   chunks=("Apply 40 kg urea per acre at tillering.\n",))

        self.assertEqual(1, chat.calls)
        self.assertEqual(reply, answer)

    def test_a_fertiliser_question_is_unaffected(self):
        reply = "For medium land apply Urea 40 kg per acre."
        answer, chat = answer_with(reply, COMPLETE,
                                   question="Which fertiliser will be applied to Pooja Dhan?",
                                   chunks=("Pooja Dhan is suitable for medium land.\n"
                                           "For medium land apply Urea 40 kg per acre.\n",))

        self.assertEqual(1, chat.calls)
        self.assertEqual(reply, answer)

    def test_a_question_whose_facet_nothing_supports_is_unaffected(self):
        reply = "The information available does not identify one."
        answer, chat = answer_with(reply, COMPLETE,
                                   chunks=("Sarala is suitable for medium land.\n",))

        self.assertEqual(1, chat.calls)
        self.assertEqual(reply, answer)


class FilteringUnchangedTests(SimpleTestCase):
    """The pool that feeds the check is the same pool as before."""

    def test_medium_duration_filtering_is_unchanged(self):
        from queries.services import entity_intent

        names = [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(QUESTION), hits())]

        self.assertCountEqual(ALL_SIX, names)

    def test_medium_land_filtering_is_unchanged(self):
        from queries.services import entity_intent

        sheet = "MTU 1010\nSuitable for medium land\nBRRI Dhan 69\nMedium-duration\n"
        names = [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse("Which variety is suitable for medium land?"),
            hits(sheet))]

        self.assertEqual(["MTU 1010"], names)

    def test_the_completeness_check_cannot_add_an_unsupported_variety(self):
        # Even if the model tries twice, an unsupported name never survives.
        invented = "Suitable varieties are: " + ", ".join(ALL_SIX) + ", and Sarala."
        answer, _chat = answer_with(MISSING_ONE, invented)

        self.assertNotIn("Sarala", answer)

    def test_the_answer_still_never_leaks_internal_wording(self):
        answer, _chat = answer_with(
            "The CONTEXT says: " + MISSING_ONE,
            "The CONTEXT says: " + COMPLETE)

        self.assertNotIn("CONTEXT", answer)
