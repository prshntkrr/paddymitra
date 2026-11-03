"""Regression tests for follow-up question detection.

Covers: entity references ("which one", "which of these"), standalone questions,
topic changes, and short clarifying replies.
"""
from django.test import SimpleTestCase

from queries.services.pipeline import _is_followup


def _hist(user_msg, assistant_msg):
    return [{"user": user_msg, "assistant": assistant_msg}]


class EntityRefFollowupTests(SimpleTestCase):
    """Questions with entity references are follow-ups when history exists."""

    HIST = _hist(
        "Which rice variety is suitable for lowland?",
        "CR Dhan 800, Swarna Sub-1, MTV-1061 are suitable for lowland."
    )

    def test_which_one_matures_fastest(self):
        self.assertTrue(
            _is_followup("Which one matures fastest?", self.HIST))

    def test_which_of_these_gives_highest_yield(self):
        self.assertTrue(
            _is_followup("Which of these gives highest yield?", self.HIST))

    def test_from_these_which_is_best(self):
        self.assertTrue(
            _is_followup("From these which is best?", self.HIST))

    def test_those_varieties_how_long(self):
        self.assertTrue(
            _is_followup("How long do those take?", self.HIST))

    def test_the_above_varieties(self):
        self.assertTrue(
            _is_followup("Tell me more about the above", self.HIST))


class StandaloneQuestionTests(SimpleTestCase):
    """Genuine standalone questions are NOT follow-ups."""

    HIST = _hist(
        "How much urea per acre?",
        "Apply 30-35 kg urea per acre at tillering."
    )

    def test_new_topic_variety(self):
        self.assertFalse(
            _is_followup("What is the best variety for upland?", self.HIST))

    def test_new_topic_pest(self):
        self.assertFalse(
            _is_followup("How do I control stem borer?", self.HIST))

    def test_no_history(self):
        self.assertFalse(
            _is_followup("Which one matures fastest?", []))


class ClarifyingReplyTests(SimpleTestCase):
    """Short replies to clarifying questions are follow-ups."""

    def test_land_type_reply(self):
        hist = _hist(
            "How much fertilizer for rice?",
            "Which type of land do you have — irrigated, rainfed, or upland?"
        )
        self.assertTrue(_is_followup("irrigated", hist))

    def test_upland_reply(self):
        hist = _hist(
            "How much fertilizer for early rice?",
            "Which type of land?"
        )
        self.assertTrue(_is_followup("upland", hist))

    def test_area_reply(self):
        hist = _hist(
            "How much fertilizer?",
            "Please tell me your cultivated area and variety?"
        )
        self.assertTrue(_is_followup("2 hectares MTU 1010", hist))

    def test_question_word_reply_to_clarifying(self):
        """A question-word reply to a clarifying question is standalone."""
        hist = _hist(
            "How much fertilizer?",
            "Which type of land do you have?"
        )
        self.assertFalse(
            _is_followup("What is the best variety for lowland?", hist))


class TopicChangeTests(SimpleTestCase):
    """Topic changes should not be treated as follow-ups."""

    def test_urea_then_variety(self):
        hist = _hist(
            "How much urea per acre?",
            "Apply 30-35 kg urea per acre."
        )
        self.assertFalse(
            _is_followup("What is the best variety for upland?", hist))

    def test_variety_then_fertilizer(self):
        hist = _hist(
            "Which variety for lowland?",
            "CR Dhan 800, Swarna Sub-1."
        )
        self.assertFalse(
            _is_followup("How much DAP per acre?", hist))
