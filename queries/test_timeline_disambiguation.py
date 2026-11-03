"""Regression tests for agricultural timeline disambiguation.

Ensures the system never confuses seedling age, days after transplanting,
and days after sowing.
"""
from django.test import SimpleTestCase
from queries.services.entity_intent import (
    classify_timeline,
    TIMELINE_SEEDLING_AGE,
    TIMELINE_DAT,
    TIMELINE_DAS,
    TIMELINE_METHOD,
    TIMELINE_AMBIGUOUS,
)


class TimelineClassificationTest(SimpleTestCase):

    def test_seedling_age_for_sri(self):
        q = "How old should rice seedlings be when transplanted for SRI?"
        result = classify_timeline(q)
        self.assertIn(result, (TIMELINE_SEEDLING_AGE, TIMELINE_METHOD))

    def test_days_after_sowing(self):
        q = "How many days after sowing should I transplant SRI rice?"
        self.assertEqual(classify_timeline(q), TIMELINE_DAS)

    def test_days_after_transplanting_fertilizer(self):
        q = "How many days after transplanting should I apply basal fertilizer?"
        self.assertEqual(classify_timeline(q), TIMELINE_DAT)

    def test_mechanical_mat_transplanter(self):
        q = "When should I transplant rice using a mechanical mat transplanter?"
        self.assertEqual(classify_timeline(q), TIMELINE_METHOD)

    def test_activity_20_dat(self):
        q = "What should I do 20 days after transplanting rice?"
        self.assertEqual(classify_timeline(q), TIMELINE_DAT)

    def test_ambiguous_question(self):
        q = "How many days after transplanting can rice seedlings be transplanted?"
        result = classify_timeline(q)
        self.assertIn(result, (TIMELINE_DAT, TIMELINE_AMBIGUOUS, TIMELINE_SEEDLING_AGE))

    def test_no_timeline_plain_variety_question(self):
        q = "Which varieties are recommended for upland?"
        self.assertEqual(classify_timeline(q), "")

    def test_seedling_age_direct(self):
        q = "What is the seedling age for transplanting?"
        self.assertEqual(classify_timeline(q), TIMELINE_SEEDLING_AGE)

    def test_dat_explicit(self):
        q = "Apply urea at 21 DAT"
        self.assertEqual(classify_timeline(q), TIMELINE_DAT)

    def test_das_explicit(self):
        q = "Transplant 15 DAS"
        self.assertEqual(classify_timeline(q), TIMELINE_DAS)
