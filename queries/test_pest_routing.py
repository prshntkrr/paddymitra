"""Regression tests for pest/disease routing.

Bug Bugia and other pest-related questions must route to the pest catalogue,
not to nutrient management.
"""
from django.test import SimpleTestCase, override_settings

from hooks.routing import classify


class PestRoutingTests(SimpleTestCase):
    """Pest-related questions classify as pest."""

    def test_bug_bugia_classifies_as_pest(self):
        result = classify("What is Bug Bugia and how to control it?")
        self.assertEqual(result["category"], "pest")

    def test_generic_bug_question(self):
        result = classify("There are bugs on my rice leaves")
        self.assertEqual(result["category"], "pest")

    def test_blast_still_pest(self):
        result = classify("What causes leaf blast?")
        self.assertEqual(result["category"], "pest")

    def test_insect_control(self):
        result = classify("How do I control the insects attacking my crop?")
        self.assertEqual(result["category"], "pest")

    def test_fertilizer_stays_nutrient(self):
        result = classify("How much urea per acre?")
        self.assertEqual(result["category"], "nutrient")

    def test_variety_stays_variety(self):
        result = classify("Which rice variety is suitable for lowland?")
        self.assertEqual(result["category"], "variety")

    def test_ambiguous_pest_name_routes_to_pest(self):
        result = classify("My crop has some bug attack, what should I spray?")
        self.assertEqual(result["category"], "pest")

    def test_disease_with_no_pest_name(self):
        result = classify("My rice leaves are turning yellow and dying")
        cat = result["category"]
        self.assertIn(cat, ("pest", ""))
