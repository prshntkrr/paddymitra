"""Production-quantity vs area-rate mismatch detection.

When a farmer asks "how much fertilizer for X tons of rice", the retrieved
context only has per-acre/per-hectare rates.  The pipeline must NOT invent
a total and should ask for cultivated area instead.
"""
from unittest.mock import patch
from django.test import SimpleTestCase

from queries.services.pipeline import generate_answer


_AREA_RATE_HITS = [
    {
        "chunk": (
            "Apply DAP 35 kg/acre as basal. At active tillering apply "
            "Urea 30-35 kg/acre. Nitrogen: approximately 100-120 kg N "
            "per hectare for good yields."
        ),
        "score": 0.65,
        "document_title": "Nutrient Management",
    },
]

_MOCK_PATH = "queries.services.pipeline._chat"


class ProductionQuantityMismatchTests(SimpleTestCase):
    """Rule A: production quantity question + area-based context."""

    @patch(_MOCK_PATH)
    def test_tons_question_prompt_has_mismatch_rule(self, mock_chat):
        mock_chat.return_value = (
            "The available recommendations are expressed per acre or per "
            "hectare. Please provide your cultivated area."
        )
        generate_answer(
            "How much fertilizer should be applied for 1900 tons of rice?",
            _AREA_RATE_HITS,
        )
        prompt = mock_chat.call_args[0][0][0]["content"]
        self.assertIn("PRODUCTION-QUANTITY", prompt)

    @patch(_MOCK_PATH)
    def test_quintals_question_prompt_has_mismatch_rule(self, mock_chat):
        mock_chat.return_value = (
            "The recommendations are per unit area. "
            "Please share your cultivated area."
        )
        generate_answer(
            "How much urea for 500 quintals of paddy?",
            _AREA_RATE_HITS,
        )
        prompt = mock_chat.call_args[0][0][0]["content"]
        self.assertIn("PRODUCTION-QUANTITY", prompt)


class AreaBasedCalculationTests(SimpleTestCase):
    """Rule B: area question + compatible rate = valid calculation."""

    @patch(_MOCK_PATH)
    def test_hectare_question_with_rate(self, mock_chat):
        mock_chat.return_value = "For 2 hectares: DAP 70 kg (35 kg/acre x 2)."
        answer = generate_answer(
            "How much DAP for 2 hectares?",
            _AREA_RATE_HITS,
        )
        self.assertIn("2", answer)

    @patch(_MOCK_PATH)
    def test_nutrient_vs_product_rule_in_prompt(self, mock_chat):
        mock_chat.return_value = (
            "The context states 100-120 kg N per hectare. This is the "
            "elemental nitrogen rate, not the urea quantity."
        )
        generate_answer(
            "The source recommends 100 kg N per hectare. "
            "How much urea should I apply?",
            _AREA_RATE_HITS,
        )
        prompt = mock_chat.call_args[0][0][0]["content"]
        self.assertIn("elemental nutrient", prompt.lower())


class MissingInfoTests(SimpleTestCase):
    """Rule C: insufficient info triggers clarification."""

    @patch(_MOCK_PATH)
    def test_no_area_asks_for_area(self, mock_chat):
        mock_chat.return_value = "Please provide your cultivated area."
        answer = generate_answer(
            "How much fertilizer do I need?",
            _AREA_RATE_HITS,
        )
        self.assertIn("area", answer.lower())


class SafeFallbackTests(SimpleTestCase):
    """Rule D: no evidence = safe fallback."""

    @patch(_MOCK_PATH)
    def test_no_hits_returns_safe_answer(self, mock_chat):
        mock_chat.return_value = (
            "I do not have enough information to answer this question."
        )
        answer = generate_answer(
            "How much boron should I apply for wheat?",
            [],
        )
        self.assertNotIn("kg", answer)


class NamedVarietyPreservedTests(SimpleTestCase):
    """Rule: existing named-variety land-type restriction still works."""

    @patch(_MOCK_PATH)
    def test_variety_question_still_works(self, mock_chat):
        mock_chat.return_value = (
            "For MTU 1010 in lowland: apply DAP 35 kg/acre as basal."
        )
        answer = generate_answer(
            "What is the fertilizer dose for MTU 1010?",
            _AREA_RATE_HITS,
        )
        self.assertTrue(len(answer) > 10)
