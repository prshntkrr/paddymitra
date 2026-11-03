"""Regression tests for dose verification, substance-free tolerance,
nitrogen→urea correction, computed-amount substance swaps,
sentence-bounded windowing in _dose_pairs, and deterministic schedule
rendering.

Covers the fixes from the Oct 2026 query-failure investigation.
"""
import re
from unittest import mock

from django.test import SimpleTestCase

from queries.services.pipeline import (
    _dose_pairs,
    _substance_free_doses,
    _unverified_doses,
    _drop_substance_free_dose_sentences,
    _context_rates,
    area_scaled_doses,
    _deterministic_schedule,
)


# ── _dose_pairs: sentence-bounded window ────────────────────────

class DosePairsSentenceBoundaryTests(SimpleTestCase):
    """The 80-char backward window must not cross sentence boundaries."""

    def test_no_cross_sentence_bleed(self):
        text = ("Apply 70 kg of DAP. At tillering, apply 60-70 kg of urea. "
                "At PI, apply 30-40 kg of MOP.")
        pairs = _dose_pairs(text)
        self.assertIn(("70", "dap"), pairs)
        self.assertIn(("60-70", "urea"), pairs)
        self.assertIn(("30-40", "mop"), pairs)
        # Cross-sentence contamination must not happen
        self.assertNotIn(("60-70", "dap"), pairs)
        self.assertNotIn(("30-40", "urea"), pairs)

    def test_same_sentence_pairing_works(self):
        text = "Apply DAP 35 kg/acre, MOP 15-20 kg/acre, and urea 30 kg/acre."
        pairs = _dose_pairs(text)
        self.assertIn(("35", "dap"), pairs)
        self.assertIn(("15-20", "mop"), pairs)
        self.assertIn(("30", "urea"), pairs)

    def test_nitrogen_abbreviation_detected(self):
        text = "Apply 60 kg N/ha total, split as 30 kg N basal."
        pairs = _dose_pairs(text)
        self.assertIn(("60", "nitrogen"), pairs)
        self.assertIn(("30", "nitrogen"), pairs)

    def test_context_paragraph_pairing(self):
        text = ("Nitrogen deficiency management: apply fertilizer in splits: "
                "give one-third basal at planting, another third at tillering, "
                "using urea or ammonium sulfate based on soil tests—about "
                "100-120 kg N per hectare in total")
        pairs = _dose_pairs(text)
        self.assertIn(("100-120", "nitrogen"), pairs)
        self.assertIn(("100-120", "urea"), pairs)


# ── Substance-free dose tolerance ────────────────────────────────

class SubstanceFreeDoseToleranceTests(SimpleTestCase):
    """Substance-free doses grounded in context are tolerated."""

    def test_tolerated_when_number_in_context(self):
        context = "Apply 30 kg N basal, 15 kg N at tillering. Total 60 kg N/ha."
        answer = "Apply 30 kg at basal stage and 15 kg at tillering."
        bad = _unverified_doses(answer, context)
        # 30 and 15 exist in context paired with nitrogen → tolerated
        substance_free_entries = [b for b in bad if "does not say of WHAT" in b]
        self.assertEqual(substance_free_entries, [])

    def test_flagged_when_number_not_in_context(self):
        context = "Apply 30 kg N basal."
        answer = "Apply 99 kg at basal stage."
        bad = _unverified_doses(answer, context)
        substance_free = [b for b in bad if "does not say of WHAT" in b]
        self.assertTrue(len(substance_free) > 0)


class DropSubstanceFreeWithContextTests(SimpleTestCase):
    """_drop_substance_free_dose_sentences tolerates context-grounded numbers."""

    def test_grounded_sentences_kept(self):
        context = "Apply 30 kg N basal, 15 kg N at tillering."
        answer = ("Apply 30 kg at basal. Apply 15 kg at tillering. "
                  "This totals 60 kg N/ha.")
        result = _drop_substance_free_dose_sentences(answer, context)
        self.assertIn("30 kg", result)
        self.assertIn("15 kg", result)

    def test_ungrounded_sentences_dropped(self):
        context = "Apply 30 kg N basal."
        answer = ("Apply 30 kg at basal. Apply 99 kg at tillering. "
                  "This totals 60 kg N/ha.")
        result = _drop_substance_free_dose_sentences(answer, context)
        self.assertNotIn("99 kg", result)
        self.assertIn("30 kg", result)


# ── Nitrogen→urea conversion detection ──────────────────────────

class NitrogenUreaConversionTests(SimpleTestCase):
    """The verifier detects nitrogen→urea wrong-substance errors."""

    def test_urea_flagged_when_context_says_nitrogen(self):
        context = "Apply 60 kg N/ha in total, 30 kg N at basal."
        answer = "Apply 60 kg of urea in total."
        bad = _unverified_doses(answer, context)
        self.assertTrue(any("urea" in b and "wrong substance" in b for b in bad))

    def test_nitrogen_in_context_passes(self):
        context = "Apply 60 kg N/ha in total."
        answer = "Apply 60 kg N/ha in total."
        bad = _unverified_doses(answer, context)
        self.assertEqual(bad, [])


# ── Area scaling ─────────────────────────────────────────────────

class AreaScalingTests(SimpleTestCase):
    """area_scaled_doses produces correct figures and allowed pairs."""

    CONTEXT = ("Apply DAP 35 kg/acre, MOP 15-20 kg/acre at sowing. "
               "At active tillering, apply Urea 30-35 kg/acre.")

    def test_two_acres(self):
        note, allowed = area_scaled_doses(
            "How much fertilizer for 2 acres?", self.CONTEXT)
        self.assertIn(("70", "dap"), allowed)
        self.assertIn(("30-40", "mop"), allowed)
        self.assertIn(("60-70", "urea"), allowed)
        self.assertIn("2 acre", note)

    def test_no_area_no_scaling(self):
        note, allowed = area_scaled_doses(
            "How much fertilizer for rice?", self.CONTEXT)
        self.assertEqual(note, "")
        self.assertEqual(allowed, set())

    def test_scaled_doses_pass_verification(self):
        _, allowed = area_scaled_doses(
            "Fertilizer for 2 acres?", self.CONTEXT)
        answer = ("For 2 acres, apply 70 kg of DAP at basal. "
                  "At tillering, apply 60-70 kg of urea. "
                  "At PI, apply 30-40 kg of MOP.")
        bad = _unverified_doses(answer, self.CONTEXT, allowed)
        self.assertEqual(bad, [])

    def test_unsupported_numbers_rejected(self):
        _, allowed = area_scaled_doses(
            "Fertilizer for 2 acres?", self.CONTEXT)
        answer = "For 2 acres, apply 99 kg of DAP."
        bad = _unverified_doses(answer, self.CONTEXT, allowed)
        self.assertTrue(any("99" in b for b in bad))

    def test_five_acres(self):
        note, allowed = area_scaled_doses(
            "How much fertilizer for 5 acres?", self.CONTEXT)
        self.assertIn(("175", "dap"), allowed)
        self.assertIn(("75-100", "mop"), allowed)
        self.assertIn(("150-175", "urea"), allowed)

    def test_ten_acres(self):
        note, allowed = area_scaled_doses(
            "How much fertilizer for 10 acres?", self.CONTEXT)
        self.assertIn(("350", "dap"), allowed)
        self.assertIn(("150-200", "mop"), allowed)
        self.assertIn(("300-350", "urea"), allowed)

    def test_only_per_acre_rates_scaled(self):
        context_no_rate = "Nitrogen is important for rice growth."
        note, allowed = area_scaled_doses(
            "How much fertilizer for 2 acres?", context_no_rate)
        self.assertEqual(note, "")
        self.assertEqual(allowed, set())

    def test_five_acres_scaled_doses_pass_verification(self):
        _, allowed = area_scaled_doses(
            "Fertilizer for 5 acres?", self.CONTEXT)
        answer = ("For 5 acres, apply 175 kg of DAP at basal. "
                  "At tillering, apply 150-175 kg of urea. "
                  "At PI, apply 75-100 kg of MOP.")
        bad = _unverified_doses(answer, self.CONTEXT, allowed)
        self.assertEqual(bad, [])

    def test_wrong_stage_correct_quantity_still_passes(self):
        """Quantities are verified, not stage assignments."""
        _, allowed = area_scaled_doses(
            "Fertilizer for 2 acres?", self.CONTEXT)
        answer = "For 2 acres, apply 70 kg of DAP at tillering."
        bad = _unverified_doses(answer, self.CONTEXT, allowed)
        self.assertEqual(bad, [])

    def test_ambiguous_mapping_rejected(self):
        """Numbers that only exist for one substance are wrong for another."""
        # Use 3 acres: DAP=105 (unique to DAP), MOP=45-60, urea=90-105.
        # 45 is unique to MOP at 3 acres, so "45 kg of urea" is wrong.
        _, allowed3 = area_scaled_doses(
            "Fertilizer for 3 acres?", self.CONTEXT)
        answer = "For 3 acres, apply 45 kg of urea."
        bad = _unverified_doses(answer, self.CONTEXT, allowed3)
        self.assertTrue(any("wrong substance" in b for b in bad))


# ── Deterministic schedule rendering ──────────────────────────────

class DeterministicScheduleTests(SimpleTestCase):
    """_deterministic_schedule renders correct amounts from structured data."""

    CONTEXT = ("Apply DAP 35 kg/acre, MOP 15-20 kg/acre at sowing. "
               "At active tillering, apply Urea 30-35 kg/acre.")

    def test_two_acres(self):
        result = _deterministic_schedule("Fertilizer for 2 acres?", self.CONTEXT)
        self.assertIsNotNone(result)
        self.assertIn("70", result)
        self.assertIn("DAP", result)
        self.assertIn("60-70", result)
        self.assertIn("UREA", result)
        self.assertIn("30-40", result)
        self.assertIn("MOP", result)

    def test_five_acres(self):
        result = _deterministic_schedule("Fertilizer for 5 acres?", self.CONTEXT)
        self.assertIsNotNone(result)
        self.assertIn("175", result)
        self.assertIn("DAP", result)
        self.assertIn("150-175", result)
        self.assertIn("UREA", result)

    def test_ten_acres(self):
        result = _deterministic_schedule("Fertilizer for 10 acres?", self.CONTEXT)
        self.assertIsNotNone(result)
        self.assertIn("350", result)

    def test_no_area_returns_none(self):
        result = _deterministic_schedule("How much fertilizer?", self.CONTEXT)
        self.assertIsNone(result)

    def test_no_rates_returns_none(self):
        result = _deterministic_schedule(
            "Fertilizer for 2 acres?", "Nitrogen is important.")
        self.assertIsNone(result)

    def test_one_acre_returns_none(self):
        result = _deterministic_schedule(
            "Fertilizer for 1 acre?", self.CONTEXT)
        self.assertIsNone(result)
