"""Dose arithmetic for the farmer's land size, done in code.

The LLM is not allowed to do this and never will be: permitting "multiply the
rate by the land size" produced 90-100 kg of urea for 2.5 acres, a figure that
appears nowhere in the corpus. So `area_scaled_doses` computes it and the model
copies the result.

Two things here are load-bearing and easy to break:

1. SUBSTANCE ATTRIBUTION. An early version took the first name in _SUBSTANCES
   order rather than the nearest one, and reported DAP's 35 kg/acre as a MOP
   rate. A dose on the wrong fertiliser is worse than no answer.
2. VALIDATOR COOPERATION. A computed figure is by definition absent from the
   documents, so `_unverified_doses` strips it as invented unless the computed
   pairs are passed through.
"""
from django.test import SimpleTestCase

from queries.services.pipeline import (_unverified_doses, area_scaled_doses,
                                       farmer_area)

CTX = ("Apply DAP 35 kg/acre, MOP 15-20 kg/acre, and ZnSO4 10 kg/acre at sowing. "
       "At 15-25 DAS, apply Urea 20 kg/acre for medium and long duration varieties. "
       "At the active tillering stage, apply Urea 30-35 kg/acre, and at panicle "
       "initiation, apply Urea 30-35 kg/acre along with MOP 15-20 kg/acre. "
       "about 100-120 kg N per hectare total for good yields")


class FarmerAreaTests(SimpleTestCase):
    def test_half_an_acre(self):
        result = farmer_area("How much urea for half an acre?")
        self.assertEqual(result[:2], (0.5, "acre"))

    def test_decimal_acres(self):
        result = farmer_area("urea for 2.5 acres")
        self.assertEqual(result[:2], (2.5, "acre"))

    def test_bare_rate_is_not_a_land_size(self):
        # "per hectare" names a basis, not the farmer's plot.
        self.assertIsNone(farmer_area("how much urea per hectare"))

    def test_no_area(self):
        self.assertIsNone(farmer_area("What is blast disease?"))


class NoScalingNeededTests(SimpleTestCase):
    def test_one_acre_is_left_alone(self):
        # The stated per-acre rate already IS the answer; scaling would only
        # add a second number saying the same thing.
        self.assertEqual(area_scaled_doses("How much potash on 1 acre?", CTX),
                         ("", set()))

    def test_question_without_land_size(self):
        self.assertEqual(area_scaled_doses("What is blast?", CTX), ("", set()))


class ScalingTests(SimpleTestCase):
    def test_half_acre_scales_each_rate(self):
        note, _ = area_scaled_doses("How much urea for half an acre?", CTX)
        self.assertIn("15-17.5", note)          # urea 30-35 kg/acre
        self.assertIn("7.5-10", note)           # MOP 15-20 kg/acre

    def test_substance_attribution_is_nearest_not_first(self):
        note, _ = area_scaled_doses("How much for half an acre?", CTX)
        self.assertIn("dap: documents say 35 kg per acre", note)
        self.assertNotIn("mop: documents say 35 kg per acre", note)

    def test_unattributable_figure_is_dropped(self):
        # "ZnSO4" is not in _SUBSTANCES, so its 10 kg/acre must be dropped
        # rather than pinned on whichever product is nearest.
        note, _ = area_scaled_doses("How much for half an acre?", CTX)
        self.assertNotIn("that is 5 kg", note)

    def test_per_hectare_rate_is_not_converted(self):
        # The 100-120 kg N/hectare figure must not be dragged into an acre
        # answer — unit conversion stays forbidden.
        note, _ = area_scaled_doses("How much urea for half an acre?", CTX)
        self.assertNotIn("100-120", note)
        self.assertNotIn("50-60", note)

    def test_the_dose_that_was_invented_before(self):
        note, _ = area_scaled_doses("How much urea for 2.5 acres?", CTX)
        self.assertIn("75-87.5", note)
        self.assertNotIn("90-100", note)


class ValidatorCooperationTests(SimpleTestCase):
    def test_computed_figure_is_flagged_without_the_whitelist(self):
        answer = "Give 15 to 17.5 kg of urea for half an acre."
        self.assertTrue(_unverified_doses(answer, CTX))

    def test_computed_figure_passes_with_the_whitelist(self):
        answer = "Give 15 to 17.5 kg of urea for half an acre."
        _, allowed = area_scaled_doses("How much urea for half an acre?", CTX)
        self.assertEqual(_unverified_doses(answer, CTX, allowed), [])

    def test_an_actually_invented_dose_is_still_caught(self):
        _, allowed = area_scaled_doses("How much urea for half an acre?", CTX)
        self.assertTrue(_unverified_doses("Give 500 kg of urea.", CTX, allowed))
