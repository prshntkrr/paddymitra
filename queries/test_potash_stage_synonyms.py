"""Potash and the two stage-dose terms must canonicalise, and must not overreach.

Four reported wrong answers traced back here. ପଟାସ had no synonym entry, so the
translator rendered it as "rice husk" / "Planchette" and retrieval never saw a
potash question at all. ପିଲସାର and ଥୋଡ଼ସାର reached English untranslated and
anchored on nothing.

The near-misses are the point of this file: ପଟାସ is a prefix of ପଟାସିୟମ
(potassium) and ପିଲ on its own means a tiller, not the tillering-stage dose.
Both must stay untouched.
"""
from django.test import SimpleTestCase

from queries.services.synonyms import canonicalise


class PotashSynonymTests(SimpleTestCase):
    def test_potash_canonicalises_and_is_pinned(self):
        out, found = canonicalise("ପଟାସ କେତେ ଦେବି?")
        self.assertIn("potash", out)
        self.assertIn("potash", found)

    def test_potash_with_case_suffix(self):
        # Odia glues the case ending straight on; the bare form is never spoken.
        out, _ = canonicalise("ପଟାସରେ କେତେ ଦେବି?")
        self.assertIn("potash", out)

    def test_potassium_is_not_touched(self):
        # ପଟାସ is a prefix of ପଟାସିୟମ. A match here would turn a deficiency
        # question into a dose question.
        out, found = canonicalise("ପଟାସିୟମ ଅଭାବ ଲକ୍ଷଣ କଣ?")
        self.assertNotIn("potash", out)
        self.assertEqual(found, [])


class StageDoseSynonymTests(SimpleTestCase):
    def test_both_stage_doses_in_one_question(self):
        out, found = canonicalise("ପିଲସାର ଓ ଥୋଡ଼ସାରରେ କେତେ ୟୁରିଆ ଦେବି?")
        self.assertIn("tillering stage fertilizer", out)
        self.assertIn("panicle initiation fertilizer", out)
        self.assertIn("tillering stage fertilizer", found)
        self.assertIn("panicle initiation fertilizer", found)

    def test_bare_tiller_word_is_not_a_dose(self):
        # ପିଲ alone is the tiller itself, not the fertiliser applied at tillering.
        out, found = canonicalise("ପିଲ ବାହାରିବା ସମୟରେ")
        self.assertNotIn("fertilizer", out)
        self.assertEqual(found, [])


class PotashRetrievalBridgeTests(SimpleTestCase):
    """The documents never write "potash" — only "MOP". Without a bridge the
    correctly translated question retrieved nothing and was answered "no
    information available"."""

    def test_potash_query_gains_an_mop_variant(self):
        from queries.services.pipeline import _synonym_variants

        variants = _synonym_variants(
            "How much potash should be applied on 1 acre of land?")
        self.assertTrue(
            any("MOP" in v for v in variants),
            f"no MOP variant generated: {variants}")

    def test_unrelated_fertiliser_query_gains_nothing(self):
        from queries.services.pipeline import _synonym_variants

        variants = _synonym_variants("How much urea for half an acre?")
        self.assertFalse([v for v in variants if "MOP" in v])


class NoRegressionTests(SimpleTestCase):
    def test_existing_entry_still_works(self):
        out, found = canonicalise("ଜିଆ ଖତ ବିଷୟରେ କୁହନ୍ତୁ।")
        self.assertIn("vermicompost", out)
        self.assertIn("vermicompost", found)
