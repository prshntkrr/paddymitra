"""The asked-for facet has to be STATED, not merely nearby.

THE REPORTED FAILURE
--------------------
    "Which medium-duration rice variety is good to plant in my land?"

came back with four varieties, of which one was medium-duration:

    BRRI Dhan 69 … suitable for lowland, medium-duration, medium-bold …
    Bina Dhan 17 … suitable for irrigated mid-early conditions
    CR Dhan 310  … suitable for irrigated and favourable shallow rainfed areas
    CR Dhan 100  … suitable for upland

Two faults, both upstream of generation.

FIRST: condition_terms() split matched phrases on SPACES, so the hyphenated
"medium-duration" stayed one token and the bare words "medium" and "duration"
were never removed from the question's content words. "my land" contributed a
bare "land" for the same reason it always would. Evidence is matched by
substring, so "land" matched up-LAND and low-LAND, and "medium" matched medium
BOLD and medium HEIGHT — a duration question collecting land and grain rows.

SECOND, and the reason the first one was fatal: candidate_pool() accepted a
candidate when ANY term matched a line. CR Dhan 100 was accepted on "Fit for
upland" while its own recorded conditions said land_type=upland and no maturity
at all. The pool knew there was no medium-duration evidence and took it anyway.

Both are fixed at the earliest point: the terms no longer carry bare words, and
a candidate must now state every facet the farmer asked for, read off ITS OWN
evidence lines — so evidence belonging to BRRI Dhan 69 can never validate
CR Dhan 100.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

# The four rows from the report, written the way a varietal sheet writes them:
# a name on its own line, its properties under it.
SHEET = [
    "BRRI Dhan 69\nMedium-duration\nSuitable for lowland\nMedium-bold grain\n"
    "Medium height\n",
    "Bina Dhan 17\nSuitable for irrigated mid-early conditions\n",
    "CR Dhan 310\nSuitable for irrigated and favourable shallow rainfed areas\n",
    "CR Dhan 100\nFit for upland\n",
]
# The two wordings from the report: hyphenated with "my land", and plain.
HYPHENATED = "Which medium-duration rice variety is good to plant in my land?"
PLAIN = "Which medium duration rice varieties are suitable?"


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet",
                                      "document_id": index}}
            for index, c in enumerate(chunks or SHEET)]


def pool(question, *chunks):
    return [c["name"] for c in entity_intent.candidate_pool(
        entity_intent.analyse(question), hits(*chunks))]


class TheReportedQueryTests(SimpleTestCase):
    """Only the variety the source calls medium-duration comes back."""

    def test_the_hyphenated_question_returns_only_the_medium_duration_variety(self):
        self.assertEqual(["BRRI Dhan 69"], pool(HYPHENATED))

    def test_the_plain_question_returns_the_same(self):
        self.assertEqual(["BRRI Dhan 69"], pool(PLAIN))

    def test_both_wordings_agree(self):
        self.assertEqual(pool(PLAIN), pool(HYPHENATED))

    def test_the_varieties_from_the_report_are_each_rejected(self):
        names = pool(HYPHENATED)

        for wrongly_returned in ("Bina Dhan 17", "CR Dhan 310", "CR Dhan 100"):
            self.assertNotIn(wrongly_returned, names)

    def test_nothing_unrelated_even_reaches_the_pool(self):
        # With the bare words gone, the land and grain rows no longer match a
        # term at all, so they are never candidates in the first place.
        report: dict = {}
        entity_intent.candidate_pool(entity_intent.analyse(HYPHENATED),
                                     hits(), report)

        self.assertEqual(["BRRI Dhan 69"],
                         [c["name"] for c in report["accepted"]])

    def test_the_facet_check_rejects_and_says_which_facet_is_missing(self):
        # A two-facet question: the upland row matches the land term and gets
        # looked at, then fails because nothing states its duration.
        question = "Which medium-duration variety is suitable for upland?"
        report: dict = {}
        pool_names = [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(question),
            hits("CR Dhan 100\nFit for upland\n",
                 "DRR 44\nMedium duration\nFit for upland\n"), report)]
        reasons = " | ".join(r["reason"] for r in report["rejected"])

        self.assertEqual(["DRR 44"], pool_names)
        self.assertIn("no evidence for the requested", reasons)
        self.assertIn("maturity=medium_duration", reasons)


class TermLeakTests(SimpleTestCase):
    """The bare words that let the unrelated rows in are gone."""

    def test_a_hyphenated_facet_does_not_leave_its_words_behind(self):
        terms = entity_intent.analyse(HYPHENATED).condition_terms

        for leaked in ("medium", "duration", "land"):
            self.assertNotIn(leaked, terms)

    def test_the_facets_family_is_still_there(self):
        terms = entity_intent.analyse(HYPHENATED).condition_terms

        self.assertIn("medium duration", terms)
        self.assertIn("medium-duration", terms)
        self.assertIn("medium maturity", terms)

    def test_my_land_is_not_read_as_a_land_type(self):
        intent = entity_intent.analyse(HYPHENATED)

        self.assertEqual("", intent.land_type)
        self.assertEqual("medium_duration", intent.maturity)
        self.assertTrue(intent.land_type_missing)


class OtherAttributesAreNotEvidenceTests(SimpleTestCase):
    """Land, irrigation and grain rows do not answer a duration question."""

    def test_upland_lowland_and_irrigated_only_rows_are_rejected(self):
        for chunk in ("CR Dhan 100\nFit for upland\n",
                      "Swarna Sub1\nSuitable for lowland\n",
                      "Bina Dhan 17\nSuitable for irrigated mid-early conditions\n",
                      "CR Dhan 310\nSuitable for favourable shallow rainfed areas\n",
                      "MTU 1010\nSuitable for medium land\n"):
            with self.subTest(chunk=chunk.replace("\n", " | ")):
                self.assertEqual([], pool(HYPHENATED, chunk))

    def test_medium_bold_and_medium_height_are_not_medium_duration(self):
        for chunk in ("BRRI Dhan 69\nMedium-bold grain\n",
                      "BRRI Dhan 69\nMedium height\n",
                      "BRRI Dhan 69\nMedium slender grain\n"):
            with self.subTest(chunk=chunk.replace("\n", " | ")):
                self.assertEqual([], pool(HYPHENATED, chunk))

    def test_medium_land_and_medium_duration_stay_separate(self):
        sheet = ("MTU 1010\nSuitable for medium land\n"
                 "BRRI Dhan 69\nMedium-duration\n")

        self.assertEqual(["BRRI Dhan 69"], pool(HYPHENATED, sheet))
        self.assertEqual(["MTU 1010"],
                         pool("Which variety is suitable for medium land?", sheet))

    def test_a_row_that_states_both_answers_both(self):
        sheet = "CR Dhan 910\nMedium duration\nSuitable for medium land\n"

        self.assertEqual(["CR Dhan 910"], pool(HYPHENATED, sheet))
        self.assertEqual(["CR Dhan 910"],
                         pool("Which variety is suitable for medium land?", sheet))


class EvidenceBelongsToOneVarietyTests(SimpleTestCase):
    """One variety's evidence can never validate another."""

    def test_a_neighbouring_records_duration_does_not_carry_over(self):
        sheet = ("BRRI Dhan 69\nMedium-duration\n"
                 "CR Dhan 100\nFit for upland\n"
                 "Bina Dhan 17\nSuitable for irrigated conditions\n")

        self.assertEqual(["BRRI Dhan 69"], pool(HYPHENATED, sheet))

    def test_the_same_holds_when_the_rows_are_in_one_chunk(self):
        sheet = ("Rice varieties for Odisha\n"
                 "BRRI Dhan 69 is a medium-duration variety.\n"
                 "CR Dhan 100 is fit for upland.\n"
                 "CR Dhan 310 is suitable for irrigated areas.\n")

        self.assertEqual(["BRRI Dhan 69"], pool(HYPHENATED, sheet))

    def test_a_heading_naming_the_facet_does_not_vouch_for_the_table(self):
        sheet = ("Medium-duration varieties\n"
                 "CR Dhan 100\nFit for upland\n"
                 "Bina Dhan 17\nSuitable for irrigated conditions\n")

        self.assertEqual([], pool(HYPHENATED, sheet))


class GenerationOnlySeesTheFilteredPoolTests(SimpleTestCase):
    """The answer step is handed the filtered pool and nothing else."""

    def prompt_for(self, question, *chunks):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(question, hits(*chunks))
        return seen["sys"]

    def test_only_the_supported_variety_is_listed_for_the_model(self):
        prompt = self.prompt_for(HYPHENATED)
        note = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split("\nCONTEXT:")[0]

        self.assertIn("BRRI Dhan 69", note)
        for excluded in ("Bina Dhan 17", "CR Dhan 310", "CR Dhan 100"):
            self.assertNotIn(excluded, note)

    def test_the_model_is_told_not_to_add_a_variety_of_its_own(self):
        prompt = self.prompt_for(HYPHENATED)

        self.assertIn("Do not add a variety that is not in this list", prompt)

    def test_no_list_is_offered_when_nothing_states_the_facet(self):
        prompt = self.prompt_for(HYPHENATED, "CR Dhan 100\nFit for upland\n")

        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)


class ExistingBehaviourUnchangedTests(SimpleTestCase):
    """The land-type, upland and singular/plural work must not have moved."""

    def test_the_upland_record_fix_still_holds(self):
        chunk = "CR Dhan 100\nFit for upland\n"

        self.assertEqual(["CR Dhan 100"],
                         pool("Which rice variety is suitable for high land?", chunk))
        self.assertEqual(["CR Dhan 100"],
                         pool("Which rice varieties are suitable for high land?", chunk))

    def test_lowland_still_excludes_upland(self):
        sheet = "CR Dhan 100\nFit for upland\nSwarna Sub1\nSuitable for lowland\n"

        self.assertEqual(["Swarna Sub1"],
                         pool("Which variety is suitable for lowland?", sheet))

    def test_a_stress_question_is_unaffected(self):
        sheet = ("Sahbhagi Dhan is drought tolerant.\n"
                 "CR Dhan 100 is fit for upland.\n")

        self.assertEqual(["Sahbhagi Dhan"],
                         pool("Which varieties are drought tolerant?", sheet))

    def test_a_question_with_no_facet_keeps_its_old_behaviour(self):
        # Nothing structured was asked for, so nothing extra is required; the
        # free-text condition still selects on its own.
        sheet = "CR Dhan 907\nSuitable for puffed rice\n"

        self.assertEqual(["CR Dhan 907"],
                         pool("Which variety is good for puffed rice?", sheet))

    def test_singular_and_plural_still_agree(self):
        for singular, plural in (
                ("Which medium-duration variety is good?",
                 "Which medium-duration varieties are good?"),
                ("Which variety is suitable for upland?",
                 "Which varieties are suitable for upland?")):
            with self.subTest(question=singular):
                self.assertEqual(pool(plural), pool(singular))
