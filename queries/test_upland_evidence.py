""""Fit for upland" is an answer to "which variety suits high land?".

THE REPORTED FAILURE
--------------------
The farmer asked which variety suits HIGH LAND. The context held a varietal
sheet saying, of CR Dhan 100, "Fit for upland" — and the answer came back "the
information available does not specify a variety suitable for high land."

The synonym layer was not at fault: high land already canonicalised to upland,
and the intent already carried land_type=upland. What broke was the LAST step.
A PDF table flattened to text puts the variety in one cell and its properties in
the next, so the context reads

    CR Dhan 100
    Fit for upland

and the collector required the name and the evidence on the SAME line. The
evidence was retrieved, read, and dropped for having no name on its line.

Two fixes are asserted here: a name-only line opens a record that the lines
under it belong to, and the canonical term is substituted INTO the question so
the index is also searched the way the corpus writes it.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

# The reported layout, verbatim in shape: name alone on its line.
TABLE = "CR Dhan 100\nFit for upland\nDuration 110 days\nYield 4.0 t/ha\n"
# The other ways a sheet writes the same thing.
PHRASINGS = (
    "CR Dhan 100\nFit for upland\n",
    "CR Dhan 100\nSuitable for upland\n",
    "CR Dhan 100\nRecommended for upland\n",
    "CR Dhan 100\nRecommended for cultivation in upland ecosystem\n",
    "CR Dhan 100 - Fit for upland.\n",
    "Variety: CR Dhan 100\nEcosystem: fit for upland\n",
)
HIGH_LAND = "Which rice variety is suitable for high land?"
HIGH_LAND_PLURAL = "Which rice varieties are suitable for high land?"


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "Varietal sheet"}}
            for c in chunks or (TABLE,)]


def pool(question, *chunks):
    return [c["name"] for c in entity_intent.candidate_pool(
        entity_intent.analyse(question), hits(*chunks))]


class TheReportedCaseTests(SimpleTestCase):
    """High land, "Fit for upland", one line apart."""

    def test_the_high_land_question_finds_the_upland_variety(self):
        self.assertEqual(["CR Dhan 100"], pool(HIGH_LAND, TABLE))

    def test_both_wordings_find_it(self):
        self.assertEqual(pool(HIGH_LAND, TABLE), pool(HIGH_LAND_PLURAL, TABLE))
        self.assertEqual(["CR Dhan 100"], pool(HIGH_LAND_PLURAL, TABLE))

    def test_every_way_the_sheet_writes_it_is_accepted(self):
        for chunk in PHRASINGS:
            with self.subTest(chunk=chunk.replace("\n", " | ")):
                self.assertEqual(["CR Dhan 100"], pool(HIGH_LAND, chunk))

    def test_the_farmers_wording_does_not_matter(self):
        for wording in ("high land", "highland", "high-land", "dry land", "upland"):
            with self.subTest(wording=wording):
                question = f"Which rice variety is suitable for {wording}?"

                self.assertEqual("upland", entity_intent.analyse(question).land_type)
                self.assertEqual(["CR Dhan 100"], pool(question, TABLE))

    def test_the_answer_is_given_the_variety_and_its_evidence(self):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "ok"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer(HIGH_LAND, hits(TABLE))

        self.assertIn("CR Dhan 100", seen["sys"])
        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", seen["sys"])


class RecordScopeTests(SimpleTestCase):
    """A name-only line vouches for the lines under it — and only those."""

    def test_each_record_keeps_its_own_evidence(self):
        sheet = ("CR Dhan 100\nFit for upland\n"
                 "Swarna Sub1\nSuitable for lowland\n")

        self.assertEqual(["CR Dhan 100"], pool(HIGH_LAND, sheet))
        self.assertEqual(["Swarna Sub1"],
                         pool("Which variety is suitable for lowland?", sheet))

    def test_a_sentence_about_a_variety_does_not_open_a_record(self):
        # The guard against over-attribution: prose that merely follows a name
        # is not that name's property sheet.
        sheet = ("CR Dhan 100 is a variety released in 2015.\n"
                 "Upland fields are suitable for direct seeding.\n")

        self.assertEqual([], pool(HIGH_LAND, sheet))

    def test_a_record_lapses_after_a_dozen_lines(self):
        filler = "".join(f"Note line {i}\n" for i in range(14))
        sheet = "CR Dhan 100\n" + filler + "Fit for upland\n"

        self.assertEqual([], pool(HIGH_LAND, sheet))

    def test_a_record_does_not_reach_across_chunks(self):
        self.assertEqual([], pool(HIGH_LAND, "CR Dhan 100\n", "Fit for upland\n"))

    def test_a_denial_under_a_record_is_still_a_denial(self):
        sheet = "CR Dhan 100\nNot suitable for upland\n"

        self.assertEqual([], pool(HIGH_LAND, sheet))

    def test_a_bare_field_value_counts_inside_a_record(self):
        # The header already said whose sheet this is, so "Ecosystem: upland"
        # needs no "is suitable for" to be the document's claim.
        for chunk in ("CR Dhan 100\nUpland\n",
                      "CR Dhan 100\nEcosystem: upland\n",
                      "CR Dhan 100\nCultivation in upland ecosystem\n"):
            with self.subTest(chunk=chunk.replace("\n", " | ")):
                self.assertEqual(["CR Dhan 100"], pool(HIGH_LAND, chunk))

    def test_a_bare_mention_outside_a_record_still_needs_a_support_word(self):
        # No header, so nearness is not assertion and prose stays prose.
        self.assertEqual([], pool(HIGH_LAND,
                                  "Upland\nCR Dhan 100 was released in 2015.\n"))

    def test_the_line_speaks_for_itself_when_it_names_a_variety(self):
        # The record header must not steal evidence that already has an owner.
        sheet = ("CR Dhan 100\nDuration 110 days\n"
                 "Sahbhagi Dhan is fit for upland.\n")

        self.assertEqual(["Sahbhagi Dhan"], pool(HIGH_LAND, sheet))


class StructuredMatchingStillStrictTests(SimpleTestCase):
    """Widening how evidence is ATTACHED must not widen what counts as evidence."""

    def test_upland_is_still_a_land_type_and_matches_nothing_else(self):
        for chunk in ("CR Dhan 100\nMedium duration variety\n",
                      "CR Dhan 100\nMedium slender grain\n",
                      "CR Dhan 100\nSuitable for lowland\n",
                      "CR Dhan 100\nUpland areas of Odisha grow rice\n"):
            with self.subTest(chunk=chunk.replace("\n", " | ")):
                self.assertEqual([], pool(HIGH_LAND, chunk))

    def test_a_record_does_not_make_a_heading_vouch_for_a_table(self):
        # The original guard: a document ABOUT upland varieties does not make
        # every name printed in it an upland variety.
        sheet = ("Varieties suitable for upland\n"
                 "CR Dhan 100\nDuration 110 days\n"
                 "Swarna Sub1\nDuration 145 days\n")

        self.assertEqual([], pool(HIGH_LAND, sheet))

    def test_the_facets_stay_apart_under_a_record(self):
        sheet = "CR Dhan 910\nMedium duration\nSuitable for shallow lowland\n"

        self.assertEqual([], pool("Which variety is suitable for medium land?", sheet))
        self.assertEqual(["CR Dhan 910"],
                         pool("Which variety is medium duration?", sheet))


class RetrievalVariantTests(SimpleTestCase):
    """The index is searched the way the corpus writes it, too."""

    def test_the_canonical_term_is_substituted_into_the_question(self):
        variants = entity_intent.retrieval_variants(HIGH_LAND)

        self.assertIn("Which rice variety is suitable for upland?", variants)
        self.assertIn("Which rice varieties are suitable for upland?", variants)
        self.assertIn("Which rice varieties are suitable for high land?", variants)
        self.assertIn("rice varieties recommended for upland", variants)
        self.assertIn("rice variety suitable for upland", variants)

    def test_the_original_wording_is_never_dropped(self):
        with patch.object(pipeline, "_chat", side_effect=RuntimeError("no LLM")):
            expanded = pipeline._expand_queries(HIGH_LAND)

        self.assertEqual(HIGH_LAND, expanded[0])
        self.assertTrue(any("upland" in q for q in expanded), expanded)

    def test_both_wordings_search_the_same_set_of_strings(self):
        one = {HIGH_LAND, *entity_intent.retrieval_variants(HIGH_LAND)}
        many = {HIGH_LAND_PLURAL,
                *entity_intent.retrieval_variants(HIGH_LAND_PLURAL)}

        self.assertEqual(many, one)


class DiagnosticsTests(SimpleTestCase):
    """Why a candidate was accepted or dropped, without re-running the pipeline."""

    def test_the_report_names_what_was_accepted(self):
        report: dict = {}
        entity_intent.candidate_pool(entity_intent.analyse(HIGH_LAND),
                                     hits(TABLE), report)

        self.assertTrue(report["evidence_found"])
        self.assertEqual(["CR Dhan 100"], [c["name"] for c in report["accepted"]])
        self.assertEqual("Fit for upland", report["accepted"][0]["evidence"])
        self.assertEqual("upland", report["accepted"][0]["conditions"]["land_type"])
        self.assertIn("upland", report["condition_terms"])
        self.assertIn("high land", report["condition_terms"])

    def test_the_report_says_why_a_line_was_dropped(self):
        report: dict = {}
        entity_intent.candidate_pool(
            entity_intent.analyse(HIGH_LAND),
            hits("CR Dhan 100 is not suitable for upland.\n"
                 "Sahbhagi Dhan matures in 105 days.\n"), report)

        self.assertFalse(report["evidence_found"])
        reasons = " | ".join(r["reason"] for r in report["rejected"])
        self.assertIn("denies the condition", reasons)
        self.assertIn("no condition term", reasons)
