"""A "which varieties exist?" question was answering with two or three names.

The varietal data lives in ONE document — a sheet of 183 rows across 39 chunks.
`_diversify` caps how many chunks any single document may contribute, at
`n_results // 3`. For an entity question at n_results=20 that is SIX chunks,
roughly 28 rows of 183, so after narrowing to the asked duration class only a
couple of names survived. The farmer was shown two of dozens, with no sign that
the rest were never read.

The cap itself is right and stays: without it one chatty document crowds out the
sheet that actually answers the question. What was wrong is applying the SAME
cap to a question whose answer is, by definition, a list held in one place.

Measured on the real shape (39 sheet chunks, 40 competing documents):

    n=20 cap=6  (old)  ->  6 sheet chunks   ~28 of 183 varieties
    n=20 no cap        -> 20 sheet chunks   ~94, but ZERO other documents
    n=30 cap=20 (new)  -> 20 sheet chunks   ~94, and 10 slots still elsewhere

Removing the cap was rejected on that middle line: the per-variety PDFs and the
FAQ rows would have been starved, losing names that live only there.
"""
from django.test import SimpleTestCase

from queries.services.pipeline import _diversify, _missing_glossary_terms


def shape(sheet_chunks=39, other_docs=40):
    """One big varietal sheet ranked first, then many competing documents."""
    ranked = [f"v{i}" for i in range(sheet_chunks)] + [f"o{i}" for i in range(other_docs)]
    holder = {f"v{i}": {"metadata": {"document_id": 1}} for i in range(sheet_chunks)}
    holder.update({f"o{i}": {"metadata": {"document_id": 100 + i}}
                   for i in range(other_docs)})
    return ranked, holder


def split(chosen):
    """(chunks from the sheet, chunks from anywhere else)."""
    sheet = sum(1 for k in chosen if k.startswith("v"))
    return sheet, len(chosen) - sheet


class TheDefaultCapIsUnchangedTests(SimpleTestCase):
    """Every question that is not about entities retrieves exactly as before."""

    def test_the_formula_still_applies_when_no_cap_is_given(self):
        ranked, holder = shape()
        sheet, _other = split(_diversify(ranked, holder, 20))

        self.assertEqual(6, sheet, "the default cap changed")

    def test_the_small_window_is_unchanged(self):
        ranked, holder = shape()
        sheet, other = split(_diversify(ranked, holder, 10))

        self.assertEqual(3, sheet)
        self.assertEqual(7, other)

    def test_the_floor_of_three_still_holds(self):
        # n_results // 3 would be 1 here; the floor keeps it at 3.
        ranked, holder = shape()
        sheet, _ = split(_diversify(ranked, holder, 5))

        self.assertEqual(3, sheet)


class TheEntityWindowTests(SimpleTestCase):
    """What an entity question now gets, and what it must not take."""

    def test_the_sheet_contributes_far_more(self):
        ranked, holder = shape()
        sheet, _other = split(_diversify(ranked, holder, 30, 20))

        self.assertEqual(20, sheet)
        self.assertGreater(sheet, 6, "no better than the old default")

    def test_other_documents_are_not_starved(self):
        # The reason the cap was not simply removed. Names that live only in a
        # per-variety PDF or an FAQ row must still reach the model.
        ranked, holder = shape()
        _sheet, other = split(_diversify(ranked, holder, 30, 20))

        self.assertEqual(10, other)

    def test_removing_the_cap_entirely_would_starve_them(self):
        # Asserted so the rejected option stays rejected on purpose.
        ranked, holder = shape()
        _sheet, other = split(_diversify(ranked, holder, 20, 20))

        self.assertEqual(0, other)

    def test_a_second_varietal_document_still_gets_in(self):
        # Two sheets, both relevant: neither may lock the other out.
        ranked = [f"a{i}" for i in range(30)] + [f"b{i}" for i in range(30)]
        holder = {f"a{i}": {"metadata": {"document_id": 1}} for i in range(30)}
        holder.update({f"b{i}": {"metadata": {"document_id": 2}} for i in range(30)})
        chosen = _diversify(ranked, holder, 30, 20)
        from_b = sum(1 for k in chosen if k.startswith("b"))

        self.assertEqual(10, from_b)


class NothingIsLostTests(SimpleTestCase):
    """The cap holds chunks back; it must never drop them while there is room."""

    def test_overflow_refills_when_there_are_few_competitors(self):
        # One sheet, almost nothing else: the window fills from the sheet.
        ranked, holder = shape(sheet_chunks=39, other_docs=2)
        sheet, other = split(_diversify(ranked, holder, 30, 20))

        self.assertEqual(30, sheet + other)
        self.assertEqual(28, sheet, "held-back chunks were dropped, not appended")

    def test_the_window_is_filled_whenever_enough_chunks_exist(self):
        ranked, holder = shape()
        for n, cap in ((10, None), (20, None), (30, 20), (20, 20)):
            with self.subTest(n=n, cap=cap):
                self.assertEqual(n, len(_diversify(ranked, holder, n, cap)))

    def test_a_chunk_with_no_document_id_is_never_capped(self):
        # FAQ context is injected without a document_id; it must not be counted
        # against any document's share.
        ranked = [f"x{i}" for i in range(10)]
        holder = {k: {"metadata": {}} for k in ranked}

        self.assertEqual(10, len(_diversify(ranked, holder, 10, 2)))


class APluralIsNotADroppedTermTests(SimpleTestCase):
    """One plural cost a question its meaning, four stages downstream.

    Live, for "ଶହେ ପାଞ୍ଚ ଦିନିଆ ଧାନ କିସମ ଗୁଡ଼ିକର ନାମ କୁହନ୍ତୁ।" (name the 105-day
    varieties):

      1. the glossary forced "rice variety" into the sentence;
      2. Google returned "Name the 105-day rice varieties." — correct;
      3. this check looked for the literal "variety", which is NOT a substring
         of "varieties", and reported the term dropped;
      4. so the OpenAI fallback ran and rephrased it as "the rice varieties
         that take 105 days";
      5. and entity_intent read that as asking about "take" and "days".

    Ordinary English endings only. Accepting more would start accepting a
    different word, which is the failure this check exists to catch.
    """

    def missing(self, translated, term):
        return _missing_glossary_terms(translated, {"odia": term})

    def test_a_plural_still_counts_as_present(self):
        self.assertEqual([], self.missing("Name the 105-day rice varieties.",
                                          "rice variety"))

    def test_the_other_ordinary_endings_too(self):
        for translated, term in (("apply two doses", "dose"),
                                 ("fertilizer applications at tillering",
                                  "fertilizer application"),
                                 ("check the fields", "field")):
            with self.subTest(term=term):
                self.assertEqual([], self.missing(translated, term))

    def test_the_singular_is_unaffected(self):
        self.assertEqual([], self.missing("Name the 105-day rice variety.",
                                          "rice variety"))

    def test_a_genuinely_missing_term_is_still_reported(self):
        # The whole point of the check: a term Google rebuilt the sentence
        # around must still route to the fallback.
        self.assertEqual(["rice variety"],
                         self.missing("Name the 105-day rice types.", "rice variety"))

    def test_one_field_operation_swapped_for_another_is_still_caught(self):
        self.assertEqual(["transplanting"],
                         self.missing("when to harvest", "transplanting"))

    def test_a_different_word_that_merely_starts_the_same_is_not_a_match(self):
        # "varied" must not satisfy "variety" — the trailing boundary is what
        # stops the tolerance becoming a wildcard.
        self.assertEqual(["variety"], self.missing("varied results", "variety"))

    def test_every_word_of_a_multi_word_term_must_survive(self):
        self.assertEqual(["rice variety"], self.missing("Name the varieties.",
                                                        "rice variety"))
