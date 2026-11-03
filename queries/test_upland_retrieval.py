"""The upland varietal chunk has to REACH the answer step.

WHAT THE TRACE FOUND
--------------------
"Which rice variety is suitable for high land?" returned "the information
available does not specify any variety", and the CR Dhan 100 chunk was not in
the context. Tracing it stage by stage (manage.py trace_retrieval) showed the
chunk missing from every phrasing's vector list AND every phrasing's keyword
list — so the loss was upstream of ranking. The indexed text for that document
turned out to be

    Cr Dhan 100 / Imported from CR-Dhan-100.pdf / All India / Rice / Advisory

i.e. its metadata, because the PDF is not in storage. That is covered by
catalogues/tests (documents indexed without their text are now reported).

The trace also showed two real retrieval faults, which these tests pin down:
a canonical variant dropped outright by the expansion cap, and fusion reading
rank off one globally-sorted list in which short canonical queries always sink.

The searches here run against a controlled in-memory index, so the assertions
are about the pipeline's own logic and never about a live corpus.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import entity_intent, pipeline

# A small corpus: one varietal sheet per ecosystem, plus FAQ-style rows of the
# kind that crowded the real context out.
CORPUS = [
    ("Cr Dhan 100", "CR Dhan 100\nFit for upland\nDuration 110 days\n"),
    ("Sahbhagi Dhan", "Sahbhagi Dhan\nSuitable for upland and drought prone areas\n"),
    ("Mtu 1010", "MTU 1010\nSuitable for medium land\nDuration 125 days\n"),
    ("Cr Dhan 910", "CR Dhan 910\nMedium duration\nSuitable for shallow lowland\n"),
    ("Swarna Sub1", "Swarna Sub1\nSuitable for lowland\n"),
    ("FAQ 19", "19. Which variety is best for fallow land?\nQuestion: fallow land variety\n"),
    ("FAQ 20", "20. Which paddy is suitable for medium land this year?\nQuestion: medium land\n"),
    ("FAQ 6", "6. Is Veena paddy variety good for cultivation?\nQuestion: Veena variety\n"),
    ("FAQ 22", "22. Suggest long-duration, waterlogging-tolerant varieties\nQuestion: lowland\n"),
    ("FAQ 5", "5. Which variety is best for lowland fields?\nQuestion: lowland fields\n"),
]


def _tokens(text):
    import re
    return set(re.findall(r"[a-z]+", text.lower()))


class FakeIndex:
    """A deterministic stand-in for Chroma + BM25.

    Vector "distance" is 1 - token overlap, and the keyword score is a raw term
    count. Both are deliberately SCALE-DEPENDENT on query length, which is the
    property that broke the old fusion: a short canonical query scores lower in
    absolute terms than a long natural one, however well it matches.
    """

    def __init__(self, corpus=CORPUS):
        self.corpus = [
            {"chunk": text,
             "metadata": {"document_id": index, "document_title": title,
                          "chunk_index": 0}}
            for index, (title, text) in enumerate(corpus)]

    def _scored(self, query):
        wanted = _tokens(query)
        out = []
        for row in self.corpus:
            overlap = len(wanted & _tokens(row["chunk"]))
            if overlap:
                out.append((overlap, row))
        out.sort(key=lambda pair: -pair[0])
        return out

    def vector(self, _catalogue, queries, n_results=6):
        results = []
        for query in queries:
            hits = [dict(row, distance=round(1.0 - overlap / max(len(_tokens(query)), 1), 4))
                    for overlap, row in self._scored(query)]
            results.append(hits[:n_results])
        return results

    def keyword(self, _catalogue, query, n_results=10):
        # Absolute BM25-like magnitude grows with query length, exactly as the
        # real scorer's does.
        return [dict(row, score=float(overlap), distance=None)
                for overlap, row in self._scored(query)][:n_results]


class RetrievalTests(SimpleTestCase):
    """What comes back for each land type, and in what company."""

    def setUp(self):
        self.index = FakeIndex()
        patches = [
            patch.object(pipeline, "search_catalogue_batch", self.index.vector),
            patch.object(pipeline, "keyword_search", self.index.keyword),
            # No LLM in tests: expansion keeps the original plus the
            # deterministic variants, which is the half under test.
            patch.object(pipeline, "_chat", side_effect=RuntimeError("no LLM")),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def context_for(self, question, n_results=10):
        expanded = pipeline._expand_queries(question)
        hits = pipeline.retrieve(None, question, n_results=n_results, queries=expanded)
        return [(h["metadata"]["document_title"], h) for h in hits]

    def titles_for(self, question, n_results=10):
        return [title for title, _hit in self.context_for(question, n_results)]

    def pool_for(self, question, n_results=10):
        hits = [hit for _title, hit in self.context_for(question, n_results)]
        return [c["name"] for c in entity_intent.candidate_pool(
            entity_intent.analyse(question), hits)]

    # A ---------------------------------------------------------------
    def test_a_high_land_question_retrieves_the_upland_sheet_and_accepts_it(self):
        question = "Which rice variety is suitable for high land?"

        self.assertIn("Cr Dhan 100", self.titles_for(question))
        self.assertIn("CR Dhan 100", self.pool_for(question))

    # B ---------------------------------------------------------------
    def test_the_plural_wording_retrieves_and_accepts_the_same(self):
        singular = "Which rice variety is suitable for high land?"
        plural = "Which rice varieties are suitable for high land?"

        self.assertEqual(set(self.titles_for(singular)), set(self.titles_for(plural)))
        self.assertEqual(self.pool_for(singular), self.pool_for(plural))
        self.assertIn("CR Dhan 100", self.pool_for(plural))

    # C ---------------------------------------------------------------
    def test_the_canonical_wording_behaves_the_same(self):
        self.assertIn("CR Dhan 100", self.pool_for("Which variety is suitable for upland?"))
        self.assertIn("CR Dhan 100", self.pool_for("Which varieties are suitable for upland?"))

    # D ---------------------------------------------------------------
    def test_dry_land_reaches_the_upland_varieties_too(self):
        pool = self.pool_for("Which varieties are suitable for dry land?")

        self.assertIn("CR Dhan 100", pool)
        self.assertIn("Sahbhagi Dhan", pool)

    # E ---------------------------------------------------------------
    def test_medium_land_still_retrieves_medium_land_varieties(self):
        question = "Which rice variety is suitable for medium land?"

        self.assertIn("Mtu 1010", self.titles_for(question))
        self.assertEqual(["MTU 1010"], self.pool_for(question))

    # F ---------------------------------------------------------------
    def test_medium_duration_does_not_accept_a_medium_land_variety(self):
        pool = self.pool_for("Which varieties are medium duration?")

        self.assertIn("CR Dhan 910", pool)
        self.assertNotIn("MTU 1010", pool)

    # G ---------------------------------------------------------------
    def test_lowland_does_not_accept_upland_varieties(self):
        pool = self.pool_for("Which varieties are suitable for lowland?")

        self.assertIn("Swarna Sub1", pool)
        self.assertNotIn("CR Dhan 100", pool)
        self.assertNotIn("Sahbhagi Dhan", pool)

    # H ---------------------------------------------------------------
    def test_an_explicit_single_request_still_asks_for_one(self):
        intent = entity_intent.analyse("Recommend only one variety for upland.")

        self.assertEqual("single_explicit", intent.quantity)
        self.assertIn("CR Dhan 100",
                      self.pool_for("Recommend only one variety for upland."))

    def test_one_document_cannot_fill_the_whole_context(self):
        # Ten near-identical rows of one FAQ document, the shape that crowded
        # the real context out.
        crowd = [("FAQ pack", f"{i}. Which variety is suitable for upland?\n")
                 for i in range(10)]
        index = FakeIndex(crowd + [("Cr Dhan 100", "CR Dhan 100\nFit for upland\n")])
        # One document id per row is what the real index gives; here the crowd
        # shares a title but not an id, so cap by id the way retrieval does.
        for row in index.corpus[:10]:
            row["metadata"]["document_id"] = 99
        with patch.object(pipeline, "search_catalogue_batch", index.vector), \
                patch.object(pipeline, "keyword_search", index.keyword):
            hits = pipeline.retrieve(None, "Which variety is suitable for upland?",
                                     n_results=6,
                                     queries=["Which variety is suitable for upland?"])

        titles = [h["metadata"]["document_title"] for h in hits]
        self.assertIn("Cr Dhan 100", titles)
        self.assertLessEqual(titles.count("FAQ pack"), 3)


class QueryExpansionTests(SimpleTestCase):
    """The canonical variants must survive the cap."""

    def expand(self, question):
        with patch.object(pipeline, "_chat",
                          side_effect=lambda *a, **k: '["rewrite one","rewrite two",'
                                                      '"rewrite three"]'):
            return pipeline._expand_queries(question)

    def test_every_canonical_variant_reaches_the_index(self):
        question = "Which rice variety is suitable for high land?"
        expanded = self.expand(question)

        self.assertEqual(question, expanded[0])
        for variant in entity_intent.retrieval_variants(question):
            self.assertIn(variant, expanded)

    def test_the_canonical_variants_come_before_the_llm_rewrites(self):
        question = "Which rice variety is suitable for high land?"
        expanded = self.expand(question)
        variants = entity_intent.retrieval_variants(question)

        last_canonical = max(expanded.index(v) for v in variants)
        rewrites = [expanded.index(q) for q in ("rewrite one", "rewrite two")
                    if q in expanded]
        for position in rewrites:
            self.assertGreater(position, last_canonical)

    def test_an_ordinary_question_is_expanded_as_before(self):
        expanded = self.expand("How much urea should I apply at tillering?")

        self.assertEqual("How much urea should I apply at tillering?", expanded[0])
        self.assertLessEqual(len(expanded), 6)


class FusionTests(SimpleTestCase):
    """Rank is comparable across phrasings; raw scores are not."""

    def test_a_chunk_only_the_canonical_phrasing_finds_is_kept(self):
        # The long natural question matches the FAQ rows; only the short
        # canonical phrasing reaches the varietal sheet. Under the old fusion
        # the sheet sank — its distance was numerically worse — and was cut.
        index = FakeIndex()
        with patch.object(pipeline, "search_catalogue_batch", index.vector), \
                patch.object(pipeline, "keyword_search", index.keyword):
            hits = pipeline.retrieve(
                None, "Which rice variety is suitable for high land?", n_results=6,
                queries=["Which rice variety is suitable for high land?",
                         "rice variety suitable for upland"])

        titles = [h["metadata"]["document_title"] for h in hits]
        self.assertIn("Cr Dhan 100", titles)

    def test_every_phrasing_keeps_its_best_hit(self):
        index = FakeIndex()
        queries = ["Which variety is best for lowland fields?",
                   "rice variety suitable for upland"]
        with patch.object(pipeline, "search_catalogue_batch", index.vector), \
                patch.object(pipeline, "keyword_search", index.keyword):
            hits = pipeline.retrieve(None, queries[0], n_results=4, queries=queries)

        titles = [h["metadata"]["document_title"] for h in hits]
        self.assertIn("Cr Dhan 100", titles)          # the upland phrasing's own best
        self.assertTrue({"Swarna Sub1", "FAQ 5", "FAQ 22"} & set(titles))

    def test_an_empty_index_returns_nothing_rather_than_raising(self):
        with patch.object(pipeline, "search_catalogue_batch",
                          lambda *a, **k: [[] for _ in a[1]]), \
                patch.object(pipeline, "keyword_search", lambda *a, **k: []):
            self.assertEqual([], pipeline.retrieve(None, "anything", n_results=5,
                                                   queries=["anything"]))


class EvidenceAcceptanceTests(SimpleTestCase):
    """The diagnostic the brief asked for: given the chunk, is it accepted?"""

    def test_when_the_chunk_is_present_the_variety_is_accepted(self):
        hits = [{"chunk": "CR Dhan 100\nFit for upland\n",
                 "metadata": {"document_title": "Cr Dhan 100", "document_id": 1}}]
        report: dict = {}
        pool = entity_intent.candidate_pool(
            entity_intent.analyse("Which rice variety is suitable for high land?"),
            hits, report)

        self.assertEqual(["CR Dhan 100"], [c["name"] for c in pool])
        self.assertTrue(report["evidence_found"])
        self.assertEqual("Fit for upland", pool[0]["evidence"])
