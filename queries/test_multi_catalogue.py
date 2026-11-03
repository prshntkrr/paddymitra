"""Retrieving across more than one catalogue, and merging what comes back.

Each catalogue is searched by the existing retrieve(), so nothing here re-tests
ranking within one corpus. What is tested is the part that is new: that the
lists are merged without one corpus dominating, that the same paragraph is not
shown twice, that provenance survives, and that a single catalogue still takes
the path it always took.
"""
from unittest.mock import patch

from django.test import TestCase

from catalogues.models import Catalogue
from queries.services import pipeline


def hit(text, title="doc"):
    return {"chunk": text, "metadata": {"document_title": title}, "score": 0.9}


class RetrieveAcrossCataloguesTests(TestCase):

    def setUp(self):
        self.nutrient = Catalogue.objects.create(pk=5, name="Nutrient Management")
        self.variety = Catalogue.objects.create(pk=6, name="Varietal recommendation")
        self.pest = Catalogue.objects.create(pk=7, name="Insect-pest and disease")

    def fake_retrieve(self, per_catalogue):
        """A stand-in retrieve() returning a fixed list per catalogue."""
        def _retrieve(catalogue, query_text_en, n_results=6, queries=None, **kw):
            return [hit(t, f"{catalogue.name} doc")
                    for t in per_catalogue.get(catalogue.pk, [])][:n_results]
        return _retrieve

    # ── the single-catalogue path is untouched ────────────────────────────
    def test_one_catalogue_uses_the_ordinary_retrieve(self):
        with patch.object(pipeline, "retrieve") as retrieve:
            retrieve.return_value = [hit("urea 30 kg")]
            hits = pipeline.retrieve_across_catalogues("urea?", [5], n_results=10)

        self.assertEqual(1, retrieve.call_count)
        self.assertEqual([hit("urea 30 kg")], hits)

    def test_no_catalogue_returns_nothing(self):
        self.assertEqual([], pipeline.retrieve_across_catalogues("urea?", []))

    def test_an_id_that_does_not_exist_is_ignored(self):
        with patch.object(pipeline, "retrieve") as retrieve:
            retrieve.return_value = [hit("urea 30 kg")]
            pipeline.retrieve_across_catalogues("urea?", [5, 999], n_results=10)

        self.assertEqual(1, retrieve.call_count)

    # ── merging ───────────────────────────────────────────────────────────
    def test_each_catalogue_is_searched_independently(self):
        with patch.object(pipeline, "retrieve",
                          side_effect=self.fake_retrieve({5: ["a"], 6: ["b"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        self.assertEqual({"a", "b"}, {h["chunk"] for h in hits})

    def test_provenance_is_kept_on_every_hit(self):
        with patch.object(pipeline, "retrieve",
                          side_effect=self.fake_retrieve({5: ["a"], 6: ["b"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        by_chunk = {h["chunk"]: h["metadata"] for h in hits}
        self.assertEqual(5, by_chunk["a"]["catalogue_id"])
        self.assertEqual("Nutrient Management", by_chunk["a"]["catalogue_name"])
        self.assertEqual("Varietal recommendation", by_chunk["b"]["catalogue_name"])
        # the document title from the ordinary hit is not overwritten
        self.assertIn("doc", by_chunk["a"]["document_title"])

    def test_the_lists_are_interleaved_so_neither_leads(self):
        # Merging by score would let the wordier corpus fill the context; the
        # merge takes each catalogue's best, then each catalogue's second.
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: ["n1", "n2", "n3"], 6: ["v1", "v2", "v3"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=4)

        self.assertEqual(["n1", "v1", "n2", "v2"], [h["chunk"] for h in hits])

    def test_no_catalogue_can_fill_the_context_on_its_own(self):
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: [f"n{i}" for i in range(20)], 6: ["v1"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        from_nutrient = [h for h in hits
                         if h["metadata"]["catalogue_id"] == 5]
        self.assertLessEqual(len(from_nutrient), 7)
        self.assertIn("v1", [h["chunk"] for h in hits])

    def test_the_total_is_capped(self):
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: [f"n{i}" for i in range(9)], 6: [f"v{i}" for i in range(9)]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=6)

        self.assertEqual(6, len(hits))

    # ── duplicates ────────────────────────────────────────────────────────
    def test_the_same_chunk_from_two_catalogues_appears_once(self):
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: ["shared paragraph", "n2"], 6: ["shared paragraph", "v2"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        chunks = [h["chunk"] for h in hits]
        self.assertEqual(1, chunks.count("shared paragraph"))

    def test_a_duplicate_records_the_other_catalogue_it_came_from(self):
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: ["shared paragraph"], 6: ["shared paragraph"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        self.assertEqual(["Varietal recommendation"],
                         hits[0]["metadata"]["also_in_catalogues"])

    def test_whitespace_differences_do_not_make_a_second_copy(self):
        # The same page ingested twice is rarely byte-identical.
        with patch.object(pipeline, "retrieve", side_effect=self.fake_retrieve(
                {5: ["Apply urea 30 kg/acre"], 6: ["Apply  urea   30 kg/acre\n"]})):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        self.assertEqual(1, len(hits))

    # ── failure of one corpus ─────────────────────────────────────────────
    def test_one_unreadable_catalogue_does_not_lose_the_other(self):
        def _retrieve(catalogue, *a, **kw):
            if catalogue.pk == 5:
                raise RuntimeError("collection unreadable")
            return [hit("v1")]

        with patch.object(pipeline, "retrieve", side_effect=_retrieve):
            hits = pipeline.retrieve_across_catalogues("q", [5, 6], n_results=10)

        self.assertEqual(["v1"], [h["chunk"] for h in hits])


class ContextBoundaryTests(TestCase):
    """What the model is shown when the context spans two catalogues."""

    def test_one_catalogue_produces_the_context_it_always_did(self):
        hits = [hit("urea 30 kg", "Nutrient doc")]
        context, names = pipeline._context_from(hits)

        self.assertEqual("[Source: Nutrient doc]\nurea 30 kg", context)
        self.assertEqual([], names)

    def test_two_catalogues_are_separated_by_a_header(self):
        hits = [
            {"chunk": "urea 30 kg", "metadata": {"document_title": "N doc",
                                                 "catalogue_name": "Nutrient"}},
            {"chunk": "blast resistant", "metadata": {"document_title": "V doc",
                                                      "catalogue_name": "Variety"}},
        ]
        context, names = pipeline._context_from(hits)

        self.assertIn("===== CATALOGUE: Nutrient =====", context)
        self.assertIn("===== CATALOGUE: Variety =====", context)
        self.assertEqual(["Nutrient", "Variety"], names)
        # the Source marker is untouched: everything downstream finds a chunk's
        # document by walking up to the nearest one
        self.assertIn("[Source: N doc]\nurea 30 kg", context)

    def test_the_model_is_told_it_need_not_use_every_catalogue(self):
        hits = [
            {"chunk": "urea 30 kg", "metadata": {"document_title": "N",
                                                 "catalogue_name": "Nutrient"}},
            {"chunk": "CR Dhan 102", "metadata": {"document_title": "V",
                                                  "catalogue_name": "Variety"}},
        ]
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "Apply urea 30 kg."

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer("How much urea?", hits)

        self.assertIn("MORE THAN ONE COLLECTION", seen["sys"])
        self.assertIn("NOT required to say something from each", seen["sys"])

    def test_a_single_catalogue_answer_gets_no_such_note(self):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "Apply urea 30 kg."

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.generate_answer("How much urea?", [hit("urea 30 kg")])

        self.assertNotIn("MORE THAN ONE COLLECTION", seen["sys"])
