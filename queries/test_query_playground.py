"""The testing page runs the real pipeline and keeps nothing.

Debugging an answer through the chat page costs a Query row, a ChatQuery row and
an audio file per attempt, and leaves a reviewer unable to tell a farmer's
question from the twentieth attempt at a test. This page runs the same pipeline
and rolls the transaction back.

The persistence tests are the point of the file: they count rows in EVERY table
the query path writes to, before and after, including the FAQ hit counter that a
`persist=False` flag would have had to remember.
"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from catalogues.models import Catalogue
from queries.models import ChatQuery, FAQ, Query
from queries.services import query_test

QUESTION = "Which medium-duration rice variety is good to plant in my land?"
ANSWER = "Suitable medium-duration varieties are: CR Dhan 310 and MTU-1140."
HITS = [{"chunk": "CR Dhan 310 is a medium-duration variety.\n"
                  "MTU-1140 is a medium-duration variety.\n",
         "metadata": {"document_title": "Varietal sheet", "document_id": 1,
                      "chunk_index": 0}}]


def stub_pipeline():
    """Patch only the outside world: retrieval, the LLM and translation.

    Everything between them — intent, candidate filtering, generation prompt
    assembly, the trace, and every database write — is the real production code.
    """
    return [
        patch("queries.services.pipeline.retrieve", return_value=list(HITS)),
        patch("queries.services.pipeline._expand_queries",
              side_effect=lambda q: [q]),
        patch("queries.services.pipeline._chat", side_effect=lambda *a, **k: ANSWER),
        patch("queries.services.pipeline.translate", side_effect=lambda t, *a: t),
        patch("queries.services.pipeline.classify_and_enrich", return_value={}),
        patch("queries.services.pipeline.faq_lookup", return_value=(None, 0.0)),
        patch("queries.services.pipeline.faq_nearest", return_value=(None, 0.0)),
    ]


class PlaygroundTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user("tester", password="x")
        cls.catalogue = Catalogue.objects.create(name="Rice KB", slug="rice-kb")

    def setUp(self):
        self.client.force_login(self.user)
        self.patches = stub_pipeline()
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def counts(self):
        return (Query.objects.count(), ChatQuery.objects.count(),
                FAQ.objects.count())

    def run_test_query(self, question=QUESTION, **extra):
        payload = {"query_text": question, "catalogue": self.catalogue.pk}
        payload.update(extra)
        return self.client.post(reverse("queries:test_query_run"), payload)


class AnswerTests(PlaygroundTestCase):
    """1 and 2 — it answers, and it answers through the existing pipeline."""

    def test_the_endpoint_returns_an_answer(self):
        response = self.run_test_query()
        body = response.json()

        self.assertEqual(200, response.status_code)
        self.assertTrue(body["ok"])
        self.assertEqual(ANSWER, body["answer"])

    def test_it_runs_the_existing_pipeline_function(self):
        with patch("queries.services.pipeline.run_pipeline_for_text") as run:
            run.return_value = Query.objects.create(original_query_text=QUESTION)
            query_test.run_test_query(catalogue=self.catalogue, question=QUESTION)

        self.assertEqual(1, run.call_count)
        kwargs = run.call_args.kwargs
        self.assertEqual(QUESTION, kwargs["query_text"])
        self.assertFalse(kwargs["generate_tts"])       # no file can be left behind
        self.assertIsNone(kwargs["hook"])              # cannot message anyone
        self.assertEqual("", kwargs["whatsapp_number"])
        self.assertIsNone(kwargs["source_audio"])

    def test_the_real_intent_and_candidates_come_back(self):
        body = self.run_test_query().json()

        self.assertEqual("variety", body["intent"]["entity"])
        self.assertEqual({"maturity": "medium_duration"},
                         body["intent"]["attributes"])
        self.assertEqual("", body["intent"]["land_type"])
        names = [c["name"] for c in body["candidates"]["accepted"]]
        self.assertCountEqual(["CR Dhan 310", "MTU-1140"], names)

    def test_the_sources_come_back(self):
        body = self.run_test_query().json()

        self.assertEqual(1, len(body["sources"]))
        self.assertEqual("Varietal sheet", body["sources"][0]["document_title"])

    def test_the_trace_comes_back_in_the_response(self):
        body = self.run_test_query().json()

        self.assertIn("steps", body["trace"])
        self.assertIn("intent", body["trace"])
        self.assertIn("candidates", body["trace"])


class NoPersistenceTests(PlaygroundTestCase):
    """3, 4, 5 and 7 — the whole point of the page."""

    def test_no_query_row_is_created(self):
        before = Query.objects.count()
        self.run_test_query()

        self.assertEqual(before, Query.objects.count())

    def test_no_chat_or_other_rows_are_created(self):
        before = self.counts()
        self.run_test_query()

        self.assertEqual(before, self.counts())

    def test_ten_test_runs_leave_the_database_exactly_as_it_was(self):
        before = self.counts()
        for i in range(10):
            self.run_test_query(question=f"{QUESTION} ({i})")

        self.assertEqual(before, self.counts())

    def test_the_faq_hit_counter_is_not_touched(self):
        # The FAQ path increments hit_count and last_used_at. A flag threaded
        # through the pipeline would have had to remember this; a rollback
        # cannot forget it.
        faq = FAQ.objects.create(question_en=QUESTION, answer_en="Stored answer.",
                                 question_norm="norm", is_active=True)
        with patch("queries.services.pipeline.faq_lookup", return_value=(faq, 0.99)):
            response = self.run_test_query()

        faq.refresh_from_db()
        self.assertTrue(response.json()["ok"])
        self.assertEqual(0, faq.hit_count)
        self.assertIsNone(faq.last_used_at)
        self.assertEqual(0, Query.objects.count())

    def test_a_failing_pipeline_leaves_no_failed_row_behind(self):
        # The pipeline's own error path saves the Query with status Failed.
        with patch("queries.services.pipeline._chat",
                   side_effect=RuntimeError("model exploded")):
            response = self.run_test_query()

        self.assertFalse(response.json()["ok"])
        self.assertEqual(0, Query.objects.count())

    def test_the_service_returns_plain_data_not_a_model(self):
        result = query_test.run_test_query(catalogue=self.catalogue,
                                           question=QUESTION)

        self.assertIsInstance(result, dict)
        self.assertNotIn("query", result)
        self.assertEqual(0, Query.objects.count())


class ProductionUnchangedTests(PlaygroundTestCase):
    """6 and 8 — the production path still saves exactly as it did."""

    def test_the_production_pipeline_still_creates_a_query(self):
        from queries.services.pipeline import run_pipeline_for_text

        before = Query.objects.count()
        query = run_pipeline_for_text(catalogue=self.catalogue,
                                      farmer_language="English",
                                      query_text=QUESTION)

        self.assertEqual(before + 1, Query.objects.count())
        self.assertEqual("Responded", query.status)
        self.assertEqual(ANSWER, query.ai_response_text)

    def test_the_production_pipeline_signature_has_no_new_required_argument(self):
        import inspect

        from queries.services.pipeline import run_pipeline_for_text

        required = [name for name, p in
                    inspect.signature(run_pipeline_for_text).parameters.items()
                    if p.default is inspect.Parameter.empty]

        self.assertEqual(["catalogue", "farmer_language", "query_text"], required)

    def test_the_demo_chat_endpoint_is_untouched(self):
        response = self.client.post(reverse("queries:demo_chat"), {
            "catalogue": self.catalogue.pk, "query_text": QUESTION,
            "farmer_language": "English"})

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["ok"])
        # Production still persists both rows.
        self.assertEqual(1, Query.objects.count())
        self.assertEqual(1, ChatQuery.objects.count())


class ErrorHandlingTests(PlaygroundTestCase):
    """9 and 10 — useful errors, no secrets, no tracebacks."""

    def test_an_empty_question_is_refused(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                response = self.run_test_query(question=value)

                self.assertEqual(400, response.status_code)
                self.assertFalse(response.json()["ok"])
                self.assertEqual(0, Query.objects.count())

    def test_a_missing_catalogue_is_refused(self):
        response = self.client.post(reverse("queries:test_query_run"),
                                    {"query_text": QUESTION, "catalogue": 99999})

        self.assertEqual(400, response.status_code)
        self.assertFalse(response.json()["ok"])

    def test_a_retrieval_error_is_reported_not_raised(self):
        with patch("queries.services.pipeline.retrieve",
                   side_effect=RuntimeError("chroma is down")):
            response = self.run_test_query()

        body = response.json()
        self.assertEqual(500, response.status_code)
        self.assertFalse(body["ok"])
        self.assertIn("chroma is down", body["error"])

    def test_an_llm_error_is_reported_not_raised(self):
        with patch("queries.services.pipeline._chat",
                   side_effect=RuntimeError("rate limited")):
            body = self.run_test_query().json()

        self.assertFalse(body["ok"])
        self.assertIn("rate limited", body["error"])

    def test_no_traceback_reaches_the_browser(self):
        with patch("queries.services.pipeline._chat",
                   side_effect=RuntimeError("boom")):
            body = self.run_test_query().json()

        self.assertNotIn("Traceback", body["error"])
        self.assertNotIn("File \"", body["error"])

    def test_a_secret_is_never_returned_in_an_error(self):
        with self.settings(OPENAI_API_KEY="sk-super-secret-value-1234"):
            with patch("queries.services.pipeline._chat",
                       side_effect=RuntimeError(
                           "auth failed for sk-super-secret-value-1234")):
                body = self.run_test_query().json()

        self.assertNotIn("sk-super-secret-value-1234", body["error"])
        self.assertIn("***", body["error"])


class AccessTests(PlaygroundTestCase):
    """Existing permission behaviour, not a new one."""

    def test_the_page_renders_for_a_signed_in_user(self):
        response = self.client.get(reverse("queries:test_query"))

        self.assertEqual(200, response.status_code)
        self.assertContains(response, "Paddy Mitra Query Testing")

    def test_both_urls_require_signing_in(self):
        self.client.logout()

        self.assertEqual(302, self.client.get(reverse("queries:test_query")).status_code)
        self.assertEqual(302, self.run_test_query().status_code)

    def test_the_run_endpoint_refuses_get(self):
        self.assertEqual(405, self.client.get(
            reverse("queries:test_query_run")).status_code)
