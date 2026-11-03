"""An Odia question must be answered in Odia — through the Odia pipeline.

THE REPORTED ANSWER
-------------------
Asked in Odia on the Query Testing page with the language dropdown left on
English, the farmer got back:

    ଯଦି ତୁମର ଗଛ ଖାଇଯାଉଛି, ତେବେ ତୁମେ ଜିଙ୍କ ସାର ଦେଖୁପାରିବା।
    ଜିଙ୍କ ଦେଖୁବାକୁ ତୁମେ ଜିଙ୍କ ସଲଫେଟ୍ ବ୍ୟବହାର କରା। ଏହା ଗଛର ଗ୍ରୋଥକୁ ସାହାଯ୍ୟ କରିବ।

which is not Odia a person would say: ଦେଖୁପାରିବା and ଜିଙ୍କ ଦେଖୁବାକୁ are not
words, ବ୍ୟବହାର କରା is the wrong imperative, ତୁମେ is the familiar address the
answers never use, and ଗ୍ରୋଥ is English in Odia letters.

None of the post-translation passes did that — fed correct Odia they all return
it untouched, which is asserted below. The cause was the LABEL: the caller
declared "English" for a question written in Odia, and the pipeline believed it.
That disabled the Odia machinery at BOTH ends —

    inbound   source_lang was "English", so the Odia -> English translation
              never ran: retrieval was given Odia text (the reported sources
              scored -0.615 at distance 1.61 — near-misses)
    outbound  target_lang was "English", so _fix_odia_fluency,
              _detransliterate_odia, to_spoken and to_imperative were ALL
              skipped, and whatever Odia the model happened to emit went
              straight to the farmer, unrepaired

— which is exactly the list of defects in that answer.
"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from catalogues.models import Catalogue
from queries.models import Query
from queries.services import pipeline
from queries.services.odia_glossary import apply_glossary
from queries.services.odia_register import to_imperative, to_spoken

ODIA_Q = ("ଏ ପୋକ ମୋର ଗଛ ଖାଇଯାଉଛି। କେଉଁ ସାର ଦେଲେ ସେ ଗଛଟା ବଢ଼ିବ "
          "ସେ ବିଷୟରେ ମତେ ଟିକେ କୁହନ୍ତୁ।")
HINDI_Q = "मुझे कितना यूरिया देना चाहिए?"
ENGLISH_Q = "How much urea should I apply?"


class LanguageOfQuestionTests(SimpleTestCase):
    """The script settles it — but a non-English label is never overridden."""

    def test_an_odia_question_declared_english_is_treated_as_odia(self):
        self.assertEqual("Odia", pipeline.language_of_question(ODIA_Q, "English"))

    def test_a_hindi_question_declared_english_is_treated_as_hindi(self):
        self.assertEqual("Hindi", pipeline.language_of_question(HINDI_Q, "English"))

    def test_an_english_question_stays_english(self):
        self.assertEqual("English",
                         pipeline.language_of_question(ENGLISH_Q, "English"))

    def test_a_declared_language_is_never_overridden(self):
        # A farmer typing English (or romanised Odia) through an Odia hook still
        # wants the answer in Odia, and the declaration is the only thing that
        # says so. This must not change.
        self.assertEqual("Odia", pipeline.language_of_question(ENGLISH_Q, "Odia"))
        self.assertEqual("Odia", pipeline.language_of_question("DAP 20 kg", "Odia"))
        self.assertEqual("Hindi", pipeline.language_of_question(ENGLISH_Q, "Hindi"))

    def test_an_odia_question_declared_odia_is_unchanged(self):
        self.assertEqual("Odia", pipeline.language_of_question(ODIA_Q, "Odia"))

    def test_digits_and_product_names_alone_are_not_another_language(self):
        for text in ("DAP 20 kg", "20-25", "NPK", ""):
            with self.subTest(text=text):
                self.assertEqual("English",
                                 pipeline.language_of_question(text, "English"))


class ThePostPassesAreInnocentTests(SimpleTestCase):
    """Correct Odia in, the same Odia out — the damage was never here."""

    GOOD = ("ଯଦି ଆପଣଙ୍କ ଗଛକୁ ପୋକ ଖାଇଯାଉଛି, ତେବେ ଆପଣ ଜିଙ୍କ ସାର ଦେଇପାରିବେ। "
            "ଜିଙ୍କ ଦେବା ପାଇଁ ଜିଙ୍କ ସଲଫେଟ୍ ବ୍ୟବହାର କରନ୍ତୁ। "
            "ଏହା ଗଛର ଅଭିବୃଦ୍ଧିରେ ସାହାଯ୍ୟ କରିବ।")

    def test_no_deterministic_pass_alters_correct_odia(self):
        for name, fn in (("apply_glossary", apply_glossary),
                         ("_correct_odia", pipeline._correct_odia),
                         ("_fix_transliterations", pipeline._fix_transliterations),
                         ("to_spoken", to_spoken),
                         ("to_imperative", to_imperative)):
            with self.subTest(pass_=name):
                self.assertEqual(self.GOOD, fn(self.GOOD))

    def test_the_broken_forms_are_not_something_a_pass_produces(self):
        out = to_imperative(to_spoken(self.GOOD))

        for broken in ("ଦେଖୁପାରିବା", "ଦେଖୁବାକୁ", "ବ୍ୟବହାର କରା", "ତୁମେ"):
            self.assertNotIn(broken, out)


class BothDirectionsRunTests(TestCase):
    """The integration point: an Odia question drives the Odia pipeline."""

    @classmethod
    def setUpTestData(cls):
        get_user_model().objects.create_user(
            username="lang", email="lang@example.com", password="x")
        cls.catalogue = Catalogue.objects.create(name="Rice KB", slug="rice-kb-lang")

    def setUp(self):
        self.calls = []

        def record(text, source, target):
            self.calls.append((source, target))
            return text

        # No document_id: the pipeline links a real Document when it sees one,
        # and this test is about the language, not the linkage.
        hits = [{"chunk": "Apply zinc sulphate to correct zinc deficiency.",
                 "metadata": {"document_title": "Nutrients"}}]
        patches = [
            patch("queries.services.pipeline.retrieve", return_value=hits),
            patch("queries.services.pipeline._expand_queries", side_effect=lambda q: [q]),
            patch("queries.services.pipeline._chat", side_effect=lambda *a, **k: "Apply zinc."),
            patch("queries.services.pipeline.translate", side_effect=record),
            patch("queries.services.pipeline.classify_and_enrich", return_value={}),
            patch("queries.services.pipeline.faq_lookup", return_value=(None, 0.0)),
            patch("queries.services.pipeline.faq_nearest", return_value=(None, 0.0)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_it(self, question, language):
        return pipeline.run_pipeline_for_text(
            catalogue=self.catalogue, farmer_language=language,
            query_text=question, generate_tts=False, hook=None,
            whatsapp_number="")

    def test_the_question_is_translated_from_odia_not_from_english(self):
        self.run_it(ODIA_Q, "English")

        self.assertIn(("Odia", "English"), self.calls)
        self.assertNotIn(("English", "English"), self.calls)

    def test_the_answer_is_translated_into_odia(self):
        # This is the call that runs the fluency repair, the de-transliteration,
        # the spoken register and the imperative pass. It was not happening.
        self.run_it(ODIA_Q, "English")

        self.assertIn(("English", "Odia"), self.calls)

    def test_the_query_records_the_language_it_was_actually_asked_in(self):
        query = self.run_it(ODIA_Q, "English")

        self.assertEqual("Odia", Query.objects.get(pk=query.pk).original_query_language)

    def test_the_correction_is_recorded_in_the_trace(self):
        query = self.run_it(ODIA_Q, "English")
        trace = Query.objects.get(pk=query.pk).pipeline_trace or {}

        self.assertEqual("English", trace.get("declared_language"))
        self.assertEqual("Odia", trace.get("detected_language"))

    def test_an_english_question_is_unaffected(self):
        self.run_it(ENGLISH_Q, "English")

        self.assertNotIn(("Odia", "English"), self.calls)
        self.assertNotIn(("English", "Odia"), self.calls)

    def test_an_odia_declaration_still_works_as_before(self):
        self.run_it(ODIA_Q, "Odia")

        self.assertIn(("Odia", "English"), self.calls)
        self.assertIn(("English", "Odia"), self.calls)

    def test_english_typed_through_an_odia_hook_still_answers_in_odia(self):
        # The case the override must not break.
        self.run_it(ENGLISH_Q, "Odia")

        self.assertIn(("English", "Odia"), self.calls)
