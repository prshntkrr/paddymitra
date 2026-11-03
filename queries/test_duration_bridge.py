"""One misheard letter invented a variety; a day count could not find its class.

THE REPORTED QUESTION
---------------------
    ସହେ ଦିନିଆ ଛୋଟ କିସମରେ କେତେ କେଜି ୟୁରିଆ ପ୍ରୟୋଗ କରିବା ଦରକାର?
    -> "How many kg of urea should be applied for the Sahia Diniya short variety?"

"Sahia Diniya" is not a variety. ଶହେ means ONE HUNDRED and ଦିନିଆ means -day, so
the farmer asked about the HUNDRED-DAY short variety. Measured on the live APIs:

    ସହେ ଦିନିଆ   -> Google: "Same day"      odia_numbers: nothing converted
    ଶହେ ଦିନିଆ   -> Google: "Hundred days"  odia_numbers: ଶହେ -> 100

Two defects, fixed separately because they fail separately:

1. THE LETTER. ସ for ଶ hid the number word from odia_numbers. Repaired before
   the numbers are read — but only where the next word makes it a count, since
   ସହେ is also an ordinary verb form.

2. THE FABRICATION. Google said "a small variety of rice"; the glossary check
   saw that "apply" had been dropped and fell back to OpenAI, which invented
   "Sahia Diniya". A vague translation is a small loss; a fabricated variety
   name sends retrieval hunting for a seed that does not exist.

AND THE FAQ CHAIN. Even translated correctly, "100-day" reaches nothing: against
the operator's own entries it scores 0.508, under the 0.60 context floor, while
the class phrasing scores 0.625 and is offered as context. So a day count is
classified into short/medium/long and offered to the FAQ matcher as a second
probe — days -> duration -> the varieties and doses written for that class.
"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from catalogues.models import Catalogue
from queries.services import pipeline
from queries.services.odia_numbers import convert_odia_numbers

AS_HEARD = "ସହେ ଦିନିଆ ଛୋଟ କିସମରେ କେତେ କେଜି ୟୁରିଆ ପ୍ରୟୋଗ କରିବା ଦରକାର?"
CORRECT = "ଶହେ ଦିନିଆ ଛୋଟ କିସମରେ କେତେ କେଜି ୟୁରିଆ ପ୍ରୟୋଗ କରିବା ଦରକାର?"


class TheMishearingRepairTests(SimpleTestCase):
    """ସହେ is a number here and a verb elsewhere."""

    def test_the_reported_question_is_repaired(self):
        self.assertEqual(CORRECT, pipeline._repair_odia_mishearings(AS_HEARD))

    def test_the_repair_lets_the_number_convert(self):
        repaired = pipeline._repair_odia_mishearings(AS_HEARD)
        _out, pairs = convert_odia_numbers(repaired)

        self.assertEqual([("ଶହେ", "100")], [(o, v) for o, v in pairs])

    def test_without_the_repair_nothing_converts(self):
        _out, pairs = convert_odia_numbers(AS_HEARD)

        self.assertEqual([], pairs)

    def test_every_time_word_triggers_it(self):
        for word in ("ଦିନିଆ", "ଦିନ", "ମାସ", "ବର୍ଷ", "ସପ୍ତାହ"):
            with self.subTest(word=word):
                self.assertEqual(f"ଶହେ {word} ପରେ",
                                 pipeline._repair_odia_mishearings(f"ସହେ {word} ପରେ"))

    def test_the_ordinary_verb_is_left_alone(self):
        # ସହେ as a form of ସହିବା (to bear). No time word follows, so no repair.
        for sentence in ("ସେ ବହୁତ ଦୁଃଖ ସହେ।",
                         "ଏହି ଧାନ ଜଳ ଜମା ସହେ।",
                         "ଏହା ସହେ ନାହିଁ।"):
            with self.subTest(sentence=sentence):
                self.assertEqual(sentence,
                                 pipeline._repair_odia_mishearings(sentence))

    def test_correct_text_is_untouched(self):
        self.assertEqual(CORRECT, pipeline._repair_odia_mishearings(CORRECT))


class TheFabricationGuardTests(SimpleTestCase):
    """A Title Case name only the fallback produced."""

    GOOGLE = "How many kg of urea should be applied for a small variety of rice?"
    INVENTED = "How many kg of urea should be applied for the Sahia Diniya short variety?"

    def test_the_reported_fabrication_is_caught(self):
        self.assertEqual("Sahia Diniya",
                         pipeline._invented_name(self.INVENTED, self.GOOGLE))

    def test_a_name_google_also_produced_is_not_flagged(self):
        both = "How much urea for Sahia Diniya?"

        self.assertEqual("", pipeline._invented_name(both, "Urea for Sahia Diniya"))

    def test_a_known_cultivar_is_never_flagged(self):
        # A real variety the farmer named must survive transliteration.
        fallback = "How much urea for Swarna Sub1 paddy?"

        self.assertEqual("", pipeline._invented_name(fallback, "How much urea?"))

    def test_a_sentence_initial_capital_is_not_a_name(self):
        self.assertEqual("", pipeline._invented_name("Urea Dose is high?", "x y"))

    def test_a_single_capitalised_word_is_not_enough(self):
        self.assertEqual("", pipeline._invented_name("How much Urea?", "How much?"))

    def test_nothing_to_compare_against_is_not_a_fabrication(self):
        self.assertEqual("", pipeline._invented_name(self.INVENTED, ""))


class TheFallbackIsJudgedTests(SimpleTestCase):
    """translate() prefers Google's wording over an invented name."""

    def translate(self, google_out, chat_out):
        # The fallback only runs when Google dropped a glossary term, and which
        # terms exist depends on the deployed glossary — this repo's copy holds
        # six. Forced here, so the test exercises the branch it is about rather
        # than whatever the local glossary happens to contain.
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: google_out), \
                patch.object(pipeline, "_missing_glossary_terms",
                             side_effect=lambda *a, **k: ["apply"]), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: chat_out):
            return pipeline.translate(CORRECT, "Odia", "English")

    def test_the_fallback_really_is_reached(self):
        # Guard for the guard: if Google's candidate were accepted, the two
        # tests below would pass without ever exercising the fabrication check.
        calls = []
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "google wording"), \
                patch.object(pipeline, "_missing_glossary_terms",
                             side_effect=lambda *a, **k: ["apply"]), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: calls.append(1) or "llm wording"):
            pipeline.translate(CORRECT, "Odia", "English")

        self.assertEqual(1, len(calls), "the OpenAI fallback did not run")

    def test_an_invented_name_is_discarded_for_googles_wording(self):
        google = "How many kg of urea for a small variety of rice?"
        out = self.translate(google, TheFabricationGuardTests.INVENTED)

        self.assertNotIn("Sahia Diniya", out)
        self.assertIn("small variety", out)

    def test_an_honest_fallback_is_still_used(self):
        # Google dropped the glossary term; the fallback fixed it without
        # inventing anything, so the fallback is what should be used.
        google = "How many kg of urea for a small variety of rice?"
        honest = "How many kg of urea should be applied for the 100-day short variety?"
        out = self.translate(google, honest)

        self.assertIn("100-day", out)


class TheDurationProbeTests(SimpleTestCase):
    """Days become the class the answers are written for."""

    def test_the_reported_question_becomes_a_class_question(self):
        probe = pipeline.duration_probe(
            "How many kg of urea should be applied for the 100-day short variety?")

        self.assertIn("short duration", probe)
        self.assertNotIn("100-day", probe)

    def test_the_nutrient_asked_about_is_carried_into_the_probe(self):
        # Measured: the entry that answers this scores 0.787 on this phrasing
        # against 0.655 for the long-duration entry. Substituting into the
        # farmer's own sentence scored 0.568 — under the floor, and level with
        # the wrong entry.
        for question, nutrient in (
                ("How many kg of urea for the 100-day variety?", "urea"),
                ("How much DAP for a 150 day variety?", "dap"),
                ("How much MOP for a 130-day variety?", "mop")):
            with self.subTest(nutrient=nutrient):
                self.assertIn(nutrient, pipeline.duration_probe(question).lower())

    def test_it_falls_back_to_fertilizer_when_none_is_named(self):
        probe = pipeline.duration_probe("What to apply for a 100-day variety?")

        self.assertIn("fertilizer", probe)

    def test_the_probe_stays_below_the_serve_threshold_by_design(self):
        # It is a sentence the farmer never said, so it may add context but
        # must not be able to serve an approved answer verbatim. Asserted as
        # shape: the probe asks "and when?", which is what kept it at 0.787
        # rather than the 0.832 of the bare fertilizer phrasing.
        probe = pipeline.duration_probe("How much urea for a 100-day variety?")

        self.assertTrue(probe.startswith("For short duration rice varieties"))
        self.assertIn("and when?", probe)

    def test_each_band(self):
        for days, label in ((100, "short"), (110, "short"), (125, "medium"),
                            (135, "medium"), (145, "long"), (160, "long")):
            with self.subTest(days=days):
                probe = pipeline.duration_probe(f"urea for a {days} day variety")

                self.assertIn(f"{label} duration", probe)

    def test_a_question_that_names_no_class_at_all_has_no_probe(self):
        # "How much urea for a short duration variety?" USED to belong here,
        # when only a day count could produce a probe. It now names its class
        # in words, so it gets one — see TheClassInWordsTests.
        for question in ("How much urea per acre?",
                         "Which variety suits upland?",
                         "When should I apply MOP?"):
            with self.subTest(question=question):
                self.assertEqual("", pipeline.duration_probe(question))

    def test_a_day_count_that_is_not_a_duration_is_ignored(self):
        # An interval, not a variety class.
        self.assertEqual("", pipeline.duration_probe(
            "How much urea 5 days after transplanting?"))
        self.assertEqual("", pipeline.duration_probe(
            "Is 400 days normal for rice?"))


class TheClassInWordsTests(SimpleTestCase):
    """A question can name its class without naming any days.

    "ଅଳ୍ପ କାଳୀନ କିସମ ପାଇଁ କେତେ ୟୁରିଆ ଦେବି?" arrives as "How much urea should I
    give for an early-maturing variety?" — no digits at all. A day-count rule
    found nothing, so no probe was built, and the generic "how much urea for my
    rice crop" entry was handed over: its 70 kg went out under an
    "early-maturing variety" heading.
    """

    def test_the_class_is_read_from_the_words(self):
        for question, label in (
                ("How much urea should I give for an early-maturing variety?", "short"),
                ("How much urea for a short duration variety?", "short"),
                ("What dose for a late-maturing variety?", "long"),
                ("How much urea for a medium duration rice variety?", "medium")):
            with self.subTest(question=question):
                self.assertEqual(label, pipeline.duration_class(question))

    def test_a_probe_is_built_without_any_day_count(self):
        probe = pipeline.duration_probe(
            "How much urea should I give for an early-maturing variety?")

        self.assertEqual("For short duration rice varieties, how much urea "
                         "to be applied and when?", probe)

    def test_a_day_count_still_wins_when_both_are_present(self):
        # The digits are the farmer's own number; they decide.
        self.assertEqual("long", pipeline.duration_class(
            "How much urea for a 150 day variety, is it short duration?"))

    def test_a_question_naming_every_class_gets_no_probe(self):
        # A classification question is generic — there is no single class to
        # look up an answer for.
        self.assertEqual("", pipeline.duration_class(
            "Classify short, medium and long duration varieties by days"))
        self.assertEqual("", pipeline.duration_probe(
            "Classify short, medium and long duration varieties by days"))

    def test_a_question_with_neither_has_no_class(self):
        self.assertEqual("", pipeline.duration_class("How much urea per acre?"))


class TheMoneyIntrusionTests(SimpleTestCase):
    """ଦେବି is "shall I give", not "shall I pay"."""

    ODIA = "ସହଭାଗୀ ଧାନ ପାଇଁ କେତେ ୟୁରିଆ ଦେବି?"
    MISTRANSLATED = "How much urea should I pay for Sahbhagi Dhan?"

    def test_the_reported_mistranslation_is_caught(self):
        self.assertEqual("pay", pipeline._money_intrusion(self.ODIA,
                                                          self.MISTRANSLATED))

    def test_every_money_word_is_caught(self):
        for english in ("What is the cost of urea?", "What is the price per kg?",
                        "How many rupees for DAP?", "Is urea expensive?"):
            with self.subTest(english=english):
                self.assertTrue(pipeline._money_intrusion(self.ODIA, english))

    def test_a_real_price_question_is_left_alone(self):
        # The farmer did ask about money, so the English is right.
        for odia in ("ୟୁରିଆର ଦାମ କେତେ?", "ଡିଏପି ଦର କେତେ ଟଙ୍କା?",
                     "ସାର ପାଇଁ କେତେ ଖର୍ଚ୍ଚ ହେବ?"):
            with self.subTest(odia=odia):
                self.assertEqual("", pipeline._money_intrusion(
                    odia, "What is the price of urea per kg?"))

    def test_a_dose_translation_is_not_flagged(self):
        self.assertEqual("", pipeline._money_intrusion(
            self.ODIA, "How much urea should I give for Sahbhagi Dhan?"))

    def test_google_is_refused_when_it_invents_a_cost_question(self):
        calls = []
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: self.MISTRANSLATED), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: calls.append(1) or
                             "How much urea should I give for Sahbhagi Dhan?"):
            out = pipeline.translate(self.ODIA, "Odia", "English")

        self.assertEqual(1, len(calls), "the fallback did not run")
        self.assertNotIn("pay", out.lower())


class TheDurationClassGuardTests(SimpleTestCase):
    """Short is not long — and nothing used to say so.

    Measured: the probe for a 150-day question scored 0.771 against the SHORT
    entry and 0.713 against the long one, so the wrong dose would have been put
    in front of the model. The guard had groups for season, product, stage,
    variety, number and unit — but none for duration class, so this was true on
    the ordinary path too, not only through the probe.
    """

    SHORT_FAQ = ("For short duration rice varieties like Sahabhagi Dhan, how "
                 "much fertilizer to be applied and at what time it should be "
                 "applied?")
    LONG_FAQ = ("How much fertilizer to be applied for a long duration rice "
                "variety like Puja Dhana, Kala Champa, Swarna sub-1?")

    def test_the_class_is_read_from_the_phrase(self):
        from queries.services.faq import _durations

        self.assertEqual({"short"}, set(_durations("short duration varieties")))
        self.assertEqual({"long"}, set(_durations("a long-duration variety")))
        self.assertEqual({"medium"}, set(_durations("medium maturing rice")))
        self.assertEqual({"short"}, set(_durations("early maturity types")))

    def test_a_bare_word_is_not_a_duration_class(self):
        from queries.services.faq import _durations

        for text in ("How long should I keep water in the field?",
                     "Which variety suits medium land?",
                     "Apply it in the short term"):
            with self.subTest(text=text):
                self.assertEqual(set(), set(_durations(text)))

    def test_short_and_long_contradict(self):
        from queries.services.faq import _contradicts

        probe = "For long duration rice varieties, how much dap to be applied and when?"

        self.assertIn("durations", _contradicts(probe, self.SHORT_FAQ))
        self.assertEqual("", _contradicts(probe, self.LONG_FAQ))

    def test_naming_no_class_is_not_a_contradiction(self):
        # Silence is not disagreement — that is what makes a near miss usable.
        from queries.services.faq import _contradicts

        self.assertEqual("", _contradicts("How much urea per acre?", self.SHORT_FAQ))

    def test_the_serve_guard_is_strict_about_it_too(self):
        # _decisive_conflict reports the FIRST group that differs, so the two
        # sentences differ in nothing else — otherwise products or varieties
        # would be reported and the duration check would never be reached.
        from queries.services.faq import _decisive_conflict

        self.assertIn("durations", _decisive_conflict(
            "How much fertilizer for a short duration rice variety?",
            "How much fertilizer for a long duration rice variety?"))

    def test_an_entry_naming_every_class_is_generic_not_contradictory(self):
        # FAQ-81 lists short, medium AND long. Strict inequality made it
        # contradict every single-class question, including the ones it answers.
        from queries.services.faq import _contradicts

        classify = ("According to number of days, classify the short, medium "
                    "and long duration of rice varieties.")

        self.assertEqual("", _contradicts(
            "What is a short duration rice variety?", classify))
        self.assertEqual("", _contradicts(
            "For long duration rice varieties, how much urea and when?", classify))

    def test_two_single_class_questions_still_contradict(self):
        from queries.services.faq import _contradicts

        self.assertIn("durations", _contradicts(
            "For long duration rice varieties, how much urea and when?",
            self.SHORT_FAQ))

    def test_the_probe_for_a_long_question_cannot_match_the_short_entry(self):
        from queries.services.faq import _contradicts

        probe = pipeline.duration_probe("How much DAP for a 150 day variety?")

        self.assertIn("long duration", probe)
        self.assertIn("durations", _contradicts(probe, self.SHORT_FAQ))


class TheVarietyClassTests(TestCase):
    """A variety names no class — so the class is read from the entries.

    Query 1013: "How much urea should be given for Sahbhagi Dhan?" found the
    GENERIC urea entry at 0.612 and answered "apply 70 kg of urea per acre".
    The entry that answers it — "For short duration rice varieties like
    Sahabhagi Dhan …" — scored 0.511 and was never seen. No class, no days, so
    nothing was probed.

    The mapping is not hardcoded: the operator's own entry names the variety and
    the class in one sentence. The only difficulty is spelling — the question
    says Sahbhagi, the entry says Sahabhagi. Measured over this project's
    cultivar list, true variants score 0.857-0.963 and DIFFERENT cultivars peak
    at 0.778 (Manaswini~Tapaswini), so 0.85 sits in the gap.
    """

    @classmethod
    def setUpTestData(cls):
        cls.catalogue = Catalogue.objects.create(name="KB", slug="kb-variety")
        from queries.models import FAQ
        cls.short = FAQ.objects.create(
            question_en=("For short duration rice varieties like Sahabhagi Dhan, "
                         "how much fertilizer to be applied?"),
            answer_en="Apply 5-8 kg Urea as basal, 25-30 kg at tillering and PI.",
            catalogue=cls.catalogue, is_active=True)

    def test_the_name_is_taken_from_the_question(self):
        from queries.services.faq import _candidate_names

        self.assertEqual(["Sahbhagi Dhan"], _candidate_names(
            "How much urea should be given for Sahbhagi Dhan?"))

    def test_products_and_question_words_are_not_names(self):
        from queries.services.faq import _candidate_names

        for question in ("How much urea per acre?",
                         "How much DAP at Active Tillering?",
                         "Which variety suits upland?"):
            with self.subTest(question=question):
                self.assertEqual([], _candidate_names(question))

    def test_a_spelling_variant_still_finds_the_name(self):
        from queries.services.faq import _name_in_text

        self.assertTrue(_name_in_text("Sahbhagi Dhan",
                                      "varieties like Sahabhagi Dhan, apply"))
        self.assertTrue(_name_in_text("Kalachampa", "Puja Dhana, Kala Champa"))

    def test_a_different_cultivar_is_not_a_match(self):
        from queries.services.faq import _name_in_text

        # 0.778 — the closest false pair in the whole cultivar list.
        self.assertFalse(_name_in_text("Manaswini", "for Tapaswini rice"))
        # Too short to match loosely: exact only.
        self.assertFalse(_name_in_text("Bina", "for Veena rice"))

    def test_the_class_is_learned_from_the_entry(self):
        from queries.services.faq import duration_for_variety

        self.assertEqual("short", duration_for_variety(
            "How much urea should be given for Sahbhagi Dhan?",
            catalogue=self.catalogue))

    def test_a_variety_no_entry_names_has_no_class(self):
        from queries.services.faq import duration_for_variety

        self.assertEqual("", duration_for_variety(
            "How much urea for Tapaswini?", catalogue=self.catalogue))

    def test_a_variety_described_two_ways_picks_no_side(self):
        from queries.models import FAQ
        from queries.services.faq import duration_for_variety
        FAQ.objects.create(
            question_en="For long duration varieties like Sahabhagi Dhan, how much urea?",
            answer_en="Apply 30-32 kg urea.", catalogue=self.catalogue, is_active=True)

        self.assertEqual("", duration_for_variety(
            "How much urea should be given for Sahbhagi Dhan?",
            catalogue=self.catalogue))

    def test_the_probe_is_built_from_the_learned_class(self):
        probe = pipeline.duration_probe(
            "How much urea should be given for Sahbhagi Dhan?", label="short")

        self.assertEqual("For short duration rice varieties, how much urea "
                         "to be applied and when?", probe)


class TheFaqChainTests(TestCase):
    """The probe reaches the FAQ matcher, and only as a second attempt."""

    @classmethod
    def setUpTestData(cls):
        get_user_model().objects.create_user(
            username="dur", email="dur@example.com", password="x")
        cls.catalogue = Catalogue.objects.create(name="Rice KB", slug="rice-kb-dur")

    def run_pipeline(self, question, lookup_results=None, nearest_results=None):
        """Returns the questions each FAQ function was asked."""
        asked = {"lookup": [], "nearest": []}

        def lookup(q, catalogue=None):
            asked["lookup"].append(q)
            return (lookup_results or {}).get(q, (None, 0.0))

        def nearest(q, catalogue=None, duration=""):
            asked["nearest"].append(q)
            return (nearest_results or {}).get(q, (None, 0.0))

        hits = [{"chunk": "Apply urea at tillering.",
                 "metadata": {"document_title": "Nutrients"}}]
        with patch.object(pipeline, "retrieve", return_value=hits), \
                patch.object(pipeline, "_expand_queries", side_effect=lambda q: [q]), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "Apply urea."), \
                patch.object(pipeline, "translate", side_effect=lambda t, s, d: t), \
                patch.object(pipeline, "classify_and_enrich", return_value={}), \
                patch.object(pipeline, "faq_lookup", side_effect=lookup), \
                patch.object(pipeline, "faq_nearest", side_effect=nearest):
            self.last_query = pipeline.run_pipeline_for_text(
                catalogue=self.catalogue, farmer_language="English",
                query_text=question, generate_tts=False, hook=None,
                whatsapp_number="")
        return asked

    QUESTION = "How many kg of urea should be applied for the 100-day short variety?"

    def test_the_duration_phrasing_is_tried_when_the_question_finds_nothing(self):
        asked = self.run_pipeline(self.QUESTION)

        self.assertEqual(self.QUESTION, asked["lookup"][0])
        self.assertTrue(any("short duration" in q for q in asked["lookup"][1:]),
                        asked["lookup"])

    def test_the_context_probe_is_tried_too(self):
        asked = self.run_pipeline(self.QUESTION)

        self.assertTrue(any("short duration" in q for q in asked["nearest"]),
                        asked["nearest"])

    def test_a_class_with_no_entry_gets_no_faq_at_all(self):
        """Query 1023: a medium-duration question, and no medium entry exists.

        The class-filtered probe found nothing, so the code fell back to the
        question's own near miss — the GENERIC "How much urea I have to apply in
        my rice crop?" at 0.741 — and its schedule went out headed "For medium
        duration rice varieties". A dose wearing a class it was never written
        for is the failure this path exists to prevent.
        """
        from queries.models import FAQ
        generic = FAQ.objects.create(
            question_en="How much urea I have to apply in my rice crop?",
            answer_en="For 1 acre, apply 70 kg of Urea.",
            catalogue=self.catalogue, is_active=True)
        question = "How many kilograms of urea for 120 days rice?"
        probe = pipeline.duration_probe(question)

        asked = self.run_pipeline(
            question,
            nearest_results={question: (generic, 0.741)})   # probe finds nothing
        self.last_query.refresh_from_db()
        trace = self.last_query.pipeline_trace or {}

        self.assertIn("medium duration", probe)
        self.assertIn(probe, asked["nearest"])
        self.assertIsNone(trace.get("faq_context"))
        self.assertNotIn("How much urea I have to apply",
                         str(trace.get("retrieved_chunks", "")))

    def test_a_classless_question_still_keeps_its_near_miss(self):
        # No class named, so there is nothing to contradict: the near miss is
        # ordinary evidence and must still be offered.
        from queries.models import FAQ
        generic = FAQ.objects.create(
            question_en="How much urea I have to apply in my rice crop?",
            answer_en="For 1 acre, apply 70 kg of Urea.",
            catalogue=self.catalogue, is_active=True)
        question = "How much urea per acre?"

        self.run_pipeline(question, nearest_results={question: (generic, 0.70)})
        self.last_query.refresh_from_db()
        trace = self.last_query.pipeline_trace or {}

        self.assertEqual(generic.pk, (trace.get("faq_context") or {}).get("id"))

    def test_a_question_with_no_day_count_probes_once(self):
        asked = self.run_pipeline("How much urea per acre?")

        self.assertEqual(1, len(asked["lookup"]))
        self.assertEqual(1, len(asked["nearest"]))

    def test_the_probe_runs_even_when_the_question_already_found_a_faq(self):
        """The reported failure: a GENERIC entry blocked the specific one.

        On the server the farmer's own wording found "How much urea I have to
        apply in my rice crop?" — over the floor, and nothing to do with short
        duration. The probe was a fallback, so it never ran, and the answer came
        back as "the information available does not specify the amount of urea
        for a 100-day short variety."
        """
        from queries.models import FAQ
        generic = FAQ.objects.create(
            question_en="How much urea I have to apply in my rice crop?",
            answer_en="For 1 acre, apply 70 kg of Urea.",
            catalogue=self.catalogue, is_active=True)
        specific = FAQ.objects.create(
            question_en=("For short duration rice varieties like Sahabhagi Dhan, "
                         "how much fertilizer to be applied?"),
            answer_en="Apply 5-8 kg Urea as basal, 25-30 kg at tillering and PI.",
            catalogue=self.catalogue, is_active=True)
        probe = pipeline.duration_probe(self.QUESTION)

        asked = self.run_pipeline(
            self.QUESTION,
            nearest_results={self.QUESTION: (generic, 0.64),
                             probe: (specific, 0.787)})
        self.last_query.refresh_from_db()
        trace = self.last_query.pipeline_trace or {}

        # The probe ran even though the question's own wording found something.
        self.assertIn(probe, asked["nearest"])
        chunks = str(trace.get("retrieved_chunks", ""))
        self.assertIn("short duration rice varieties", chunks)
        # AND THE GENERIC ONE IS GONE. Measured with the real model: given both,
        # it applied the generic 70 kg figure to the 100-day SHORT variety.
        self.assertNotIn("How much urea I have to apply", chunks)
        self.assertEqual(specific.pk, trace["faq_context"]["id"])
        self.assertNotIn("faq_context_extra", trace)

    def test_the_class_is_spelled_out_for_generation(self):
        """Without it the model refuses: nothing says 100-day IS short duration."""
        seen = {}

        def generate(question, hits, user_type="farmer"):
            seen["question"] = question
            return "Apply 5-8 kg of urea."

        with patch.object(pipeline, "generate_answer", side_effect=generate):
            self.run_pipeline(self.QUESTION)

        self.assertIn("Interpreted for clarity as", seen["question"])
        self.assertIn("short duration rice varieties", seen["question"])
        # The farmer's own question is still there, above it.
        self.assertIn(self.QUESTION, seen["question"])
