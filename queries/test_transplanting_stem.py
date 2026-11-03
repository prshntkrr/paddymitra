"""The verb stem, not just the endings someone happened to list.

THE PRODUCTION CASE
-------------------
    ଧାନ ରୋଇ ସାରିବା ପରେ ଗୋଟେ ଏକରରେ ମୁଁ କେତେ ଡିଏପି … ମୂଳସାର ଭାବରେ?

    glossary matches = {'ଧାନ': 'rice'}          <- ରୋଇ matched nothing
    google           = "… in 1 acre after HARVESTING rice?"

The glossary held ରୋଇବା, ରୋଇବ, ରୋଇଲା and ରୋପଣ — four of the endings this verb
takes. The farmer used a fifth: the bare stem ରୋଇ, in "ରୋଇ ସାରିବା ପରେ" (after
finishing transplanting). Nothing matched, so Google was free to guess, and the
question moved to the other end of the season.

Listing the STEM covers every construction built on it. It cannot match inside
the longer forms, because the scanner only permits a NOUN case ending after a
term and ବା is not one — so ରୋଇବା still matches its own entry, and ରୋଇ catches
what is left.

WHY NOT A GENERAL "ANY VERB ENDING" RULE: it would let ସାର (fertilizer) match
inside ସାରିବା ("to finish") — a word in this very sentence — and turn a timing
clause into a fertiliser one. That case is asserted below.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.odia_glossary import (_reverse_map, normalize_odia,
                                            relevant_reverse_glossary)
from queries.services.odia_numbers import convert_odia_numbers

PRODUCTION_Q = ("ଧାନ ରୋଇ ସାରିବା ପରେ ଗୋଟେ ଏକରରେ ମୁଁ କେତେ ଡିଏପି, କେତେ ୟୁରିଆ, "
                "କେତେ ପୋଟାସ ପ୍ରୟୋଗ କରିବି ମୂଳସାର ଭାବରେ?")
# Captured live from Google for the protected sentence.
PRODUCTION_EN = ("How much DAP, how much urea, how much potassium should I apply "
                 "as basal fertilizer in 1 acre after transplanting rice?")


def matches(text):
    return {normalize_odia(k): v
            for k, v in relevant_reverse_glossary(text).items()}


def key(odia):
    return normalize_odia(odia)


class StemRecognitionTests(SimpleTestCase):
    """Every form of the verb reaches the same English concept."""

    def test_the_bare_stem_is_recognised(self):
        self.assertEqual("transplanting", _reverse_map()[key("ରୋଇ")])

    def test_every_listed_form_means_transplanting(self):
        for form in ("ରୋଇ", "ରୋଇବା", "ରୋଇବ", "ରୋଇଲା", "ରୋଇବାର", "ରୋପଣ"):
            with self.subTest(form=form):
                self.assertEqual("transplanting", _reverse_map()[key(form)])

    def test_the_stem_matches_in_a_sentence(self):
        self.assertEqual("transplanting", matches("ଧାନ ରୋଇ ପରେ")[key("ରୋଇ")])

    def test_the_stem_matches_before_the_completion_verb(self):
        self.assertEqual("transplanting",
                         matches("ରୋଇ ସାରିବା ପରେ")[key("ରୋଇ")])

    def test_the_longer_forms_still_win_over_the_stem(self):
        # Leftmost-longest: the entry for the inflected form is used, and the
        # stem does not fire a second time inside it.
        for phrase, expected in (("ଧାନ ରୋଇବା କେତେ ଦିନ", "ଧାନ ରୋଇବା"),
                                 ("ଧାନ ରୋଇଲା ପରେ", "ରୋଇଲା"),
                                 ("ରୋଇବ କେବେ", "ରୋଇବ")):
            with self.subTest(phrase=phrase):
                found = matches(phrase)

                self.assertIn(key(expected), found)
                self.assertNotIn(key("ରୋଇ"), found)

    def test_the_completion_verb_is_not_read_as_fertiliser(self):
        # The reason a blanket verb-ending rule was rejected: ସାର inside
        # ସାରିବା. Nothing may match here.
        self.assertEqual({}, matches("ସାରିବା ପରେ"))

    def test_the_stem_is_never_harvesting(self):
        for form in ("ରୋଇ", "ରୋଇବା", "ରୋଇଲା"):
            with self.subTest(form=form):
                english = _reverse_map()[key(form)].lower()

                self.assertNotIn("harvest", english)
                self.assertNotIn("cut", english)


class ProductionSentenceTests(SimpleTestCase):
    """The reported query, end to end."""

    def translate(self, reply=PRODUCTION_EN):
        sent = {}

        def google(text, source, target):
            sent["input"] = text
            return reply

        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate(PRODUCTION_Q, "Odia", "English")
        return out, sent.get("input", ""), chat

    def test_the_transplanting_term_is_matched_before_google_is_called(self):
        found = matches(PRODUCTION_Q)

        self.assertTrue(any("transplanting" in v for v in found.values()), found)

    def test_the_english_carries_every_required_term(self):
        out, _sent, _chat = self.translate()
        low = out.lower()

        for term in ("rice", "transplanting", "dap", "urea", "apply",
                     "basal fertilizer", "1 acre", "after transplanting"):
            self.assertIn(term, low)

    def test_the_english_is_never_about_harvest(self):
        out, _sent, _chat = self.translate()
        low = out.lower()

        for wrong in ("harvesting", "harvested", "harvest", "cutting"):
            self.assertNotIn(wrong, low)

    def test_google_is_given_the_protected_terms(self):
        _out, sent, _chat = self.translate()

        self.assertIn("transplanting", sent)
        self.assertIn("basal fertilizer", sent)
        self.assertIn("DAP", sent)          # pinned by protect_input_terms

    def test_the_acre_quantity_is_not_glued_to_the_input_name(self):
        _out, sent, _chat = self.translate()

        self.assertIn("1", sent)
        self.assertNotIn("DAP 1", sent)

    def test_a_reply_that_loses_transplanting_falls_back_to_openai(self):
        # The original production output.
        losing = ("How much DAP, how much urea, how much potash should I apply "
                  "as basic fertilizer in 1 acre after harvesting rice?")
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: losing), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: "fallback") as chat:
            out = pipeline.translate(PRODUCTION_Q, "Odia", "English")

        chat.assert_called()
        # reassert_tokens() re-appends the pinned DAP token when the fallback
        # reply dropped it, so the answer is "fallback (DAP)".
        self.assertIn("fallback", out)


class BasalFertiliserTests(SimpleTestCase):
    """ମୂଳସାର is basal, which is the word the corpus uses."""

    def test_it_resolves_to_basal_not_basic(self):
        self.assertEqual("basal fertilizer", _reverse_map()[key("ମୂଳସାର")])

    def test_the_spaced_form_resolves_too(self):
        self.assertEqual("basal fertilizer", _reverse_map()[key("ମୂଳ ସାର")])


class PreviousRegressionsTests(SimpleTestCase):
    """Nothing earlier may have moved."""

    def google_reply(self, question, reply):
        sent = {}

        def google(text, source, target):
            sent["input"] = text
            return reply

        with patch.object(pipeline, "_google_translate", google), \
                patch.object(pipeline, "_chat") as chat:
            out = pipeline.translate(question, "Odia", "English")
        return out, sent.get("input", ""), chat

    def test_the_transplanting_ploughing_question(self):
        reply = "How many days before rice transplanting should ploughing be done?"
        out, _sent, chat = self.google_reply(
            "ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?", reply)

        chat.assert_not_called()
        for term in ("rice", "transplanting", "before", "ploughing"):
            self.assertIn(term, out.lower())
        self.assertNotIn("harvest", out.lower())

    def test_the_dap_per_acre_question(self):
        out, sent, _chat = self.google_reply(
            "ଡିଏପି ଗୋଟେ ଏକରରେ କେତେ ପକେଇବା?", "How much DAP to plant in 1 acre?")

        self.assertIn("DAP", sent)
        self.assertIn("1", sent)
        self.assertNotIn("dap 1", out.lower())
        self.assertNotIn("harvest", out.lower())

    def test_the_spoken_range_still_never_becomes_37(self):
        out, conversions = convert_odia_numbers("ରୋଗ ବାର ଦଶ ପନ୍ଦର ଦିନ ହେଇଗଲାଣି।")

        self.assertNotIn("37", out)
        self.assertEqual([], conversions)

    def test_the_fraction_still_converts(self):
        _out, conversions = convert_odia_numbers("ଅଢ଼େଇ ଏକରରେ କେତେ ଦେବି?")

        self.assertEqual("2.5", conversions[0][1])

    def test_the_pest_and_weather_terms_are_unchanged(self):
        self.assertEqual("pest and disease", _reverse_map()[key("ରୋଗପୋକ")])
        self.assertEqual("cyclone", _reverse_map()[key("ବାତ୍ୟା")])

    def test_the_ploughing_terms_are_unchanged(self):
        self.assertEqual("ploughing",
                         matches("ହଳ କରିବା ଉଚିତ")[key("ହଳ କରିବା")])


class DebugLoggingTests(SimpleTestCase):
    """The matched term is visible before Google is called."""

    def test_the_glossary_match_is_logged_before_the_google_call(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: PRODUCTION_EN), \
                patch.object(pipeline, "_chat"):
            with self.assertLogs("irri", level="INFO") as logs:
                pipeline.translate(PRODUCTION_Q, "Odia", "English")
        written = "\n".join(logs.output)

        self.assertIn("5 glossary matches", written)
        self.assertIn("transplanting", written)
        self.assertIn("6 protected input to google", written)
        self.assertIn("7 raw google output", written)
        self.assertIn("9 final output", written)

    def test_the_stem_appears_in_the_matches_when_it_is_the_form_used(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "after transplanting rice"), \
                patch.object(pipeline, "_chat"):
            with self.assertLogs("irri", level="INFO") as logs:
                pipeline.translate("ଧାନ ରୋଇ ପରେ କେତେ ସାର ଦେବି?", "Odia", "English")
        written = "\n".join(logs.output)

        self.assertIn("transplanting", written)
