"""A misheard transcript is a different question, answered confidently.

The pipeline answers whatever the STT model GUESSED the farmer said. Nothing
repaired that guess: the raw transcript went straight into translation and
retrieval, so one misheard word produced a fluent answer to a question nobody
asked. This pass recovers the intended QUESTION before retrieval sees it.

It runs inside stt_transcribe(), the single funnel every audio path uses — the
web demo, the API, the catalogues endpoint and WhatsApp (through
transcribe_audio_bytes) — so no caller can miss it and none had to change.

WHAT THE GUARDS ENFORCE, since a prompt cannot enforce itself:

    added or dropped number   refused (new / lost information)
    changed number VALUE      allowed by rule 8, and always logged
    a clarification (rule 13) refused as a query, and logged
    answered / padded / cut   refused by the length window
    no Odia, empty, failure   original kept, usually with no model call at all

Every refusal has the same outcome: keep the ORIGINAL transcript. A transcript
with a misheard word is recoverable — the farmer's recording is stored — but a
question rewritten into a different question is not.

NOTE ON SETTINGS: these use override_settings, never
patch.object(settings, ..., create=True). `settings` is a lazy proxy, so the
real attribute lives on the object behind it; mock therefore treats it as
non-local and DELETES it on exit, which removes GEMINI_API_KEY from the whole
process and breaks every later test that reads it.
"""
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from queries.services import pipeline

# A plausible Odia voice question, and the same question cleaned up.
HEARD = "ଧାନରେ କେତେ ୟୁରିଆ ଦେବି 20 କିଲୋ ପ୍ରତି ଏକରରେ"
FIXED = "ଧାନରେ କେତେ ୟୁରିଆ ଦେବି 20 କିଲୋ ପ୍ରତି ଏକରରେ?"
CLARIFICATION = "CLARIFY: ଆପଣ କେଉଁ ଫସଲ ବିଷୟରେ ପଚାରୁଛନ୍ତି?"


def correct(reply, transcript=HEARD, gemini=True, enabled=None):
    """Run the recovery with the model's reply stubbed.

    Returns (output, calls) so a test can assert that NO model call was made.
    """
    calls = []

    def model(*args, **kwargs):
        calls.append(args)
        if isinstance(reply, Exception):
            raise reply
        return reply

    overrides = {"GEMINI_API_KEY": "k" if gemini else ""}
    if enabled is not None:
        overrides["STT_TRANSCRIPT_CORRECTION"] = enabled
    with override_settings(**overrides), \
            patch.object(pipeline, "_gemini_text", side_effect=model), \
            patch.object(pipeline, "_chat", side_effect=model):
        out = pipeline._correct_odia_transcript(transcript)
    return out, calls


class TheCorrectionIsAppliedTests(SimpleTestCase):
    """A likely mis-hearing is repaired."""

    def test_a_recovered_question_is_returned(self):
        out, calls = correct(FIXED)

        self.assertEqual(FIXED, out)
        self.assertEqual(1, len(calls))

    def test_the_transcript_is_what_the_model_is_given(self):
        seen = {}

        def gemini(prompt, text):
            seen["text"] = text
            return FIXED

        with override_settings(GEMINI_API_KEY="k"), \
                patch.object(pipeline, "_gemini_text", side_effect=gemini):
            pipeline._correct_odia_transcript(HEARD)

        self.assertEqual(HEARD, seen["text"])

    def test_openai_is_the_fallback_when_gemini_has_no_key(self):
        out, calls = correct(FIXED, gemini=False)

        self.assertEqual(FIXED, out)
        self.assertEqual(1, len(calls))

    def test_markdown_the_model_added_is_stripped(self):
        out, _calls = correct(f"**{FIXED}**")

        self.assertEqual(FIXED, out)
        self.assertNotIn("*", out)

    def test_a_doubled_echo_is_collapsed(self):
        out, _calls = correct(FIXED + " " + FIXED)

        self.assertNotIn(FIXED + " " + FIXED, out)


class ThePromptTests(SimpleTestCase):
    """The agreed instructions, verbatim."""

    def prompt(self):
        seen = {}

        def gemini(prompt, text):
            seen["prompt"] = prompt
            return FIXED

        with override_settings(GEMINI_API_KEY="k"), \
                patch.object(pipeline, "_gemini_text", side_effect=gemini):
            pipeline._correct_odia_transcript(HEARD)
        return seen["prompt"]

    def test_it_is_framed_as_question_recovery_for_retrieval(self):
        prompt = self.prompt()

        self.assertIn("recover the most likely intended question", prompt)
        self.assertIn("before it is sent for knowledge-base retrieval", prompt)

    def test_the_spoken_question_outranks_the_transcription(self):
        self.assertIn("Assume the farmer's spoken question is more likely to be "
                      "correct than the transcription", self.prompt())

    def test_every_rule_that_bounds_the_change_is_present(self):
        prompt = self.prompt()

        for rule in ("Do not answer the question",
                     "Do not add new information that was not implied",
                     "Do not change numbers, quantities, dates, crop stages, "
                     "or measurements unless they are clearly transcription "
                     "errors",
                     "Preserve the farmer's original intent",
                     "Avoid unnecessary rewriting",
                     "restore the most likely intended meaning",
                     "infer the most likely intended agricultural term",
                     "clean, natural Odia question suitable for knowledge-base "
                     "retrieval",
                     "a single mistranscribed word changes the entire meaning",
                     "verify that the correction improves clarity"):
            self.assertIn(rule, prompt)

    def test_the_clarification_rule_and_its_marker_are_stated(self):
        prompt = self.prompt()

        self.assertIn("return a short clarification question instead of "
                      "guessing", prompt)
        self.assertIn("CLARIFY:", prompt)

    def test_it_asks_for_the_question_only(self):
        prompt = self.prompt()

        self.assertIn("return ONLY the corrected Odia question", prompt)
        self.assertIn("no explanation, note or comment", prompt)


class NumbersTests(SimpleTestCase):
    """Rule 8: a misheard number may be fixed; one may not appear or vanish."""

    def test_a_misheard_number_may_be_corrected(self):
        # "5" heard for "15" is a transcription error like any other.
        heard = "ରୋଇବା ପରେ 5 ଦିନରେ ସାର ଦେବି କି?"
        fixed = "ରୋଇବା ପରେ 15 ଦିନରେ ସାର ଦେବି କି?"
        with self.assertLogs("irri", level="INFO"):
            out, _calls = correct(fixed, transcript=heard)

        self.assertEqual(fixed, out)

    def test_a_changed_number_is_always_logged(self):
        heard = "ରୋଇବା ପରେ 5 ଦିନରେ ସାର ଦେବି କି?"
        fixed = "ରୋଇବା ପରେ 15 ଦିନରେ ସାର ଦେବି କି?"
        with self.assertLogs("irri", level="WARNING") as logs:
            correct(fixed, transcript=heard)

        self.assertIn("changed a number", "\n".join(logs.output))

    def test_a_dropped_number_is_refused(self):
        out, _calls = correct(FIXED.replace("20 କିଲୋ ", ""))

        self.assertEqual(HEARD, out)

    def test_an_added_number_is_refused(self):
        out, _calls = correct(FIXED.replace("ପ୍ରତି ଏକରରେ", "ପ୍ରତି 1 ଏକରରେ"))

        self.assertEqual(HEARD, out)

    def test_an_unchanged_number_is_not_logged_as_changed(self):
        with self.assertLogs("irri", level="INFO") as logs:
            correct(FIXED)

        self.assertNotIn("changed a number", "\n".join(logs.output))


class TheClarificationTests(SimpleTestCase):
    """Rule 13: right for a person, wrong to retrieve on."""

    def test_a_clarification_never_becomes_the_query(self):
        out, _calls = correct(CLARIFICATION)

        self.assertEqual(HEARD, out)

    def test_the_clarification_is_logged_so_it_is_not_lost(self):
        with self.assertLogs("irri", level="WARNING") as logs:
            correct(CLARIFICATION)
        written = "\n".join(logs.output)

        self.assertIn("asked for clarification", written)
        self.assertIn("ଆପଣ କେଉଁ ଫସଲ", written)

    def test_the_marker_is_recognised_whatever_its_case(self):
        out, _calls = correct(CLARIFICATION.replace("CLARIFY:", "clarify:"))

        self.assertEqual(HEARD, out)


class TheGuardsKeepTheOriginalTests(SimpleTestCase):
    """Every other way the recovery can go wrong ends with the original."""

    def test_an_answer_instead_of_a_question_is_refused(self):
        # Rule 6. An answer is much longer than the question.
        answered = ("ଧାନରେ ଏକର ପ୍ରତି 20 କିଲୋ ୟୁରିଆ ତିନି ଥରରେ ଦିଅନ୍ତୁ। ପ୍ରଥମେ ମୂଳ "
                    "ସାର ଭାବରେ, ତାପରେ ପିଲ ବାହାରିବା ସମୟରେ ଏବଂ କେଣ୍ଡା ବାହାରିବା "
                    "ସମୟରେ ପ୍ରୟୋଗ କରନ୍ତୁ। ଏହା ଅମଳ ବଢ଼ାଇବ ଏବଂ ଗଛ ସୁସ୍ଥ ରହିବ।")
        out, _calls = correct(answered)

        self.assertEqual(HEARD, out)

    def test_a_reply_that_dropped_most_of_the_question_is_refused(self):
        out, _calls = correct("ୟୁରିଆ")

        self.assertEqual(HEARD, out)

    def test_a_model_failure_keeps_the_original(self):
        out, _calls = correct(RuntimeError("model down"))

        self.assertEqual(HEARD, out)

    def test_an_empty_reply_keeps_the_original(self):
        out, _calls = correct("   ")

        self.assertEqual(HEARD, out)

    def test_a_reply_with_no_odia_keeps_the_original(self):
        # An English explanation is not a recovered Odia question.
        out, _calls = correct("The farmer asks about urea dose.")

        self.assertEqual(HEARD, out)


class NothingToCorrectTests(SimpleTestCase):
    """No model call at all — this is on the path of every voice question."""

    def test_an_empty_transcript_is_untouched_and_costs_nothing(self):
        for empty in ("", "   ", None):
            with self.subTest(transcript=empty):
                out, calls = correct(FIXED, transcript=empty)

                self.assertEqual(empty, out)
                self.assertEqual([], calls)

    def test_an_english_transcript_is_untouched_and_costs_nothing(self):
        english = "how much urea per acre for paddy"
        out, calls = correct(FIXED, transcript=english)

        self.assertEqual(english, out)
        self.assertEqual([], calls)

    def test_a_romanised_odia_transcript_is_untouched(self):
        # The Whisper fallback produces these; correcting romanisation is a
        # different job (canonicalise/synonyms already handle it downstream).
        romanised = "dhan re kete urea debi"
        out, calls = correct(FIXED, transcript=romanised)

        self.assertEqual(romanised, out)
        self.assertEqual([], calls)

    def test_the_off_switch_skips_the_call(self):
        out, calls = correct(FIXED, enabled=False)

        self.assertEqual(HEARD, out)
        self.assertEqual([], calls)

    def test_it_is_on_by_default(self):
        out, calls = correct(FIXED)

        self.assertEqual(FIXED, out)
        self.assertEqual(1, len(calls))


class TheFunnelTests(SimpleTestCase):
    """stt_transcribe() applies it, for every backend, without changing its
    contract."""

    def run_stt(self, raw, language="Odia", reply=FIXED):
        with override_settings(GEMINI_API_KEY="k"), \
                patch.object(pipeline, "_stt_transcribe_raw",
                             side_effect=lambda path, hint="": (raw, "sarvam_odia")), \
                patch.object(pipeline, "_gemini_text", side_effect=lambda p, t: reply):
            return pipeline.stt_transcribe("/tmp/voice.ogg", language_hint=language)

    def test_the_question_is_recovered_on_the_way_out(self):
        text, _backend = self.run_stt(HEARD)

        self.assertEqual(FIXED, text)

    def test_the_backend_label_is_unchanged(self):
        _text, backend = self.run_stt(HEARD)

        self.assertEqual("sarvam_odia", backend)

    def test_it_still_returns_a_two_tuple(self):
        result = self.run_stt(HEARD)

        self.assertEqual(2, len(result))

    def test_a_clarification_does_not_reach_the_pipeline(self):
        text, _backend = self.run_stt(HEARD, reply=CLARIFICATION)

        self.assertEqual(HEARD, text)
        self.assertNotIn("CLARIFY", text)

    def test_an_english_transcript_passes_straight_through(self):
        text, _backend = self.run_stt("how much urea", language="English")

        self.assertEqual("how much urea", text)


class TheSharedEmphasisHelperTests(SimpleTestCase):
    """One implementation, used by the fluency rewrite and by this pass."""

    def test_it_strips_both_markers_and_keeps_the_words(self):
        self.assertEqual("ସାର ପ୍ରୟୋଗ କରନ୍ତୁ",
                         pipeline._strip_emphasis("**ସାର ପ୍ରୟୋଗ** କରନ୍ତୁ"))
        self.assertEqual("ସାର ପ୍ରୟୋଗ",
                         pipeline._strip_emphasis("__ସାର ପ୍ରୟୋଗ__"))

    def test_a_stray_marker_is_removed(self):
        self.assertNotIn("*", pipeline._strip_emphasis("**ସାର ପ୍ରୟୋଗ"))

    def test_text_without_markers_is_returned_unchanged(self):
        self.assertEqual(FIXED, pipeline._strip_emphasis(FIXED))
