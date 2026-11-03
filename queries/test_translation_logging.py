"""What Google actually said, before the glossary got involved.

The glossary is applied BEFORE Google sees the text — English spliced into the
question, opaque markers standing in for terms in the answer — so every existing
log line shows an already-influenced translation. There was no way to see what
Google would have produced on its own, which is the one thing you need in order
to tell a glossary problem from a translation problem.

So: the answer direction now logs its stages (it logged nothing at all), and
BOTH directions can log a glossary-free baseline of the same sentence.

The baseline costs a SECOND API call, so it is off unless asked for. Two things
are asserted about that: no extra call when the flag is off, and a baseline
failure can never break the translation it is diagnosing.
"""
import os
import re
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from queries.services import pipeline

ENGLISH = "Apply urea at tillering."
ODIA_Q = "ଧାନରେ ପିଲ କମ ହେଉଛି"


def translate_out(flag=None, google=None):
    """English -> Odia with the outside world stubbed. Returns (log, calls)."""
    calls = []

    def default_google(text, source, target):
        # Echo any glossary markers, as real Google does. A stub that swallows
        # them now looks like Google DROPPING a term, which translate() answers
        # with a second call (see test_contextual_glossary) — and these tests
        # are about the baseline log's call count, not about that retry.
        calls.append(text)
        return " ".join(re.findall(r"ZQX\d+XQZ", text) + ["ଅନୁବାଦ"])

    patches = [
        patch.object(pipeline, "_google_translate",
                     side_effect=google or default_google),
        patch.object(pipeline, "_fix_odia_fluency",
                     side_effect=lambda t, s="", glossary_terms=(): t),
        patch.object(pipeline, "_detransliterate_odia",
                     side_effect=lambda t, s="": t),
        patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"),
    ]
    override = None
    if flag is not None:
        # override_settings, NOT patch.dict(os.environ).
        #
        # The flag is read as `getattr(settings, NAME, None)` and only falls back
        # to the environment when the setting is undefined. LOG_RAW_TRANSLATION
        # is now declared in settings/base.py — it had to be, because decouple
        # reads .env without exporting to os.environ, so a flag set in .env was
        # invisible and STT_TRANSCRIPT_CORRECTION stayed on however often it was
        # switched off. With the setting defined, an os.environ patch is shadowed
        # and silently does nothing.
        #
        # Still NEVER patch.object(settings, ..., create=True): `settings` is a
        # lazy proxy, the attribute lives on the object behind it, and mock
        # deletes it on exit — removing the real setting from the whole process.
        # enable()/disable(), NOT start()/stop() — override_settings is not a
        # mock patcher and does not share that interface.
        override = override_settings(LOG_RAW_TRANSLATION=bool(flag))
    for p in patches:
        p.start()
    if override is not None:
        override.enable()
    try:
        import logging
        logger = logging.getLogger("irri")
        with patch.object(logger, "handlers", []):
            import unittest
            with unittest.TestCase().assertLogs("irri", level="INFO") as logs:
                out = pipeline.translate(ENGLISH, "English", "Odia")
        return "\n".join(logs.output), calls, out
    finally:
        if override is not None:
            override.disable()
        for p in patches:
            p.stop()


class TheAnswerDirectionIsLoggedTests(SimpleTestCase):
    """It used to log nothing, so an Odia answer could not be traced at all."""

    def test_every_stage_is_logged_in_order(self):
        written, _calls, _out = translate_out(flag=False)

        for stage in ("translate_out | 1 english answer",
                      "translate_out | 2 glossary forced",
                      "translate_out | 3 marked input to google",
                      "translate_out | 4 raw google output",
                      "translate_out | 5 glossary restored",
                      "translate_out | 6 after fluency rewrite"):
            self.assertIn(stage, written)

    def test_the_stages_appear_in_numbered_order(self):
        written, _calls, _out = translate_out(flag=False)
        positions = [written.index(f"translate_out | {n}") for n in range(1, 7)]

        self.assertEqual(sorted(positions), positions)

    def test_the_english_that_went_in_is_shown(self):
        written, _calls, _out = translate_out(flag=False)

        self.assertIn(repr(ENGLISH), written)


class TheBaselineTests(SimpleTestCase):
    """The glossary-free translation, on request only."""

    def test_it_is_logged_when_the_flag_is_on(self):
        written, _calls, _out = translate_out(flag=True)

        self.assertIn("BASELINE google, no glossary", written)

    def test_it_is_the_untouched_text_that_is_sent(self):
        # The point of the comparison: no markers in the baseline call.
        _written, calls, _out = translate_out(flag=True)

        self.assertIn(ENGLISH, calls)
        self.assertTrue(any("ZQX" not in c for c in calls))

    def test_nothing_is_logged_and_nothing_is_called_when_off(self):
        written, calls, _out = translate_out(flag=False)

        self.assertNotIn("BASELINE", written)
        self.assertEqual(1, len(calls), "the baseline cost an extra API call")

    def test_it_is_off_by_default(self):
        # No flag set either way — a second call per answer must not be the
        # default, since it is billed per character.
        _written, calls, _out = translate_out()

        self.assertEqual(1, len(calls))

    def test_a_baseline_failure_cannot_break_the_translation(self):
        calls = []

        def flaky(text, source, target):
            calls.append(text)
            if len(calls) == 1:          # the baseline call is the first one
                raise RuntimeError("baseline API down")
            return "ଅନୁବାଦ"

        written, _calls, out = translate_out(flag=True, google=flaky)

        self.assertIn("baseline translation unavailable", written)
        self.assertIn("ଅନୁବାଦ", out)

    def test_the_question_direction_logs_a_baseline_too(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "The rice has fewer tillers."), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"), \
                override_settings(LOG_RAW_TRANSLATION=True):
            with self.assertLogs("irri", level="INFO") as logs:
                pipeline.translate(ODIA_Q, "Odia", "English")

        self.assertIn("translate_in | BASELINE google, no glossary",
                      "\n".join(logs.output))
