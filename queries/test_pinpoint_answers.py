"""Same question, same answer — and only the answer.

TWO REPORTS, ONE FILE
---------------------
"I asked the same question yesterday and got an answer; today it says it is out
of context." and "you give too much extra information — answer the question
directly."

The second is a prompt problem. The first is NOT: generation is already pinned
(temperature=0, seed=1), so the same question can only drift if its INPUT
drifted. It had. The answer path learns any English word left in Latin script
when an answer is prepared for speech, and those learned terms are inverted into
the map that translates the next farmer's QUESTION. Ordinary words got learned
before `is_auto_learn_blocked()` existed, and nothing filters by key at load
time, so they are live in both directions:

    ଅଧିକ ସାର ଦେଲେ କି ହେବ?      "if I apply MORE fertiliser"
        -> high fertiliser ଦେଲେ କି ହେବ?          ("more" replaced by "high")

    ଏହି ଋତୁରେ କେତେ ସାର ଦେବି?   "how much fertiliser THIS SEASON"
        -> that time କେତେ fertiliser ଦେବି?       (the season is gone)

The second question cannot retrieve anything sensible, and that is what reaches
the farmer as "out of context". These are DATA, not code — which is exactly why
the behaviour changed from one day to the next.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from queries.services import odia_glossary as g
from queries.services import pipeline

# Every ordinary word found live in the deployed glossary, with the Odia it was
# recorded against. Kept verbatim so the guard is tested against real damage.
LEARNED_JUNK = {
    "high": "ଅଧିକ",
    "is": "ହେଉଛି",
    "in": "ରେ",
    "it": "ଏହା",
    "this": "ଏହା",
    "to": "କୁ",
    "you": "ଆପଣ",
    "with": "ସହିତ",
    "or": "ଅଥବା",
    "after": "ପରେ",
    "during": "ସମୟରେ",
    "first": "ପ୍ରଥମ",
    "then": "ବେଳେ",
    "usually": "ସାଧାରଣତଃ",
    "helps": "ସାହାଯ୍ୟ କରେ",
    "apply": "ପ୍ରୟୋଗ କରିବା",
    "control": "ନିୟନ୍ତ୍ରଣ",
    "content": "ଅଂଶ",
    "released": "ପ୍ରକାଶିତ",
    "shortly": "ଅଳ୍ପ ସମୟ ପରେ",
    "that time": "ଏହି ଋତୁ",
    "suitable for": "ପାଇଁ ଉପଯୁକ୍ତ",
    "matures in": "ପରିପକ୍ୱ ହୁଏ",
}


class TheGuardRefusesOrdinaryWordsTests(SimpleTestCase):
    """None of these may ever be learned again."""

    def test_every_word_found_live_is_refused_today(self):
        for word in LEARNED_JUNK:
            with self.subTest(word=word):
                self.assertTrue(g.is_auto_learn_blocked(word),
                                f"{word!r} could be learned again")

    def test_real_agricultural_terms_are_still_learnable(self):
        # The guard must not have been widened into uselessness.
        for word in ("rice", "dap", "urea", "tillering", "blast", "azolla"):
            with self.subTest(word=word):
                self.assertFalse(g.is_auto_learn_blocked(word))

    def test_the_learner_drops_them_even_if_asked_to_record_them(self):
        with patch.object(g, "_atomic_write_glossary") as write:
            g.add_terms({"high": "ଅଧିକ", "usually": "ସାଧାରଣତଃ"})

        write.assert_not_called()


class GlossaryFileTests(SimpleTestCase):
    """The cleaner, against a file holding exactly the live damage."""

    def setUp(self):
        self.real_path = g._GLOSSARY_PATH
        self.tmp = Path(tempfile.mkdtemp()) / "odia_glossary.json"
        keep = {"rice": "ଧାନ", "urea": "ୟୁରିଆ", "tillering": "କୁଆଁରା ବାହାରିବା"}
        self.tmp.write_text(json.dumps({**keep, **LEARNED_JUNK}, ensure_ascii=False),
                            encoding="utf-8")
        g._GLOSSARY_PATH = self.tmp
        g.reload_glossary()
        self.addCleanup(self._restore)

    def _restore(self):
        g._GLOSSARY_PATH = self.real_path
        g.reload_glossary()

    def clean(self, apply=False):
        from io import StringIO
        out = StringIO()
        call_command("glossary_clean", stdout=out, **({"apply": True} if apply else {}))
        return out.getvalue()


class TheQuestionIsHijackedTests(GlossaryFileTests):
    """What the farmer asked, and what the translator was told they asked."""

    MORE = "ଅଧିକ ସାର ଦେଲେ କି ହେବ?"            # "if I apply MORE fertiliser"
    SEASON = "ଏହି ଋତୁରେ କେତେ ସାର ଦେବି?"        # "how much fertiliser THIS SEASON"

    def test_more_is_read_as_high_before_the_fix(self):
        marked, _applied = pipeline._force_reverse_glossary_terms(self.MORE)

        self.assertIn("high", marked)

    def test_this_season_becomes_that_time_before_the_fix(self):
        marked, _applied = pipeline._force_reverse_glossary_terms(self.SEASON)

        self.assertIn("that time", marked)
        self.assertNotIn("season", marked)

    def test_cleaning_leaves_more_alone(self):
        self.clean(apply=True)
        marked, _applied = pipeline._force_reverse_glossary_terms(self.MORE)

        self.assertNotIn("high", marked)
        self.assertIn("ଅଧିକ", marked)          # left for the translator to render

    def test_cleaning_leaves_the_season_alone(self):
        self.clean(apply=True)
        marked, _applied = pipeline._force_reverse_glossary_terms(self.SEASON)

        self.assertNotIn("that time", marked)
        self.assertIn("ଋତୁ", marked)

    def test_the_real_terms_are_still_forced_after_cleaning(self):
        self.clean(apply=True)
        marked, applied = pipeline._force_reverse_glossary_terms("ଧାନ ପାଇଁ କେତେ ୟୁରିଆ?")

        self.assertIn("rice", marked)
        self.assertIn("urea", marked)
        self.assertTrue(applied)


class TheCleanerTests(GlossaryFileTests):
    """It reports before it touches anything."""

    def test_it_lists_every_ordinary_word(self):
        report = self.clean()

        for word in LEARNED_JUNK:
            self.assertIn(repr(word), report)

    def test_it_changes_nothing_without_apply(self):
        before = self.tmp.read_text(encoding="utf-8")
        self.clean()

        self.assertEqual(before, self.tmp.read_text(encoding="utf-8"))

    def test_apply_removes_them_and_keeps_the_real_terms(self):
        self.clean(apply=True)
        data = json.loads(self.tmp.read_text(encoding="utf-8"))

        self.assertEqual({"rice", "urea", "tillering"}, set(data))

    def test_a_backup_is_written_before_changing_anything(self):
        self.clean(apply=True)
        backup = self.tmp.with_name("odia_glossary.backup.json")

        self.assertTrue(backup.exists())
        self.assertIn("high", json.loads(backup.read_text(encoding="utf-8")))

    def test_it_is_safe_to_run_twice(self):
        self.clean(apply=True)
        report = self.clean(apply=True)

        self.assertIn("Nothing to clean", report)

    def test_the_glossary_is_live_again_immediately(self):
        self.clean(apply=True)

        self.assertNotIn("high", g.get_glossary())
        self.assertEqual("ଧାନ", g.get_glossary().get("rice"))


def prompt_for(question):
    """The grounding prompt built for `question`, with no network calls."""
    seen = {}

    def chat(messages, **kwargs):
        seen.setdefault("sys", messages[0]["content"])
        return "answer"

    with patch.object(pipeline, "_chat", side_effect=chat), \
            patch.object(pipeline, "_gemini_text", side_effect=lambda *a, **k: "answer"):
        pipeline.generate_answer(question, [{"chunk": "Apply 20 kg urea at tillering.",
                                             "metadata": {"document_title": "Guide"}}])
    return seen.get("sys", "")


class TheAnswerShapeTests(SimpleTestCase):
    """Pin-point: the answer opens with the answer, and stops."""

    QUESTION = "How much urea should be applied at tillering?"

    def test_the_shape_rules_are_in_the_prompt(self):
        prompt = prompt_for(self.QUESTION)

        self.assertIn("ANSWER SHAPE", prompt)
        self.assertIn("Open with the answer itself", prompt)

    def test_it_forbids_restating_the_question(self):
        prompt = prompt_for(self.QUESTION)

        self.assertIn("Do not restate or rephrase the question", prompt)
        self.assertIn("regarding your question", prompt)

    def test_it_forbids_a_closing_line(self):
        prompt = prompt_for(self.QUESTION)

        self.assertIn("No closing line", prompt)

    def test_it_forbids_framing_sentences(self):
        prompt = prompt_for(self.QUESTION)

        self.assertIn("must carry a fact the farmer asked for", prompt)

    def test_it_states_no_character_or_word_limit(self):
        # A limit makes the model drop a figure the farmer needed. The rule is
        # about padding, never about length.
        prompt = prompt_for(self.QUESTION)
        shape = prompt.split("ANSWER SHAPE")[1][:1200]

        for banned in ("characters", "words or fewer", "at most", "maximum",
                       "sentences or fewer", "one sentence only"):
            self.assertNotIn(banned, shape)

    def test_an_ordinary_question_now_gets_a_scope_note(self):
        # Every question used to fall through with no scope note at all.
        prompt = prompt_for("Which varieties suit upland?")

        self.assertIn("ANSWER SCOPE for THIS question", prompt)
        self.assertIn("Answer this one directly", prompt)

    def test_the_default_note_does_not_narrow_what_may_be_said(self):
        # "What fertiliser should I apply in DSR?" is default scope AND must
        # still get the complete schedule — the default note must not fight it.
        prompt = prompt_for("What fertiliser should I apply in DSR?")

        self.assertIn("Everything the CONTEXT supports FOR WHAT WAS ASKED", prompt)
        self.assertIn("give the COMPLETE schedule", prompt)

    def test_the_existing_scopes_are_unchanged(self):
        for question, expected in (
                ("How much DAP per acre?", pipeline.SCOPE_QUANTITY),
                ("What is the duration of BRRI Dhan-69?", pipeline.SCOPE_SINGLE_FACT),
                ("How should I manage fertilizer for rice?", pipeline.SCOPE_BROAD)):
            with self.subTest(question=question):
                self.assertEqual(expected, pipeline.response_scope(question))
