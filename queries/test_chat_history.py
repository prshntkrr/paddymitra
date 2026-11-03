"""Tests for chat history persistence and follow-up detection.

Covers: JSON serialization, pkl migration, file locking, phone sanitization,
max_turns edge cases, follow-up detection false positives/negatives, and
history isolation between users.
"""
import json
import pickle
import shutil
import tempfile
import threading
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from queries.services.chat_history import (
    _clamp_max_turns,
    _sanitize_phone,
    _validate_turn,
    format_history_for_prompt,
    load_chat_history,
    save_chat_turn,
)
from queries.services.pipeline import _is_followup, _ENTITY_REF_RE


# ──────────────────────── chat_history.py ────────────────────────

class PhoneSanitizationTests(SimpleTestCase):
    def test_digits_only(self):
        self.assertEqual(_sanitize_phone("+91-9178-623-980"), "919178623980")

    def test_whatsapp_prefix(self):
        self.assertEqual(_sanitize_phone("whatsapp:+919876543210"), "919876543210")

    def test_empty_returns_empty(self):
        self.assertEqual(_sanitize_phone(""), "")
        self.assertEqual(_sanitize_phone(None), "")

    def test_pure_letters_returns_empty(self):
        self.assertEqual(_sanitize_phone("abc"), "")


class TurnValidationTests(SimpleTestCase):
    def test_valid_turn(self):
        self.assertTrue(_validate_turn({"user": "hello", "assistant": "hi"}))

    def test_user_only(self):
        self.assertTrue(_validate_turn({"user": "hello"}))

    def test_empty_strings_invalid(self):
        self.assertFalse(_validate_turn({"user": "", "assistant": ""}))

    def test_non_dict_invalid(self):
        self.assertFalse(_validate_turn("string"))
        self.assertFalse(_validate_turn(42))
        self.assertFalse(_validate_turn(None))


class MaxTurnsTests(SimpleTestCase):
    def test_zero_clamped_to_one(self):
        self.assertEqual(_clamp_max_turns(0), 1)

    def test_negative_clamped_to_one(self):
        self.assertEqual(_clamp_max_turns(-5), 1)

    def test_positive_unchanged(self):
        self.assertEqual(_clamp_max_turns(10), 10)

    def test_none_uses_default(self):
        result = _clamp_max_turns(None)
        self.assertGreaterEqual(result, 1)


class PersistenceTests(SimpleTestCase):
    """JSON read/write, pkl migration, corruption handling."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = mock.patch(
            "queries.services.chat_history._history_dir",
            return_value=Path(self._tmpdir),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        save_chat_turn("9191", "hello", "hi")
        hist = load_chat_history("9191")
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["user"], "hello")
        self.assertEqual(hist[0]["assistant"], "hi")

    def test_json_format_on_disk(self):
        save_chat_turn("9191", "q", "a")
        jp = Path(self._tmpdir) / "9191.json"
        self.assertTrue(jp.exists())
        data = json.loads(jp.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)

    def test_no_pkl_created(self):
        """New saves should create .json, not .pkl."""
        save_chat_turn("9191", "q", "a")
        self.assertFalse((Path(self._tmpdir) / "9191.pkl").exists())

    def test_max_turns_trims(self):
        for i in range(10):
            save_chat_turn("9191", f"q{i}", f"a{i}", max_turns=3)
        hist = load_chat_history("9191", max_turns=3)
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[0]["user"], "q7")

    def test_pkl_migration(self):
        """A legacy .pkl is migrated to .json on first load."""
        pkl_path = Path(self._tmpdir) / "9191.pkl"
        legacy = [{"user": "old q", "assistant": "old a"}]
        with open(pkl_path, "wb") as f:
            pickle.dump(legacy, f)
        hist = load_chat_history("9191")
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["user"], "old q")
        self.assertFalse(pkl_path.exists())
        self.assertTrue((Path(self._tmpdir) / "9191.json").exists())

    def test_pkl_migration_preserves_turns_then_appends(self):
        """After migration, new turns append to migrated data."""
        pkl_path = Path(self._tmpdir) / "9191.pkl"
        legacy = [{"user": "old", "assistant": "old reply"}]
        with open(pkl_path, "wb") as f:
            pickle.dump(legacy, f)
        load_chat_history("9191")  # triggers migration
        save_chat_turn("9191", "new", "new reply")
        hist = load_chat_history("9191")
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["user"], "old")
        self.assertEqual(hist[1]["user"], "new")

    def test_corrupted_json_returns_empty(self):
        jp = Path(self._tmpdir) / "9191.json"
        jp.write_text("NOT VALID JSON", encoding="utf-8")
        hist = load_chat_history("9191")
        self.assertEqual(hist, [])

    def test_malformed_turns_filtered(self):
        jp = Path(self._tmpdir) / "9191.json"
        data = [{"user": "valid", "assistant": "ok"}, "garbage", 42, None,
                {"user": "", "assistant": ""}]
        jp.write_text(json.dumps(data), encoding="utf-8")
        hist = load_chat_history("9191")
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["user"], "valid")

    def test_empty_phone_returns_empty(self):
        self.assertEqual(load_chat_history(""), [])
        save_chat_turn("", "q", "a")  # should not crash

    def test_user_isolation(self):
        save_chat_turn("1111", "q1", "a1")
        save_chat_turn("2222", "q2", "a2")
        h1 = load_chat_history("1111")
        h2 = load_chat_history("2222")
        self.assertEqual(len(h1), 1)
        self.assertEqual(len(h2), 1)
        self.assertEqual(h1[0]["user"], "q1")
        self.assertEqual(h2[0]["user"], "q2")


class ConcurrencyTests(SimpleTestCase):
    """Two concurrent writes for the same user should not lose turns."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = mock.patch(
            "queries.services.chat_history._history_dir",
            return_value=Path(self._tmpdir),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_concurrent_writes_no_lost_turns(self):
        barrier = threading.Barrier(2)
        errors = []

        def writer(msg):
            try:
                barrier.wait(timeout=5)
                save_chat_turn("9191", msg, f"reply to {msg}")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=("first",))
        t2 = threading.Thread(target=writer, args=("second",))
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)
        self.assertEqual(errors, [])
        hist = load_chat_history("9191")
        self.assertEqual(len(hist), 2, f"Expected 2 turns, got {len(hist)}: {hist}")


class FormatTests(SimpleTestCase):
    def test_empty_history(self):
        self.assertEqual(format_history_for_prompt([]), "")

    def test_formats_turns(self):
        hist = [{"user": "How much urea?", "assistant": "Apply 20 kg/acre."}]
        out = format_history_for_prompt(hist)
        self.assertIn("Farmer: How much urea?", out)
        self.assertIn("PaddyMitra: Apply 20 kg/acre.", out)


# ──────────────── Follow-up detection (pipeline.py) ────────────────

VARIETY_HIST = [
    {"user": "Tell me about 150 day rice varieties",
     "assistant": "The 150-day varieties include CR 1009 SUB-1, JAGANNATH..."},
]

CLARIFYING_HIST = [
    {"user": "Which variety should I plant?",
     "assistant": "Which type of land do you have — irrigated, rainfed, or upland?"},
]

FERTILIZER_HIST = [
    {"user": "How much urea for rice?",
     "assistant": "Apply 20 kg urea per acre at tillering stage."},
]


class FollowupDetectionTests(SimpleTestCase):
    """H1: follow-up detection must not fire on standalone questions."""

    # --- Should fire (true follow-ups) ---

    def test_pronoun_this_fires(self):
        self.assertTrue(_is_followup(
            "What land is suitable for this variety?", VARIETY_HIST))

    def test_pronoun_these_fires(self):
        self.assertTrue(_is_followup(
            "Which of these can grow in lowland?", VARIETY_HIST))

    def test_short_reply_to_clarifying_question(self):
        self.assertTrue(_is_followup("irrigated", CLARIFYING_HIST))
        self.assertTrue(_is_followup("medium land", CLARIFYING_HIST))
        self.assertTrue(_is_followup("rainfed", CLARIFYING_HIST))

    def test_the_same_fires(self):
        self.assertTrue(_is_followup(
            "And the same for upland?", FERTILIZER_HIST))

    def test_which_one_matures_fastest(self):
        self.assertTrue(_is_followup(
            "Which one matures fastest?", VARIETY_HIST))

    # --- Should NOT fire (standalone questions) ---

    def test_what_is_urea_standalone(self):
        """'What is urea?' is a complete question, not a follow-up."""
        self.assertFalse(_is_followup("What is urea?", FERTILIZER_HIST))

    def test_best_rice_variety_standalone(self):
        self.assertFalse(_is_followup("Best rice variety?", FERTILIZER_HIST))

    def test_how_much_dap_standalone(self):
        self.assertFalse(_is_followup("How much DAP?", FERTILIZER_HIST))

    def test_tell_me_about_blast(self):
        self.assertFalse(_is_followup(
            "Tell me about blast disease", VARIETY_HIST))

    def test_no_history_never_fires(self):
        self.assertFalse(_is_followup("irrigated", []))
        self.assertFalse(_is_followup("this variety", []))

    def test_short_reply_without_clarifying_question(self):
        """'irrigated' after a statement (not a ?) is NOT a follow-up."""
        statement_hist = [
            {"user": "Tell me about rice",
             "assistant": "Rice is a staple crop."},
        ]
        self.assertFalse(_is_followup("irrigated", statement_hist))

    def test_long_standalone_no_fire(self):
        self.assertFalse(_is_followup(
            "How much urea should I apply for rice on two acres?",
            VARIETY_HIST))

    # --- Case A: upland variety follow-up ---
    def test_upland_which_of_above(self):
        hist = [{"user": "Which upland rice variety is better?",
                 "assistant": "Suitable upland varieties are IR 55419-42, "
                              "Sugandha-1, IET 18941, IR 77080-B, IET-16430, "
                              "RR-347-166."}]
        self.assertTrue(_is_followup(
            "Which of the above rice has high yield timing?", hist))

    # --- Case B: lowland variety follow-up ---
    def test_lowland_which_of_these(self):
        hist = [{"user": "Which rice seed is good for lowland?",
                 "assistant": "Suitable lowland varieties include Swarna Sub-1, "
                              "CR Dhan 800, CR Dhan 801, MTU 7029."}]
        self.assertTrue(_is_followup(
            "Which of these gives the highest yield?", hist))

    # --- Case C: fertilizer continuation with "it" ---
    def test_fertilizer_continuation_with_it(self):
        hist = [{"user": "How much urea DAP should be applied in 5 acres?",
                 "assistant": "175 kg DAP and 150-175 kg urea for 5 acres."}]
        self.assertTrue(_is_followup(
            "I cannot find urea in our area. Where can I get it and how "
            "will I do farming?", hist))

    # --- Topic change: rice to maize ---
    def test_topic_change_no_pronouns(self):
        self.assertFalse(_is_followup(
            "How to grow maize in kharif season?", VARIETY_HIST))

    # --- Empty/malformed history ---
    def test_none_history(self):
        self.assertFalse(_is_followup("these varieties", None))


class EntityRefTests(SimpleTestCase):
    """_ENTITY_REF_RE detects references to a list of previous entities."""

    def test_which_of_these(self):
        self.assertTrue(_ENTITY_REF_RE.search("Which of these gives highest yield?"))

    def test_the_above(self):
        self.assertTrue(_ENTITY_REF_RE.search("Which of the above rice has high yield?"))

    def test_from_these(self):
        self.assertTrue(_ENTITY_REF_RE.search("From these which is best?"))

    def test_no_entity_ref(self):
        self.assertIsNone(_ENTITY_REF_RE.search("How much urea per acre?"))

    def test_it_not_entity_ref(self):
        self.assertIsNone(_ENTITY_REF_RE.search("Where can I get it?"))

    def test_this_not_entity_ref(self):
        self.assertIsNone(_ENTITY_REF_RE.search(
            "What land is suitable for this variety?"))


# Multi-turn clarification history: user answered "irrigated", now says variety+duration
MULTI_TURN_CLARIFYING_HIST = [
    {"user": "Fertilizer for 2 hectares?",
     "assistant": "Which type of land do you have — irrigated, rainfed, or upland?"},
    {"user": "irrigated",
     "assistant": "Which variety and crop duration are you growing?"},
]


class LongFollowupToQuestionTests(SimpleTestCase):
    """Follow-up replies longer than 3 words to a clarifying question."""

    def test_variety_duration_answer_is_followup(self):
        self.assertTrue(_is_followup(
            "MTU 1010, 120 days", MULTI_TURN_CLARIFYING_HIST))

    def test_four_word_answer_to_clarifying(self):
        self.assertTrue(_is_followup(
            "Swarna Sub-1 lowland irrigated", MULTI_TURN_CLARIFYING_HIST))

    def test_long_standalone_question_not_followup(self):
        """A real question (starts with question word) is never a follow-up."""
        self.assertFalse(_is_followup(
            "What is the best variety for lowland irrigated?",
            MULTI_TURN_CLARIFYING_HIST))

    def test_long_reply_to_statement_not_followup(self):
        """Long reply after a statement (no ?) is NOT a follow-up."""
        statement_hist = [
            {"user": "Tell me about rice",
             "assistant": "Rice is the primary food crop of Odisha."},
        ]
        self.assertFalse(_is_followup(
            "MTU 1010 is a good variety", statement_hist))

    def test_irrigated_in_multi_turn(self):
        hist_with_irrigated = [
            {"user": "Fertilizer for 1900 tons?",
             "assistant": "Please provide your cultivated area and land type?"},
        ]
        self.assertTrue(_is_followup("irrigated", hist_with_irrigated))

    def test_both_area_and_production_in_history(self):
        hist = [
            {"user": "Fertilizer for 1900 tons? Fertilizer for 2 hectares?",
             "assistant": "Which land type — irrigated, rainfed, or upland?"},
            {"user": "irrigated",
             "assistant": "Which variety and duration?"},
        ]
        self.assertTrue(_is_followup("MTU 1010, 120 days", hist))


