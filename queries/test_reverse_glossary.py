"""One missing word turned a ploughing question into a harvesting one.

THE REPORTED FAILURE (Q-982)
----------------------------
    ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?
    = "How many days before TRANSPLANTING rice should PLOUGHING be done?"

came back as

    "How many days before harvesting should rice be cut?"

The reverse glossary was not at fault for the term everyone suspected: the trace
shows ହଳ କରିବା -> ploughing WAS matched and WAS passed to the translator. What
had no entry was its neighbour ରୋଇବା (transplanting) — the glossary held only
ରୁଆ, the written form, not the verb a farmer speaks. The model guessed the
opposite end of the season, and then rewrote the correctly-matched ploughing as
"cut" so the sentence would agree with its own guess.

So both halves are fixed here: the missing word, and a prompt that treats a
matched term as a constraint rather than a hint.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline
from queries.services.odia_glossary import (_reverse_map, normalize_odia,
                                            relevant_reverse_glossary)


def matches(text: str) -> dict:
    """Glossary matches keyed by NFD-normalised Odia.

    The scanner works on normalised text and returns normalised keys — Odia
    nukta letters and vowel signs have two valid encodings that NFC does not
    unify, which is why the module normalises everywhere. A test that compares a
    literal source string against a returned key is comparing two encodings of
    the same word, so both sides are normalised here.
    """
    return {normalize_odia(od): en
            for od, en in relevant_reverse_glossary(text).items()}


def key(odia: str) -> str:
    return normalize_odia(odia)

QUESTION = "ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?"


class TheReportedQuestionTests(SimpleTestCase):
    """Every term in Q-982 now resolves, and to the right English."""

    def test_the_question_matches_all_of_its_terms(self):
        found = matches(QUESTION)

        self.assertEqual("rice", found[key("ଧାନ")])
        self.assertEqual("transplanting", found[key("ରୋଇବା")])
        self.assertEqual("ploughing", found[key("ହଳ କରିବା")])

    def test_the_word_that_caused_it_now_has_an_entry(self):
        # ରୋଇବା had no entry at all; that is the whole bug.
        self.assertEqual("transplanting", matches("ଧାନ ରୋଇବା")[key("ରୋଇବା")])

    def test_the_ordering_word_resolves_too(self):
        # Without it the model was free to read "X ଆଗରୁ Y" as "Y before X" and
        # invert which operation comes first.
        self.assertEqual("before", matches(QUESTION)[key("ଆଗରୁ")])

    def test_transplanting_is_never_read_as_harvesting(self):
        for english in matches(QUESTION).values():
            self.assertNotIn("harvest", english.lower())


class TransplantingFormsTests(SimpleTestCase):
    """The spoken forms, not just the written one the glossary had."""

    def test_every_spoken_form_resolves(self):
        for spoken in ("ରୋଇବା", "ରୋଇବ", "ରୋଇଲା", "ରୋପଣ", "ରୋପଣ କରିବା"):
            with self.subTest(spoken=spoken):
                found = matches(f"ଧାନ {spoken} ପାଇଁ")

                self.assertEqual("transplanting", found[key(spoken)])

    def test_the_written_form_still_works(self):
        # ରୁଆ was the only form the glossary had; it must keep working.
        self.assertEqual("transplanting",
                         _reverse_map()[normalize_odia("ରୁଆ")])


class LongestPhraseWinsTests(SimpleTestCase):
    """An overlapping shorter term must not take the longer one's place."""

    def test_ploughing_beats_tillage(self):
        found = matches("ହଳ କରିବା ଉଚିତ")

        self.assertEqual({key("ହଳ କରିବା"): "ploughing"}, found)
        self.assertNotIn("tillage", found.values())

    def test_the_other_ploughing_phrasing_also_wins(self):
        found = matches("ହଳ ଧରିବା ଉଚିତ")

        self.assertEqual({key("ହଳ ଧରିବା"): "ploughing"}, found)

    def test_tillage_is_still_recorded_for_the_bare_word(self):
        # The mapping is intact in the map itself; see the KNOWN LIMIT test
        # below for why the bare word is not scannable.
        self.assertEqual("tillage", _reverse_map()[normalize_odia("ହଳ")])

    def test_a_two_character_term_is_not_scanned(self):
        # KNOWN LIMIT, pre-existing and deliberate: _reverse_scanner() skips
        # terms shorter than 3 characters, because a two-letter Odia word
        # matches inside far too much. ହଳ alone therefore yields no hint.
        self.assertEqual({}, relevant_reverse_glossary("ହଳ ଦରକାର"))


class MultipleConstraintsTests(SimpleTestCase):
    """Several matched terms have to survive together."""

    def test_four_terms_coexist_in_one_sentence(self):
        found = matches(QUESTION)

        self.assertEqual(4, len(found))
        self.assertCountEqual(["rice", "transplanting", "before", "ploughing"],
                              list(found.values()))

    def test_an_unrelated_sentence_keeps_its_own_terms(self):
        found = matches("ଜିଆ ଖତ କେତେ ଦେବି?")

        self.assertEqual("vermicompost", found[key("ଜିଆ ଖତ")])
        self.assertNotIn("earthworm", found.values())


class CollisionTests(SimpleTestCase):
    """A discarded entry must be visible, not silent."""

    def test_collisions_are_logged_when_the_map_is_built(self):
        _reverse_map.cache_clear()
        with self.assertLogs("irri", level="INFO") as logs:
            _reverse_map()
        written = "\n".join(logs.output)

        self.assertIn("reverse map", written)
        self.assertIn("collision", written)

    def test_the_log_names_both_the_survivor_and_the_loser(self):
        _reverse_map.cache_clear()
        with self.assertLogs("irri", level="INFO") as logs:
            _reverse_map()
        written = "\n".join(logs.output)

        # The known one: ଅମଳ is recorded for both, and "yield" wins on length.
        self.assertIn("dropped", written)
        self.assertIn("harvesting", written)

    def test_the_rule_is_unchanged_and_deterministic(self):
        # Behaviour deliberately NOT changed — 108 entries would move at once.
        # Two builds must agree, and the shortest English key still wins.
        _reverse_map.cache_clear()
        first = dict(_reverse_map())
        _reverse_map.cache_clear()
        second = dict(_reverse_map())

        self.assertEqual(first, second)
        self.assertEqual("yield", first[normalize_odia("ଅମଳ")])

    def test_the_reverse_only_table_still_overrides_the_inverted_glossary(self):
        # _ODIA_SYNONYMS is applied last on purpose: it is how a bad collision
        # is corrected without touching the forward glossary.
        self.assertEqual("transplanting", _reverse_map()[normalize_odia("ରୋଇବା")])


class PromptTests(SimpleTestCase):
    """The matches reach the model as constraints, not suggestions."""

    def prompt_for(self, text):
        seen = {}

        def capture(messages, **kwargs):
            seen["sys"] = messages[0]["content"]
            return "translated"

        with patch.object(pipeline, "_chat", side_effect=capture):
            pipeline.translate(text, "Odia", "English")
        return seen["sys"]

    def test_the_block_is_marked_authoritative(self):
        prompt = self.prompt_for(QUESTION)

        self.assertIn("AUTHORITATIVE TERMINOLOGY CONSTRAINTS", prompt)
        self.assertIn("constraints, not suggestions", prompt)

    def test_every_matched_term_appears_in_the_prompt(self):
        prompt = self.prompt_for(QUESTION)

        for odia, english in (("ଧାନ", "rice"), ("ରୋଇବା", "transplanting"),
                              ("ହଳ କରିବା", "ploughing"), ("ଆଗରୁ", "before")):
            self.assertIn(key(odia), normalize_odia(prompt))
            self.assertIn(english, prompt)

    def test_the_rules_that_were_missing_are_stated(self):
        prompt = self.prompt_for(QUESTION)

        self.assertIn("Do NOT replace a matched term with a synonym", prompt)
        self.assertIn("Do NOT reinterpret or omit a matched term", prompt)
        self.assertIn("ALL of the matches must survive together", prompt)
        self.assertIn("NEVER turn one field operation into another", prompt)

    def test_the_specific_rewrite_that_happened_is_named(self):
        prompt = self.prompt_for(QUESTION)

        self.assertIn("Transplanting is not harvesting", prompt)
        self.assertIn("ploughing is not cutting", prompt)

    def test_a_sentence_with_no_matches_gets_no_block(self):
        prompt = self.prompt_for("ନମସ୍କାର")

        self.assertNotIn("AUTHORITATIVE TERMINOLOGY CONSTRAINTS", prompt)

    def test_the_translation_is_returned_unchanged(self):
        # Translation only, no explanation bolted on by this change.
        with patch.object(pipeline, "_chat",
                          side_effect=lambda *a, **k:
                          "How many days before transplanting rice should "
                          "ploughing be done?"):
            out = pipeline.translate(QUESTION, "Odia", "English")

        self.assertEqual("How many days before transplanting rice should "
                         "ploughing be done?", out)
        self.assertNotIn("harvest", out.lower())

    def test_the_forward_direction_is_untouched(self):
        # English -> Odia is a different path entirely: it goes through Google
        # Translate, not _chat, and must never consult the REVERSE glossary.
        # Both outbound calls are stubbed so the test makes no network request.
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "ଅନୁବାଦ"), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "ଅନୁବାଦ"), \
                patch.object(pipeline, "relevant_reverse_glossary") as reverse:
            pipeline.translate("Apply urea at tillering.", "English", "Odia")

        reverse.assert_not_called()
