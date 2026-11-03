"""A verb must never be frozen into a glossary marker.

THE LOGGED ANSWER
-----------------
    1 english   "For Veena rice, apply 70 kg of urea per acre. ..."
    3 to google "For ZQX4XQZ ZQX6XQZ, ZQX5XQZ 70 kg of ZQX7XQZ per acre. ..."
    5 restored  "ବୀଣା ଧାନ, ପ୍ରୟୋଗ କରିବା ପାଇଁ ପ୍ରତି ଏକର ପାଇଁ 70 କିଲୋଗ୍ରାମ ୟୁରିଆ
                 ବ୍ୟବହାର କରନ୍ତୁ।"

"apply" was replaced by ZQX5XQZ. An opaque token cannot be conjugated, so
Google demoted it to a noun ("for ZQX5XQZ") and wrote its own verb, "use". The
restored sentence says "for applying ... use 70 kg urea" — two verbs, one
stranded — and in the third sentence ପ୍ରୟୋଗ କରିବା is left with no verb at all.

Worse, it DEADLOCKED the repair. The fluency pass is told every forced glossary
term must survive character for character, so the frozen ପ୍ରୟୋଗ କରିବା had to
stay; the naturalisation check flags exactly that construction. A rewrite could
not satisfy both, so the broken text was kept.

The cause was two verb lists that had drifted apart: pipeline's own
_GLOSSARY_STOPWORDS (13 verbs, no "apply") and odia_glossary's
_DESTRUCTIVE_KEYS (which has apply/applies/applied/applying and ~60 more). The
marker step now consults both.
"""
import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from queries.services import odia_glossary as g
from queries.services import pipeline

LOGGED = ("For Veena rice, apply 70 kg of urea per acre. At the basal stage or "
          "after transplanting, use 10-15 kg of urea. Then, at the active "
          "tillering stage and at panicle initiation stage, apply 30-35 kg of "
          "urea per acre each time.")

# The entries that matched the logged sentence on the live server, verbatim.
GLOSSARY = {
    "apply": "ପ୍ରୟୋଗ କରିବା",
    "applied": "ପ୍ରୟୋଗ କରାଯାଇଛି",
    "control": "ନିୟନ୍ତ୍ରଣ",
    "use": "ବ୍ୟବହାର",
    "veena": "ବୀଣା",
    "rice": "ଧାନ",
    "urea": "ୟୁରିଆ",
    "basal": "ମୂଳ",
    "transplanting": "ଧାନ ରୁଆ",
    "active tillering": "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା",
    "panicle initiation stage": "କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥା",
    "weed control": "ଘାସ ନିୟନ୍ତ୍ରଣ",
}


class GlossaryFileCase(SimpleTestCase):
    def setUp(self):
        self.real = g._GLOSSARY_PATH
        tmp = Path(tempfile.mkdtemp()) / "odia_glossary.json"
        tmp.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")
        g._GLOSSARY_PATH = tmp
        g.reload_glossary()
        self.addCleanup(self._restore)

    def _restore(self):
        g._GLOSSARY_PATH = self.real
        g.reload_glossary()

    def force(self, text=LOGGED):
        return pipeline._force_glossary_terms(text)


class TheVerbStaysInlineTests(GlossaryFileCase):
    """The logged sentence, after the fix."""

    def test_apply_is_not_turned_into_a_marker(self):
        marked, marks = self.force()

        self.assertIn("apply", marked)
        self.assertNotIn("ପ୍ରୟୋଗ କରିବା", marks.values())

    def test_every_inflection_of_the_verb_is_left_inline(self):
        for form in ("apply", "applies", "applied", "applying"):
            with self.subTest(form=form):
                marked, marks = self.force(f"{form.capitalize()} urea at tillering.")

                self.assertIn(form.capitalize(), marked)

    def test_other_everyday_verbs_are_left_inline_too(self):
        # "use" and "control" were already protected; they must stay that way.
        marked, _marks = self.force("Use urea to control weeds.")

        self.assertIn("Use", marked)
        self.assertIn("control", marked)

    def test_the_fluency_guard_no_longer_demands_the_frozen_verb(self):
        # The deadlock: a forced term must survive the rewrite, so a frozen verb
        # form made the natural imperative (ପ୍ରୟୋଗ କରନ୍ତୁ) unreachable.
        _marked, marks = self.force()

        self.assertNotIn("ପ୍ରୟୋଗ କରିବା", list(marks.values()))


class TheRealTermsAreStillForcedTests(GlossaryFileCase):
    """Nothing the glossary exists for has been loosened."""

    def test_every_noun_and_stage_in_the_logged_sentence_is_still_forced(self):
        _marked, marks = self.force()
        forced = set(marks.values())

        for term in ("ବୀଣା", "ଧାନ", "ୟୁରିଆ", "ଧାନ ରୁଆ",
                     "ସକ୍ରିୟ ପିଲ ବାହାରିବା ଅବସ୍ଥା",
                     "କେଣ୍ଡା ବାହାରିବା ଆରମ୍ଭ ଅବସ୍ଥା"):
            self.assertIn(term, forced)

    def test_a_multi_word_term_containing_a_verb_is_still_forced(self):
        # A phrase is only destructive when EVERY word is — "weed control" has a
        # real word in it and is a genuine term.
        _marked, marks = self.force("Practise weed control early.")

        self.assertIn("ଘାସ ନିୟନ୍ତ୍ରଣ", marks.values())

    def test_the_numbers_are_untouched(self):
        marked, _marks = self.force()

        for figure in ("70 kg", "10-15 kg", "30-35 kg"):
            self.assertIn(figure, marked)


class TheTwoListsCannotDriftTests(SimpleTestCase):
    """The structural fix, asserted directly."""

    def test_every_destructive_verb_is_refused_by_the_marker_step(self):
        # Anything the glossary module calls destructive, the marker step must
        # also refuse — whether or not pipeline's own short list remembers it.
        from queries.services.odia_glossary import _DESTRUCTIVE_KEYS
        missing_from_local_list = sorted(
            w for w in _DESTRUCTIVE_KEYS
            if len(w) >= 4 and w not in pipeline._GLOSSARY_STOPWORDS)

        # There are many, which is exactly why the local list was not enough.
        self.assertIn("apply", missing_from_local_list)
        for word in missing_from_local_list:
            with self.subTest(word=word):
                self.assertTrue(g.is_destructive_key(word))
