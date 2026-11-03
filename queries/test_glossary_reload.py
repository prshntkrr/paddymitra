"""A word added in the browser has to be there on the next request.

THE REPORT
----------
An operator added ବିଲ on the live server, the POST returned 302, the page
reloaded — and the word could not be found afterwards. Two separate cache bugs
produced that, both of them invisible in a single-process test:

1. reload_glossary() cleared a HAND-MAINTAINED list of caches, and
   _reverse_scanner() — the Odia->English alternation, added to this module
   after that function was written — was not on the list. So a new term was
   written to disk, appeared in the editor's table, and was still not
   recognised in a farmer's question until the process restarted.

2. reload_glossary() only clears the caches of the process that CALLS it. Under
   any multi-process server the worker that serves the next request is usually
   not the one that saved, so that worker kept serving its old lru_cache. The
   term was on disk and invisible.

Both are covered here against a temporary glossary file, so the real one is
never touched.
"""
import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase, TestCase

from queries.services import odia_glossary as g

WORD = "ବିଲ"          # the reported word
OTHER = "ହିଂଳ"


class GlossaryFileMixin:
    """Each test gets its own glossary file; the real one is never touched."""

    def setUp(self):
        super().setUp()
        self.real_path = g._GLOSSARY_PATH
        self.tmp = Path(tempfile.mkdtemp()) / "odia_glossary.json"
        self.tmp.write_text(json.dumps({"water": "ପାଣି"}, ensure_ascii=False),
                            encoding="utf-8")
        g._GLOSSARY_PATH = self.tmp
        g.reload_glossary()
        self.addCleanup(self._restore)

    def _restore(self):
        g._GLOSSARY_PATH = self.real_path
        g.reload_glossary()

    def write_behind_its_back(self, english, odia):
        """Simulate ANOTHER process saving a term (another worker, or a hand edit)."""
        data = json.loads(self.tmp.read_text(encoding="utf-8"))
        data[english] = odia
        self.tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class GlossaryFileTestCase(GlossaryFileMixin, SimpleTestCase):
    """No database needed for the service-level tests."""


class TheAddedWordIsVisibleTests(GlossaryFileTestCase):
    """What the operator does in the browser, in the order they do it."""

    def test_it_is_written_to_disk(self):
        self.assertTrue(g.upsert_term("field", WORD))

        self.assertIn(WORD, self.tmp.read_text(encoding="utf-8"))

    def test_it_appears_in_the_table_the_page_renders(self):
        g.upsert_term("field", WORD)

        # glossary_manage() builds its rows from exactly this call.
        self.assertEqual([WORD], g.get_all_variants().get("field"))

    def test_it_is_searchable_by_the_odia_word(self):
        # The page's filter matches against the Odia column too.
        g.upsert_term("field", WORD)
        rendered = " ".join(" ".join(v) for v in g.get_all_variants().values())

        self.assertIn(WORD, rendered)

    def test_it_is_recognised_in_a_farmers_question_immediately(self):
        # BUG 1. The scanner is warmed first, as a live process has served
        # traffic before the operator adds anything.
        g.relevant_reverse_glossary("ପାଣି କେତେ ଦେବି?")
        g.upsert_term("field", WORD)

        found = g.relevant_reverse_glossary(f"ମୋ {WORD}ରେ କେତେ ସାର ଦେବି?")

        self.assertEqual("field", found.get(g.normalize_odia(WORD)))

    def test_the_forward_direction_sees_it_immediately_too(self):
        g.apply_glossary("water")                      # warm the compiled pattern
        g.upsert_term("field", WORD)

        self.assertEqual(WORD, g.get_glossary().get("field"))


class AnotherProcessWroteItTests(GlossaryFileTestCase):
    """BUG 2: the saving worker is not the worker that serves the next request."""

    def test_the_editor_table_picks_up_another_workers_add(self):
        g.get_all_variants()                           # this worker has its own cache
        self.write_behind_its_back("bund", OTHER)

        self.assertEqual([OTHER], g.get_all_variants().get("bund"))

    def test_a_farmers_question_picks_up_another_workers_add(self):
        g.relevant_reverse_glossary("ପାଣି")
        self.write_behind_its_back("bund", OTHER)

        found = g.relevant_reverse_glossary(f"ମୋ {OTHER} ପାଇଁ")

        self.assertEqual("bund", found.get(g.normalize_odia(OTHER)))

    def test_a_hand_edit_needs_no_restart(self):
        g.get_glossary()
        self.write_behind_its_back("bund", OTHER)

        self.assertEqual(OTHER, g.get_glossary().get("bund"))

    def test_a_deletion_by_another_process_is_picked_up_too(self):
        g.upsert_term("field", WORD)
        self.assertIn("field", g.get_all_variants())

        data = json.loads(self.tmp.read_text(encoding="utf-8"))
        data.pop("field")
        self.tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        self.assertNotIn("field", g.get_all_variants())

    def test_an_unchanged_file_is_not_reloaded_on_every_call(self):
        # The guard is a stat(), not a reload — otherwise every request would
        # rebuild the regex and the reverse map.
        g.get_all_variants()
        before = g._load_variants.cache_info().hits
        for _ in range(5):
            g.get_all_variants()

        self.assertEqual(before + 5, g._load_variants.cache_info().hits)


class EveryCacheIsClearedTests(GlossaryFileTestCase):
    """BUG 1's root cause: the list was maintained by hand and went stale."""

    def test_no_cache_in_the_module_survives_a_reload(self):
        # Warm everything, then assert reload_glossary() emptied all of it —
        # including any cache added after this test was written.
        g.get_all_variants()
        g.apply_glossary("water")
        g.relevant_reverse_glossary("ପାଣି")
        g.reload_glossary()

        # reload_glossary() ends by calling _load_glossary(), so the load caches
        # are warm again on purpose — with the NEW file. Everything DERIVED from
        # them must be empty, and that is where the missing one was.
        rewarmed = {"_load_glossary", "_load_variants", "_value_stopwords_normalised"}
        stale = []
        for name, obj in vars(g).items():
            info = getattr(obj, "cache_info", None)
            if callable(info) and info().currsize and name not in rewarmed:
                stale.append(name)

        self.assertEqual([], stale)

    def test_the_scanner_cache_is_among_them(self):
        # Named explicitly: this is the one that was missing.
        g.relevant_reverse_glossary("ପାଣି")
        self.assertTrue(g._reverse_scanner.cache_info().currsize)

        g.reload_glossary()

        self.assertFalse(g._reverse_scanner.cache_info().currsize)


class NothingElseChangedTests(GlossaryFileTestCase):
    """The existing behaviour the rest of the system relies on."""

    def test_the_builtin_fallback_still_shows_through(self):
        self.assertEqual("ସବୁଜ ସାର", g.get_glossary().get("green manure"))

    def test_the_file_still_wins_over_the_builtin(self):
        g.upsert_term("rice", WORD)

        self.assertEqual(WORD, g.get_glossary().get("rice"))

    def test_alternatives_are_still_kept_when_the_primary_changes(self):
        g.upsert_term("field", WORD)
        g.add_variant("field", OTHER)
        g.upsert_term("field", "ଜମି")

        self.assertEqual(["ଜମି", WORD, OTHER], g.get_all_variants()["field"])

    def test_a_missing_file_falls_back_without_erroring(self):
        self.tmp.unlink()

        self.assertEqual("ଧାନ", g.get_glossary().get("rice"))


class TheRenderedPageTests(GlossaryFileMixin, TestCase):
    """The last layer: the actual HTTP POST, and the HTML the search reads.

    Everything above tests the service. This drives the real view and template,
    because the page's filter matches against a data-s attribute rather than the
    visible cells — an escaping or normalisation problem there would look exactly
    like "I added it and it is not found".
    """

    def setUp(self):
        super().setUp()
        from django.contrib.auth import get_user_model
        User = get_user_model()
        self.user = User.objects.create_user(
            username="op", email="op@example.com", password="pw12345!")
        self.client.force_login(self.user)

    def post(self, english, odia, action="upsert"):
        return self.client.post("/queries/glossary/",
                                {"action": action, "english": english, "odia": odia},
                                follow=True)

    def test_the_word_is_in_the_page_after_saving_it(self):
        response = self.post("field", WORD)
        html = response.content.decode("utf-8")

        self.assertIn(WORD, html)

    def test_the_word_is_in_the_attribute_the_search_matches_against(self):
        self.post("field", WORD)
        html = self.client.get("/queries/glossary/").content.decode("utf-8")

        # glFilter() does rows[i].dataset.s.indexOf(query) — so the Odia has to
        # be inside data-s, unescaped, or searching for it finds nothing.
        row = [line for line in html.splitlines() if "data-s=" in line and WORD in line]
        self.assertTrue(row, "the saved term is in no data-s attribute")

    def test_the_operator_is_told_it_was_saved(self):
        response = self.post("field", WORD)
        html = response.content.decode("utf-8")

        self.assertIn("Saved", html)

    def test_odia_in_the_english_box_is_refused_with_a_visible_reason(self):
        # The mistake this page makes easy: the Odia word typed into the English
        # field and the English into the Odia field. It is refused — and the
        # operator must be able to SEE that, or a rejected save looks like a
        # broken one.
        response = self.post(WORD, "field")
        html = response.content.decode("utf-8")

        self.assertIn("must not contain English letters", html)
        self.assertNotIn("Saved", html)

    def test_the_row_count_grows_by_one(self):
        before = len(g.get_all_variants())
        self.post("field", WORD)

        self.assertEqual(before + 1, len(g.get_all_variants()))
