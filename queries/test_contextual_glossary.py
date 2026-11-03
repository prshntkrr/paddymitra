"""Freeze a glossary term only when its exact surface form is genuinely required.

THE PRODUCTION FAILURE
----------------------
    english  "Fertilizer application should be done at the time of transplanting
              or within 5 days after transplanting for effective nutrient uptake."
    forced   ['ରୁଆ ପରର ଦିନ', 'ସାର ପ୍ରୟୋଗ', 'ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ', 'ଧାନ ରୁଆ']
    marked   "ZQX1 should be done at the time of ZQX3 or within 5 ZQX0 for
              effective ZQX2."
    google   "ZQX1 ପ୍ରଭାବଶାଳୀ ZQX3 ସମୟରେ କିମ୍ବା 5 ZQX0 ମଧ୍ୟରେ କରାଯିବା ଉଚିତ।"
    final    "... କିମ୍ବା ୫ ଦିନ ରୁଆ ପରର ଦିନ ମଧ୍ୟରେ କରନ୍ତୁ।"

Two separate defects:

1. A RELATIONAL PHRASE WAS FROZEN. "days after transplanting" -> ରୁଆ ପରର ଦିନ is
   not a thing, it is a relation: its number lives OUTSIDE the term ("5 days
   after transplanting"). Frozen, the number cannot bind to ଦିନ, so Odia gets
   "5 ରୁଆ ପରର ଦିନ" — and the fluency pass, required to keep that exact string,
   patched it into "୫ ଦିନ ରୁଆ ପରର ଦିନ". Such phrases must translate
   compositionally, with the TERMINOLOGY inside them (transplanting -> ଧାନ ରୁଆ)
   still frozen.

2. A PLACEHOLDER WAS SILENTLY LOST. Google dropped ZQX2 altogether — "for
   effective nutrient uptake" vanished and ପ୍ରଭାବଶାଳୀ was left stranded. Restoring
   cannot bring back a marker that is not there, and nothing checked. A dropped
   marker is now detected and the sentence re-translated with that term inline.

THE CLASSIFICATION
------------------
    action      the value's verb must inflect (unchanged from the last fix)
    contextual  a relational/temporal phrase — translate compositionally
    protected   noun-headed terminology — freeze exactly

Odia is head-final, so the test reads the ends of the value: a value that ENDS
in a relator (X ପରେ, ଏହି ସମୟରେ), STARTS with one (ପାଇଁ ଉପଯୁକ୍ତ) or ends in a
counted time noun (ରୁଆ ପରର ଦିନ) is contextual. A relator INSIDE a noun-headed
term does not count — foliar spray -> ପତ୍ର ଉପରେ ସିଞ୍ଚନ is terminology.
"""
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import odia_glossary as g
from queries.services import pipeline
from queries.services.technical_terms import normalize_odia

PRODUCTION = ("Fertilizer application should be done at the time of transplanting "
              "or within 5 days after transplanting for effective nutrient uptake.")

# Verbatim live values.
GLOSSARY = {
    # the four production terms
    "fertilizer application": "ସାର ପ୍ରୟୋଗ",
    "transplanting": "ଧାନ ରୁଆ",
    "days after transplanting": "ରୁଆ ପରର ଦିନ",
    "nutrient uptake": "ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ",
    # contextual: relational / temporal phrases
    "after submergence": "ଜଳମଗ୍ନ ହେବା ପରେ",
    "suitable for": "ପାଇଁ ଉପଯୁକ୍ତ",
    "during this season": "ଏହି ସମୟରେ",
    # protected despite a relator INSIDE the term
    "foliar spray": "ପତ୍ର ଉପରେ ସିଞ୍ଚନ",
    "pre-emergence application": "ଗଜା ବାହାରିବା ପୂର୍ବରୁ ପ୍ରୟୋଗ",
    "machine transplanted rice": "ମେସିନ୍ ଦ୍ୱାରା ରୁଆ ହୋଇଥିବା ଧାନ",
    "time of planting": "ଲଗାଇବା ସମୟ",
    "muriate of potash": "ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ",
    # entities
    "rice": "ଧାନ",
    "urea": "ୟୁରିଆ",
    "veena": "ବୀଣା",
    "swarna": "ସ୍ୱର୍ଣ୍ଣ",
    "tillering": "ପିଲ ଧରିବା",
    # action verbs
    "apply": "ପ୍ରୟୋଗ କରିବା",
    "use": "ବ୍ୟବହାର କରିବା",
    "give": "ଦେବା",
}

_ASCII = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")


def n(text):
    return normalize_odia(text or "").translate(_ASCII)


class GlossaryCase(SimpleTestCase):
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

    def forced(self, text):
        marked, marks = pipeline._force_glossary_terms(text)
        return marked, [n(v) for v in marks.values()]


# ── Unit level: selection ─────────────────────────────────────────────────

class RelationalPhrasesTranslateCompositionallyTests(GlossaryCase):

    def test_a_within_5_days_after_transplanting(self):
        marked, forced = self.forced("Apply urea within 5 days after transplanting.")

        self.assertNotIn(n("ରୁଆ ପରର ଦିନ"), forced)
        self.assertIn("5 days after", marked)        # left for the translator
        # ...but the TERMINOLOGY inside the phrase is still frozen.
        self.assertIn(n("ଧାନ ରୁଆ"), forced)

    def test_b_at_the_time_of_transplanting(self):
        marked, forced = self.forced("Apply urea at the time of transplanting.")

        self.assertIn("at the time of", marked)
        self.assertIn(n("ଧାନ ରୁଆ"), forced)

    def test_c_the_production_sentence(self):
        _marked, forced = self.forced(PRODUCTION)

        self.assertNotIn(n("ରୁଆ ପରର ଦିନ"), forced)
        for term in ("ସାର ପ୍ରୟୋଗ", "ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ", "ଧାନ ରୁଆ"):
            self.assertIn(n(term), forced)

    def test_d_effective_nutrient_uptake_stays_protected_terminology(self):
        _marked, forced = self.forced("Irrigate for effective nutrient uptake.")

        self.assertIn(n("ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ"), forced)

    def test_other_relational_phrases_are_not_frozen(self):
        for sentence, value in (("Drain the field after submergence.", "ଜଳମଗ୍ନ ହେବା ପରେ"),
                                ("This variety is suitable for upland.", "ପାଇଁ ଉପଯୁକ୍ତ"),
                                ("Apply urea during this season.", "ଏହି ସମୟରେ")):
            with self.subTest(sentence=sentence):
                _marked, forced = self.forced(sentence)
                self.assertNotIn(n(value), forced)


class ProtectedTerminologyStaysFrozenTests(GlossaryCase):
    """Requirement 10e: the glossary is not weakened for real terms."""

    def test_e_rice_urea_and_variety_names(self):
        _marked, forced = self.forced("Give urea to Swarna and Veena rice at tillering.")

        for term in ("ୟୁରିଆ", "ସ୍ୱର୍ଣ୍ଣ", "ବୀଣା", "ଧାନ", "ପିଲ ଧରିବା"):
            self.assertIn(n(term), forced)

    def test_terms_with_a_relator_inside_a_noun_head_stay_frozen(self):
        for sentence, value in (("Use foliar spray of zinc.", "ପତ୍ର ଉପରେ ସିଞ୍ଚନ"),
                                ("Do pre-emergence application now.",
                                 "ଗଜା ବାହାରିବା ପୂର୍ବରୁ ପ୍ରୟୋଗ"),
                                ("Grow machine transplanted rice.",
                                 "ମେସିନ୍ ଦ୍ୱାରା ରୁଆ ହୋଇଥିବା ଧାନ"),
                                ("Note the time of planting.", "ଲଗାଇବା ସମୟ"),
                                ("Apply muriate of potash.", "ମ୍ୟୁରିଏଟ୍ ଅଫ୍ ପୋଟାଶ")):
            with self.subTest(sentence=sentence):
                _marked, forced = self.forced(sentence)
                self.assertIn(n(value), forced)


class ActionVerbsStillInflectTests(GlossaryCase):
    """Requirement 10f and 11: the previous fix is intact."""

    def test_f_apply_use_give_are_not_frozen(self):
        marked, forced = self.forced("Apply 20 kg urea, use DAP, give MOP.")

        for verb in ("Apply", "use", "give"):
            self.assertIn(verb, marked)
        for value in ("ପ୍ରୟୋଗ କରିବା", "ବ୍ୟବହାର କରିବା", "ଦେବା"):
            self.assertNotIn(n(value), forced)


class TheClassifierTests(SimpleTestCase):
    """One classification, three classes."""

    def cls(self, english, odia):
        from queries.services.odia_glossary import glossary_term_class
        return glossary_term_class(english, odia)

    def test_the_four_production_terms(self):
        self.assertEqual("protected", self.cls("fertilizer application", "ସାର ପ୍ରୟୋଗ"))
        self.assertEqual("protected", self.cls("transplanting", "ଧାନ ରୁଆ"))
        self.assertEqual("contextual", self.cls("days after transplanting", "ରୁଆ ପରର ଦିନ"))
        self.assertEqual("protected", self.cls("nutrient uptake", "ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ"))

    def test_relational_phrases_are_contextual(self):
        for english, odia in (("after submergence", "ଜଳମଗ୍ନ ହେବା ପରେ"),
                              ("before submergence", "ଜମିରେ ପାଣି ଜମିବା ପୂର୍ବରୁ"),
                              ("suitable for", "ପାଇଁ ଉପଯୁକ୍ତ"),
                              ("during this season", "ଏହି ସମୟରେ"),
                              ("shortly", "ଅଳ୍ପ ସମୟ ପରେ")):
            with self.subTest(english=english):
                self.assertEqual("contextual", self.cls(english, odia))

    def test_a_relator_inside_a_noun_headed_term_is_protected(self):
        for english, odia in (("foliar spray", "ପତ୍ର ଉପରେ ସିଞ୍ଚନ"),
                              ("pre-emergence application", "ଗଜା ବାହାରିବା ପୂର୍ବରୁ ପ୍ରୟୋଗ"),
                              ("climate change adaptation", "ଜଳବାୟୁ ପରିବର୍ତ୍ତନ ସହ ଅନୁକୂଳନ"),
                              ("nutrient manager for rice", "ଧାନ ପାଇଁ ସାର ପରିଚାଳନା"),
                              ("time of planting", "ଲଗାଇବା ସମୟ")):
            with self.subTest(english=english):
                self.assertEqual("protected", self.cls(english, odia))

    def test_action_verbs_keep_precedence(self):
        # The previous classifier is not reverted or overridden.
        from queries.services.odia_glossary import is_action_verb_entry

        self.assertTrue(is_action_verb_entry("apply", "ପ୍ରୟୋଗ କରିବା"))
        self.assertEqual("action", self.cls("apply", "ପ୍ରୟୋଗ କରିବା"))
        self.assertEqual("action", self.cls("puddling", "କାଦୁଅ କରିବା"))


class InnerTermsTests(GlossaryCase):
    def test_the_terminology_inside_a_contextual_phrase_is_found(self):
        from queries.services.odia_glossary import freezable_sub_terms

        self.assertEqual({"transplanting": "ଧାନ ରୁଆ"},
                         freezable_sub_terms("days after transplanting"))


# ── Second selection path: LLM-fallback prompt ────────────────────────────

class TheFallbackPromptTests(GlossaryCase):
    def fallback_prompt(self, text=PRODUCTION):
        seen = {}

        def chat(messages, **kwargs):
            seen.setdefault("sys", messages[0]["content"])
            return "ଅନୁବାଦ"

        with patch.object(pipeline, "_google_translate",
                          side_effect=RuntimeError("google down")), \
                patch.object(pipeline, "_chat", side_effect=chat), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t):
            pipeline.translate(text, "English", "Odia")
        return seen.get("sys", "")

    def exact_list(self, prompt):
        return prompt.split("Output EXACTLY")[1].split("\n\n")[0] if "Output EXACTLY" in prompt else ""

    def test_the_relational_phrase_is_not_in_the_output_exactly_list(self):
        exact = self.exact_list(self.fallback_prompt())

        self.assertNotIn("days after transplanting → ରୁଆ ପରର ଦିନ", exact)

    def test_the_protected_terms_are_still_listed_exactly(self):
        exact = self.exact_list(self.fallback_prompt())

        self.assertIn("fertilizer application → ସାର ପ୍ରୟୋଗ", exact)
        self.assertIn("nutrient uptake → ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ", exact)

    def test_the_relational_phrase_is_offered_as_guidance(self):
        prompt = self.fallback_prompt()

        self.assertIn("ରୁଆ ପରର ଦିନ", prompt)
        self.assertIn("naturally", prompt)


# ── Full English -> Odia pipeline ─────────────────────────────────────────

class GoogleStub:
    """A deterministic stand-in with Google's logged behaviour.

    A marker right after an adjective ("effective ZQXn") is dropped the way the
    production call dropped ZQX2; a frozen relational phrase keeps its number
    detached ("5 ZQX0 ମଧ୍ୟରେ"); an unfrozen one is rendered compositionally.
    """

    RULES = [
        (r"within (\d+) days after (ZQX\d+XQZ)", r"\2 ପରେ \1 ଦିନ ମଧ୍ୟରେ"),
        (r"within (\d+) (ZQX\d+XQZ)", r"\1 \2 ମଧ୍ୟରେ"),
        (r"at the time of (ZQX\d+XQZ)", r"\1 ସମୟରେ"),
        (r"for effective nutrient uptake", r"ପ୍ରଭାବଶାଳୀ ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ ପାଇଁ"),
        (r"for effective (ZQX\d+XQZ)", r"ପ୍ରଭାବଶାଳୀ \1 ପାଇଁ"),
        (r"(ZQX\d+XQZ) should be done", r"\1"),
        (r"\bor\b", "କିମ୍ବା"),
        (r"\bApply\b|\bapply\b", "ପ୍ରୟୋଗ କରନ୍ତୁ"),
        (r"\burea\b", "ୟୁରିଆ"),
        (r"\.\s*$", " କରନ୍ତୁ।"),
    ]

    def __init__(self, drop_after_adjective=False):
        self.drop = drop_after_adjective
        self.calls = []

    def __call__(self, text, source, target):
        self.calls.append(text)
        out = text
        if self.drop:
            out = re.sub(r"for effective ZQX\d+XQZ", "ପ୍ରଭାବଶାଳୀ", out)
        for rx, repl in self.RULES:
            out = re.sub(rx, repl, out)
        return re.sub(r"\s{2,}", " ", out).strip()


class TheFullPipelineTests(GlossaryCase):
    def translate(self, text, google):
        with patch.object(pipeline, "_google_translate", side_effect=google), \
                patch.object(pipeline, "_fix_odia_fluency",
                             side_effect=lambda t, s="", **k: t), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            return n(pipeline.translate(text, "English", "Odia"))

    def test_a_the_number_binds_to_days_in_natural_order(self):
        out = self.translate("Apply urea within 5 days after transplanting.", GoogleStub())

        self.assertNotIn(n("5 ରୁଆ ପରର ଦିନ"), out)
        self.assertIn(n("ଧାନ ରୁଆ ପରେ 5 ଦିନ ମଧ୍ୟରେ"), out)

    def test_b_at_the_time_of_transplanting_takes_its_postposition(self):
        out = self.translate("Apply urea at the time of transplanting.", GoogleStub())

        self.assertIn(n("ଧାନ ରୁଆ ସମୟରେ"), out)

    def test_c_production_sentence_has_no_detached_or_duplicated_days(self):
        out = self.translate(PRODUCTION, GoogleStub())

        self.assertNotIn(n("ରୁଆ ପରର ଦିନ"), out)
        self.assertNotIn(n("ଦିନ ରୁଆ ପରର ଦିନ"), out)
        self.assertIn(n("ସାର ପ୍ରୟୋଗ"), out)

    def test_d_a_dropped_marker_is_detected_and_the_meaning_kept(self):
        google = GoogleStub(drop_after_adjective=True)
        out = self.translate(PRODUCTION, google)

        # The first call lost the nutrient-uptake marker; it was re-sent inline.
        self.assertEqual(2, len(google.calls))
        self.assertIn("nutrient uptake", google.calls[1])
        self.assertIn(n("ପୋଷକ ଉପାଦାନ ଗ୍ରହଣ"), out)

    def test_11_the_action_verb_bug_is_not_reintroduced(self):
        from queries.test_action_verb_glossary import google_like, PRODUCTION as VEENA

        out = self.translate(VEENA, google_like)

        self.assertIsNone(re.search(n("ପ୍ରୟୋଗ କରିବା ପାଇଁ") + ".*" + n("ବ୍ୟବହାର କରନ୍ତୁ"), out))
