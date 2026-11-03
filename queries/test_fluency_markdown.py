"""Markdown from the fluency model must never reach a farmer's written answer.

FOUND BY THE REAL-API VALIDATION
--------------------------------
For "Fertilizer application should be done at the time of transplanting or
within 5 days after transplanting for effective nutrient uptake.", the real
Gemini fluency rewrite returned:

    ପ୍ରଭାବଶାଳୀ ପୋଷକ ତତ୍ତ୍ୱ ଗ୍ରହଣ ପାଇଁ, ଧାନ ରୁଆ ସମୟରେ କିମ୍ବା ଧାନ ରୁଆ ପରେ 5 ଦିନ
    ଭିତରେ **ସାର ପ୍ରୟୋଗ** କରନ୍ତୁ।

The asterisks disappeared only because the NEXT LLM pass, _detransliterate_odia,
happened to drop them. That pass returns its input unchanged on failure, and
replaying the real strings with it failing put **ସାର ପ୍ରୟୋଗ** in the final answer.
The voice was already safe (prepare_for_tts strips them); the written answer —
screen, WhatsApp, the stored response — was not.

Stripped in _fix_odia_fluency, the stage that introduces them.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline

EN = ("Fertilizer application should be done at the time of transplanting or "
      "within 5 days after transplanting for effective nutrient uptake.")
# Verbatim from the real-API run.
DRAFT = ("ପ୍ରଭାବଶାଳୀ ପୋଷକ ତତ୍ତ୍ୱ ଗ୍ରହଣ ପାଇଁ ସାର ପ୍ରୟୋଗ ଧାନ ରୁଆ ସମୟରେ କିମ୍ବା "
         "ଧାନ ରୁଆ ପରେ 5 ଦିନ ମଧ୍ୟରେ କରାଯିବା ଉଚିତ।")
BOLDED = ("ପ୍ରଭାବଶାଳୀ ପୋଷକ ତତ୍ତ୍ୱ ଗ୍ରହଣ ପାଇଁ, ଧାନ ରୁଆ ସମୟରେ କିମ୍ବା ଧାନ ରୁଆ ପରେ 5 ଦିନ "
          "ଭିତରେ **ସାର ପ୍ରୟୋଗ** କରନ୍ତୁ।")
CLEAN = ("ପ୍ରଭାବଶାଳୀ ପୋଷକ ତତ୍ତ୍ୱ ଗ୍ରହଣ ପାଇଁ, ଧାନ ରୁଆ ସମୟରେ କିମ୍ବା ଧାନ ରୁଆ ପରେ 5 ଦିନ "
         "ଭିତରେ ସାର ପ୍ରୟୋଗ କରନ୍ତୁ।")


def fluency(reply, draft=DRAFT, **kwargs):
    with patch.object(pipeline, "_gemini_text", side_effect=lambda p, t: reply):
        return pipeline._fix_odia_fluency(draft, EN, **kwargs)


class TheFluencyPassStripsMarkdownTests(SimpleTestCase):

    def test_the_real_bolded_reply_comes_back_clean(self):
        self.assertEqual(CLEAN, fluency(BOLDED))

    def test_the_bolded_word_itself_is_kept(self):
        out = fluency(BOLDED)

        self.assertIn("ସାର ପ୍ରୟୋଗ", out)
        self.assertNotIn("*", out)

    def test_underscore_emphasis_is_stripped_too(self):
        out = fluency(BOLDED.replace("**", "__"))

        self.assertEqual(CLEAN, out)

    def test_a_stray_unpaired_marker_is_removed(self):
        out = fluency(CLEAN.replace("ସାର ପ୍ରୟୋଗ", "**ସାର ପ୍ରୟୋଗ"))

        self.assertNotIn("**", out)

    def test_a_bolded_rewrite_is_used_not_discarded(self):
        # Before the fix, stripping happened nowhere; the rewrite was used WITH
        # asterisks. It must now be used, clean — not rejected for them.
        self.assertNotEqual(DRAFT, fluency(BOLDED))

    def test_the_glossary_guard_still_applies_after_stripping(self):
        # A bolded approved term counts as present once the markers are gone.
        out = fluency(BOLDED, glossary_terms=["ସାର ପ୍ରୟୋଗ"])

        self.assertEqual(CLEAN, out)

    def test_the_number_guard_still_applies_after_stripping(self):
        changed = BOLDED.replace("5 ଦିନ", "7 ଦିନ")

        self.assertEqual(DRAFT, fluency(changed))


class TheWrittenAnswerTests(SimpleTestCase):
    """End to end, with the pass that used to hide the bug FAILING."""

    def test_no_asterisks_reach_the_final_answer_when_detransliteration_fails(self):
        with patch.object(pipeline, "_force_glossary_terms",
                          side_effect=lambda text, exclude=(): (text, {})), \
                patch.object(pipeline, "_google_translate",
                             side_effect=lambda t, s, d: DRAFT), \
                patch.object(pipeline, "_gemini_text",
                             side_effect=lambda p, t: BOLDED), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            final = pipeline.translate(EN, "English", "Odia")

        self.assertNotIn("*", final)
        self.assertIn("ସାର ପ୍ରୟୋଗ", final)
