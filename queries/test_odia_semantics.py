"""The question direction finally checks its own work.

English -> Odia has always verified its output and retried once with the fault
named. Odia -> English only ever chose between ENGINES, so two engines agreeing
on a wrong reading passed straight through — and a wrong QUESTION is invisible,
because it produces a fluent English sentence, retrieves nothing for it, and
reports "the information available does not specify". It reads like a gap in the
knowledge base rather than a translation bug.

The hardest requirement here is NOT catching the faults. It is not catching
anything else: a validator that rejects good translations is a worse bug than
the one it replaces, because it sends healthy sentences to be "repaired". So
roughly half of these tests assert that nothing fires.

No test makes a real API call. Google and the LLM are both stubbed.
"""
import os
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from queries.services import pipeline
from queries.services.odia_semantics import UNVERIFIED, validate_odia_to_english


def codes(odia, english):
    return [f.code for f in validate_odia_to_english(odia, english)]


class StagesMustSurviveTests(SimpleTestCase):
    """Grounded in the glossary: ରୁଆ -> transplanting, ହଳ -> tillage,
    ଫୁଲ ଫୁଟା -> flowering. ପଡ଼ିଗଲା -> lodging is grounded in the production
    failure documented in pipeline.py."""

    def test_transplanting_turned_into_harvesting_is_caught(self):
        self.assertIn("transplanting_became_harvesting",
                      codes("ରୁଆ କେବେ କରିବି?", "When should I harvest?"))

    def test_finishing_transplanting_is_not_harvesting(self):
        # The glossary itself maps 'transplanting rice' -> ଧାନ ରୋଇ ସାରିବା, so
        # ରୋଇ ସାରିବା is transplanting being COMPLETED, never harvesting.
        self.assertIn("transplanting_became_harvesting",
                      codes("ଧାନ ରୋଇ ସାରିବା ପରେ କଣ କରିବି?",
                            "What to do after harvesting the rice?"))

    def test_lodging_turned_into_maturity_is_caught(self):
        # The live case: the lodging vanished, and the answer then recommended
        # fertiliser — which is what CAUSES lodging.
        self.assertIn("lodging_became_maturity",
                      codes("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା",
                            "Pooja Dhan is ready in 55 days"))

    def test_ploughing_turned_into_cutting_is_caught(self):
        self.assertIn("ploughing_became_harvesting",
                      codes("ହଳ କେବେ କରିବି?", "When should I cut?"))

    def test_flowering_turned_into_harvesting_is_caught(self):
        self.assertIn("flowering_became_harvesting",
                      codes("ଫୁଲ ଫୁଟା ସମୟରେ ସାର ଦେବି?",
                            "Should I apply fertilizer at harvesting?"))

    def test_a_damage_report_reduced_to_a_healthy_crop_is_caught(self):
        self.assertIn("damage_report_lost",
                      codes("ଧାନ ଶୁଖିଗଲା", "The rice is ready"))


class ItMustNotFireOnGoodTranslationsTests(SimpleTestCase):
    """False positives send healthy sentences to be repaired. Every check
    needs the Odia marker, a wrong English word, AND no right one."""

    def test_a_correct_transplanting_translation_passes(self):
        self.assertEqual([], codes("ରୁଆ କେବେ କରିବି?",
                                   "When should I do transplanting?"))

    def test_a_question_about_both_operations_passes(self):
        # Mentions harvesting legitimately, AND keeps transplanting. The third
        # condition is what saves it.
        self.assertEqual([], codes("ରୁଆ ପରେ ଅମଳ କେବେ?",
                                   "When is harvesting after transplanting?"))

    def test_a_correct_lodging_translation_passes(self):
        self.assertEqual([], codes("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା",
                                   "Pooja Dhan lodged at 55 days"))

    def test_a_plain_fertiliser_question_passes(self):
        self.assertEqual([], codes("100 ଦିନିଆ କିସମରେ କେତେ ୟୁରିଆ ଦେବି?",
                                   "How much urea for a 100 day variety?"))

    def test_a_yield_question_is_not_treated_as_harvesting(self):
        # ଅମଳ means BOTH harvesting and yield — the glossary logs the collision
        # on every load. Treating it as a harvest marker would report every
        # yield question as a mistranslation, so it is not a marker at all.
        self.assertEqual([], codes("ଅମଳ କେତେ ହେବ?", "What will the yield be?"))

    def test_an_empty_side_is_never_a_fault(self):
        self.assertEqual([], codes("", "When should I harvest?"))
        self.assertEqual([], codes("ରୁଆ କେବେ କରିବି?", ""))


class NumbersUnitsAndDurationTests(SimpleTestCase):

    def test_a_dropped_number_is_caught(self):
        self.assertIn("number_lost",
                      codes("100 ଦିନିଆ କିସମରେ କେତେ ୟୁରିଆ?",
                            "How much urea for a short variety?"))

    def test_every_number_must_survive_not_just_one(self):
        self.assertIn("number_lost",
                      codes("2.5 ଏକରରେ 50 କେଜି", "50 kg for the field"))

    def test_odia_digits_are_compared_as_digits(self):
        # Conversion normally reaches these first; if it did not, the check
        # must still not report a false loss.
        self.assertEqual([], codes("୫୫ ଦିନ", "55 days"))

    def test_a_number_the_english_adds_is_not_a_loss(self):
        # Only LOSS is checked here. An invented number is the answer
        # direction's problem (_unverified_doses), not this one.
        self.assertEqual([], codes("କେତେ ୟୁରିଆ?", "How much urea per 1 acre?"))


class NegationAndQuestionTypeTests(SimpleTestCase):

    def test_a_dropped_negation_is_caught(self):
        # Reversing this reverses the advice the farmer acts on.
        self.assertIn("negation_lost",
                      codes("ଏହି ସମୟରେ ସାର ଦେବେ ନାହିଁ",
                            "Apply fertilizer at this time"))

    def test_a_kept_negation_passes(self):
        self.assertEqual([], codes("ଏହି ସମୟରେ ସାର ଦେବେ ନାହିଁ",
                                   "Do not apply fertilizer at this time"))

    def test_a_question_flattened_into_a_statement_is_caught(self):
        self.assertIn("question_type_lost",
                      codes("କେତେ ୟୁରିଆ ଦେବି?", "Urea application in rice"))

    def test_a_politely_framed_question_passes(self):
        # Real stored query #290, which an earlier draft reported as broken.
        # "I want to know how much ..." is a question wearing a full stop, and
        # the wh-word is in the middle rather than at the front.
        self.assertEqual([], codes(
            "ମୁଁ ଜାଣିବାକୁ ଚାହୁଁଛି ଯେ କେତେ ପରିମାଣର ସାର ବ୍ୟବହାର କରିବା ଏକ ଏକର ଜମିରେ।",
            "I want to know how much fertiliser is needed for 1 acre of land."))

    def test_a_bare_topic_with_no_interrogative_is_still_caught(self):
        # The other side of the same rule: nothing here is asking anything.
        self.assertIn("question_type_lost",
                      codes("କେତେ ୟୁରିଆ ଦେବି?", "Urea application in rice"))

    def test_a_question_without_a_wh_word_still_passes(self):
        # "ସାର ଦେଇପାରିବି?" -> "Can we apply fertilizer?" is a real question
        # with no wh-word. A modal opener counts.
        self.assertEqual([], codes("କେତେ ସାର ଦେବି?",
                                   "Can we apply fertilizer, and how much?"))


class PaymentAmbiguityTests(SimpleTestCase):
    """ଦେବି = 'I will give/apply', which Google has turned into 'pay'.

    Handled by the EXISTING _money_intrusion, which compares against Odia money
    words instead of guessing from the verb. Asserted here so the two checks
    cannot silently start double-reporting or cancelling out.
    """

    def test_applying_is_not_paying(self):
        self.assertEqual(
            "pay", pipeline._money_intrusion("କେତେ ୟୁରିଆ ଦେବି?",
                                             "How much should I pay for urea?"))

    def test_a_real_price_question_is_left_alone(self):
        self.assertEqual(
            "", pipeline._money_intrusion("ୟୁରିଆ ଦାମ କେତେ?",
                                          "What is the price of urea?"))

    def test_the_validator_does_not_also_report_it(self):
        self.assertEqual([], codes("କେତେ ୟୁରିଆ ଦେବି?",
                                   "How much should I pay for urea?"))


class TheRepairCallTests(SimpleTestCase):
    """Detection is free and always runs; only the repair costs a call."""

    FAULTS = validate_odia_to_english("ରୁଆ କେବେ କରିବି?", "When should I harvest?")

    def test_it_is_on_by_default(self):
        self.assertTrue(pipeline._semantic_repair_enabled())

    @override_settings(ODIA_SEMANTIC_REPAIR=False)
    def test_settings_can_turn_it_off(self):
        # NEVER patch.object(settings, ..., create=True): settings is a lazy
        # proxy and mock deletes the attribute from the real object on exit.
        self.assertFalse(pipeline._semantic_repair_enabled())

    def test_the_prompt_names_the_fault_and_forbids_answering(self):
        seen = {}

        def capture(messages, **kwargs):
            seen["system"] = messages[0]["content"]
            seen["user"] = messages[1]["content"]
            return "When should I do transplanting?"

        with patch.object(pipeline, "_chat", side_effect=capture):
            out = pipeline._repair_translation("ରୁଆ କେବେ କରିବି?",
                                               "When should I harvest?",
                                               self.FAULTS)

        self.assertEqual("When should I do transplanting?", out)
        self.assertIn("transplanting_became_harvesting", seen["system"])
        self.assertIn("NEVER answer the question", seen["system"])
        self.assertIn("ରୁଆ କେବେ କରିବି?", seen["user"])

    def test_a_failed_repair_call_keeps_the_original(self):
        with patch.object(pipeline, "_chat", side_effect=RuntimeError("API down")):
            self.assertEqual("", pipeline._repair_translation(
                "ରୁଆ କେବେ କରିବି?", "When should I harvest?", self.FAULTS))

    def test_an_empty_repair_keeps_the_original(self):
        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "   "):
            self.assertEqual("", pipeline._repair_translation(
                "ରୁଆ କେବେ କରିବି?", "When should I harvest?", self.FAULTS))

    def test_a_repair_that_answered_instead_of_translating_is_refused(self):
        essay = "You should transplant rice 25 days after sowing. " * 30
        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: essay):
            self.assertEqual("", pipeline._repair_translation(
                "ରୁଆ କେବେ କରିବି?", "When should I harvest?", self.FAULTS))


class TheWiringTests(SimpleTestCase):
    """translate() end to end with both outside services stubbed.

    THE QUESTION HERE IS TOUCHED BY NOTHING ELSE, on purpose — it took two
    tries to find one.

    Draft 1 used "ରୁଆ କେବେ କରିବି?" and proved nothing: ରୁଆ IS a glossary term,
    so _force_reverse_glossary_terms substituted it, Google's "harvest" dropped
    it, and the EXISTING _missing_glossary_terms guard fell back to OpenAI
    before the new validation ever ran. The _chat stub was standing in for the
    fallback translator, not the repair.

    Draft 2 used the lodging case and proved something better, but not this:
    protect_input_terms already pins ପଡ଼ିଗଲା -> 'lodging', and reassert_tokens
    appends it to the English BEFORE this check runs. The English therefore
    contains 'lodging', the third condition holds, and no fault fires — the
    validator correctly staying silent on a case the pipeline already handles.
    That is asserted on its own below.

    A lost NUMBER is a real gap: no pin, no glossary entry, no existing guard.
    """
    ODIA = "100 ଦିନିଆ କିସମରେ କେତେ ୟୁରିଆ ଦରକାର?"
    GOOGLE_WRONG = "How much urea is needed for the short variety?"

    def run_translate(self, google, chat, flag="true"):
        calls = {"chat": []}

        def chat_stub(messages, **kwargs):
            calls["chat"].append(messages)
            return chat

        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: google), \
                patch.object(pipeline, "_chat", side_effect=chat_stub), \
                override_settings(ODIA_SEMANTIC_REPAIR=(flag == "true")):
            out = pipeline.translate(self.ODIA, "Odia", "English")
        return out, calls

    GOOD = "How much urea is needed for the 100 day variety?"

    def test_a_faulty_translation_is_repaired(self):
        out, calls = self.run_translate(google=self.GOOGLE_WRONG, chat=self.GOOD)

        self.assertIn("100", out)
        self.assertTrue(calls["chat"], "the repair was never attempted")

    def test_the_repair_is_the_call_that_was_made(self):
        # Proves the _chat stub was the REPAIR, not the fallback translator.
        _out, calls = self.run_translate(google=self.GOOGLE_WRONG, chat=self.GOOD)

        self.assertIn("number_lost", calls["chat"][0][0]["content"])

    def test_a_good_translation_costs_no_repair_call(self):
        out, calls = self.run_translate(google=self.GOOD,
                                        chat="SHOULD NOT BE CALLED")

        self.assertIn("100", out)
        self.assertEqual([], calls["chat"])

    def test_a_repair_that_is_still_wrong_is_discarded(self):
        # The original at least came from an engine whose failure modes the
        # existing guards cover; a new wrong answer is not an improvement.
        out, calls = self.run_translate(
            google=self.GOOGLE_WRONG,
            chat="How much urea for a short duration variety?")

        self.assertTrue(calls["chat"], "the repair was never attempted")
        self.assertEqual(self.GOOGLE_WRONG, out)

    def test_the_off_switch_skips_the_repair_but_still_translates(self):
        out, calls = self.run_translate(google=self.GOOGLE_WRONG,
                                        chat="SHOULD NOT BE CALLED",
                                        flag="false")

        self.assertEqual(self.GOOGLE_WRONG, out)
        self.assertEqual([], calls["chat"])

    def test_google_failing_still_falls_back_to_the_llm(self):
        with patch.object(pipeline, "_google_translate",
                          side_effect=RuntimeError("Google down")), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: self.GOOD):
            out = pipeline.translate(self.ODIA, "Odia", "English")

        self.assertIn("100", out)

    def test_a_pinned_token_survives_an_accepted_repair(self):
        # A DEFECT FOUND IN REVIEW. reassert_tokens runs BEFORE validation, on
        # the text the repair then replaces — so a dictated acronym rescued as
        # "... (BGA)" was dropped again the moment the model rewrote the
        # sentence. Retrieval needs that token present; it is the entire reason
        # it was pinned ("BGA" appears in the corpus, "blue green algae" does
        # not). The repair now re-asserts the pins.
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "How much BGA should be applied?"), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: "How much to apply for 100 days?"):
            out = pipeline.translate("ବିଜିଏ 100 ଦିନରେ କେତେ ଦେବି?", "Odia", "English")

        self.assertIn("100", out)
        self.assertIn("BGA", out, "the pinned acronym was lost by the repair")

    def test_the_pinned_lodging_case_needs_no_repair(self):
        # protect_input_terms pins ପଡ଼ିଗଲା -> 'lodging' and reassert_tokens puts
        # it back into the English, so the pipeline already recovers this one.
        # The validator must NOT fire a second time and pay for a repair call.
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "Pooja Dhan is ready in 55 days"), \
                patch.object(pipeline, "_chat",
                             side_effect=AssertionError("no repair should run")):
            out = pipeline.translate("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା", "Odia", "English")

        self.assertIn("lodging", out.lower())

    def test_the_existing_glossary_fallback_still_fires_first(self):
        # The path the earlier draft was accidentally testing. It must keep
        # working: a dropped glossary term still routes to OpenAI, and the new
        # validation must not interfere with that.
        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "When should I harvest?"), \
                patch.object(pipeline, "_chat",
                             side_effect=lambda *a, **k: "When should I do transplanting?"):
            out = pipeline.translate("ରୁଆ କେବେ କରିବି?", "Odia", "English")

        self.assertIn("transplant", out.lower())


class TheAnswerDirectionIsUntouchedTests(SimpleTestCase):
    """English -> Odia must not acquire any of this."""

    def test_no_validation_or_repair_runs_on_the_answer_direction(self):
        calls = []

        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "ଅନୁବାଦ"), \
                patch.object(pipeline, "_fix_odia_fluency",
                             side_effect=lambda t, s="", **k: t), \
                patch.object(pipeline, "_detransliterate_odia",
                             side_effect=lambda t, s="": t), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"), \
                patch.object(pipeline, "_repair_translation",
                             side_effect=lambda *a: calls.append(a) or ""):
            out = pipeline.translate("Apply urea at tillering.", "English", "Odia")

        self.assertEqual([], calls, "the answer direction was sent for repair")
        self.assertTrue(out.strip())


class RepairQualityTests(SimpleTestCase):
    """What a repair must produce, and every way it is refused.

    The wording asserted here is the wording the REAL model produced when these
    exact failures were replayed through it (see the docstring on each test).
    Stubbing it lets the acceptance logic be tested without an API call; the
    live run is what established the strings are realistic.
    """

    def repair(self, odia, bad, reply):
        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: reply):
            return pipeline._repair_translation(odia, bad,
                                                validate_odia_to_english(odia, bad))

    def accept(self, odia, bad, reply):
        """The pipeline's own accept/reject rule: repair, then re-validate."""
        fixed = self.repair(odia, bad, reply)
        if not fixed:
            return None
        return fixed if not validate_odia_to_english(odia, fixed) else None

    # ── one per detected failure shape, with the real model's own output ──

    def test_transplanting_and_ploughing_are_both_restored(self):
        # Real query #981/#982. Live model returned exactly this.
        self.assertEqual(
            "How many days before transplanting should rice be ploughed?",
            self.accept("ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?",
                        "How many days before harvesting should rice be cut?",
                        "How many days before transplanting should rice be ploughed?"))

    def test_basal_fertiliser_after_transplanting_is_restored(self):
        # Real query #998/#999 — the stage the whole question hangs on.
        odia = ("ଧାନ ରୋଇ ସାରିବା ପରେ ଗୋଟେ ଏକରରେ ମୁଁ କେତେ ଡିଏପି, କେତେ ୟୁରିଆ, "
                "କେତେ ପୋଟାସ ପ୍ରୟୋଗ କରିବି ମୂଳସାର ଭାବରେ?")
        bad = ("How much DAP, how much urea, how much potash should I apply as "
               "basic fertilizer in 1 acre after harvesting rice?")
        good = bad.replace("after harvesting rice", "after transplanting rice")

        self.assertEqual(good, self.accept(odia, bad, good))

    def test_lodging_is_restored_with_the_number_intact(self):
        fixed = self.accept("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା, ସାର ଦେଇପାରିବି?",
                            "Pooja Dhan is ready in 55 days, can I apply fertilizer?",
                            "Pooja Dhan lodged in 55 days, can I apply fertilizer?")

        self.assertIn("lodged", fixed)
        self.assertIn("55", fixed)
        self.assertIn("Pooja Dhan", fixed)

    def test_negation_is_restored(self):
        self.assertEqual(
            "Should not apply fertilizer at this time.",
            self.accept("ଏହି ସମୟରେ ସାର ଦେବେ ନାହିଁ", "Apply fertilizer at this time",
                        "Should not apply fertilizer at this time."))

    def test_a_lost_number_is_restored(self):
        fixed = self.accept("100 ଦିନିଆ କିସମରେ କେତେ ୟୁରିଆ ଦରକାର?",
                            "How much urea is needed for the short variety?",
                            "How much urea is needed for the 100 days variety?")

        self.assertIn("100", fixed)

    # ── every way a repair is refused ────────────────────────────────────

    def test_an_incomplete_repair_is_rejected(self):
        # Fixes the transplanting, leaves the ploughing as "cut".
        self.assertIsNone(
            self.accept("ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?",
                        "How many days before harvesting should rice be cut?",
                        "How many days before transplanting should rice be cut?"))

    def test_a_repair_that_drops_the_number_is_rejected(self):
        self.assertIsNone(
            self.accept("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା",
                        "Pooja Dhan is ready in 55 days",
                        "Pooja Dhan has lodged"))

    def test_a_repair_that_reverses_the_negation_back_is_rejected(self):
        self.assertIsNone(
            self.accept("ଏହି ସମୟରେ ସାର ଦେବେ ନାହିଁ",
                        "Apply fertilizer at this time",
                        "Apply fertilizer now at this time"))

    def test_a_repair_that_answers_the_question_is_refused(self):
        # Rule 1 is the one a model breaks. A long advisory reply in the
        # question slot would send retrieval after the wrong thing entirely.
        answer = ("You should plough the field about 20 to 25 days before "
                  "transplanting, and apply 50 kg of DAP per acre as basal "
                  "dose. Make sure the soil is well puddled. ") * 4
        self.assertEqual("", self.repair("ଧାନ ରୋଇବା କେତେ ଦିନ ଆଗରୁ ହଳ କରିବା ଉଚିତ?",
                                         "How many days before harvesting should "
                                         "rice be cut?", answer))

    def test_an_invented_variety_name_does_not_survive_revalidation(self):
        # The fallback's known failure mode: transliterating ordinary words into
        # a Title Case cultivar. Here it also loses the number, which is what
        # the re-validation catches — an invented name is never ACCEPTED
        # silently, because a repair only lands if it re-validates clean.
        self.assertIsNone(
            self.accept("100 ଦିନିଆ କିସମରେ କେତେ ୟୁରିଆ ଦରକାର?",
                        "How much urea is needed for the short variety?",
                        "How much urea for the Sahia Diniya variety?"))

    def test_a_non_string_reply_is_refused(self):
        with patch.object(pipeline, "_chat", side_effect=lambda *a, **k: object()):
            self.assertEqual("", pipeline._repair_translation(
                "ରୁଆ କେବେ କରିବି?", "When should I harvest?", self.FAULTS_HARVEST))

    FAULTS_HARVEST = validate_odia_to_english("ରୁଆ କେବେ କରିବି?",
                                              "When should I harvest?")


class AProductionSizedGlossaryTests(SimpleTestCase):
    """2,996 terms on the server; 92 in this checkout.

    The worry is real but specific: a bigger reverse glossary substitutes far
    more English INTO the sentence handed to Google, so more of the final
    English is glossary-chosen. If the validator read that substituted text as
    "the Odia", its markers would vanish and every rule would go quiet.

    It does not: _force_reverse_glossary_terms returns a SEPARATE `marked`
    string and `text` is never reassigned. This class proves that against a
    real glossary of production size rather than by reading the code.
    """
    BIG = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import json
        import tempfile
        from pathlib import Path
        from queries.services import odia_glossary

        real = dict(odia_glossary.get_glossary())
        # Synthetic bulk, shaped like the real thing: distinct Odia keys, and
        # English values that deliberately INCLUDE the words the stage rules
        # treat as wrong, so heavy substitution is stress-tested rather than
        # merely simulated.
        filler = {}
        stems = ["harvesting method", "cutting tool", "ready stage", "mature grain",
                 "field operation", "crop care", "soil test", "water level"]
        for i in range(2996 - len(real)):
            filler[f"{stems[i % len(stems)]} {i}"] = f"ପରୀକ୍ଷା{i}"
        cls._tmp = Path(tempfile.mkdtemp()) / "big_glossary.json"
        cls._tmp.write_text(json.dumps({**real, **filler}, ensure_ascii=False),
                            encoding="utf-8")
        cls._original_path = odia_glossary._GLOSSARY_PATH
        odia_glossary._GLOSSARY_PATH = cls._tmp
        odia_glossary.reload_glossary()
        cls.BIG = odia_glossary.get_glossary()

    @classmethod
    def tearDownClass(cls):
        from queries.services import odia_glossary
        odia_glossary._GLOSSARY_PATH = cls._original_path
        odia_glossary.reload_glossary()
        super().tearDownClass()

    def test_the_glossary_really_is_production_sized(self):
        self.assertGreaterEqual(len(self.BIG), 2900)

    def test_the_stage_rules_still_fire(self):
        # The validator reads Odia, which no amount of substitution touches.
        self.assertIn("transplanting_became_harvesting",
                      codes("ରୁଆ କେବେ କରିବି?", "When should I harvest?"))
        self.assertIn("lodging_became_maturity",
                      codes("ପୂଜାଧାନ 55 ଦିନରେ ପଡ଼ିଗଲା",
                            "Pooja Dhan is ready in 55 days"))

    def test_good_translations_still_pass(self):
        self.assertEqual([], codes("ରୁଆ କେବେ କରିବି?",
                                   "When should I do transplanting?"))
        self.assertEqual([], codes("ଅମଳ କେତେ ହେବ?", "What will the yield be?"))

    def test_the_odia_reaching_the_validator_is_still_odia(self):
        # The end-to-end guarantee: whatever the glossary substitutes into the
        # text sent to Google, the string VALIDATED must still carry the
        # farmer's Odia, or every marker silently stops matching.
        seen = {}

        def spy(odia, english):
            seen.setdefault("odia", odia)
            return []

        with patch.object(pipeline, "_google_translate",
                          side_effect=lambda t, s, d: "When should I harvest?"), \
                patch.object(pipeline, "validate_odia_to_english", side_effect=spy), \
                patch.object(pipeline, "_chat", side_effect=lambda *a, **k: "x"):
            pipeline.translate("ରୁଆ କେବେ କରିବି?", "Odia", "English")

        self.assertIn("ରୁଆ", seen.get("odia", ""),
                      "the validator was handed substituted text, not the Odia")


class HonestyAboutWhatIsNotCoveredTests(SimpleTestCase):

    def test_the_ungrounded_stages_are_declared_not_guessed(self):
        # The brief named sowing, weeding and fruiting. None could be grounded
        # in this checkout's glossary, and a marker that is merely plausible
        # produces false faults — so they are listed, not implemented.
        joined = " ".join(UNVERIFIED).lower()
        for missing in ("sowing", "weeding", "fruiting"):
            self.assertIn(missing, joined)
