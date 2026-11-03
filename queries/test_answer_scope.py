"""The answer matches the size of the question, not the size of the context.

THE REPORTED CASE
-----------------
    "When and how much fertiliser should be applied to rice?"
    -> "At the basal stage, apply 25-30 kg of DAP, 10-15 kg of Urea, 15 kg of
        MOP, and 10 kg of Zinc fertilizer per acre. At the active tillering
        stage, apply 30-35 kg of Urea per acre. Finally, at the panicle
        initiation stage, apply 15 kg of MOP per acre."

Every figure was real. The answer was still wrong, because the farmer had to
find their question inside three stages, four products and seven figures.

IT WAS A RULE CONFLICT, NOT A MISSING RULE. "ANSWER ONLY WHAT WAS ASKED" sat a
few lines above "IF THE QUESTION ASKS GENERALLY WHAT FERTILISER A CROP NEEDS,
with no stage named ... give the COMPLETE schedule" — a rule added because a
farmer asking what to apply in DSR had once been told the basal dose alone. This
question names no stage, so the model obeyed the second rule. Both were right;
nothing said which governed.

The scope is now decided BEFORE the prompt is built and stated in it. Nothing is
truncated afterwards — a scope decision made after generation is just deletion,
and it removes exactly the figures a short answer needed.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from queries.services import pipeline

SCHEDULE = (
    "Basal: apply 25-30 kg DAP, 10-15 kg Urea, 15 kg MOP and 10 kg Zinc per acre.\n"
    "Active tillering: apply 30-35 kg Urea per acre.\n"
    "Panicle initiation: apply 15 kg MOP per acre.\n"
    "Nitrogen builds the crop canopy and improves tillering.\n"
)
VARIETIES = (
    "BRRI Dhan-69 is a medium-duration variety with a duration of 145 days.\n"
    "BRRI Dhan-69 yields 6 t/ha and has medium-bold grain and medium height.\n"
    "CR Dhan 310 is suitable for irrigated medium land.\n"
    "MTU 1010 is suitable for irrigated medium land.\n"
)


def hits(*chunks):
    return [{"chunk": c, "metadata": {"document_title": "sheet", "document_id": i}}
            for i, c in enumerate(chunks or (SCHEDULE,))]


def prompt_for(question, *chunks):
    seen = {}

    def capture(messages, **kwargs):
        seen["sys"] = messages[0]["content"]
        return "ok"

    with patch.object(pipeline, "_chat", side_effect=capture):
        pipeline.generate_answer(question, hits(*chunks))
    return seen["sys"]


def scope_block(prompt):
    """The per-question scope note.

    The prompt contains "ANSWER SCOPE (apply in this order)" — the fixed
    hierarchy — followed by "ANSWER SCOPE for THIS question", which is the line
    derived from the question. These tests are about the second one.
    """
    return prompt.split("ANSWER SCOPE for THIS question")[1].split("\nCONTEXT:")[0]


class ScopeDetectionTests(SimpleTestCase):
    """The decision is made before the prompt, so it can be asserted."""

    def test_each_kind_of_question_is_classified(self):
        for question, expected in (
                ("When and how much fertiliser should be applied to rice?",
                 pipeline.SCOPE_TIMING_QUANTITY),
                ("How much DAP should be applied per acre?",
                 pipeline.SCOPE_QUANTITY),
                ("Give me the complete fertilizer schedule for rice.",
                 pipeline.SCOPE_COMPLETE),
                ("What fertilizers should be applied at each stage?",
                 pipeline.SCOPE_COMPLETE),
                ("What is the duration of BRRI Dhan-69?",
                 pipeline.SCOPE_SINGLE_FACT),
                ("How should I manage fertilizer for rice?",
                 pipeline.SCOPE_BROAD)):
            with self.subTest(question=question):
                self.assertEqual(expected, pipeline.response_scope(question))

    def test_an_unrecognised_question_keeps_the_ordinary_behaviour(self):
        self.assertEqual(pipeline.SCOPE_DEFAULT,
                         pipeline.response_scope("Which varieties suit upland?"))
        self.assertEqual(pipeline.SCOPE_DEFAULT,
                         pipeline.response_scope("What fertiliser should I apply in DSR?"))

    def test_the_scope_hierarchy_is_stated_at_the_top_of_the_rules(self):
        prompt = prompt_for("How much DAP should be applied per acre?")
        rules = prompt.split("GROUNDING RULES (MANDATORY):")[1]

        # Prominent: inside the first 1200 characters of the rules, not buried.
        self.assertIn("ANSWER SCOPE", rules[:1200])
        for line in ("Answer the question the farmer actually asked",
                     "Match the SCOPE of the question",
                     "Add nothing further",
                     "Never invent what is missing"):
            self.assertIn(line, rules)


class TestA_QuantityQuestion(SimpleTestCase):
    """How much DAP per acre — the DAP figure, not the schedule."""

    QUESTION = "How much DAP should be applied per acre?"

    def test_the_prompt_asks_for_the_one_figure(self):
        block = scope_block(prompt_for(self.QUESTION))

        self.assertIn("quantity", block)
        self.assertIn("Give the figure for what they named", block)
        self.assertIn("Do not add the other products, the other stages", block)

    def test_the_whole_schedule_rule_is_explicitly_stood_down(self):
        prompt = prompt_for(self.QUESTION)

        self.assertIn("does NOT apply when the farmer asked how much of one "
                      "input", prompt)

    def test_the_completeness_check_requires_nothing_here(self):
        # The check that could force expansion only ever holds VARIETY options.
        self.assertEqual([], pipeline._missing_candidates(
            "Apply 25-30 kg DAP per acre at basal.", []))


class TestB_TimingAndQuantity(SimpleTestCase):
    """The reported question: both facts, said compactly."""

    QUESTION = "When and how much fertiliser should be applied to rice?"

    def test_the_scope_is_timing_and_quantity(self):
        self.assertEqual(pipeline.SCOPE_TIMING_QUANTITY,
                         pipeline.response_scope(self.QUESTION))

    def test_both_halves_are_required_and_kept_compact(self):
        block = scope_block(prompt_for(self.QUESTION))

        self.assertIn("asked WHEN and HOW MUCH", block)
        self.assertIn("one short line per stage", block)

    def test_the_explanatory_padding_is_ruled_out(self):
        block = scope_block(prompt_for(self.QUESTION))

        self.assertIn("Do not add the explanation of why, the benefits", block)

    def test_no_length_limit_is_imposed_anywhere(self):
        # Scope is not solved by truncation: a limit would drop a figure the
        # farmer needed.
        prompt = prompt_for(self.QUESTION)
        rules = prompt.split("GROUNDING RULES (MANDATORY):")[1].split("\nCONTEXT:")[0]

        for banned in ("maximum 2 sentences", "at most two sentences",
                       "no more than", "characters or less", "word limit"):
            self.assertNotIn(banned, rules.lower())


class TestC_CompleteSchedule(SimpleTestCase):
    """Asked for all of it — all of it is right."""

    def test_the_complete_schedule_is_allowed(self):
        block = scope_block(prompt_for("Give me the complete fertilizer schedule "
                                       "for rice."))

        self.assertIn("COMPLETE set", block)
        self.assertIn("leaving none out", block)

    def test_a_stage_wise_question_is_also_complete(self):
        block = scope_block(prompt_for("What fertilizers should be applied at "
                                       "each stage?"))

        self.assertIn("COMPLETE set", block)

    def test_the_original_dsr_protection_survives(self):
        # The harm the old rule was written for: "what fertiliser in DSR?" must
        # still get the timetable, not the basal dose alone.
        prompt = prompt_for("What fertiliser should I apply in DSR?")

        self.assertIn("COMPLETE schedule the CONTEXT supports", prompt)


class TestD_SingleFact(SimpleTestCase):
    """One property of one variety — not the whole profile."""

    QUESTION = "What is the duration of BRRI Dhan-69?"

    def test_the_scope_is_a_single_fact(self):
        self.assertEqual(pipeline.SCOPE_SINGLE_FACT,
                         pipeline.response_scope(self.QUESTION))

    def test_the_rest_of_the_profile_is_ruled_out(self):
        block = scope_block(prompt_for(self.QUESTION, VARIETIES))

        self.assertIn("ONE property of ONE thing", block)
        self.assertIn("Do not recite the rest of its profile", block)

    def test_no_variety_list_is_forced_for_a_single_fact_question(self):
        prompt = prompt_for(self.QUESTION, VARIETIES)

        self.assertNotIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)


class TestE_VarietyListingUnchanged(SimpleTestCase):
    """The variety behaviour built earlier must not have moved."""

    QUESTION = "Which rice varieties are suitable for irrigated medium land?"

    def test_the_supported_varieties_are_still_listed(self):
        prompt = prompt_for(self.QUESTION, VARIETIES)

        self.assertIn("EVERY VARIETY THE CONTEXT SUPPORTS", prompt)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split(
            "\nCONTEXT:")[0]
        self.assertIn("CR Dhan 310", listed)
        self.assertIn("MTU 1010", listed)

    def test_an_unrelated_variety_is_not_listed(self):
        prompt = prompt_for(self.QUESTION, VARIETIES)
        listed = prompt.split("EVERY VARIETY THE CONTEXT SUPPORTS")[1].split(
            "\nCONTEXT:")[0]

        self.assertNotIn("BRRI Dhan-69", listed)

    def test_the_criterion_note_still_frames_the_answer(self):
        prompt = prompt_for(self.QUESTION, VARIETIES)

        self.assertIn("PRIMARY ANSWER CRITERION", prompt)


class GroundingPreservedTests(SimpleTestCase):
    """Scope control must not loosen anything about the facts themselves."""

    def test_the_dose_rules_are_still_present(self):
        prompt = prompt_for("How much DAP should be applied per acre?")

        self.assertIn("Every figure must come from the CONTEXT", prompt)
        self.assertIn("Do NOT convert a per-hectare rate", prompt)

    def test_the_claim_strength_rules_are_still_present(self):
        prompt = prompt_for("Which variety resists blast?", VARIETIES)

        self.assertIn("A HEDGE IS PART OF THE RATING", prompt)
        self.assertIn("NEVER GENERALISE A LIST INTO A BLANKET", prompt)

    def test_the_answer_still_never_leaks_internal_wording(self):
        with patch.object(pipeline, "_chat",
                          side_effect=lambda *a, **k:
                          "The CONTEXT says: apply 25 kg DAP."):
            answer = pipeline.generate_answer(
                "How much DAP should be applied per acre?", hits())

        self.assertNotIn("CONTEXT", answer)

    def test_the_verifiers_still_run_after_generation(self):
        # An overstated claim is still caught and retried once.
        replies = ["CR Dhan 910 is resistant to blast.",
                   "CR Dhan 910 is moderately resistant to blast."]
        context = "CR Dhan 910 is moderately resistant to blast.\n"
        with patch.object(pipeline, "_chat",
                          side_effect=lambda *a, **k: replies.pop(0)):
            answer = pipeline.generate_answer("Which variety resists blast?",
                                              hits(context))

        self.assertIn("moderately resistant", answer)
