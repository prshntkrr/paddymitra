"""Complete variety lists for a duration question.

"Name the 150-day varieties" has exactly one correct answer: all nine of them.
They live in seven near-identical table chunks, so whether all nine reach the
answer was a property of the RANKING, not of the data — which is why the answer
named two or three. The list is now read off the catalogue table directly.

The gate is the dangerous half. Firing too narrowly gives a short list again;
firing too broadly forces 53 variety names into a fertiliser answer, which is
much worse. The negative cases below are the ones that matter.
"""
from django.test import SimpleTestCase

from queries.services.pipeline import _DURATION_ROW_RE, duration_requested

# Rows exactly as chunking.extract_xlsx_text emits them.
CHUNK = (
    "Sl No. | Variety Name | Duration | Duration Category | Yield (t/ha) | "
    "Source Institute | Suitable Land | Stress Tolerant\n"
    "3 | CR 1009 SUB-1 | 150 | Long | 5 | ICAR-NRRI, Cuttack | Irrigated/medium | Submergence\n"
    "8 | JAGANNATH | 150 | Long | 4.5 | OUAT, Odisha | Irrigated/medium | None reported\n"
    "16 | POOJA (IET-12241) | 150 | Long | 5 | ICAR-NRRI, Cuttack | Irrigated/medium | None reported\n"
    "123 | Sampada | 140 | Medium | 5 | ICAR-NRRI, Cuttack | Irrigated/medium | None reported\n"
)


def _names(chunk, days):
    return [m.group(1).strip() for m in _DURATION_ROW_RE.finditer(chunk)
            if int(m.group(2)) == days]


class DurationGateTests(SimpleTestCase):
    def test_variety_list_questions_fire(self):
        for q, want in [
            ("Tell me about 150 day rice variety", 150),
            ("name some 150 day rice", 150),
            ("105 day paddy varieties", 105),
            ("Which are the 130 day rice varieties?", 130),
            ("list 145 day dhan", 145),
            ("suggest a 110 day paddy", 110),
        ]:
            self.assertEqual(duration_requested(q), want, q)

    def test_dose_questions_never_fire(self):
        # The expensive mistake: these would otherwise force the complete
        # variety list into an answer about fertiliser.
        for q in ["How much urea for 120 day rice?",
                  "What fertilizer for 150 day rice variety?",
                  "How many kg of MOP for a 120 day variety?"]:
            self.assertIsNone(duration_requested(q), q)

    def test_timing_phrases_are_not_durations(self):
        for q in ["How much urea 20 days after transplanting?",
                  "When should I transplant 25 day old seedlings?",
                  "What is blast disease?"]:
            self.assertIsNone(duration_requested(q), q)

    def test_implausible_duration_ignored(self):
        self.assertIsNone(duration_requested("rice variety for 400 days"))


class DurationRowParsingTests(SimpleTestCase):
    def test_exact_duration_only(self):
        # A 150-day question is not asking for the 140s.
        self.assertEqual(_names(CHUNK, 150),
                         ["CR 1009 SUB-1", "JAGANNATH", "POOJA (IET-12241)"])
        self.assertNotIn("Sampada", _names(CHUNK, 150))

    def test_multi_word_name_is_not_split(self):
        # _names_in() used to return ['CR 1009', 'SUB-1'] for this row.
        names = _names(CHUNK, 150)
        self.assertIn("CR 1009 SUB-1", names)
        self.assertNotIn("SUB-1", names)

    def test_parenthesised_code_kept_with_name(self):
        self.assertIn("POOJA (IET-12241)", _names(CHUNK, 150))

    def test_header_row_is_not_a_variety(self):
        self.assertNotIn("Variety Name", _names(CHUNK, 150))
