"""
Regression tests for stage-specific answer scoping.
Tests that the pipeline returns only the requested stage, not the full schedule.
"""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\parsh\Downloads\git\paddymitra_new\paddymitra_rollback\PaddyMitra")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.dev")
import django
django.setup()

from queries.services.pipeline import _deterministic_schedule, _context_rates

# A typical FAQ fertilizer schedule covering 3 stages
FAQ_SCHEDULE = (
    "For long duration varieties, for acre, apply 28-30 kg DAP, 15-17 kg Urea, "
    "16-18 kg MOP, 10 kg zinc sulphate as basal dose, apply 30-32 kg Urea at "
    "active tillering stage, at panicle initiation stage apply 30-32 kg Urea "
    "and 16-18 kg MOP fertilizer."
)

results = []
def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print("REGRESSION TESTS: Stage-specific answer scoping")
print("=" * 70)

# 1. Tillering-only query with area → deterministic schedule should show only tillering
det = _deterministic_schedule(
    "How much tillering stage fertilizer should I give to the rice on 1.5 acres?",
    FAQ_SCHEDULE)
if det:
    has_tillering = "tillering" in det.lower()
    has_basal = "basal" in det.lower()
    has_panicle = "panicle" in det.lower()
    test("Tillering query: includes tillering", has_tillering, det[:120])
    test("Tillering query: excludes basal", not has_basal)
    test("Tillering query: excludes panicle", not has_panicle)
else:
    test("Tillering query: produces output", False, "got None")

# 2. Basal-only query
det_basal = _deterministic_schedule(
    "What is the basal dose for rice on 2 acres?",
    FAQ_SCHEDULE)
if det_basal:
    test("Basal query: includes basal", "basal" in det_basal.lower() or "sowing" in det_basal.lower(),
         det_basal[:120])
    test("Basal query: excludes tillering", "tillering" not in det_basal.lower())
    test("Basal query: excludes panicle", "panicle" not in det_basal.lower())
else:
    test("Basal query: produces output", False, "got None")

# 3. Panicle initiation query
det_pi = _deterministic_schedule(
    "How much fertilizer at panicle initiation for 1.5 acre rice?",
    FAQ_SCHEDULE)
if det_pi:
    test("PI query: includes panicle", "panicle" in det_pi.lower(), det_pi[:120])
    test("PI query: excludes basal", "basal" not in det_pi.lower())
    test("PI query: excludes tillering", "tillering" not in det_pi.lower())
else:
    test("PI query: produces output", False, "got None")

# 4. Full schedule query (no specific stage) should return ALL stages
det_full = _deterministic_schedule(
    "What fertilizer should I apply for rice on 2 acres?",
    FAQ_SCHEDULE)
if det_full:
    test("Full schedule: includes basal",
         "basal" in det_full.lower() or "sowing" in det_full.lower())
    test("Full schedule: includes tillering", "tillering" in det_full.lower())
    test("Full schedule: includes panicle", "panicle" in det_full.lower())
else:
    test("Full schedule: produces output", False, "got None")

# 5. Stage absent from context → should return None
det_flower = _deterministic_schedule(
    "What fertilizer at flowering stage for 1.5 acres of rice?",
    FAQ_SCHEDULE)
test("Absent stage (flowering): returns None", det_flower is None,
     f"got: {det_flower}" if det_flower else "correct")

# 6. 1-acre query (no area scaling) → _deterministic_schedule returns None (expected)
det_1ac = _deterministic_schedule(
    "How much tillering stage fertilizer for 1 acre rice?",
    FAQ_SCHEDULE)
test("1-acre tillering: returns None (no scaling needed)", det_1ac is None,
     f"got: {det_1ac}" if det_1ac else "correct")

# 7. Verify _context_rates still extracts all stages
rates = _context_rates(FAQ_SCHEDULE)
stages_found = set(r[5] for r in rates if r[5])
test("Rate extraction: finds basal", any("basal" in s for s in stages_found))
test("Rate extraction: finds tillering",
     any("tillering" in s for s in stages_found))
test("Rate extraction: finds panicle",
     any("panicle" in s for s in stages_found))

# 8. Stage completeness guard — check that the regex detects query stages
from queries.services.pipeline import _STAGE_LABEL_RE, _STAGE_ORDER
q_stages = set(s.strip().lower() for s in _STAGE_LABEL_RE.findall(
    "How much tillering stage fertilizer should I give?"))
test("Stage detection: 'tillering' detected in query", "tillering" in q_stages)
q_orders = {_STAGE_ORDER.get(s, -1) for s in q_stages} - {-1}
test("Stage detection: maps to order 1 (tillering)", 1 in q_orders)

# No stages in a generic query
q_generic = set(_STAGE_LABEL_RE.findall("What fertilizer for rice?"))
test("Stage detection: generic query has no stages", len(q_generic) == 0)


print(f"\n{'=' * 70}")
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
if failed:
    print("\nFailed:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
print("=" * 70)
