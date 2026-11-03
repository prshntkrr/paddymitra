"""
Diagnostic test: verify "tillering stage" glossary resolution.
Ensures the intended glossary entry is selected, not a hardcoded fallback.
"""
import sys, io, os, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\parsh\Downloads\git\paddymitra_new\paddymitra_rollback\PaddyMitra")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.dev")
import django
django.setup()

from pathlib import Path
from queries.services.odia_glossary import (
    _load_glossary, _load_raw, _BUILTIN_FALLBACK, relevant_glossary,
    reload_glossary,
)

results = []
def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print("DIAGNOSTIC: Tillering stage glossary resolution")
print("=" * 70)

# ── 1. Read authoritative source: odia_glossary.json ──
json_path = Path(__file__).parent / "services" / "odia_glossary.json"
raw_json = json.loads(json_path.read_text(encoding="utf-8"))
json_tillering = raw_json.get("tillering")
json_tillering_stage = raw_json.get("tillering stage")

print(f"\n  JSON 'tillering':       {json_tillering}")
print(f"  JSON 'tillering stage': {json_tillering_stage}")

test("JSON has 'tillering' entry", json_tillering is not None)
test("JSON has 'tillering stage' entry", json_tillering_stage is not None)

# ── 2. Check _BUILTIN_FALLBACK has phrase-level entries ──
fb_tillering = _BUILTIN_FALLBACK.get("tillering")
fb_tillering_stage = _BUILTIN_FALLBACK.get("tillering stage")
fb_active = _BUILTIN_FALLBACK.get("active tillering stage")

print(f"\n  Fallback 'tillering':              {fb_tillering}")
print(f"  Fallback 'tillering stage':        {fb_tillering_stage}")
print(f"  Fallback 'active tillering stage': {fb_active}")

test("Fallback has 'tillering stage' entry",
     fb_tillering_stage is not None,
     "MISSING — fallback-only mode would split 'tillering' + 'stage'" if not fb_tillering_stage else "")
test("Fallback has 'active tillering stage' entry",
     fb_active is not None,
     "MISSING" if not fb_active else "")

# ── 3. Verify merged glossary resolves correctly ──
reload_glossary()
glossary = _load_glossary()
merged_tillering = glossary.get("tillering")
merged_tillering_stage = glossary.get("tillering stage")

print(f"\n  Merged 'tillering':       {merged_tillering}")
print(f"  Merged 'tillering stage': {merged_tillering_stage}")

# JSON should win over fallback
if json_tillering:
    expected_tillering = json_tillering if isinstance(json_tillering, str) else json_tillering[0]
    test("Merged 'tillering' matches JSON (JSON wins)",
         merged_tillering == expected_tillering,
         f"expected {expected_tillering!r}, got {merged_tillering!r}")

if json_tillering_stage:
    expected_ts = json_tillering_stage if isinstance(json_tillering_stage, str) else json_tillering_stage[0]
    test("Merged 'tillering stage' matches JSON (JSON wins)",
         merged_tillering_stage == expected_ts,
         f"expected {expected_ts!r}, got {merged_tillering_stage!r}")

# ── 4. Test relevant_glossary matching ──
text1 = "Apply 30-35 kg of Urea per acre at the tillering stage."
rel1 = relevant_glossary(text1)
print(f"\n  relevant_glossary for 'tillering stage' text:")
for k, v in sorted(rel1.items()):
    if "tiller" in k.lower():
        print(f"    {k!r}: {v!r}")

test("relevant_glossary matches 'tillering stage' as phrase",
     "tillering stage" in rel1,
     f"keys with 'tiller': {[k for k in rel1 if 'tiller' in k.lower()]}")

# The forced value should come from JSON, not fallback
if "tillering stage" in rel1:
    forced_val = rel1["tillering stage"]
    if isinstance(forced_val, list):
        forced_val = forced_val[0]
    test("Forced value for 'tillering stage' matches JSON",
         json_tillering_stage and forced_val == (json_tillering_stage if isinstance(json_tillering_stage, str) else json_tillering_stage[0]),
         f"forced={forced_val!r}, json={json_tillering_stage!r}")

# ── 5. Test 'active tillering stage' separately ──
text2 = "Apply 30-35 kg of Urea per acre at the active tillering stage."
rel2 = relevant_glossary(text2)
print(f"\n  relevant_glossary for 'active tillering stage' text:")
for k, v in sorted(rel2.items()):
    if "tiller" in k.lower():
        print(f"    {k!r}: {v!r}")

test("'active tillering stage' matched as phrase (not split)",
     "active tillering stage" in rel2 or "tillering stage" in rel2,
     f"tiller keys: {[k for k in rel2 if 'tiller' in k.lower()]}")

# ── 6. Simulate fallback-only mode ──
print(f"\n  Simulating fallback-only mode:")
fb_only = dict(_BUILTIN_FALLBACK)
fb_has_phrase = "tillering stage" in fb_only
test("Fallback-only dict has 'tillering stage'", fb_has_phrase,
     "Without this, 'stage' leaks to Google Translate → wrong ଅବସ୍ଥା suffix")

# ── 7. Simulate the full force_glossary flow ──
from queries.services.pipeline import _force_glossary_terms
marked, marks = _force_glossary_terms(text1)
print(f"\n  _force_glossary_terms('{text1[:50]}...'):")
for placeholder, odia in marks.items():
    print(f"    {placeholder}: {odia}")

# Check that "stage" is NOT left unfrozen in the marked text
test("'stage' is frozen (not left for Google Translate)",
     "stage" not in marked.lower().replace("tillering stage", ""),
     f"marked text: {marked}")


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
