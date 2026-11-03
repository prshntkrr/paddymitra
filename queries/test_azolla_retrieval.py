"""
Regression tests for Azolla query retrieval and answer grounding.
Tests entity-relevance reranking and substance identity guard.
"""
import sys, io, os, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\parsh\Downloads\git\paddymitra_new\paddymitra_rollback\PaddyMitra")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.dev")
import django
django.setup()

from queries.services.pipeline import _rerank_by_substance

# ── Unit tests for _rerank_by_substance ────────────────────────────────────

def test_rerank_promotes_azolla():
    """Azolla query: chunks mentioning azolla should come first."""
    hits = [
        {"chunk": "Apply 200 kg lime and 5 kg urea for Dhaincha", "metadata": {"document_title": "Dhaincha"}},
        {"chunk": "Apply 4 quintals of Azolla per acre", "metadata": {"document_title": "Nutrient Management"}},
        {"chunk": "Nitrogen split: 60-120 kg/ha", "metadata": {"document_title": "Split application"}},
        {"chunk": "Apply 500 kg Azolla per hectare", "metadata": {"document_title": "Benefits of Azolla in Rice"}},
    ]
    result = _rerank_by_substance("How much Azolla should be applied on 1 acre?", hits)
    # First two should be the azolla hits
    assert "azolla" in result[0]["chunk"].lower(), f"Expected azolla chunk first, got: {result[0]['chunk'][:60]}"
    assert "azolla" in result[1]["chunk"].lower(), f"Expected azolla chunk second, got: {result[1]['chunk'][:60]}"
    assert len(result) == 4, "All hits preserved"
    print("  [PASS] test_rerank_promotes_azolla")


def test_rerank_promotes_dhaincha():
    """Dhaincha query: chunks mentioning dhaincha should come first."""
    hits = [
        {"chunk": "Apply 4 quintals of Azolla per acre", "metadata": {"document_title": "Benefits of Azolla"}},
        {"chunk": "Sow Dhaincha seeds at 40 kg/ha", "metadata": {"document_title": "Dhaincha"}},
        {"chunk": "Nitrogen split: 60 kg/ha", "metadata": {"document_title": "Nitrogen"}},
    ]
    result = _rerank_by_substance("How to apply Dhaincha as green manure?", hits)
    assert "dhaincha" in result[0]["chunk"].lower()
    print("  [PASS] test_rerank_promotes_dhaincha")


def test_rerank_no_effect_on_urea():
    """Urea/DAP/MOP queries should NOT trigger reranking (too common)."""
    hits = [
        {"chunk": "Apply 30 kg urea at tillering", "metadata": {"document_title": "N splits"}},
        {"chunk": "Apply 4 quintals of Azolla", "metadata": {"document_title": "Azolla"}},
    ]
    result = _rerank_by_substance("How much urea to apply?", hits)
    # Should be unchanged — urea is not in _AGRI_INPUT_RE
    assert result[0]["chunk"] == hits[0]["chunk"], "Urea query should not rerank"
    print("  [PASS] test_rerank_no_effect_on_urea")


def test_rerank_no_relevant_hits():
    """When no hits mention the substance, order unchanged."""
    hits = [
        {"chunk": "Apply 30 kg DAP at basal", "metadata": {"document_title": "Fertilizer"}},
        {"chunk": "Nitrogen split: 60 kg/ha", "metadata": {"document_title": "Nitrogen"}},
    ]
    result = _rerank_by_substance("How much Azolla should I apply?", hits)
    assert result == hits, "No azolla hits → order unchanged"
    print("  [PASS] test_rerank_no_relevant_hits")


def test_rerank_title_match():
    """Chunks matching by document title (not chunk text) are also promoted."""
    hits = [
        {"chunk": "General nutrient info", "metadata": {"document_title": "Nitrogen"}},
        {"chunk": "Green manure application rates", "metadata": {"document_title": "Benefits of Azolla in Rice"}},
    ]
    result = _rerank_by_substance("How much Azolla to use?", hits)
    assert "Azolla" in result[0]["metadata"]["document_title"]
    print("  [PASS] test_rerank_title_match")


def test_rerank_preserves_all_hits():
    """All hits are preserved, just reordered."""
    hits = [{"chunk": f"chunk {i}", "metadata": {"document_title": f"doc {i}"}} for i in range(10)]
    hits[7]["chunk"] = "Apply Azolla in standing water"
    result = _rerank_by_substance("Azolla application rate?", hits)
    assert len(result) == 10
    assert "Azolla" in result[0]["chunk"]
    print("  [PASS] test_rerank_preserves_all_hits")


def test_substance_guard_lime():
    """Lime query should trigger reranking."""
    hits = [
        {"chunk": "Apply 30 kg DAP", "metadata": {"document_title": "Fertilizer"}},
        {"chunk": "Apply 200 kg lime per acre", "metadata": {"document_title": "Soil amendment"}},
    ]
    result = _rerank_by_substance("How much lime to apply for acidic soil?", hits)
    assert "lime" in result[0]["chunk"].lower()
    print("  [PASS] test_substance_guard_lime")


def test_fym_query():
    """FYM (farm yard manure) query should trigger reranking."""
    hits = [
        {"chunk": "Apply 30 kg urea", "metadata": {"document_title": "Nitrogen"}},
        {"chunk": "Apply 5 tonnes FYM per acre", "metadata": {"document_title": "Organic"}},
    ]
    result = _rerank_by_substance("How much FYM should I apply?", hits)
    assert "FYM" in result[0]["chunk"]
    print("  [PASS] test_fym_query")


if __name__ == "__main__":
    print("=" * 70)
    print("REGRESSION TESTS: Azolla retrieval and entity-relevance reranking")
    print("=" * 70)

    tests = [
        test_rerank_promotes_azolla,
        test_rerank_promotes_dhaincha,
        test_rerank_no_effect_on_urea,
        test_rerank_no_relevant_hits,
        test_rerank_title_match,
        test_rerank_preserves_all_hits,
        test_substance_guard_lime,
        test_fym_query,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)
