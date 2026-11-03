#!/usr/bin/env python
"""
Production ChromaDB chunk quality audit — READ-ONLY.
Run on the server: cd /home/paddymitra_dev/PaddyMitra && python docs/rag_audit/production_chunk_audit.py

Writes results to docs/rag_audit/production_chunk_results.json
and a human-readable summary to stdout.

Does NOT modify any data, embeddings, indexes, or configurations.
"""
import sys, os, re, json, hashlib, statistics
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb

# Direct path — no Django needed for read-only ChromaDB access
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
CHROMA_DIR = str(PROJECT_DIR / "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR)

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = OUTPUT_DIR / "production_chunk_results.json"

# ============================================================
# 1. Collect all chunks from all collections
# ============================================================

collections = client.list_collections()
print(f"Collections found: {len(collections)}")

all_chunks = []       # list of dicts
per_col_stats = {}    # collection -> summary dict
content_hashes = Counter()  # for duplicate detection

for col_meta in collections:
    col = client.get_collection(col_meta.name)
    count = col.count()
    print(f"\n{'='*60}")
    print(f"Collection: {col_meta.name} | Total chunks: {count}")

    if count == 0:
        per_col_stats[col_meta.name] = {"total": 0}
        continue

    # Fetch ALL chunks in batches
    batch_size = 200
    offset = 0
    col_chunks = []

    while offset < count:
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )
        if not result["documents"]:
            break
        for i, doc in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            chunk_id = result["ids"][i]
            doc_title = (meta or {}).get("document_title", "?")
            doc_id = (meta or {}).get("document_id", "?")
            chunk_idx = (meta or {}).get("chunk_index", "?")

            text = doc or ""
            raw_len = len(text)
            non_ws = len(re.sub(r"\s", "", text))
            ws_count = raw_len - non_ws
            ws_ratio = ws_count / raw_len if raw_len > 0 else 1.0

            # Whitespace breakdown
            spaces = text.count(" ")
            newlines = text.count("\n")
            tabs = text.count("\t")
            consec_3 = len(re.findall(r" {3,}", text))
            consec_5 = len(re.findall(r" {5,}", text))
            consec_10 = len(re.findall(r" {10,}", text))
            blank_lines = len(re.findall(r"\n\s*\n", text))

            # Content heuristics
            has_numbers = bool(re.search(r"\d+\.?\d*\s*(?:kg|g|ton|t|ml|l|ha|acre|hectare|%)", text, re.I))
            has_fertilizer = bool(re.search(r"\b(urea|dap|mop|npk|potash|nitrogen|phosphorus)\b", text, re.I))
            has_seed = bool(re.search(r"\b(seed\s*rate|seed|bihan)\b", text, re.I))
            has_fym = bool(re.search(r"\b(fym|farmyard|farm\s*yard|manure|compost|vermicompost)\b", text, re.I))
            has_azolla = bool(re.search(r"\bazolla\b", text, re.I))
            has_stage = bool(re.search(r"\b(basal|tillering|panicle|flowering|heading|grain.?fill)\b", text, re.I))

            # Content hash for duplicate detection
            norm_text = re.sub(r"\s+", " ", text.lower().strip())
            content_hash = hashlib.md5(norm_text.encode()).hexdigest()
            content_hashes[content_hash] += 1

            chunk = {
                "collection": col_meta.name,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "chunk_index": chunk_idx,
                "raw_len": raw_len,
                "non_ws_len": non_ws,
                "ws_ratio": round(ws_ratio, 4),
                "spaces": spaces,
                "newlines": newlines,
                "tabs": tabs,
                "consec_3": consec_3,
                "consec_5": consec_5,
                "consec_10": consec_10,
                "blank_lines": blank_lines,
                "has_numbers": has_numbers,
                "has_fertilizer": has_fertilizer,
                "has_seed": has_seed,
                "has_fym": has_fym,
                "has_azolla": has_azolla,
                "has_stage": has_stage,
                "content_hash": content_hash,
                "preview": repr(text[:300]),
            }
            col_chunks.append(chunk)
            all_chunks.append(chunk)

        offset += batch_size

    # Per-collection summary
    if col_chunks:
        ws_ratios = [c["ws_ratio"] for c in col_chunks]
        raw_lens = [c["raw_len"] for c in col_chunks]
        non_ws_lens = [c["non_ws_len"] for c in col_chunks]

        stats = {
            "total": len(col_chunks),
            "ws_ratio_min": round(min(ws_ratios), 4),
            "ws_ratio_p25": round(sorted(ws_ratios)[len(ws_ratios)//4], 4),
            "ws_ratio_median": round(statistics.median(ws_ratios), 4),
            "ws_ratio_mean": round(statistics.mean(ws_ratios), 4),
            "ws_ratio_p75": round(sorted(ws_ratios)[int(len(ws_ratios)*0.75)], 4),
            "ws_ratio_p90": round(sorted(ws_ratios)[int(len(ws_ratios)*0.90)], 4),
            "ws_ratio_p95": round(sorted(ws_ratios)[int(len(ws_ratios)*0.95)], 4),
            "ws_ratio_max": round(max(ws_ratios), 4),
            "ws_gt_50": sum(1 for r in ws_ratios if r > 0.5),
            "ws_gt_60": sum(1 for r in ws_ratios if r > 0.6),
            "ws_gt_70": sum(1 for r in ws_ratios if r > 0.7),
            "ws_gt_80": sum(1 for r in ws_ratios if r > 0.8),
            "non_ws_lt_50": sum(1 for c in col_chunks if c["non_ws_len"] < 50),
            "non_ws_lt_20": sum(1 for c in col_chunks if c["non_ws_len"] < 20),
            "consec_5_any": sum(1 for c in col_chunks if c["consec_5"] > 0),
            "consec_10_any": sum(1 for c in col_chunks if c["consec_10"] > 0),
            "raw_len_min": min(raw_lens),
            "raw_len_median": round(statistics.median(raw_lens)),
            "raw_len_max": max(raw_lens),
            "non_ws_len_min": min(non_ws_lens),
            "non_ws_len_median": round(statistics.median(non_ws_lens)),
            "non_ws_len_max": max(non_ws_lens),
            "has_fertilizer": sum(1 for c in col_chunks if c["has_fertilizer"]),
            "has_seed": sum(1 for c in col_chunks if c["has_seed"]),
            "has_fym": sum(1 for c in col_chunks if c["has_fym"]),
            "has_azolla": sum(1 for c in col_chunks if c["has_azolla"]),
            "has_stage": sum(1 for c in col_chunks if c["has_stage"]),
        }
        per_col_stats[col_meta.name] = stats

        print(f"  WS ratio: min={stats['ws_ratio_min']}, median={stats['ws_ratio_median']}, "
              f"p90={stats['ws_ratio_p90']}, max={stats['ws_ratio_max']}")
        print(f"  WS > 50%: {stats['ws_gt_50']} ({stats['ws_gt_50']/len(col_chunks)*100:.1f}%)")
        print(f"  WS > 70%: {stats['ws_gt_70']} ({stats['ws_gt_70']/len(col_chunks)*100:.1f}%)")
        print(f"  Non-WS < 50 chars: {stats['non_ws_lt_50']}")
        print(f"  5+ consec spaces: {stats['consec_5_any']} ({stats['consec_5_any']/len(col_chunks)*100:.1f}%)")
        print(f"  10+ consec spaces: {stats['consec_10_any']} ({stats['consec_10_any']/len(col_chunks)*100:.1f}%)")
        print(f"  Contains fertilizer terms: {stats['has_fertilizer']}")
        print(f"  Contains seed terms: {stats['has_seed']}")
        print(f"  Contains FYM terms: {stats['has_fym']}")
        print(f"  Contains azolla: {stats['has_azolla']}")
        print(f"  Contains stage names: {stats['has_stage']}")

# ============================================================
# 2. Global summary
# ============================================================

print(f"\n{'='*60}")
print(f"GLOBAL SUMMARY")
print(f"{'='*60}")
print(f"Total chunks: {len(all_chunks)}")

if all_chunks:
    ws_ratios = [c["ws_ratio"] for c in all_chunks]
    print(f"\nWHITESPACE DISTRIBUTION")
    print(f"  min:    {min(ws_ratios):.4f}")
    print(f"  p10:    {sorted(ws_ratios)[int(len(ws_ratios)*0.10)]:.4f}")
    print(f"  p25:    {sorted(ws_ratios)[len(ws_ratios)//4]:.4f}")
    print(f"  median: {statistics.median(ws_ratios):.4f}")
    print(f"  mean:   {statistics.mean(ws_ratios):.4f}")
    print(f"  p75:    {sorted(ws_ratios)[int(len(ws_ratios)*0.75)]:.4f}")
    print(f"  p90:    {sorted(ws_ratios)[int(len(ws_ratios)*0.90)]:.4f}")
    print(f"  p95:    {sorted(ws_ratios)[int(len(ws_ratios)*0.95)]:.4f}")
    print(f"  p99:    {sorted(ws_ratios)[int(len(ws_ratios)*0.99)]:.4f}")
    print(f"  max:    {max(ws_ratios):.4f}")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = sum(1 for r in ws_ratios if r > threshold)
        print(f"  WS > {threshold*100:.0f}%: {n} ({n/len(all_chunks)*100:.1f}%)")

    print(f"\nCHUNK SIZE")
    raw_lens = [c["raw_len"] for c in all_chunks]
    non_ws_lens = [c["non_ws_len"] for c in all_chunks]
    print(f"  Raw:    min={min(raw_lens)}, median={statistics.median(raw_lens):.0f}, max={max(raw_lens)}")
    print(f"  Non-WS: min={min(non_ws_lens)}, median={statistics.median(non_ws_lens):.0f}, max={max(non_ws_lens)}")
    print(f"  Non-WS < 20: {sum(1 for n in non_ws_lens if n < 20)}")
    print(f"  Non-WS < 50: {sum(1 for n in non_ws_lens if n < 50)}")
    print(f"  Non-WS < 100: {sum(1 for n in non_ws_lens if n < 100)}")

    # Duplicates
    dupes = sum(v for v in content_hashes.values() if v > 1)
    unique_duped = sum(1 for v in content_hashes.values() if v > 1)
    print(f"\nDUPLICATES")
    print(f"  Unique content hashes: {len(content_hashes)}")
    print(f"  Duplicate groups: {unique_duped}")
    print(f"  Total duplicate chunks: {dupes}")

    # Content coverage
    print(f"\nCONTENT COVERAGE (keyword presence)")
    print(f"  Fertilizer terms: {sum(1 for c in all_chunks if c['has_fertilizer'])}")
    print(f"  Seed terms: {sum(1 for c in all_chunks if c['has_seed'])}")
    print(f"  FYM/manure terms: {sum(1 for c in all_chunks if c['has_fym'])}")
    print(f"  Azolla: {sum(1 for c in all_chunks if c['has_azolla'])}")
    print(f"  Stage names (basal/tillering/panicle/flowering): {sum(1 for c in all_chunks if c['has_stage'])}")

# ============================================================
# 3. Worst chunks
# ============================================================

print(f"\n{'='*60}")
print(f"WORST 25 CHUNKS BY WHITESPACE RATIO")
print(f"{'='*60}")
worst = sorted(all_chunks, key=lambda c: c["ws_ratio"], reverse=True)[:25]
for c in worst:
    print(f"\n  [{c['collection']}] {c['chunk_id']} | doc='{c['doc_title']}' (id={c['doc_id']})")
    print(f"  raw={c['raw_len']} non_ws={c['non_ws_len']} ws_ratio={c['ws_ratio']} "
          f"consec5={c['consec_5']} consec10={c['consec_10']}")
    print(f"  {c['preview'][:200]}")

# ============================================================
# 4. Shortest non-WS chunks
# ============================================================

print(f"\n{'='*60}")
print(f"SHORTEST 20 CHUNKS BY NON-WHITESPACE LENGTH")
print(f"{'='*60}")
shortest = sorted(all_chunks, key=lambda c: c["non_ws_len"])[:20]
for c in shortest:
    print(f"\n  [{c['collection']}] {c['chunk_id']} | doc='{c['doc_title']}' (id={c['doc_id']})")
    print(f"  raw={c['raw_len']} non_ws={c['non_ws_len']} ws_ratio={c['ws_ratio']}")
    print(f"  {c['preview'][:200]}")

# ============================================================
# 5. Worst documents (by avg whitespace)
# ============================================================

print(f"\n{'='*60}")
print(f"WORST 15 DOCUMENTS BY AVG WHITESPACE RATIO")
print(f"{'='*60}")
doc_ws = defaultdict(list)
for c in all_chunks:
    key = f"{c['doc_id']}|{c['doc_title']}"
    doc_ws[key].append(c["ws_ratio"])

doc_avg = [(k, statistics.mean(v), len(v)) for k, v in doc_ws.items()]
doc_avg.sort(key=lambda x: x[1], reverse=True)
for k, avg, count in doc_avg[:15]:
    ratios = doc_ws[k]
    print(f"  {k}: {count} chunks, avg_ws={avg:.3f}, max_ws={max(ratios):.3f}")

# ============================================================
# 6. Fertilizer/seed/FYM/azolla chunk inspection
# ============================================================

print(f"\n{'='*60}")
print(f"CHUNKS WITH SPECIFIC AGRICULTURAL CONTENT")
print(f"{'='*60}")

for label, key in [("Seed", "has_seed"), ("FYM/manure", "has_fym"),
                    ("Azolla", "has_azolla"), ("Fertilizer+Stage", None)]:
    if key:
        matches = [c for c in all_chunks if c[key]]
    else:
        matches = [c for c in all_chunks if c["has_fertilizer"] and c["has_stage"]]

    print(f"\n--- {label}: {len(matches)} chunks ---")
    if matches:
        ws = [c["ws_ratio"] for c in matches]
        print(f"  WS ratio: min={min(ws):.3f}, median={statistics.median(ws):.3f}, max={max(ws):.3f}")
        print(f"  WS > 50%: {sum(1 for r in ws if r > 0.5)}")
        # Show top 3 by WS ratio
        matches_sorted = sorted(matches, key=lambda c: c["ws_ratio"], reverse=True)
        for c in matches_sorted[:3]:
            print(f"    [{c['collection']}] {c['chunk_id']} ws={c['ws_ratio']} "
                  f"non_ws={c['non_ws_len']} doc='{c['doc_title']}'")
            print(f"    {c['preview'][:150]}")

# ============================================================
# 7. Cross-collection duplicates detail
# ============================================================

print(f"\n{'='*60}")
print(f"CROSS-COLLECTION DUPLICATES (same content in multiple collections)")
print(f"{'='*60}")

hash_to_chunks = defaultdict(list)
for c in all_chunks:
    hash_to_chunks[c["content_hash"]].append(c)

cross_col_dupes = 0
for h, chunks in hash_to_chunks.items():
    if len(chunks) > 1:
        cols = set(c["collection"] for c in chunks)
        if len(cols) > 1:
            cross_col_dupes += 1
            if cross_col_dupes <= 10:
                print(f"\n  Hash {h[:12]}... appears in {len(chunks)} chunks across {cols}")
                for c in chunks:
                    print(f"    [{c['collection']}] {c['chunk_id']} doc='{c['doc_title']}'")

print(f"\nTotal cross-collection duplicate groups: {cross_col_dupes}")

# ============================================================
# Save full results
# ============================================================

results = {
    "summary": {
        "total_chunks": len(all_chunks),
        "collections": len(collections),
    },
    "per_collection": per_col_stats,
    "worst_25": [{k: v for k, v in c.items() if k != "preview"} for c in worst],
    "shortest_20": [{k: v for k, v in c.items() if k != "preview"} for c in shortest],
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {RESULTS_FILE}")
print("Done.")
