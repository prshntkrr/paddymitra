# Whitespace and Chunk Quality Audit

**Date:** 2026-10-27
**Scope:** Investigation only — no modifications made
**Data source:** Local ChromaDB (357 chunks, dev subset) + extraction code analysis + server retrieval traces from client issue investigation

---

## 1. Measured Whitespace Statistics (Local DB — 357 Chunks)

### Overall Distribution

| Metric | Value |
|---|---|
| Total chunks | 357 |
| Min whitespace ratio | 6.5% |
| P25 | 16.4% |
| Median | 18.4% |
| Mean | 17.8% |
| P75 | 19.4% |
| P90 | 20.6% |
| P95 | 21.7% |
| Max | 23.6% |

### Problem Counts

| Problem | Count | Percentage |
|---|---|---|
| WS > 50% | 0 | 0.0% |
| WS > 60% | 0 | 0.0% |
| WS > 70% | 0 | 0.0% |
| Non-WS < 50 chars | 17 | 4.8% |
| 3+ consecutive spaces | 2 | 0.6% |
| 5+ consecutive spaces | 0 | 0.0% |

### Chunk Length

| Metric | Raw chars | Non-WS chars |
|---|---|---|
| Min | 31 | 29 |
| Median | 229 | 185 |
| Max | 798 | 694 |

### Per-Collection

| Collection | Chunks | Median WS | WS > 50% |
|---|---|---|---|
| rice-knowledge-base | 193 | 18.3% | 0 (0%) |
| Varietal_Recommendation | 106 | 19.0% | 0 (0%) |
| Nutrient_Management | 56 | 16.0% | 0 (0%) |
| Insect-pest-disease-management | 2 | 16.0% | 0 (0%) |

### Assessment

The local database shows **no excessive whitespace problem**. Whitespace ratios are healthy (15-22%), consistent with normal English prose. No chunks exceed 50% whitespace.

**However:** The local DB contains only 357 chunks — a dev/test subset. The production server has 3,447+ chunks in rice-knowledge-base alone, plus 2,551 in Nutrient_Management. The whitespace problem reported by the user likely exists in the production data, which includes PDF-extracted content from table-heavy agricultural documents not available locally.

---

## 2. Extraction Code Analysis

### Source: `catalogues/services/chunking.py`

#### PDF Extraction (`extract_pdf_text()`, line 35)

```python
text = page.extract_text(extraction_mode="layout") or ""
```

**Layout mode** preserves column alignment using spaces. For a fertilizer table like:

```
Fertilizer    Per acre    Per hectare    Time
Urea            5 kg        12.5 kg      Basal
DAP            35 kg        87.5 kg      Basal
MOP            16 kg         40 kg       Basal
```

pypdf layout mode produces text with **embedded spaces** that encode column positions. This is by design — the code comment (line 41-46) explicitly explains why layout mode was chosen: it preserves table column relationships that default mode flattens.

**Key behavior:**
- Layout mode tries first per page; falls back to default if layout returns empty (line 66)
- Pages where layout mode works keep column-aligned spaces
- No post-extraction whitespace normalization is applied
- The raw layout text goes directly to `chunk_text()`

#### Chunking (`chunk_text()`, line 28)

```python
def chunk_text(text: str) -> List[str]:
    text = (text or "").strip()
    return [c.strip() for c in _splitter().split_text(text) if c.strip()]
```

- Outer `.strip()` removes leading/trailing whitespace
- Each chunk is `.strip()`-ed and empty chunks are filtered
- **But internal whitespace is preserved** — column-alignment spaces survive into the stored chunks
- The splitter uses `len` (character count), so **whitespace counts toward the 800-char limit**

### Root Cause Chain

1. pypdf layout mode → spaces encode column alignment
2. No whitespace normalization between extraction and chunking
3. `RecursiveCharacterTextSplitter` counts spaces toward 800-char limit
4. Column-aligned text produces chunks where 40-70% may be spaces
5. These spaces consume chunk capacity, reducing actual information per chunk

### Why This Matters for Table-Heavy Documents

A fertilizer schedule table extracted in layout mode might produce text like:

```
Fertilizer sources                Per acre                Per hectare           Time of application
 Lime application                   200 kg                    500 kg                     Basal
 Urea                                5 kg                     12.5 kg
 DAP                                 35 kg                    87.5 kg
 MOP                                 16 kg                     40 kg
```

Each row has ~30 chars of data but ~80 chars of spacing. An 800-char chunk could hold ~10 table rows in layout mode but ~25+ rows if whitespace were collapsed. More importantly, the 800-char boundary may split the table between rows 4 and 5, separating some fertilizers from their header row.

---

## 3. Evidence from Server Retrieval Traces

During the client issue investigation (2026-10-26), the user shared server retrieval results for "How much fertilizer application will be done on 1 acre of land?" showing actual stored chunks:

### Example: Source[4] — Dhaincha seed production (doc_id=36, chunk=2)

```
Green manuring seed production / Nutrient management
60 cm

Nutrient    management

 Dose: 20-30:40:20 kg/ha
 Fertilizer sources                Per acre                Per hectare           Time of application
 Lime application                   200 kg                    500 kg                     Basal
 Urea                                5 kg                     12.5 kg
 DAP                                 35 kg                    87.5 kg
 MOP                                 16 kg                     40 kg
 Foliar spray of DAP (2%)                                                               35 DAS
                                                                                        45 DAS
```

**Observations:**
- Heavy column-alignment whitespace (estimated 50-60% of chunk)
- Table structure is preserved (Urea/DAP/MOP with per-acre values readable)
- But no stage names (basal/tillering/panicle) — only "Basal" for lime and "Time of application" header
- "60 cm" appears as orphaned text (likely row spacing from a previous table)
- `_context_rates()` was able to parse "5 kg" + "Urea" + "per acre" from this format

### Example: Source[7] — Research paper (doc_id=72, chunk=143)

```
Fertilizer           applied           (kg           ha-1)                                    0–60                                    0–30                                    0–30                                    0–0.5
```

**Observations:**
- Extreme whitespace — this is a table row with column-alignment spaces
- The actual data ("0–60", "0–30", etc.) is separated by vast stretches of spaces
- Column headers are in a different chunk
- Meaningless without the header row — `_context_rates()` cannot parse this

### Example: Source[8] — Rice Production Manual (doc_id=7, chunk=32)

```
•                                                             Application of Mn                                                                                 amount of fertilizer
```

**Observations:**
- Almost entirely whitespace — a bullet point and two phrases separated by ~100 spaces
- This is a header/caption from a multi-column PDF page
- Contains no parseable agricultural data
- Occupies a retrieval slot that could hold a useful chunk

---

## 4. How Whitespace Affects Chunk Boundaries

### Character budget waste

The 800-char `CHUNK_SIZE` counts all characters including whitespace. For a chunk that is 60% whitespace:
- 800 total chars → only 320 chars of actual content
- Equivalent to a 320-char chunk with no whitespace waste
- A normal English paragraph at 18% whitespace would hold ~660 chars of content in the same space

### Table splitting

Tables are the primary casualty. A fertilizer schedule with 4 stages × 3 fertilizers × 2 columns ≈ 24 data cells. In layout mode with column alignment, each row might be 80-120 chars. The full table is 960-2880 chars — always exceeds the 800-char limit.

The splitter splits at the nearest `\n` boundary, which means:
- Some stages end up in one chunk, others in the next
- The first chunk may lack the column headers
- The second chunk starts mid-table with no context

### Heading separation

Layout mode can produce text like:

```
Nutrient Management                                                                   Chapter 3

    Recommended fertilizer doses for medium-duration rice varieties

Fertilizer                Per acre
```

The heading and subheading may end up in a previous chunk, while the table data starts a new chunk without any heading context.

---

## 5. Impact on Embeddings and Retrieval

### Embedding quality

Embedding models (text-embedding-3-small) process the raw text including whitespace. While transformer tokenizers generally handle multiple spaces gracefully (collapsing them), the embedding vector is diluted:
- A chunk that is 60% whitespace produces a weaker semantic signal
- The meaningful content is spread across the token sequence with space tokens in between
- **Hypothesis (unverified):** Chunks with high whitespace may have lower similarity to well-formed queries, reducing their retrieval rank

### Retrieval displacement

From the server traces, Source[8] (`"• Application of Mn amount of fertilizer"` with massive whitespace) occupied a top-10 retrieval position. This chunk is essentially empty data but its embedding was close enough to "fertilizer" queries to rank. It displaced what could have been a chunk with actual fertilizer schedule data.

### `_context_rates()` parsing

The deterministic schedule parser (`_context_rates()` in pipeline.py) uses regex to find patterns like `35 kg/acre DAP`. Layout-mode spaces between the number and the substance can exceed the 60-character window used for substance matching (`_SUBSTANCE_MAX_GAP = 25`). If "DAP" is 30+ spaces away from "35 kg", the parser fails to associate them.

**Verified from code (pipeline.py line 3981):**
```python
win_start = max(0, m.start() - 60)
window = flat[win_start: m.end() + 60].lower()
```

The 60-char window and 25-char substance gap limit mean layout-mode table spacing can push substance names outside the matching window.

---

## 6. Potential Remedies (Evaluation Only)

### A. Whitespace Normalization Before Chunking

**Approach:** Collapse runs of 2+ spaces to single space; collapse 3+ newlines to 2.

**Pros:**
- Immediately increases information density per chunk
- More content per 800-char window
- `_context_rates()` substance matching works better
- Embedding quality improves (less noise tokens)

**Cons:**
- **Destroys table column alignment.** "Urea 5 kg" and "DAP 35 kg" become ambiguous without column spacing. A naive collapse makes it impossible to tell which quantity belongs to which fertilizer.
- The code comment at line 41 explicitly warns about this: layout mode was chosen BECAUSE it preserves table columns.

**Risk:** High if applied blindly. Must distinguish tables from prose.

### B. Table Detection + Selective Handling

**Approach:** Detect table regions (consistent column alignment patterns, `|`-separated cells, grid-like spacing), preserve them intact, normalize only non-table text.

**Pros:** Preserves table structure while cleaning prose
**Cons:** Requires table detection heuristic; false positives could damage prose, false negatives leave tables bloated
**Complexity:** Medium

### C. Remove Empty and Near-Empty Chunks

**Approach:** Filter chunks where `non_ws_chars < 50` during build.

**Pros:** Removes useless chunks like Source[8] that occupy retrieval slots
**Cons:** Some short chunks may be valid (FAQ answers, single-fact statements)
**Risk:** Low — the 50-char threshold is conservative

### D. Header and Footer Filtering

**Approach:** Detect and remove repeated page headers/footers before chunking.

**Pros:** Removes "Chapter 3" / page numbers / document titles that repeat on every page
**Cons:** Requires page-level analysis; some headers carry useful section context
**Risk:** Low if implemented per-document with manual review

### E. Increase Chunk Size for Tables

**Approach:** Use larger chunks (1200-1600 chars) for table-detected regions so entire tables fit.

**Pros:** Keeps complete tables with headers + all rows in one chunk
**Cons:** Larger chunks may reduce retrieval precision for non-table content; embedding model handles longer text differently
**Risk:** Medium — requires benchmark

### F. Table-Aware Extraction

**Approach:** Use a table extraction library (pdfplumber, camelot) to detect and extract tables as structured data, then format them as "Header: Value" pairs instead of column-aligned text.

**Example output:**
```
Basal stage: Urea 5 kg/acre, DAP 35 kg/acre, MOP 16 kg/acre
Active tillering: Urea 30-35 kg/acre
Panicle initiation: Urea 30-35 kg/acre, MOP 15-20 kg/acre
```

**Pros:** Eliminates whitespace waste; produces `_context_rates()`-parseable text; complete table in one chunk
**Cons:** Requires additional library; table detection may fail on complex layouts
**Risk:** Medium — needs validation against actual documents

---

## 7. Whether Re-indexing Would Be Necessary

**Yes.** Any change to extraction or chunking requires rebuilding the vector index because:
1. Chunk text changes → different embeddings
2. Chunk boundaries change → different chunk IDs
3. Metadata may change → different filter results
4. The existing `build_catalogue_vector_db()` already does a full rebuild (delete + recreate)

The safe approach from the Gate 1 report applies: build a staging index, benchmark against the current one, and switch only after verification.

---

## 8. Limitations of This Audit

1. **Local DB is a dev subset (357 chunks).** Production has 6,000+ chunks across all collections. The whitespace distribution measured here may not represent production.

2. **No source PDFs available locally.** Cannot run side-by-side comparisons of original pages vs. extracted text vs. stored chunks. This comparison requires access to the server's `media/documents/` directory.

3. **Server chunk sampling not performed.** To measure production whitespace distribution, run the audit script on the server where chromadb is installed and the full database exists.

### Recommended Next Step

Run this audit on the production server to get accurate whitespace statistics across all 6,000+ chunks. The script from this investigation can be adapted for the server environment where chromadb is available.

---

*Investigation only. No code, embeddings, or indexes were modified.*
