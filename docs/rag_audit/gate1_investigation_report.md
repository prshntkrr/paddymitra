# PaddyMitra RAG — Gate 1 Investigation Report

**Date:** 2026-10-27
**Scope:** Investigation only — no code changes, no re-indexing
**Status:** Complete

---

## 1. Current Architecture Overview

### Data Flow: Document Upload → Farmer Response

```
Document Upload (Web UI / CSV import / PDF matching)
    ↓
Text Extraction (pypdf / python-docx / openpyxl / csv)
    ↓
Chunking (LangChain RecursiveCharacterTextSplitter, 800 chars, 120 overlap)
    ↓
Embedding (OpenAI text-embedding-3-small, 1536 dims)
    ↓
ChromaDB Storage (PersistentClient, one collection per Catalogue)
    ↓
--- Query Time ---
    ↓
Odia → English Translation (glossary → Google Translate → OpenAI fallback)
    ↓
FAQ Check (cosine similarity, threshold 0.80)
    ↓ (miss)
Query Expansion (LLM generates 3 variants + domain synonyms, max ~10 phrasings)
    ↓
Hybrid Retrieval (ChromaDB vector + BM25 keyword, fused via RRF k=60)
    ↓
Rule-based Reranking (substance match, diversification, per-doc cap)
    ↓
Context Assembly (top chunks + FAQ near-misses as context)
    ↓
LLM Answer Generation (OpenAI gpt-4o-mini, with topic guards in system prompt)
    ↓
Answer Validation (_unverified_doses, stage completeness, overstated claims)
    ↓
Deterministic Schedule Override (for area-scaled fertilizer queries)
    ↓
English → Odia Translation (Google + Gemini fluency rewrite + glossary enforcement)
    ↓
Farmer Response
```

### Source File Map

| Component | File | Key Functions |
|---|---|---|
| Document model | `documents/models.py` | `Document`, `text_for_embedding()` |
| Catalogue model | `catalogues/models.py` | `Catalogue` (slug = collection name) |
| Text extraction | `catalogues/services/chunking.py` | `document_text()`, `_extract_pdf()` |
| Chunking | `catalogues/services/chunking.py` | `chunk_text()` (line 28) |
| Embeddings | `catalogues/services/embeddings.py` | `embed_texts()`, `embedding_model_name()` |
| Vector store | `catalogues/services/vector_store.py` | `build_catalogue_vector_db()`, `search_catalogue_batch()`, `keyword_search()` |
| FAQ matching | `queries/services/faq.py` | `lookup()` (line 594), `nearest()` (line 524) |
| Pipeline (main) | `queries/services/pipeline.py` | `run_pipeline_for_text()` (line 6153), 7147 lines |
| Translation | `queries/services/pipeline.py` | `translate()` (line 1930) |
| Area math | `queries/services/area_math.py` | `farmer_area()`, `_context_rates()`, `scale_for_area()` |
| Catalogue routing | `hooks/routing.py` | Category-based routing (nutrient/variety/pest) |
| Settings | `irri_ai/settings/base.py` | CHUNK_SIZE=800, CHUNK_OVERLAP=120, models |

---

## 2. Document Extraction Findings

### Current Implementation

| Format | Library | Method |
|---|---|---|
| PDF | `pypdf.PdfReader` | Layout extraction mode first, fallback to default |
| DOCX | `python-docx` | Paragraphs + table cells (joined with ` \| `) |
| XLSX | `openpyxl` | All sheets, all rows (joined with ` \| `) |
| CSV | stdlib `csv` | All rows (joined with ` \| `) |
| TXT | `open()` | Plain read |

### Critical Gaps

1. **No OCR.** Scanned PDF pages that yield no text from pypdf silently return empty string. No Tesseract, no `unstructured`, no image-to-text. Scanned pages contribute zero chunks.

2. **No image extraction.** Photographs, diagrams, and illustrations in PDFs are silently discarded. Agricultural manuals often contain visual recommendations (pest identification photos, growth stage diagrams).

3. **Tables lose structure.** pypdf's layout mode preserves column spacing as whitespace, but the text output is still linear. DOCX tables are flattened to `cell | cell | cell` rows. Column headers are not associated with data rows after chunking splits them.

4. **No page number tracking.** Extracted text has no page provenance. Metadata includes `document_title` and `chunk_index` but not `page_start`/`page_end`.

5. **Extraction failures are silent.** If a PDF page returns empty text, no warning is logged. The document appears indexed but may have large gaps.

### Verified: What Works

- pypdf layout mode handles most digital PDFs acceptably
- DOCX paragraph extraction is complete
- Excel/CSV extraction captures all cell data
- Metadata (category, crop, doc_type) is attached per chunk

---

## 3. Chunking Implementation

### Algorithm
- **Library:** `langchain_text_splitters.RecursiveCharacterTextSplitter`
- **Chunk size:** 800 characters
- **Overlap:** 120 characters
- **Separators:** `["\n\n", "\n", ". ", " ", ""]`
- **Length function:** `len` (character count, NOT tokens)
- **No minimum chunk size** — empty or near-empty chunks are possible

### Known Weaknesses

1. **Character-based, not token-based.** 800 characters ≈ 150-200 tokens for English. The embedding model (text-embedding-3-small) has an 8191 token limit, so chunks are well under capacity, but semantic density per chunk is low.

2. **Tables split from headers.** A table spanning >800 characters gets split mid-row. Column headers end up in one chunk, data rows in the next. The data rows become meaningless without their headers.

3. **Numerical values separated from context.** A fertilizer schedule like "Basal: 5 kg Urea, 35 kg DAP" can be split from "Active tillering: 30 kg Urea" — the LLM then sees incomplete schedules.

4. **No section heading inheritance.** When a chunk starts mid-paragraph, it loses the heading that contextualized it. A chunk saying "apply 25 kg per acre" without knowing it's under "Swarna variety — Panicle initiation" is ambiguous.

5. **No table-aware chunking.** Tables are treated as plain text. Row-column relationships are not preserved.

6. **No deduplication.** The same document in multiple catalogues gets chunked and embedded independently. Near-duplicate chunks across documents are not detected.

### Chunk Statistics (from server data)

| Collection | Chunks | Notes |
|---|---|---|
| rice-knowledge-base | 3,447 | Bootstrap catch-all, all published docs |
| Nutrient_Management | 2,551 | Topic-specific subset |
| Varietal_Recommendation | 205 | Variety-specific docs |
| + 3 others | varies | |

---

## 4. Catalogue and Vector Database Structure

### Architecture

- **Database:** ChromaDB PersistentClient, stored at `<BASE_DIR>/chroma_db/chroma.sqlite3`
- **One Chroma collection per Django Catalogue record**
- **Collection name = `catalogue.slug`** (e.g. `rice-knowledge-base`, `Nutrient_Management`)
- **All collections use the same embedding model** (OpenAI text-embedding-3-small, 1536 dims)
- **Collection metadata:** `{catalogue_id, catalogue_name}`
- **Chunk metadata:** `{document_id, document_title, chunk_index, category, subcategory, crop, year, source_url, doc_type, content_type}`

### Catalogue Routing

Query routing is configured via `WHATSAPP_CATALOGUE_ROUTES` env var (format: `nutrient:5,variety:6,pest:7`).

**Router logic** (`hooks/routing.py`):
- Categories: `nutrient`, `variety`, `seed_management`, `pest`
- Each category has subject keywords (with Odia glossary expansion) and intent regex
- Scoring: subject=2pts, intent=1pt, both=+1 bonus
- Winner needs score ≥ 2.0 AND margin ≥ 2.0 over runner-up
- When TWO categories are both "confident" (score ≥ 4.0), BOTH catalogues are searched
- Unroutable queries go to `WHATSAPP_DEFAULT_CATALOGUE_ID`

### Cross-Catalogue Retrieval

`retrieve_across_catalogues()` (pipeline.py line 2951):
- Searches each catalogue independently
- Merges by RANK interleaving (not raw scores — scores across collections are not comparable)
- Deduplication by content fingerprint (SHA1 of normalized text)

### Findings

1. **Document overlap.** The same document can exist in multiple catalogues. `rice-knowledge-base` (bootstrap) likely contains ALL documents, while topic-specific catalogues contain subsets. This means the same chunk may be retrieved from two collections.

2. **Routing can miss evidence.** A generic "how much fertilizer" query may route to `Nutrient_Management` but miss variety-specific schedules in `Varietal_Recommendation`. The multi-catalogue path only fires when two categories score ≥ 4.0.

3. **No unified search.** There is no option to search all collections simultaneously. The router must pick correctly or evidence is invisible.

4. **Hypothesis (unverified):** A unified logical catalogue with metadata filtering could replace routing, eliminating the evidence-loss risk. This requires benchmarking — separate collections may have retrieval advantages from smaller, more focused embedding spaces.

---

## 5. Search and Retrieval Implementation

### Retrieval Strategy: Hybrid (Vector + BM25 + RRF)

**Vector search:**
- ChromaDB `collection.query(query_embeddings=vecs, n_results=pool)`
- Embedding: OpenAI text-embedding-3-small
- Similarity: cosine (ChromaDB default)
- Pool size: `max(n_results * 2, 12)` per query phrasing

**BM25 keyword search:**
- Custom in-memory implementation in `vector_store.py` (line 256)
- Parameters: k1=1.5, b=0.75
- Tokenizer: lowercase alphanumeric `[a-z0-9]+`
- Index cached per collection, auto-rebuilt when chunk count changes

**Fusion:**
- Reciprocal Rank Fusion (RRF) with k=60
- RRF score: `sum(1/(k + rank + 1))` across all (query, modality) ranked lists
- Each phrasing's #1 hit is reserved (up to `n_results//2` slots)

### Top-K Values

| Scenario | n_results | per_document_cap |
|---|---|---|
| Entity/area/stage query | 30 | 20 |
| Normal single-catalogue | 10 | max(3, n_results//3) |
| Multi-catalogue | 10 total | shared across catalogues |

### Query Expansion

LLM generates 3 alternative phrasings + deterministic domain synonyms + entity retrieval variants + area/stage probes. Capped at ~10 phrasings total.

### Reranking

**No neural reranker.** All reranking is rule-based:
- `_rerank_by_substance()` — promotes hits mentioning the queried substance (Azolla, FYM, etc.)
- `_diversify()` — caps chunks from one document
- RRF fusion is the primary ranking mechanism

### Thresholds and Filters

| Rule | Value | Effect |
|---|---|---|
| FAQ direct-serve threshold | 0.80 cosine | Below → FAQ becomes context, not answer |
| FAQ context floor | 0.60 cosine | Below 0.60 → FAQ ignored entirely |
| Decisive token conflict | per-group | Blocks FAQ serve if query/FAQ name different products/stages |
| Duration probe never direct-served | always | Day-count questions always go to RAG |
| Area != 1 acre skips FAQ direct-serve | always | Scaled queries always go to RAG |

---

## 6. Answer Validation and Restrictions

### Post-Generation Validation Chain

1. **`_unverified_doses()`** — every (number, substance) in answer must exist in context. Retry once. If still bad: try substance swaps, N→urea fixes, sentence stripping, deterministic schedule fallback, then "could not verify" message.

2. **Stage completeness** — if context has ≥2 fertilizer stages but answer has fewer, retry once.

3. **`overstated_claims()`** — detects promoted ratings ("moderately resistant" → "resistant"). Retry once.

4. **`_missing_candidates()`** — checks all required varieties are named. Retry once.

5. **`_substance_free_doses()`** — drops sentences with bare doses ("100 kg per acre" without naming substance).

6. **`_unrequested_entities()`** — detects narrowing to unasked variety. Retry once.

### Deterministic Schedule Override

`_deterministic_schedule()` (line 4181):
- Fires for any query with a parseable area AND fertilizer rates in context
- Parses rates via `_context_rates()` regex from context text
- Maps to 4 canonical stages: basal, tillering, panicle initiation, flowering
- Skips bulk amendments: lime, compost, FYM, azolla, vermicompost, BGA, gypsum
- Scales quantities to farmer's area
- **Bypasses LLM entirely** — computed answer, no hallucination possible

---

## 7. Analysis of 12 Client-Reported Issues

### Issue Classification

| # | Query Summary | Problem Type | Root Cause Area | Severity |
|---|---|---|---|---|
| 1 | FYM per acre | Wrong answer (fertilizer instead of FYM) | Retrieval: no FYM chunks in vector store | High |
| 2 | Seed rate 2.5 acres | Wrong answer (fertilizer instead of seed) | Multi-stage: regex miss + dose verifier kill | Critical |
| 3 | Seedling transplant age | Bad translation ("boil" instead of "transplant") | Translation: glossary gap for ତଳି | High |
| 4 | Seedling transplant age | Bad translation ("money buried") | Translation: ଧନ vs ଧାନ (1-char difference) | High |
| 5 | Urea+DAP+MOP schedule | Hallucinated "after weeding" stage | Generation: LLM invented stage not in docs | Medium |
| 6 | Generic "ସାର" query | "sara" not translated | Translation: ସାର not mapped in glossary | High |
| 7 | Azolla per acre | Returns fertilizer schedule | Retrieval: no azolla chunks; generation: no topic guard | High |
| 8 | Generic fertilizer per acre | "Could not verify" | Validation: dose verifier kills answer; deterministic schedule fails (incomplete context) | High |
| 9 | FYM per acre (=Issue #1) | Same as #1 | Same as #1 | High |
| 10 | FYM per acre (variant) | Returns fertilizer schedule | Same as #1 | High |
| 11 | Swarna urea-only | Returns all fertilizers | Generation: single-fertilizer guard may not fire properly | Medium |
| 12 | Half acre fertilizer | Includes azolla, only basal | Retrieval: incomplete context; azolla not filtered by LLM | High |

### Root Cause Distribution

| Root Cause Area | Issues | Count |
|---|---|---|
| **Retrieval (missing content)** | #1, #7, #8, #9, #10, #12 | 6 |
| **Translation (glossary gaps)** | #3, #4, #6 | 3 |
| **Generation (hallucination)** | #5 | 1 |
| **Multi-stage (regex + validation)** | #2 | 1 |
| **Generation (guard not firing)** | #11 | 1 |

### Key Insight

**50% of client issues (6/12) trace back to missing or incomplete content in the vector store.** The LLM, dose verifier, and deterministic schedule all work correctly when given proper context. The primary failure is that relevant evidence either doesn't exist in the knowledge base or isn't retrievable.

---

## 8. Seed-Quantity Failure Trace (Issue #2)

### Query
"How many kilos of seed are needed for 2.5 acres of land?"

### Failure Chain (verified from logs and code)

1. **Translation:** Odia ବିହନ → Google translates to "seeds" (plural). Original seed regex `seed\s*rate` didn't match "seeds".

2. **Seed detection failed:** `_is_seed_query` returned False. Query proceeded as fertilizer query.

3. **Retrieval:** Query rewrites generated fertilizer-oriented variants ("fertilizer dose per hectare rice"). ChromaDB returned fertilizer chunks. **Zero seed rate chunks exist in any collection** (verified by searching 3,447+ chunks across rice-knowledge-base).

4. **FAQ match:** FAQ-92 ("How much seed rate is required for rice?") scored 0.637 — below 0.80 threshold. Added as context chunk, not direct-served. Also: area=2.5 acres (not 1), so direct-serve was blocked even if score was high enough.

5. **Generation:** LLM saw fertilizer context + FAQ seed context. Generated answer mixing seed rate with fertilizer.

6. **Dose verifier:** `_unverified_doses()` checked the LLM's seed calculation (62.5 kg = 25 kg/acre × 2.5) against context. Rejected it because "62.5 kg" appeared as "does not say of WHAT" — seed FAQ text format wasn't parseable.

7. **Fallback:** Deterministic schedule returned None (no seed rates in its regex patterns). → "Could not verify" or fertilizer answer depending on timing.

### Fixes Applied (commits on test-aug13 branch)

- `ad78bec`: Broadened seed regex to match plural "seeds", "How many", etc.
- `380873b`: Generalized FAQ boost — injects FAQ text directly into topic guard system prompt
- `96f045b`: Skip dose verifier for seed queries (seed rates come from FAQ, not vector store)
- FAQ-93 created on server: seed rate FAQ with proper embedding

### Remaining Risk

The seed FAQ scores 0.637-0.690, never reaching the 0.80 direct-serve threshold. The fix relies on FAQ boost (topic guard injection) which works but is a workaround. The underlying retrieval problem — zero seed rate content in the vector store — is not addressed.

---

## 9. Top Three Highest-Impact Problems

### Problem 1: Missing Content in Vector Store (6/12 issues)

**Evidence:** FYM quantities, azolla application rates, complete stage-wise fertilizer schedules, and seed rates are absent from the vector store. The current knowledge base documents contain this information in tables and structured formats, but extraction/chunking may have lost it, or the source documents simply don't cover these topics.

**Impact:** The pipeline cannot answer questions when the evidence doesn't exist, regardless of how good retrieval, generation, or validation are. FAQ entries are a stopgap, not a solution.

**Investigation needed:** Audit which source documents contain FYM, azolla, seed rate, and complete fertilizer schedule information. Check whether the data was lost during extraction (scanned pages, tables), lost during chunking (split across boundaries), or genuinely absent from source documents.

### Problem 2: Table and Structured Data Loss During Extraction/Chunking

**Evidence:** Fertilizer schedules are typically presented as tables with stage names as row headers and quantities as cell values. pypdf layout mode preserves spacing but produces linear text. The 800-character chunker then splits tables mid-row, separating quantities from stage names and column headers from data.

**Impact:** Even when a document contains a complete fertilizer schedule, the chunk that reaches the LLM may have "5 kg Urea 35 kg DAP 16 kg MOP" without any stage context, or stage names without quantities. `_context_rates()` fails to parse rates from malformed table text.

**Investigation needed:** Sample actual chunks from the vector store for known table-heavy documents. Measure how many chunks contain parseable fertilizer rates with intact stage associations.

### Problem 3: No OCR for Scanned Documents

**Evidence:** pypdf returns empty strings for scanned pages. No OCR library is invoked. Scanned pages contribute zero chunks. This is silent — no error logged, no warning to the user.

**Impact:** Unknown until we audit which documents are scanned vs. digital. If key agricultural manuals (government advisories, district-level recommendations) are scanned PDFs, entire documents may be missing from the knowledge base.

**Investigation needed:** Check each source document for extractable text. Count pages with zero text output. Determine if missing content correlates with client-reported failures.

---

## 10. Additional Verified Weaknesses

### Translation Gaps (3/12 issues)

- **ତଳି** (seedling/transplanting) not in glossary → mistranslated as "boil" or "money"
- **ସାର** (fertilizer) → translated as "sara" (transliteration, not translation)
- **ଧନ vs ଧାନ** — single character difference causes "money" vs "paddy"

These are glossary/synonym additions, not architectural problems.

### No Neural Reranker

All reranking is rule-based (substance regex, RRF fusion, diversification). A cross-encoder reranker could improve precision by re-scoring the top candidates using the full query-chunk pair. This is a potential improvement but not the root cause of current failures.

### FAQ Threshold Too Rigid

The 0.80 cosine threshold for FAQ direct-serve is high. FAQ-92 (seed rate) scores 0.637-0.690 for closely related queries. The decisive-token conflict guard adds another rejection path. Near-miss FAQs become context chunks (above 0.60) but the LLM may not prioritize them over higher-ranked but irrelevant fertilizer chunks.

### Deterministic Schedule Parser Limited

`_context_rates()` uses regex to parse rates from context text. It requires explicit "per acre" or "kg/acre" patterns near each number. FAQ text formatted as "5 kg Urea + 35 kg DAP + 16 kg MOP per acre" only has "per acre" at the end — only the last item gets parsed. This format sensitivity limits the deterministic schedule's effectiveness.

---

## 11. Recommended Next Steps (Prioritized)

### Immediate (no code changes needed)

1. **Content audit:** Check which source documents contain FYM, azolla, seed rate, and complete fertilizer schedules. Determine if the data is in the documents but lost during extraction, or genuinely absent.

2. **Chunk quality sampling:** Extract and inspect 50-100 actual chunks from the vector store. Identify chunks with broken tables, split recommendations, and missing context.

3. **Scanned document audit:** Run pypdf on each source PDF and count pages returning empty text. Identify documents that need OCR.

### Short-term (targeted fixes, after audit confirms root cause)

4. **Glossary additions:** Add ତଳି, ସାର, ଧନ→ଧାନ mappings. Low risk, high impact for 3/12 issues.

5. **FAQ entries:** Add FYM, azolla, and complete fertilizer schedule FAQs with `_context_rates()`-parseable format. Addresses 6/12 issues as stopgap.

6. **FAQ threshold tuning:** Evaluate lowering from 0.80 to 0.70 with the decisive-token guard as the safety net. Requires benchmark on existing FAQ set.

### Medium-term (requires benchmark-driven redesign)

7. **Table-aware chunking:** Detect tables in PDFs (pypdf layout text has recognizable column alignment patterns). Keep table chunks intact with header repetition.

8. **OCR integration:** Add Tesseract or similar for scanned pages. Requires per-document audit first.

9. **Neural reranker:** Evaluate a cross-encoder (e.g., ms-marco-MiniLM) on the top-30 candidates. May improve precision significantly for ambiguous queries.

10. **Unified catalogue evaluation:** Benchmark single-collection retrieval with metadata filtering vs. current multi-catalogue routing. Requires controlled experiment.

### Long-term (Phase 5+ of the audit plan)

11. **Re-indexing pipeline:** Structure-aware chunking + table preservation + OCR + metadata enrichment. Requires the full evaluation dataset and acceptance criteria before production deployment.

---

## 12. Unanswered Questions

1. How many source documents are scanned vs. digital PDFs?
2. Which specific documents contain FYM, azolla, and complete fertilizer schedule data?
3. What is the actual chunk quality distribution — how many chunks have intact tables vs. broken ones?
4. Would a lower FAQ threshold (0.70) increase false positives unacceptably?
5. Does the multi-catalogue architecture measurably hurt retrieval compared to a single collection?
6. What is the current end-to-end response latency breakdown (translation vs. retrieval vs. generation)?

---

*This report contains verified findings from code inspection. Hypotheses are explicitly marked. No code, configuration, or production data was modified during this investigation.*
