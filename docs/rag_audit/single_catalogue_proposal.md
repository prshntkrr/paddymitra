# Single-Catalogue Consolidation — Investigation Report & Proposal

**Date:** 2026-10-27
**Status:** AWAITING APPROVAL — no files modified, no indexes changed
**Scope:** Investigation + proposal only

---

## 1. Current Architecture

### Source Files

| Component | File | Key Lines |
|---|---|---|
| Catalogue model | `catalogues/models.py` | M2M to Document (line 33) |
| Collection build | `catalogues/services/vector_store.py` | `build_catalogue_vector_db()` (line 41) |
| Chunking | `catalogues/services/chunking.py` | `chunk_text()` (line 28), `extract_pdf_text()` (line 35) |
| Embeddings | `catalogues/services/embeddings.py` | OpenAI text-embedding-3-small |
| Routing | `hooks/routing.py` | `classify()` (line 248), `plan()` (line 321) |
| Pipeline routing | `queries/services/pipeline.py` | Lines 5687-5995 |
| Multi-catalogue retrieval | `queries/services/pipeline.py` | `retrieve_across_catalogues()` (line 2910) |
| Default catalogue | `hooks/service.py` | `_resolve_catalogue()` (line 85) |
| Bootstrap command | `catalogues/management/commands/bootstrap_catalogue.py` | Creates `rice-knowledge-base` |

### Current Chunking Configuration

| Setting | Value |
|---|---|
| Splitter | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| Chunk size | 800 characters |
| Overlap | 120 characters |
| Separators | `["\n\n", "\n", ". ", " ", ""]` |
| Length function | `len` (character count) |

### Current Embedding Configuration

| Setting | Value |
|---|---|
| Provider | OpenAI |
| Model | `text-embedding-3-small` (from `settings.OPENAI_EMBED_MODEL`) |
| Dimensions | 1536 |
| Fallback | `sentence-transformers/all-MiniLM-L6-v2` (384 dims, when `USE_LOCAL_EMBEDDINGS=True`) |
| Query embedding | Same model as index (via `catalogues/services/embeddings.py`) |

All collections use the same embedding model. No per-collection model config.

### Current Catalogue Inventory

**Collections found in local ChromaDB:**

| Collection | Chunks (local) | Chunks (server, approx) |
|---|---|---|
| rice-knowledge-base | 193 | 3,447 |
| Nutrient_Management | 56 | 2,551 |
| Varietal_Recommendation | 106 | 205 |
| Insect-pest-disease-management | 2 | unknown |
| kb-probe | unknown | unknown |

**Note:** Local DB is a dev subset. Production counts come from server inspection during client issue investigation.

### Document-Catalogue Relationship

- Django M2M: one document CAN belong to multiple catalogues
- Each `build_catalogue_vector_db()` indexes only `catalogue.documents.all()`
- The bootstrap command creates `rice-knowledge-base` with whatever documents are assigned to its M2M
- Documents are NOT automatically added to all catalogues — each catalogue contains only its explicitly assigned documents
- Chunk IDs are `"{doc.id}:{chunk_index}"` — if the same document is in 2 catalogues, it gets the SAME chunk IDs in both collections

### Current Query-Time Catalogue Selection

**Web form:** User picks the catalogue manually.

**WhatsApp:** Multi-step routing:
1. `_resolve_catalogue()` picks a default from `WHATSAPP_DEFAULT_CATALOGUE_ID` env var, or the sole published catalogue
2. `classify()` scores the query against 4 categories: `nutrient`, `variety`, `seed_management`, `pest`
3. Scoring: subject keywords = 2 pts, intent regex = 1 pt, both = +1 bonus
4. Confident threshold = 4.0 (needs all three matches), margin threshold = 2.0
5. Winner category → look up catalogue PK from `WHATSAPP_CATALOGUE_ROUTES` env var
6. No confident winner → fall back to default catalogue
7. TWO confident categories (score ≥ 4.0 each) → multi-catalogue retrieval

**Single vs multi retrieval:**
- Single catalogue: `retrieve()` → ChromaDB vector + BM25, fused via RRF
- Multiple catalogues: `retrieve_across_catalogues()` → each catalogue searched independently, results merged by rank interleaving (not score), deduplicated by content hash

---

## 2. Problems with Current Multi-Catalogue Architecture

### Problem A: Routing misses evidence

A generic "how much fertilizer" query routes to Nutrient_Management. But if variety-specific fertilizer schedules (e.g., Swarna rice) are only in Varietal_Recommendation, that evidence is invisible. Multi-catalogue retrieval only fires when TWO categories both score ≥ 4.0 — a high bar.

**Evidence:** Issue #11 (Swarna urea-only) — the query mentions a variety AND asks about fertilizer, but may route to only one catalogue depending on scoring.

### Problem B: Document overlap is uncontrolled

The same document can exist in multiple catalogues, producing duplicate chunks with identical IDs but in separate collections. `retrieve_across_catalogues()` deduplicates, but single-catalogue queries never see evidence from other catalogues.

### Problem C: Incomplete catalogues

Each topic-specific catalogue contains only its assigned documents. If a document about "pest-resistant varieties" is assigned to Varietal_Recommendation but not Insect-pest-disease-management, pest queries routed to the pest catalogue miss it.

### Problem D: BM25 index is per-collection

The BM25 keyword index is built per collection. A keyword search for "azolla" in Nutrient_Management finds nothing if azolla content is only in rice-knowledge-base.

### Problem E: Maintenance burden

Adding a new document requires deciding which catalogue(s) it belongs to. Wrong assignment = invisible evidence. The bootstrap `rice-knowledge-base` is a catch-all, but if routing sends queries to topic-specific catalogues, the catch-all isn't searched.

---

## 3. Proposed Single-Catalogue Design

### Core Change

**One canonical catalogue** (e.g., `rice-knowledge-base` or a new `paddy-mitra-canonical`) containing ALL published agricultural documents. All queries search this single collection.

### What Changes

1. **Catalogue assignment:** All published documents added to the canonical catalogue's M2M
2. **Vector DB build:** One `build_catalogue_vector_db()` call indexes all documents into one collection
3. **Query routing:** Simplified — all queries use the canonical catalogue. The router still classifies intent (for topic guards in the system prompt), but no longer selects which collection to search
4. **Default catalogue:** `WHATSAPP_DEFAULT_CATALOGUE_ID` points to the canonical catalogue

### What Does NOT Change

- **Chunking algorithm:** Same RecursiveCharacterTextSplitter, same 800/120 settings
- **Embedding model:** Same text-embedding-3-small, same 1536 dims
- **Retrieval method:** Same hybrid vector + BM25 + RRF
- **Answer generation:** Same LLM, same topic guards, same dose verifier
- **Translation pipeline:** Unchanged
- **Existing catalogues:** NOT deleted — kept for reference/rollback
- **Existing collections:** NOT deleted — the canonical collection is built alongside them
- **Document model:** Unchanged
- **Catalogue model:** Unchanged (documents just added to the canonical catalogue's M2M)
- **Web form:** Can still select specific catalogues if desired
- **FAQ system:** Unchanged

### Metadata Preservation

Each chunk retains its existing metadata:
- `document_id`, `document_title`, `chunk_index`
- `category`, `subcategory`, `crop`, `year`, `doc_type`, `content_type`

The `category` and `subcategory` fields already exist per-chunk from the Document model. These can serve as metadata filters if topic-level filtering is ever needed within the unified collection.

### Duplicate Detection

- Chunk IDs are `"{doc.id}:{chunk_index}"` — unique per document
- If a document was in multiple old catalogues, it appears only ONCE in the canonical catalogue (one M2M entry)
- No content-level deduplication is proposed initially — near-duplicate chunks from different documents are a separate concern

### Backward Compatibility

| Component | Impact |
|---|---|
| WhatsApp API | Compatible — `_resolve_catalogue()` returns the canonical catalogue |
| Web form | Compatible — catalogue dropdown still works, canonical is the default |
| Admin UI | Compatible — catalogues still visible, documents still manageable |
| `build_vector_db` command | Compatible — can rebuild the canonical catalogue by slug |
| Existing hooks | Compatible — hooks reference catalogue by FK, not collection name |
| `retrieve_across_catalogues()` | Still works but rarely needed — single catalogue handles most queries |

---

## 4. Migration Plan

### Step 1: Create canonical catalogue (Django admin or management command)

```python
# New management command: create_canonical_catalogue
cat, created = Catalogue.objects.get_or_create(
    slug="paddy-mitra-canonical",
    defaults={"name": "Paddy Mitra Canonical", "status": "Published"}
)
# Add ALL published documents
from documents.models import Document
all_docs = Document.objects.filter(status="Published")
cat.documents.set(all_docs)
```

### Step 2: Build the canonical collection

```bash
python manage.py build_vector_db paddy-mitra-canonical
```

This creates a NEW Chroma collection alongside existing ones. Nothing is deleted.

### Step 3: Update default catalogue

Set `WHATSAPP_DEFAULT_CATALOGUE_ID` env var to the canonical catalogue's PK.

### Step 4: Simplify routing (optional, can defer)

Update `hooks/routing.py` `plan()` to always return the canonical catalogue ID, regardless of category classification. The category classification is still useful for topic guards in the system prompt — it just no longer selects the collection.

### Step 5: Verify

Run retrieval tests for the 12 client-reported queries against the canonical catalogue. Compare results with the current multi-catalogue setup.

### Rollback

If the canonical catalogue performs worse:
1. Set `WHATSAPP_DEFAULT_CATALOGUE_ID` back to the original value
2. Revert routing changes (if any)
3. The old collections and catalogues are untouched

---

## 5. Required Code Changes

### Minimal set (estimated ~20 lines changed + 1 new management command):

| File | Change | Lines |
|---|---|---|
| `catalogues/management/commands/create_canonical.py` | **New file** — management command to create canonical catalogue and assign all published documents | ~30 lines |
| `.env` (server) | Set `WHATSAPP_DEFAULT_CATALOGUE_ID` to canonical catalogue PK | 1 line |
| `hooks/routing.py` `plan()` | **Optional:** Always include canonical catalogue ID in `catalogue_ids` | ~5 lines |

### NOT changed:
- `pipeline.py` — no changes needed; it already handles single-catalogue retrieval
- `vector_store.py` — no changes; `build_catalogue_vector_db()` works as-is
- `chunking.py` — no changes
- `embeddings.py` — no changes
- `faq.py` — no changes

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Larger collection = slower retrieval | ChromaDB HNSW scales well to 10K+ chunks. 6,000 chunks is well within range. BM25 index is in-memory and fast. |
| Diluted topic focus | The current `n_results=10` may need tuning — more diverse content means more candidates needed. Monitor retrieval quality. |
| Loss of topic isolation | Topic guards in the system prompt still constrain answer scope. The LLM doesn't change behavior based on collection name. |
| Duplicate chunks from overlapping documents | Each document appears once in the canonical catalogue. No duplication at the M2M level. |
| Existing catalogue users disrupted | Old catalogues and collections remain. Web form still shows them. Only WhatsApp default changes. |

---

## 7. Baseline Retrieval Results (from client issue investigation)

These are the current production results for the 12 client-reported queries, which would serve as the baseline to compare against after consolidation:

| # | Query | Current Result | Expected Improvement |
|---|---|---|---|
| 1 | FYM per acre | Wrong (fertilizer schedule) | May improve if FYM content exists in any catalogue's documents |
| 2 | Seed rate 2.5 acres | Fixed via code changes | No change expected |
| 3-4 | Seedling transplant age | Bad translation | No change (translation issue) |
| 5 | Urea+DAP+MOP schedule | Hallucinated stage | May improve with more complete context |
| 6 | ସାର (fertilizer) | Untranslated | No change (glossary issue) |
| 7 | Azolla per acre | Wrong (fertilizer) | May improve if azolla content is in any document |
| 8 | Generic fertilizer | "Could not verify" | Should improve — canonical catalogue has all fertilizer docs |
| 11 | Swarna urea-only | Returns all fertilizers | Should improve — variety + nutrient docs in same collection |
| 12 | Half acre fertilizer | Incomplete | Should improve — same as #8 |

---

## 8. Recommended Test Cases

After building the canonical catalogue, test these queries and compare retrieval hits (not just final answers):

1. "How much seed is needed for 2.5 acres of paddy?" — expect seed rate chunk
2. "How much FYM should I apply per acre?" — expect FYM/manure chunk
3. "How much azolla per acre?" — expect azolla chunk (if any document contains it)
4. "How much urea, DAP and MOP per acre?" — expect complete stage-wise schedule
5. "How much fertilizer for half an acre?" — expect fertilizer schedule, scaled
6. "Swarna rice urea dosage per acre" — expect Swarna-specific schedule
7. "Which variety is blast resistant?" — expect variety + pest content in same retrieval

For each: record the top-10 retrieved chunks, their source documents, similarity scores, and whether the expected evidence appears.

---

## APPROVAL GATE

**This report is the Phase 8 deliverable. No files have been modified.**

To proceed, I need your approval on:

1. **Create the canonical catalogue** with all published documents
2. **Build its vector index** (new collection, existing ones untouched)
3. **Update the default catalogue** env var on the server
4. **Optional:** Simplify routing to always use canonical catalogue

The changes are small (~30 lines new code + 1 env var) and fully reversible.

Shall I proceed?
