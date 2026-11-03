"""
ChromaDB persistent vector store. One collection per Catalogue (collection_name = slug).
"""
from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path
from typing import List

from django.conf import settings
from django.utils import timezone

from .chunking import chunk_text, document_chunks, document_text
from .embeddings import embed_query, embed_texts, embedding_model_name

log = logging.getLogger("irri")

_client = None


def get_client():
    global _client
    if _client is None:
        import chromadb
        Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    return _client


def get_or_create_collection(catalogue):
    client = get_client()
    return client.get_or_create_collection(
        name=catalogue.vector_store_collection_name or catalogue.slug,
        metadata={"catalogue_id": catalogue.id, "catalogue_name": catalogue.name},
    )


def build_catalogue_vector_db(catalogue) -> dict:
    """
    Rebuild the Chroma collection for one catalogue.
    Returns stats: {"documents": int, "chunks": int}.
    """
    from catalogues.models import Catalogue

    catalogue.vector_db_status = Catalogue.VDB_BUILDING
    catalogue.vector_db_error = ""
    catalogue.save(update_fields=["vector_db_status", "vector_db_error"])

    client = get_client()
    coll_name = catalogue.vector_store_collection_name or catalogue.slug
    # Reset collection for a clean rebuild
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={"catalogue_id": catalogue.id, "catalogue_name": catalogue.name},
    )

    total_chunks = 0
    learned_corpus: List[str] = []   # chunks fed to glossary auto-learn after build
    docs = list(catalogue.documents.all().prefetch_related("keywords"))
    for doc in docs:
        doc_chunks = document_chunks(doc)
        if not doc_chunks:
            continue
        texts = [c["text"] for c in doc_chunks]
        learned_corpus.extend(texts)
        ids = [f"{doc.id}:{i}" for i in range(len(doc_chunks))]
        base_meta = {
            "document_id": doc.id,
            "document_title": doc.title,
            "category": doc.category.name if doc.category else "",
            "subcategory": doc.subcategory.name if doc.subcategory else "",
            "crop": doc.crop or "",
            "year": doc.year or 0,
            "source_url": doc.source_url or "",
            "doc_type": doc.doc_type or "",
        }
        metadatas = []
        for i, c in enumerate(doc_chunks):
            m = {**base_meta, "chunk_index": i}
            # Merge Sherpa metadata (content_type, page, section, table_quality, etc.)
            for k, v in c.get("metadata", {}).items():
                if v is not None:
                    m[k] = v
            if "content_type" not in m:
                m["content_type"] = doc.content_type or ""
            metadatas.append(m)
        try:
            vecs = embed_texts(texts)
        except Exception as exc:
            log.exception("Embedding failed for doc %s", doc.id)
            catalogue.vector_db_status = Catalogue.VDB_FAILED
            catalogue.vector_db_error = str(exc)
            catalogue.save(update_fields=["vector_db_status", "vector_db_error"])
            raise
        # Chroma caps upserts at ~166 per call; send in safe 150-row batches.
        CHROMA_BATCH = 150
        for start in range(0, len(texts), CHROMA_BATCH):
            end = start + CHROMA_BATCH
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=vecs[start:end],
            )
        total_chunks += len(doc_chunks)

    catalogue.vector_db_status = Catalogue.VDB_READY
    catalogue.vector_db_generated_at = timezone.now()
    catalogue.vector_db_chunks_count = total_chunks
    catalogue.embedding_model = embedding_model_name()
    catalogue.save(update_fields=[
        "vector_db_status", "vector_db_generated_at",
        "vector_db_chunks_count", "embedding_model",
    ])

    # Auto-update the Odia glossary from the freshly indexed text (background
    # thread — only genuinely new terms are translated, so this is cheap on
    # rebuilds). Never let a glossary failure break the vector-DB build.
    if getattr(settings, "GLOSSARY_AUTO_LEARN", True):
        try:
            from queries.services.glossary_learn import learn_from_texts_async
            learn_from_texts_async(learned_corpus, label=f"catalogue:{catalogue.slug}")
        except Exception as exc:
            log.warning("glossary auto-learn trigger failed: %s", exc)

    return {"documents": len(docs), "chunks": total_chunks}


def _hits_from_query_result(docs, metas, dists) -> list:
    """Shape one Chroma per-query result (parallel lists) into hit dicts."""
    hits = []
    for i, doc_chunk in enumerate(docs or []):
        meta = metas[i] if metas and i < len(metas) else {}
        dist = dists[i] if dists and i < len(dists) else None
        hits.append({
            "chunk": doc_chunk,
            "metadata": meta or {},
            "distance": dist,
            "score": round(1 - dist, 4) if dist is not None else None,
        })
    return hits


def search_catalogue(catalogue, query_text: str, n_results: int = 6):
    collection = get_or_create_collection(catalogue)
    vec = embed_query(query_text)
    if not vec:
        return []
    res = collection.query(query_embeddings=[vec], n_results=n_results)
    return _hits_from_query_result(
        (res.get("documents") or [[]])[0],
        (res.get("metadatas") or [[]])[0],
        (res.get("distances") or [[]])[0] if res.get("distances") else [],
    )


def search_catalogue_batch(catalogue, query_texts, n_results: int = 6) -> list:
    """Semantic search for SEVERAL query phrasings at once.

    Embeds all phrasings in ONE embeddings API call and runs a SINGLE Chroma
    query, instead of one round-trip per phrasing. Returns a list of hit-lists,
    one per input query (same shape as search_catalogue()), so callers can merge
    them exactly as before. Falls back to per-query search on any error."""
    texts = [t for t in (query_texts or []) if t and t.strip()]
    if not texts:
        return []
    collection = get_or_create_collection(catalogue)
    try:
        vecs = embed_texts(texts)
        if not vecs:
            return [[] for _ in texts]
        res = collection.query(query_embeddings=vecs, n_results=n_results)
        docs_all = res.get("documents") or []
        metas_all = res.get("metadatas") or []
        dists_all = res.get("distances") or []
        out = []
        for qi in range(len(texts)):
            out.append(_hits_from_query_result(
                docs_all[qi] if qi < len(docs_all) else [],
                metas_all[qi] if qi < len(metas_all) else [],
                dists_all[qi] if qi < len(dists_all) else [],
            ))
        return out
    except Exception as exc:
        log.warning("search_catalogue_batch failed (%s); falling back to per-query", exc)
        return [search_catalogue(catalogue, t, n_results=n_results) for t in texts]


# ───────────────────────── Keyword (BM25) search ──────────────────────────
# A dependency-free BM25 index built from the chunks already stored in Chroma.
# Used alongside the vector search (hybrid retrieval) so exact terms — variety
# codes (7061), "seed rate", "upland" — still match even when the embedding of
# a paraphrased/translated query drifts away from the document wording.

_bm25_cache: dict = {}
_bm25_lock = threading.Lock()
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def _build_bm25(collection) -> dict:
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    corpus_tokens = [_tokenize(d) for d in docs]
    doc_len = [len(t) for t in corpus_tokens]
    N = len(corpus_tokens)
    avgdl = (sum(doc_len) / N) if N else 0.0
    df: dict = {}
    tf: list = []
    for toks in corpus_tokens:
        counts: dict = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        tf.append(counts)
        for term in counts:
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log(1 + (N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return {"ids": ids, "docs": docs, "metas": metas, "tf": tf,
            "doc_len": doc_len, "avgdl": avgdl, "idf": idf, "N": N}


def _get_bm25(collection) -> dict:
    name = collection.name
    try:
        count = collection.count()
    except Exception:
        count = -1
    key = (name, count)            # rebuilds automatically when the DB changes
    with _bm25_lock:
        cached = _bm25_cache.get(name)
        if cached and cached.get("_key") == key:
            return cached
        idx = _build_bm25(collection)
        idx["_key"] = key
        _bm25_cache[name] = idx
        return idx


def chunks_containing(catalogue, needle: str, limit: int = 500) -> list:
    """Every indexed chunk containing `needle`, with ranking bypassed entirely.

    Similarity search answers "what is most like this question". That is the
    wrong question for "name ALL the 150-day varieties": the 9 matching rows sit
    in 7 near-identical table chunks, and whether all 7 clear the cut-off is a
    property of the ranking, not of the data. A question whose correct answer is
    a COMPLETE enumeration cannot be served by a ranked top-N, so this fetches
    by substring and lets the caller decide what matters.
    """
    collection = get_or_create_collection(catalogue)
    try:
        res = collection.get(where_document={"$contains": needle},
                             include=["documents"], limit=limit)
    except Exception as exc:
        log.warning("chunks_containing(%r) failed: %s", needle, exc)
        return []
    return [d for d in (res.get("documents") or []) if d]


def keyword_search(catalogue, query_text: str, n_results: int = 10,
                   k1: float = 1.5, b: float = 0.75):
    """BM25 keyword search over the catalogue's chunks. Returns hits in the same
    shape as search_catalogue()."""
    collection = get_or_create_collection(catalogue)
    idx = _get_bm25(collection)
    if not idx["N"]:
        return []
    q_terms = [t for t in _tokenize(query_text) if t in idx["idf"]]
    if not q_terms:
        return []
    avgdl = idx["avgdl"] or 1.0
    scored = []
    for i in range(idx["N"]):
        tf_i = idx["tf"][i]
        dl = idx["doc_len"][i]
        s = 0.0
        for term in q_terms:
            f = tf_i.get(term, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            s += idx["idf"][term] * (f * (k1 + 1)) / denom
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    hits = []
    for s, i in scored[:n_results]:
        hits.append({
            "chunk": idx["docs"][i],
            "metadata": idx["metas"][i] or {},
            "distance": None,
            "score": round(s, 4),
            "keyword": True,
        })
    return hits
