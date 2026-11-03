"""
Per-catalogue image vector store.
A single Chroma collection (`image_bank`) holds every image embedding;
queries are filtered by catalogue_id metadata so each catalogue has its
own private visual knowledge.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from django.conf import settings

from .clip_embeddings import embed_image, embed_text_clip

log = logging.getLogger("irri")

COLLECTION_NAME = "image_bank"

_client = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    return _client


def _collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_image(knowledge_image) -> int:
    """Embed one KnowledgeImage and upsert into the global image collection."""
    image_path = knowledge_image.image.path
    emb = embed_image(image_path)
    desc = (
        (knowledge_image.group.description if knowledge_image.group_id else "")
        or knowledge_image.description
        or ""
    ).strip()
    document_text = desc or f"Image: {knowledge_image.original_filename or knowledge_image.prefix}"
    metadata = {
        "ki_id": knowledge_image.pk,
        "group_id": knowledge_image.group_id or 0,
        "catalogue_id": knowledge_image.catalogue_id or 0,
        "prefix": knowledge_image.prefix or (knowledge_image.group.prefix if knowledge_image.group_id else ""),
        "source": knowledge_image.original_filename or "",
        "image_url": knowledge_image.image.url,
        "image_path": image_path,
        "description": desc,
        "has_description": bool(desc),
    }
    _collection().upsert(
        ids=[f"ki:{knowledge_image.pk}"],
        documents=[document_text],
        embeddings=[emb],
        metadatas=[metadata],
    )
    knowledge_image.embedding_dim = len(emb)
    knowledge_image.indexed = True
    knowledge_image.description = desc
    knowledge_image.save(update_fields=["embedding_dim", "indexed", "description", "updated_at"])
    return 1


def remove_image(ki_id: int):
    try:
        _collection().delete(ids=[f"ki:{ki_id}"])
    except Exception as exc:
        log.warning("Failed to delete image %s from vector store: %s", ki_id, exc)


def remove_group(group_id: int, prefix: str = ""):
    """Delete every vector belonging to an ImageGroup, including orphans whose
    `ki_id` no longer exists in the DB. We match by metadata `group_id` AND
    (optionally) by metadata `prefix` so stale entries from before this group
    existed (same prefix, different group_id) are also removed."""
    coll = _collection()
    try:
        coll.delete(where={"group_id": int(group_id)})
    except Exception as exc:
        log.warning("Failed to delete group %s by group_id: %s", group_id, exc)
    if prefix:
        try:
            coll.delete(where={"prefix": prefix})
        except Exception as exc:
            log.warning("Failed to delete group %s by prefix '%s': %s", group_id, prefix, exc)


def purge_catalogue(catalogue_id: int):
    """Delete every vector belonging to a catalogue."""
    try:
        _collection().delete(where={"catalogue_id": int(catalogue_id)})
    except Exception as exc:
        log.warning("Failed to purge catalogue %s vectors: %s", catalogue_id, exc)


def reset_collection():
    try:
        _get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return _collection()


def query_by_image_path(
    image_path: str, top_k: int = 5, catalogue_id: Optional[int] = None,
) -> List[Dict]:
    emb = embed_image(image_path)
    return _query(emb, top_k, catalogue_id)


def query_by_text(
    text: str, top_k: int = 5, catalogue_id: Optional[int] = None,
) -> List[Dict]:
    emb = embed_text_clip(text)
    return _query(emb, top_k, catalogue_id)


def _query(embedding: List[float], top_k: int, catalogue_id: Optional[int]) -> List[Dict]:
    coll = _collection()
    if coll.count() == 0:
        return []
    where = {"catalogue_id": catalogue_id} if catalogue_id else None
    res = coll.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    out: List[Dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "description": doc,
            "metadata": meta or {},
            "distance": dist,
            "score": round(1 - dist, 4) if dist is not None else None,
        })
    return out


def count(catalogue_id: Optional[int] = None) -> int:
    try:
        if catalogue_id:
            return _collection().count(where={"catalogue_id": catalogue_id})
        return _collection().count()
    except Exception:
        return 0
