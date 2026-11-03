"""
Embedding wrapper. Primary: OpenAI text-embedding-3-small.
Fallback (set USE_LOCAL_EMBEDDINGS=1): sentence-transformers MiniLM — fully offline.
"""
from __future__ import annotations

import logging
from typing import Iterable, List

from django.conf import settings

log = logging.getLogger("irri")

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        _local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _local_model


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    texts = [t for t in texts if t]
    if not texts:
        return []

    if settings.USE_LOCAL_EMBEDDINGS or not settings.OPENAI_API_KEY:
        log.info("Embedding %d texts with local MiniLM", len(texts))
        model = _get_local_model()
        return [v.tolist() for v in model.encode(texts, show_progress_bar=False)]

    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=list(texts))
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> List[float]:
    vecs = embed_texts([text])
    return vecs[0] if vecs else []


def embedding_model_name() -> str:
    if settings.USE_LOCAL_EMBEDDINGS or not settings.OPENAI_API_KEY:
        return "sentence-transformers/all-MiniLM-L6-v2"
    return settings.OPENAI_EMBED_MODEL
