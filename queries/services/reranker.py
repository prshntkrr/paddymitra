"""Optional cross-encoder reranking stage.

Disabled by default. Enable with USE_CROSS_ENCODER=1 in .env.
Model is loaded once on first use and cached for the process lifetime.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

from django.conf import settings

log = logging.getLogger("irri")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        model_name = getattr(
            settings, "CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        log.info("Loading cross-encoder: %s", model_name)
        _model = CrossEncoder(model_name)
    return _model


def cross_encoder_rerank(
    query: str,
    candidates: List[Tuple],  # [(key, hit_dict), ...]
) -> List:
    """Re-score candidates with the cross-encoder and return keys sorted by score."""
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c[1].get("chunk") or "") for c in candidates]

    t0 = time.perf_counter()
    scores = model.predict(pairs)
    elapsed = time.perf_counter() - t0
    log.info("cross-encoder reranked %d candidates in %.2fs", len(candidates), elapsed)

    scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [item[0][0] for item in scored]  # return keys in new order
