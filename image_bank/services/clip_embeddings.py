"""
CLIP embeddings (multimodal). Mirrors AgriModel/app/embeddings.py.

Uses sentence-transformers `clip-ViT-B-32`. The model file (~600 MB) is
downloaded lazily on first use and cached under HF_HOME.
"""
from __future__ import annotations

import logging
from typing import List

log = logging.getLogger("irri")

_clip_model = None


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer  # lazy
        log.info("Loading CLIP model clip-ViT-B-32 (first call only)...")
        _clip_model = SentenceTransformer("clip-ViT-B-32")
    return _clip_model


def embed_image(image_path: str) -> List[float]:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    vec = _get_clip_model().encode(img, normalize_embeddings=True)
    return vec.tolist()


def embed_text_clip(text: str) -> List[float]:
    vec = _get_clip_model().encode(text, normalize_embeddings=True)
    return vec.tolist()
