"""
Sidecar resolution: given an image filename, find the .txt/.md describing
the visual group it belongs to (mirrors AgriModel/app/image_processor.py).

Convention: many images share a filename prefix (e.g. bacterial_leaf_blight_1.jpg,
bacterial_leaf_blight_2.jpg, ...) and a single sidecar file `<prefix>.txt`
describes them all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_EXT = (".txt", ".md")


def _read(path: Path) -> str:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""


def find_sidecar(image_path: Path, search_dirs=None) -> Tuple[str, str]:
    """
    Returns (prefix, description). prefix is the matched sidecar stem
    (e.g. 'bacterial_leaf_blight'); description is the file's contents.
    Falls back to (image_stem, '') if no sidecar is found.
    """
    base_lower = image_path.stem.lower()

    # 1) exact same-stem sidecar
    for ext in TEXT_EXT:
        same = image_path.with_suffix(ext)
        text = _read(same)
        if text:
            return image_path.stem, text

    # 2) prefix-based search across the supplied folders + the image's own folder
    dirs = list(search_dirs or [])
    dirs.append(image_path.parent)

    best_prefix = ""
    best_text = ""
    best_len = 0
    seen = set()
    for d in dirs:
        try:
            entries = list(Path(d).glob("*.txt")) + list(Path(d).glob("*.md"))
        except Exception:
            continue
        for f in entries:
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            stem_lower = f.stem.lower()
            if not stem_lower:
                continue
            if (
                base_lower == stem_lower
                or base_lower.startswith(stem_lower + "_")
                or base_lower.startswith(stem_lower + "-")
                or stem_lower in base_lower.split("_")
                or stem_lower in base_lower.split("-")
            ):
                if len(stem_lower) > best_len:
                    best_prefix = f.stem
                    best_text = _read(f)
                    best_len = len(stem_lower)

    if best_prefix:
        return best_prefix, best_text
    return image_path.stem, ""
