"""
Document text extraction and chunking.

Two extraction paths for PDFs:
  1. LLM Sherpa (structure-aware): preserves headings, tables, lists, sections.
     Requires LLM_SHERPA_API_URL in settings. Falls back to pypdf on failure.
  2. pypdf + RecursiveCharacterTextSplitter (character-based): the original path.

Non-PDF formats (DOCX, XLSX, CSV, TXT) use their own extractors unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("irri")

# ─────────────────────── Text splitter (unchanged) ───────────────────────


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [c.strip() for c in _splitter().split_text(text) if c.strip()]


# ─────────────────────── Sherpa helpers ──────────────────────────────────

_JUNK_RE = [
    re.compile(r"^\d+\.\d+\s*\(Unported\)\.$"),
    re.compile(r"^www\.\S+$"),
]


def _is_junk(text: str) -> bool:
    text = text.strip()
    return any(p.match(text) for p in _JUNK_RE) or len(text) < 5


def _clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_table_row(line: str) -> str:
    """Collapse consecutive empty pipe cells."""
    return re.sub(r"\|(\s*\|)+", "|", line)


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


_OCR_ARTIFACT_RE = re.compile(r"\d\s+[.,]\s*\d|\b\d\s+\d{2}\b")


def _assess_table_quality(table_text: str) -> str:
    """Return 'clean' or 'ocr_artifacts'."""
    if _OCR_ARTIFACT_RE.search(table_text):
        return "ocr_artifacts"
    empty_cells = table_text.count("| |") + table_text.count("||")
    total_cells = table_text.count("|")
    if total_cells > 0 and empty_cells / total_cells > 0.3:
        return "ocr_artifacts"
    return "clean"


def _is_header_row(line: str) -> bool:
    """Heuristic: a pipe-row is a header if most cells are non-numeric text."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if not cells:
        return False
    non_numeric = sum(1 for c in cells if not re.match(r"^[\d.,\-\s%]+$", c))
    return non_numeric / len(cells) > 0.6


def _split_table_chunk(text: str) -> List[tuple]:
    """Split a merged table into logical sub-tables at header boundaries."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    table_start = next(
        (i for i, line in enumerate(lines) if _is_table_row(line)), None
    )
    if table_start is None:
        return [("table", text)]

    context = "\n".join(lines[:table_start])
    table_lines = [
        _clean_table_row(line) for line in lines[table_start:]
        if _is_table_row(line)
    ]
    if not table_lines:
        return [("table", text)]

    # Find boundaries: a header-like row after at least one data row.
    saw_data = False
    boundaries = [0]
    for i, line in enumerate(table_lines):
        if i == 0:
            continue
        if _is_header_row(line) and saw_data:
            boundaries.append(i)
            saw_data = False
        elif not _is_header_row(line):
            saw_data = True

    if len(boundaries) <= 1:
        content = "\n".join(table_lines)
        if context:
            content = f"{context}\n\n{content}"
        return [("table", content)]

    sections = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(table_lines)
        rows = table_lines[start:end]
        content = "\n".join(rows)
        if context:
            content = f"{context}\n\n{content}"
        label = f"table_part_{idx}"
        sections.append((label, content))
    return sections


def _dedup_list_header(texts: List[str], section: str) -> str:
    """Merge list items, removing repeated section headers."""
    deduped = []
    for text in texts:
        for line in text.splitlines():
            if line.strip() == section:
                continue
            deduped.append(line)
    return section + "\n" + "\n".join(deduped)


def _get_chunk_text(chunk) -> str:
    try:
        text = chunk.to_context_text(include_section_info=True)
    except TypeError:
        text = chunk.to_context_text()
    return _clean_text(text)


def _get_page(chunk) -> Optional[int]:
    page_idx = getattr(chunk, "page_idx", -1)
    try:
        page_idx = int(page_idx)
    except (TypeError, ValueError):
        return None
    return page_idx + 1 if page_idx >= 0 else None


def extract_pdf_sherpa(path: Path | str) -> Optional[List[Dict]]:
    """Structure-aware PDF extraction via LLM Sherpa.

    Returns a list of {"text": str, "metadata": dict} dicts, or None on failure.
    Each dict is one structural chunk (paragraph, table section, grouped list).
    """
    api_url = getattr(settings, "LLM_SHERPA_API_URL", "")
    if not api_url:
        return None

    try:
        from llmsherpa.readers import LayoutPDFReader
    except ImportError:
        log.warning("llmsherpa not installed; falling back to pypdf")
        return None

    try:
        reader = LayoutPDFReader(api_url)
        sherpa_doc = reader.read_pdf(str(path))
    except Exception as exc:
        log.warning("Sherpa extraction failed for %s: %s", path, exc)
        return None

    sherpa_chunks = sherpa_doc.chunks()
    extracted: List[Dict] = []

    for idx, chunk in enumerate(sherpa_chunks):
        text = _get_chunk_text(chunk)
        if not text:
            continue

        # Strip section header to check if body is junk.
        lines = text.splitlines()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        if _is_junk(body):
            continue

        tag = str(getattr(chunk, "tag", "unknown")).lower()
        page = _get_page(chunk)
        section = text.splitlines()[0][:250]

        if tag == "table":
            table_sections = _split_table_chunk(text)
            quality = _assess_table_quality(text)
            if quality == "ocr_artifacts":
                log.info("Table chunk %d in %s has OCR artifacts", idx, path)
            for t_idx, (t_type, t_text) in enumerate(table_sections):
                extracted.append({
                    "text": t_text,
                    "metadata": {
                        "content_type": "table",
                        "table_type": t_type,
                        "table_quality": quality,
                        "page": page,
                        "section": section,
                        "sherpa_chunk_id": idx,
                        "table_part": t_idx,
                    },
                })
        elif "list_item" in tag:
            extracted.append({
                "text": text,
                "metadata": {
                    "content_type": "list_item",
                    "page": page,
                    "section": section,
                    "sherpa_chunk_id": idx,
                },
            })
        else:
            extracted.append({
                "text": text,
                "metadata": {
                    "content_type": tag,
                    "page": page,
                    "section": section,
                    "sherpa_chunk_id": idx,
                },
            })

    # Group consecutive list items from the same page/section.
    grouped: List[Dict] = []
    pending_texts: List[str] = []
    pending_meta: Optional[Dict] = None
    pending_key: Optional[tuple] = None

    def _flush_list():
        nonlocal pending_texts, pending_meta, pending_key
        if pending_texts and pending_meta:
            section = pending_meta["section"]
            merged = _dedup_list_header(pending_texts, section)
            grouped.append({
                "text": _clean_text(merged),
                "metadata": {**pending_meta, "content_type": "list"},
            })
        pending_texts = []
        pending_meta = None
        pending_key = None

    for item in extracted:
        if item["metadata"]["content_type"] != "list_item":
            _flush_list()
            grouped.append(item)
            continue
        key = (item["metadata"]["page"], item["metadata"]["section"])
        if pending_key is not None and key != pending_key:
            _flush_list()
        if not pending_texts:
            pending_meta = {k: v for k, v in item["metadata"].items()
                           if k != "sherpa_chunk_id"}
        pending_key = key
        pending_texts.append(item["text"])
    _flush_list()

    # Safety net: split oversized chunks with RecursiveCharacterTextSplitter.
    max_size = settings.CHUNK_SIZE
    final: List[Dict] = []
    for item in grouped:
        if len(item["text"]) <= max_size:
            final.append(item)
        else:
            section_prefix = item["metadata"].get("section", "")
            sub_chunks = chunk_text(item["text"])
            for si, sc in enumerate(sub_chunks):
                if not sc.startswith(section_prefix) and section_prefix:
                    sc = section_prefix + "\n" + sc
                final.append({
                    "text": sc,
                    "metadata": {**item["metadata"], "sub_chunk": si},
                })
    return final


# ─────────────────────── pypdf extraction (unchanged) ────────────────────


def extract_pdf_text(path: Path | str) -> str:
    """Extract text from a PDF file with pypdf. Returns empty string on failure.

    LAYOUT MODE FIRST, PER PAGE, WITH A FALLBACK.

    The default reading-order mode flattens a table into a token stream: the
    varietal sheets' Positioning grid came out as "Companion Sahabhagi Dhan
    None" with the column headers emitted separately and one cell dropped
    altogether, so nothing downstream could tell that Sahabhagi Dhan is the
    COMPANION for IRRIGATED LOWLAND rather than a recommended variety. Layout
    mode keeps the cells in their columns, which is what makes that readable
    again.

    It is not a blanket replacement: `doc/Speciality rice.pdf` returns an empty
    string from layout mode on every one of its 12 pages — no exception, just
    nothing — and a blanket switch made 26,712 characters of it invisible to
    retrieval. So the choice is made PER PAGE: layout when it produced
    something, the default when it did not. A page that yields text either way
    keeps the layout version.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = ""
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                text = ""
            if not text.strip():
                try:
                    text = page.extract_text() or ""
                except Exception:
                    continue
            pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def extract_docx_text(path: Path | str) -> str:
    """Extract text from a .docx Word file. Pulls paragraph text + table cell text."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        parts: list[str] = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def extract_xlsx_text(path: Path | str) -> str:
    """Extract text from a .xlsx Excel file (all sheets, all non-empty rows)."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        for sheet in wb.worksheets:
            parts.append(f"### Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_csv_text(path: Path | str) -> str:
    """Extract text from a .csv file (all rows, all columns)."""
    try:
        import csv as _csv
        parts: list[str] = []
        with open(str(path), "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = _csv.reader(fh)
            for row in reader:
                cells = [c.strip() for c in row if c and c.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_txt_text(path: Path | str) -> str:
    """Read a plain .txt file."""
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except Exception:
        return ""


_EXTRACTORS = {
    ".pdf":  extract_pdf_text,
    ".docx": extract_docx_text,
    ".xlsx": extract_xlsx_text,
    ".csv":  extract_csv_text,
    ".txt":  extract_txt_text,
}


def document_text(document) -> str:
    """
    Return the best available text body for a Document:
      1. Uploaded file (PDF / DOCX / XLSX / CSV / TXT — extracted)
      2. Fallback body built from metadata + keywords
    """
    if document.file and document.file.path:
        path = Path(document.file.path)
        if path.exists():
            extractor = _EXTRACTORS.get(path.suffix.lower())
            if extractor is not None:
                txt = extractor(path)
                if txt:
                    return txt
    return document.text_for_embedding()


# ─────────────────────── Docling (cached, offline) ───────────────────────
#
# Docling parsing runs OUT OF PROCESS, in an isolated virtualenv (see
# docling_ingest.py at the project root), because its dependency set (numpy 2,
# transformers 5, pydantic 2, pandas 3) conflicts with this project's pins.
# That script writes one JSON cache file per PDF, keyed by the PDF's content
# SHA-256. Here we only READ that cache — no docling import, no new runtime
# dependency. match_pdfs copies PDFs byte-for-byte into media/documents/, so the
# content hash (and therefore the cache hit) is identical to the source corpus.


def _file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_cached_chunks(path: Path | str) -> Optional[List[Dict]]:
    """Pre-built structured chunks cached offline, keyed by the file's content hash.

    Covers PDFs pre-converted by Docling (docling_ingest.py) and other sources
    pre-rendered into clean per-entry chunks (variety factsheets, catalogues,
    spreadsheet tables, Q&A sheets). Returns the cached ``[{"text", "metadata"}]``
    list, or None when caching is disabled, no cache dir is configured, or this
    file was never pre-rendered — so the caller cleanly falls back to Sherpa /
    pypdf / generic extraction. Works for any file type (the cache key is the
    content hash, not the extension).
    """
    if not getattr(settings, "USE_DOCLING", False):
        return None
    cache_dir = getattr(settings, "DOCLING_CACHE_DIR", "")
    if not cache_dir:
        return None
    try:
        digest = _file_sha256(path)
    except Exception:
        return None
    cache_file = Path(cache_dir) / f"{digest}.json"
    if not cache_file.exists():
        log.info("Cache miss for %s (%s…)", Path(path).name, digest[:12])
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Cache unreadable for %s: %s", Path(path).name, exc)
        return None
    chunks = data.get("chunks") or []
    return chunks or None


def document_chunks(document) -> List[Dict]:
    """Return structured chunks for a Document.

    Preference order:
      1. Content-hash cache (structure-aware: correct reading order, intact
         tables, heading context, or clean per-entry rows for tabular sources) —
         produced offline by docling_ingest.py and the spreadsheet/factsheet
         renderers. Works for any file type.
      2. LLM Sherpa (structure-aware, remote API) for PDFs with no cache.
      3. pypdf / openpyxl / csv extraction + character splitting as the final
         fallback.

    The fallback returns character-split chunks with minimal metadata, backward
    compatible with the existing pipeline.
    """
    if document.file and document.file.path:
        path = Path(document.file.path)
        if path.exists():
            cached = extract_cached_chunks(path)
            if cached:
                log.info("Cache: %d chunks from %s", len(cached), path.name)
                return cached
            if path.suffix.lower() == ".pdf":
                sherpa_chunks = extract_pdf_sherpa(path)
                if sherpa_chunks:
                    log.info("Sherpa: %d chunks from %s", len(sherpa_chunks), path.name)
                    return sherpa_chunks

    # Fallback: existing extraction + character splitting.
    text = document_text(document)
    chunks = chunk_text(text)
    return [{"text": c, "metadata": {}} for c in chunks]
