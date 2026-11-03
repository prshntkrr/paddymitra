"""
Offline Docling pre-conversion for PaddyMitra RAG.

Runs in an ISOLATED virtualenv that has `docling` installed (NOT the Django app
env). For every PDF in the source folder it:

  1. Converts the PDF with Docling (layout-aware reading order + TableFormer
     tables), auto-enabling OCR only for scanned pages.
  2. Chunks the DoclingDocument with the HybridChunker (token-aware, keeps
     tables intact, prepends heading context).
  3. Writes a cache file keyed by the PDF's content SHA-256:
         <cache>/<sha256>.json  ==  {"source", "chunks": [{"text","metadata"}], ...}

The Django side (catalogues/services/chunking.py :: extract_pdf_docling) hashes
each Document's PDF and loads the matching cache file. Keying by content hash
means it does not matter that match_pdfs copies the PDF into media/documents/ —
the bytes are identical, so the hash (and the cache hit) is the same.

ROBUSTNESS: the batch driver converts each PDF in a SEPARATE child process, so a
native crash or out-of-memory on one pathological PDF (dense vector geometry,
giant scans) is logged as a failure and the batch keeps going instead of dying.
A PDF with no cache file simply falls back to Sherpa/pypdf in the app, so a
failed file is not fatal to the system.

Idempotent: a PDF whose cache file already exists is skipped unless --force.

Usage (from the docling venv):
    D:\Artificial\IRRI\.venv\Scripts\python.exe -I docling_ingest.py \
        --src "D:\Artificial\IRRI\documents\documents" \
        --cache "D:\Artificial\IRRI\PaddyMitra\PaddyMitra\docling_cache"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import traceback

HARD_CHAR_CAP = 2000
MIN_SCANNED_CHARS_PER_PAGE = 100   # below this (avg) a PDF is treated as scanned
DEFAULT_PER_FILE_TIMEOUT = 600     # seconds; a child exceeding this is abandoned


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def looks_scanned(path: str) -> bool:
    """True when the PDF carries almost no embedded text (image-only / scanned)."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        n = len(pdf)
        chars = 0
        for i in range(min(n, 5)):
            tp = pdf[i].get_textpage()
            chars += len(tp.get_text_range())
            tp.close()
        pdf.close()
        return (chars / max(min(n, 5), 1)) < MIN_SCANNED_CHARS_PER_PAGE
    except Exception:
        return False


# ── Docling converters (built lazily, reused within a single process) ───────
_converters: dict = {}
_TABLE_MODE = "accurate"   # accurate | fast | off  (set from --table-mode)


def get_converter(ocr: bool):
    key = (ocr, _TABLE_MODE)
    if key in _converters:
        return _converters[key]
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    if _TABLE_MODE == "off":
        opts.do_table_structure = False
    else:
        opts.do_table_structure = True
        opts.table_structure_options.mode = (
            TableFormerMode.FAST if _TABLE_MODE == "fast" else TableFormerMode.ACCURATE
        )
    opts.do_ocr = ocr
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    _converters[key] = conv
    return conv


_chunker = None


def get_chunker(max_tokens: int):
    global _chunker
    if _chunker is None:
        from docling.chunking import HybridChunker
        _chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    return _chunker


def _content_type(meta) -> str:
    labels = []
    for item in (getattr(meta, "doc_items", None) or []):
        labels.append(str(getattr(item, "label", "")).lower())
    blob = " ".join(labels)
    if "table" in blob:
        return "table"
    if "list" in blob:
        return "list"
    if "picture" in blob or "figure" in blob:
        return "picture"
    if "section_header" in blob or "title" in blob:
        return "section_header"
    return "text"


def _page_of(meta):
    pages = []
    for item in (getattr(meta, "doc_items", None) or []):
        for prov in (getattr(item, "prov", None) or []):
            p = getattr(prov, "page_no", None)
            if isinstance(p, int):
                pages.append(p)
    return min(pages) if pages else None


def _section_of(meta) -> str:
    headings = getattr(meta, "headings", None) or []
    return " > ".join(h for h in headings if h)[:250]


def chunks_for(path: str, max_tokens: int):
    ocr = looks_scanned(path)
    conv = get_converter(ocr)
    result = conv.convert(path)
    doc = result.document
    chunker = get_chunker(max_tokens)

    out = []
    for i, ch in enumerate(chunker.chunk(doc)):
        meta = getattr(ch, "meta", None)
        try:
            text = chunker.contextualize(chunk=ch)
        except Exception:
            text = ch.text
        text = (text or "").strip()
        if not text or len(text) < 5:
            continue
        md = {
            "content_type": _content_type(meta),
            "chunk_index": i,
            "docling": True,
            "ocr": bool(ocr),
            "table_mode": _TABLE_MODE,
            "source_file": os.path.basename(path),
        }
        page = _page_of(meta)
        if page is not None:
            md["page"] = page
        section = _section_of(meta)
        if section:
            md["section"] = section
        if len(text) > HARD_CHAR_CAP:
            for j in range(0, len(text), HARD_CHAR_CAP):
                piece = text[j:j + HARD_CHAR_CAP].strip()
                if piece:
                    out.append({"text": piece, "metadata": {**md, "sub_chunk": j // HARD_CHAR_CAP}})
        else:
            out.append({"text": text, "metadata": md})
    return out, ocr


def _docling_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "?")
    except Exception:
        return "?"


def convert_one(pdf: str, cache_dir: str, max_tokens: int) -> int:
    """Convert a single PDF and write its cache file. Returns process exit code."""
    os.makedirs(cache_dir, exist_ok=True)
    digest = sha256_of(pdf)
    out_path = os.path.join(cache_dir, f"{digest}.json")
    chunks, ocr = chunks_for(pdf, max_tokens)
    payload = {
        "source": os.path.basename(pdf),
        "sha256": digest,
        "ocr": bool(ocr),
        "table_mode": _TABLE_MODE,
        "docling_version": _docling_version(),
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)   # atomic: a half-written cache file never appears
    tag = " [OCR]" if ocr else ""
    print(f"    OK{tag} -> {len(chunks)} chunks", flush=True)
    return 0


def run_batch(src: str, cache: str, max_tokens: int, timeout: int, force: bool) -> int:
    os.makedirs(cache, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(src, "**", "*.pdf"), recursive=True))
    print(f"Found {len(pdfs)} PDFs under {src}")
    print(f"Cache dir: {cache}\n", flush=True)

    done = skipped = failed = 0
    failures = []
    me = os.path.abspath(__file__)
    for idx, pdf in enumerate(pdfs, 1):
        name = os.path.basename(pdf)
        try:
            digest = sha256_of(pdf)
        except Exception as exc:
            failed += 1
            failures.append((name, f"hash: {exc}"))
            print(f"[{idx}/{len(pdfs)}] HASH-FAIL {name}: {exc}", flush=True)
            continue
        out_path = os.path.join(cache, f"{digest}.json")
        if os.path.exists(out_path) and not force:
            skipped += 1
            print(f"[{idx}/{len(pdfs)}] skip (cached)  {name}", flush=True)
            continue

        print(f"[{idx}/{len(pdfs)}] convert  {name}", flush=True)
        cmd = [sys.executable, "-I", me, "--one", pdf,
               "--cache", cache, "--max-tokens", str(max_tokens),
               "--table-mode", _TABLE_MODE]
        try:
            r = subprocess.run(cmd, timeout=timeout)
            if r.returncode == 0:
                done += 1
            else:
                failed += 1
                failures.append((name, f"exit {r.returncode}"))
                print(f"    FAIL (child exit {r.returncode}) {name}", flush=True)
        except subprocess.TimeoutExpired:
            failed += 1
            failures.append((name, f"timeout >{timeout}s"))
            print(f"    TIMEOUT (>{timeout}s) {name}", flush=True)

    print(f"\nDone. converted={done} skipped={skipped} failed={failed} "
          f"total={len(pdfs)}", flush=True)
    if failures:
        print("Failures (these PDFs will fall back to Sherpa/pypdf in the app):")
        for n, why in failures:
            print(f"  - {n}: {why}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Docling offline pre-conversion for PaddyMitra.")
    ap.add_argument("--src", default=r"D:\Artificial\IRRI\documents\documents",
                    help="Folder of source PDFs (searched recursively).")
    ap.add_argument("--cache", default=r"D:\Artificial\IRRI\PaddyMitra\PaddyMitra\docling_cache",
                    help="Output cache folder (one JSON per PDF, keyed by content hash).")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=DEFAULT_PER_FILE_TIMEOUT,
                    help="Per-file conversion timeout in seconds (batch mode).")
    ap.add_argument("--table-mode", choices=["accurate", "fast", "off"], default="accurate",
                    help="TableFormer mode: 'accurate' (default), 'fast' (lower memory; avoids "
                         "heap crashes on dense tables), or 'off' (skip table structure).")
    ap.add_argument("--force", action="store_true", help="Reconvert even if a cache file exists.")
    ap.add_argument("--one", metavar="PDF", default=None,
                    help="Internal: convert exactly ONE pdf (used by the batch driver).")
    args = ap.parse_args()

    global _TABLE_MODE
    _TABLE_MODE = args.table_mode

    if args.one:
        try:
            sys.exit(convert_one(args.one, args.cache, args.max_tokens))
        except Exception as exc:
            print(f"    ERROR {os.path.basename(args.one)}: {exc}", flush=True)
            traceback.print_exc()
            sys.exit(1)

    sys.exit(run_batch(args.src, args.cache, args.max_tokens, args.timeout, args.force))


if __name__ == "__main__":
    main()
