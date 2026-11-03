"""
Walk a folder of downloaded PDFs and:

  1. Match each PDF to an existing Document by file_path (exact / normalized / fuzzy).
  2. Copy matched PDF into MEDIA_ROOT/documents/ and attach it as Document.file.
  3. Any unmatched (orphan) PDF becomes a NEW Document record (published, crop=Rice)
     so its content ends up in the vector DB too.
  4. Re-build the vector DB of every catalogue the affected documents belong to.

Idempotent: running twice won't duplicate.
"""
from __future__ import annotations

import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify

from accounts.models import User
from catalogues.models import Catalogue
from catalogues.services.vector_store import build_catalogue_vector_db
from documents.models import Category, Document, Subcategory


def _norm(s: str) -> str:
    """Normalise filename for fuzzy compare: lowercase, strip extension, collapse non-alnum."""
    s = Path(s).stem if s else ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


_STOPWORDS = {
    "the", "for", "and", "with", "from", "in", "of", "to", "a", "an",
    "rice", "paddy", "english", "document",
}


def _tokens(s: str) -> set[str]:
    """Significant word tokens (>=4 chars, not stopwords)."""
    words = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def _best_match(csv_path: str, title: str, index: dict[str, Path]) -> Path | None:
    """Return the best-matching Path, combining exact / substring / fuzzy / token-overlap."""
    key_parts = [p for p in (csv_path, title) if p]
    if not key_parts:
        return None

    # pass 1 — exact normalised filename
    for kp in key_parts:
        n = _norm(kp)
        if n and n in index:
            return index[n]

    # pass 2 — substring both ways on normalised keys
    for kp in key_parts:
        n = _norm(kp)
        if not n:
            continue
        for k, p in index.items():
            if n in k or k in n:
                return p

    # pass 3 — combined score (SequenceMatcher ratio + token-overlap)
    query_tokens = _tokens(" ".join(key_parts))
    best = (0.0, None)
    for k, p in index.items():
        ratio = max(SequenceMatcher(a=_norm(kp), b=k).ratio() for kp in key_parts)
        ptokens = _tokens(p.stem.replace("-", " ").replace("_", " "))
        overlap = (
            len(query_tokens & ptokens) / max(len(query_tokens), 1)
            if query_tokens else 0
        )
        score = 0.4 * ratio + 0.6 * overlap
        if score > best[0]:
            best = (score, p)
    return best[1] if best[0] >= 0.45 else None


def _guess_content_type(title: str) -> str:
    t = title.lower()
    if "bulletin" in t or "manual" in t:
        return "SOP / Practice"
    if "variety" in t or re.search(r"(?:mtu|crdhan|bina|ccdn|pooja|pratikshya)", t):
        return "Advisory"
    if "policy" in t or "kharif manual" in t:
        return "Policy"
    if "research" in t:
        return "Research Paper"
    return "Advisory"


def _guess_doc_type(title: str) -> str:
    t = title.lower()
    if "bulletin" in t:
        return "Bulletin"
    if "manual" in t:
        return "Training Manual"
    if "variety" in t or "varietal" in t or "catalogue" in t:
        return "Booklet"
    if ".ppt" in t:
        return "PPT"
    return "Factsheet"


class Command(BaseCommand):
    help = "Match downloaded PDFs to Documents, attach them, and rebuild affected vector DBs."

    def add_arguments(self, parser):
        parser.add_argument("folder", nargs="?", default="doc",
                            help="Folder (relative to project root or absolute) holding the PDFs.")
        parser.add_argument("--no-rebuild", action="store_true",
                            help="Skip vector DB rebuild.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Show matches without copying files or creating records.")

    def handle(self, *args, **options):
        folder = Path(options["folder"])
        if not folder.is_absolute():
            folder = Path(settings.BASE_DIR) / folder
        if not folder.exists():
            raise CommandError(f"Folder not found: {folder}")

        pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        self.stdout.write(f"Found {len(pdfs)} PDF files in {folder}")
        if not pdfs:
            return

        # Build lookup index of normalised-stem -> Path
        pdf_index: dict[str, Path] = {_norm(p.name): p for p in pdfs}

        admin = User.objects.filter(email="admin@irri.local").first() \
                 or User.objects.filter(is_superuser=True).first()

        media_docs = Path(settings.MEDIA_ROOT) / "documents"
        media_docs.mkdir(parents=True, exist_ok=True)

        matched = 0
        skipped = 0
        unmatched_docs = []
        used_pdfs: set[Path] = set()

        # 1. Walk every existing Document, try to find its PDF
        for doc in Document.objects.all():
            if doc.file:
                skipped += 1
                used_pdfs.add(Path(doc.file.path).resolve() if doc.file else None)  # type: ignore
                continue
            hit = _best_match(doc.file_path, doc.title, pdf_index)
            if not hit:
                unmatched_docs.append(doc)
                continue
            self.stdout.write(f"  MATCH  DOC-{doc.pk:04d}: {doc.title[:45]:45s}  ->  {hit.name}")
            if options["dry_run"]:
                used_pdfs.add(hit)
                matched += 1
                continue
            dest = media_docs / hit.name
            if not dest.exists():
                shutil.copy2(hit, dest)
            doc.file.name = f"documents/{hit.name}"
            doc.save()
            used_pdfs.add(hit)
            matched += 1

        # 2. Orphan PDFs -> new Documents
        orphan_pdfs = [p for p in pdfs if p not in used_pdfs]
        created = 0
        for pdf in orphan_pdfs:
            title = pdf.stem.replace("_", " ").replace("-", " ").strip()
            title = re.sub(r"\s+", " ", title).strip().title()
            # Skip "duplicate" names like "CR-Dhan-203 (1)"
            if re.search(r"\(\d+\)$", pdf.stem):
                self.stdout.write(f"  SKIP   (duplicate-looking filename) {pdf.name}")
                continue
            self.stdout.write(f"  NEWDOC {title[:60]}  <-  {pdf.name}")
            if options["dry_run"]:
                created += 1
                continue
            dest = media_docs / pdf.name
            if not dest.exists():
                shutil.copy2(pdf, dest)
            title_key = slugify(title)[:500] or pdf.stem
            doc, new = Document.objects.update_or_create(
                title_key=title_key,
                defaults={
                    "title": title,
                    "doc_type": _guess_doc_type(title),
                    "content_type": _guess_content_type(title),
                    "crop": "Rice",
                    "summary": f"Imported from {pdf.name}",
                    "geography": "Odisha" if "odisha" in title.lower() else "All India",
                    "source": "IRRI",
                    "file_path": pdf.name,
                    "status": Document.STATUS_PUBLISHED,
                    "uploaded_by": admin,
                },
            )
            doc.file.name = f"documents/{pdf.name}"
            doc.save()
            if new:
                created += 1

        # 3. Summary
        self.stdout.write(self.style.SUCCESS(
            f"\nDone: matched={matched}, skipped(already had file)={skipped}, "
            f"new docs from orphans={created}, unmatched docs={len(unmatched_docs)}"
        ))
        if unmatched_docs:
            self.stdout.write("Unmatched documents (no PDF found):")
            for d in unmatched_docs:
                self.stdout.write(f"  - DOC-{d.pk:04d} {d.title[:60]}  [csv file_path='{d.file_path}']")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry run — no files copied, no records saved."))
            return

        # 4. Rebuild affected catalogues' vector DBs
        if options["no_rebuild"]:
            self.stdout.write("Skipping vector DB rebuild (--no-rebuild).")
            return

        # Ensure every published document is included in the bootstrap catalogue
        bootstrap = Catalogue.objects.filter(slug="rice-knowledge-base").first()
        if bootstrap:
            bootstrap.documents.set(Document.objects.filter(status="Published"))
            bootstrap.save()
            self.stdout.write(f"Re-attached {bootstrap.documents.count()} docs to '{bootstrap.name}'.")

        affected = set(Catalogue.objects.filter(vector_db_status="Ready"))
        if bootstrap:
            affected.add(bootstrap)
        for cat in affected:
            self.stdout.write(f"\nRebuilding vector DB: {cat.name} ...")
            stats = build_catalogue_vector_db(cat)
            self.stdout.write(self.style.SUCCESS(
                f"  -> {stats['chunks']} chunks from {stats['documents']} documents."
            ))
