"""
Import the PaddyMitra document classification CSV into the database.

Handles:
  - trailing-space header "Category "
  - UTF-8-BOM and Windows-1252 encodings
  - comma / plus / semicolon separated keyword, category, subcategory cells
  - "india" vs "India" casing
  - URL vs filename in File Path
  - multi-line cells (csv module handles with quotes)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify

from accounts.models import User
from documents.models import Category, Document, Subcategory
from keywords.models import Keyword


SPLIT_RE = re.compile(r"[,;+]")


def _split_tags(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in SPLIT_RE.split(raw)]
    return [p for p in parts if p]


def _normalise_country(c: str) -> str:
    c = (c or "").strip()
    return c.title() if c.lower() == "india" else c


def _guess_doc_type(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "Other"
    raw_l = raw.lower()
    for choice, _ in Document.DOC_TYPE_CHOICES:
        if choice.lower() in raw_l:
            return choice
    return "Other"


def _looks_like_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


class Command(BaseCommand):
    help = "Import the PaddyMitra document classification CSV."

    def add_arguments(self, parser):
        parser.add_argument("path", help="Path to CSV file")
        parser.add_argument("--reset", action="store_true",
                            help="Delete existing Documents/Categories/Subcategories first (keeps keywords).")

    def handle(self, *args, **options):
        path = Path(options["path"])
        if not path.exists():
            raise CommandError(f"CSV not found: {path}")

        if options["reset"]:
            self.stdout.write("Resetting Documents, Categories, Subcategories...")
            Document.objects.all().delete()
            Subcategory.objects.all().delete()
            Category.objects.all().delete()

        admin = (
            User.objects.filter(email="admin@irri.local").first()
            or User.objects.filter(is_superuser=True).first()
        )
        if not admin:
            raise CommandError("No admin user found. Run: python manage.py seed_roles_users first.")

        # Try encodings
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = path.read_text(encoding=enc)
                self.stdout.write(f"Read {path.name} with encoding={enc}")
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise CommandError("Could not decode CSV with any known encoding.")

        reader = csv.DictReader(text.splitlines())
        created = 0
        updated = 0
        skipped = 0

        for raw_row in reader:
            # strip whitespace from keys (handles "Category " trailing space)
            row = {(k or "").strip(): (v or "").strip() for k, v in raw_row.items()}
            title = row.get("Title", "")
            if not title:
                skipped += 1
                continue

            # Categories / Subcategories (free-text, comma-separated)
            cat_names = _split_tags(row.get("Category", ""))
            sub_names = _split_tags(row.get("Subcategory", ""))

            cat_obj = None
            if cat_names:
                cat_obj, _ = Category.objects.get_or_create(name_key=cat_names[0].lower(),
                                                            defaults={"name": cat_names[0]})
            sub_obj = None
            if sub_names and cat_obj:
                sub_obj, _ = Subcategory.objects.get_or_create(
                    category=cat_obj,
                    name_key=sub_names[0].lower(),
                    defaults={"name": sub_names[0]},
                )

            # File Path: URL vs filename
            raw_path = row.get("File Path", "")
            source_url = raw_path if _looks_like_url(raw_path) else ""
            file_path = "" if source_url else raw_path

            # Year
            year = None
            date_raw = row.get("Date", "").strip()
            m = re.search(r"(19|20)\d{2}", date_raw)
            if m:
                try:
                    year = int(m.group(0))
                except ValueError:
                    pass

            # Abstract / Summary — header might be "Abstract/Summary"
            summary = row.get("Abstract/Summary") or row.get("Abstract / Summary") or ""

            countries = _normalise_country(row.get("Country(ies)", ""))

            # Upsert Document
            title_key = slugify(title)[:500] or f"doc-{created + updated + 1}"
            doc, is_new = Document.objects.update_or_create(
                title_key=title_key,
                defaults={
                    "title": title,
                    "doc_type": _guess_doc_type(row.get("Type", "")),
                    "content_type": "Research Paper" if "Research" in row.get("Type", "") else "Advisory",
                    "crop": "Rice",  # CSV is rice-focused
                    "summary": summary,
                    "authors": row.get("Author", ""),
                    "organizations": row.get("Organizations", ""),
                    "journal_or_book": row.get("Journal/ Conferences/ Book Name", "")
                                       or row.get("Journal/Conferences/Book Name", ""),
                    "countries": countries,
                    "geography": countries or "All India",
                    "year": year,
                    "source": row.get("Organizations", ""),
                    "source_url": source_url,
                    "file_path": file_path,
                    "category": cat_obj,
                    "subcategory": sub_obj,
                    "status": Document.STATUS_PUBLISHED,
                    "uploaded_by": admin,
                },
            )
            if is_new:
                created += 1
            else:
                updated += 1

            # Keywords — dedupe case-insensitively
            kw_tokens = _split_tags(row.get("Keyword", ""))
            kw_objs = []
            for tok in kw_tokens:
                if len(tok) > 200:
                    tok = tok[:200]
                kw, _ = Keyword.objects.get_or_create(
                    keyword_key=tok.lower(),
                    defaults={
                        "keyword": tok,
                        "parent_category": Keyword.PARENT_OTHER,
                        "region": "All India",
                        "language": "English",
                        "status": Keyword.STATUS_PUBLISHED,
                    },
                )
                kw_objs.append(kw)
            if kw_objs:
                doc.keywords.set(kw_objs)

        self.stdout.write(self.style.SUCCESS(
            f"Import done. created={created}, updated={updated}, skipped={skipped}"
        ))
        self.stdout.write(f"  Categories: {Category.objects.count()}")
        self.stdout.write(f"  Subcategories: {Subcategory.objects.count()}")
        self.stdout.write(f"  Keywords: {Keyword.objects.count()}")
        self.stdout.write(f"  Documents: {Document.objects.count()}")
