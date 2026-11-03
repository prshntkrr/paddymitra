"""
glossary_import
===============
Read queries/services/odia_glossary.xlsx (columns: English term | Odia translation)
and merge it into odia_glossary.json, then hot-reload the live glossary.

Human edits in the Excel are AUTHORITATIVE: existing terms are overwritten with
the Excel value. Rows whose Odia value still contains Latin letters are skipped
with a warning (they would put English back into the TTS text).

Usage:
    python manage.py glossary_import
    python manage.py glossary_import --file path/to/sheet.xlsx
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

GLOSSARY_DIR = Path(settings.BASE_DIR) / "queries" / "services"
JSON_PATH = GLOSSARY_DIR / "odia_glossary.json"
XLSX_PATH = GLOSSARY_DIR / "odia_glossary.xlsx"


class Command(BaseCommand):
    help = "Merge odia_glossary.xlsx into odia_glossary.json (Excel wins)."

    def add_arguments(self, parser):
        parser.add_argument("--file", help="Path to the .xlsx (default: odia_glossary.xlsx)")

    def handle(self, *args, **opts):
        try:
            import openpyxl
        except ImportError:
            raise CommandError("openpyxl is not installed. Run: pip install openpyxl")

        xlsx = Path(opts["file"]) if opts.get("file") else XLSX_PATH
        if not xlsx.exists():
            raise CommandError(f"Excel file not found: {xlsx}")

        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active

        existing = {}
        if JSON_PATH.exists():
            existing = json.loads(JSON_PATH.read_text(encoding="utf-8"))

        added, updated, skipped = 0, 0, 0
        for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if i == 1:  # header
                continue
            if not row or len(row) < 2:
                continue
            en, od = row[0], row[1]
            en = (str(en).strip().lower() if en is not None else "")
            od = (str(od).strip() if od is not None else "")
            if not en or not od:
                continue
            if re.search(r"[A-Za-z]", od):
                self.stdout.write(self.style.WARNING(f"  skip (Latin in Odia): {en} -> {od}"))
                skipped += 1
                continue
            if en in existing:
                if existing[en] != od:
                    existing[en] = od
                    updated += 1
            else:
                existing[en] = od
                added += 1

        JSON_PATH.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # hot-reload the in-memory glossary so it's live without a restart
        try:
            from queries.services.odia_glossary import reload_glossary
            reload_glossary()
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"reload skipped: {exc}"))

        self.stdout.write(self.style.SUCCESS(
            f"Imported: +{added} new, {updated} updated, {skipped} skipped. "
            f"Total terms: {len(existing)}."))
        self.stdout.write(f"Wrote {JSON_PATH}")
