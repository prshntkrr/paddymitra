"""
glossary_export
===============
Write the current odia_glossary.json out to odia_glossary.xlsx (English | Odia),
so the editable sheet always reflects the live glossary — including terms added
automatically by the self-learning pipeline.

Usage:
    python manage.py glossary_export
    python manage.py glossary_export --file path/to/out.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

GLOSSARY_DIR = Path(settings.BASE_DIR) / "queries" / "services"
JSON_PATH = GLOSSARY_DIR / "odia_glossary.json"
XLSX_PATH = GLOSSARY_DIR / "odia_glossary.xlsx"


class Command(BaseCommand):
    help = "Export odia_glossary.json to an editable odia_glossary.xlsx."

    def add_arguments(self, parser):
        parser.add_argument("--file", help="Output .xlsx path (default: odia_glossary.xlsx)")

    def handle(self, *args, **opts):
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment
        except ImportError:
            raise CommandError("openpyxl is not installed. Run: pip install openpyxl")

        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        out = Path(opts["file"]) if opts.get("file") else XLSX_PATH

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Glossary"
        ws.append(["English term", "Odia translation"])
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(vertical="center")
        for en in sorted(data, key=str.lower):
            ws.append([en, data[en]])
        ws.column_dimensions["A"].width = 38
        ws.column_dimensions["B"].width = 40
        ws.freeze_panes = "A2"
        wb.save(out)

        self.stdout.write(self.style.SUCCESS(f"Exported {len(data)} terms to {out}"))
