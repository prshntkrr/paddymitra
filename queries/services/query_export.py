"""The Query Listing, as an .xlsx.

WHY A SERVICE AND NOT A VIEW
----------------------------
The view's job is to decide WHICH queries (see `views.filtered_queries`, the one
definition the table and this file share). Turning them into a workbook is the
part with the awkward details — Excel's per-cell limits, control characters that
raise on write, a row count that must not be loaded into memory — and none of
that belongs beside a request handler.

MEMORY. A season's queries are tens of thousands of rows, each carrying a full
answer and a pipeline trace. Both halves are streamed: `write_only` workbooks
flush rows to the temp file as they are appended instead of holding a cell grid,
and the queryset is walked with `.iterator()` so the ORM never builds the whole
result list. Peak memory is one chunk of rows, not the export.
"""
from __future__ import annotations

import io
import logging

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE, WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

log = logging.getLogger("irri")

# Excel refuses a cell longer than 32767 characters outright. Answers are well
# under that; a pipeline trace pasted into a cell would not be, so everything
# text-shaped goes through _clean().
_MAX_CELL = 32000

# (header, width, accessor). Kept as data so a new column is one line, and so
# the header row and the value row can never fall out of step.
COLUMNS = [
    ("Query ID",            12, lambda q: q.code),
    ("Date / Time",         18, lambda q: q.timestamp),
    ("District",            16, lambda q: q.district),
    ("Language",            12, lambda q: q.original_query_language),
    ("Channel",             12, lambda q: "WhatsApp" if q.came_from_whatsapp else "Web"),
    ("WhatsApp Number",     16, lambda q: q.whatsapp_number),
    ("Catalogue",           22, lambda q: q.catalogue.name if q.catalogue else ""),
    ("Original Query",      46, lambda q: q.original_query_text),
    ("Query (English)",     46, lambda q: q.translated_query_text),
    ("Canonical Rewrite",   46, lambda q: _trace(q, "canonical_query")),
    ("Canonical Terms",     22, lambda q: ", ".join(_trace(q, "synonyms") or [])),
    ("STT Backend",         14, lambda q: _trace(q, "stt_backend")),
    ("AI Response",         60, lambda q: q.ai_response_text),
    ("AI Response (local)", 60, lambda q: q.ai_response_text_local),
    ("Sources / Documents", 46, lambda q: _sources(q)),
    ("FAQ Match",           40, lambda q: _faq_match(q)),
    ("Category",            16, lambda q: q.category),
    ("Task Type",           20, lambda q: q.task_type),
    ("Problem / Entity",    22, lambda q: q.problem_entity),
    ("Season",              10, lambda q: q.season),
    ("Crop Stage",          14, lambda q: q.crop_stage),
    ("Missing Context",     28, lambda q: q.missing_context),
    ("Status",              12, lambda q: q.status),
    ("Farmer Feedback",     14, lambda q: q.farmer_feedback),
    ("Flagged",             9,  lambda q: "Yes" if q.flagged_for_review else "No"),
    ("Response Time (s)",   16, lambda q: q.response_time_seconds),
]


def _trace(query, key):
    """One key out of the pipeline trace, never raising on a malformed trace."""
    trace = query.pipeline_trace
    return trace.get(key) if isinstance(trace, dict) else None


def _sources(query) -> str:
    """The documents the answer was built from, titles separated by ' | '.

    Prefers the m2m — it is what the detail page shows — and falls back to the
    titles recorded in the trace for a query whose documents were since deleted.
    """
    titles = [d.title for d in query.source_documents.all()]
    if not titles:
        seen = []
        for chunk in (_trace(query, "retrieved_chunks") or []):
            title = (chunk or {}).get("document_title")
            if title and title not in seen:
                seen.append(title)
        titles = seen
    return " | ".join(titles)


def _faq_match(query) -> str:
    """Which FAQ answered this, if one did — the served hit, else the near miss.

    A hit means retrieval never ran; a near miss means the entry went in as one
    more source. Both are worth telling apart in the sheet, so the kind is part
    of the value rather than a second column.
    """
    for key, label in (("faq", "served"), ("faq_context", "context")):
        entry = _trace(query, key)
        if isinstance(entry, dict) and entry.get("id"):
            question = (entry.get("question") or "").strip()
            return (f"FAQ-{entry['id']} ({label}, score "
                    f"{entry.get('score', '')}): {question}").strip()
    return ""


def _clean(value):
    """A value Excel will accept: numbers and dates as they are, text trimmed.

    ILLEGAL_CHARACTERS_RE is not cosmetic — openpyxl raises on a control
    character, and transcripts arrive with them. An export that dies on one row
    out of forty thousand is the failure mode worth spending a regex on.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "isoformat"):          # datetime / date, written natively
        return value
    text = str(value)
    text = ILLEGAL_CHARACTERS_RE.sub("", text)
    return text[:_MAX_CELL]


def queries_workbook(queryset, *, chunk_size: int = 500) -> io.BytesIO:
    """Every query in `queryset`, as an .xlsx in a buffer ready to be sent.

    `queryset` is used exactly as given — ordering and filters included — so the
    file matches the listing it was downloaded from.
    """
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet(title="Queries")

    for index, (_, width, _) in enumerate(COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    # In a write-only workbook the sheet's view settings are flushed with the
    # first row, so the frozen header has to be asked for before it is written.
    sheet.freeze_panes = "A2"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F7A3F")     # the IRRI green
    header_row = []
    for title, _, _ in COLUMNS:
        cell = WriteOnlyCell(sheet, value=title)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center")
        header_row.append(cell)
    sheet.append(header_row)

    wrap = Alignment(vertical="top", wrap_text=True)
    rows = 0
    # prefetch_related through .iterator() needs an explicit chunk_size (Django
    # >= 4.1); without it the source-document titles would be one extra query
    # per row.
    for query in (queryset.prefetch_related("source_documents")
                          .iterator(chunk_size=chunk_size)):
        row = []
        for _, _, accessor in COLUMNS:
            try:
                value = accessor(query)
            except Exception as exc:                        # noqa: BLE001
                # One unreadable field must not cost the whole export.
                log.warning("query export: field failed for %s (%s)",
                            getattr(query, "pk", "?"), exc)
                value = ""
            value = _clean(value)
            cell = WriteOnlyCell(sheet, value=value)
            cell.alignment = wrap
            # A native datetime sorts and filters in Excel; a string does not.
            if hasattr(value, "isoformat"):
                cell.number_format = "yyyy-mm-dd hh:mm"
            row.append(cell)
        sheet.append(row)
        rows += 1

    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()          # write_only workbooks hold a temp file open
    buffer.seek(0)
    log.info("query export: %d row(s), %d column(s)", rows, len(COLUMNS))
    return buffer
