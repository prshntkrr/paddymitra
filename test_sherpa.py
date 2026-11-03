# from llmsherpa.readers import LayoutPDFReader

# reader = LayoutPDFReader(
#     "http://13.235.198.26:5010/api/parseDocument?renderFormat=all"
# )

# doc = reader.read_pdf("doc\\using-organic-materials-and-manures-english.pdf")

# for table in doc.tables():
#     print(table.to_html())



import argparse
import json
import re
from pathlib import Path

from llmsherpa.readers import LayoutPDFReader
from langchain_core.documents import Document


DEFAULT_API = (
    "http://13.235.198.26:5010/api/parseDocument?renderFormat=all"
)


JUNK_PATTERNS = [
    re.compile(r"^\d+\.\d+\s*\(Unported\)\.$"),
    re.compile(r"^www\.\S+$"),
]


def is_junk(text):
    text = text.strip()
    return any(p.match(text) for p in JUNK_PATTERNS) or len(text) < 5


def clean_table_row(line):
    """Collapse consecutive empty pipe cells: '| | |' -> '|'."""
    return re.sub(r"\|(\s*\|)+", "|", line)


def clean_text(text):
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_chunk_text(chunk):
    """Extract text while retaining Sherpa's section context."""
    try:
        text = chunk.to_context_text(include_section_info=True)
    except TypeError:
        text = chunk.to_context_text()

    return clean_text(text)


def get_page(chunk):
    """Convert Sherpa's zero-based page index to a page number."""
    page_idx = getattr(chunk, "page_idx", -1)

    try:
        page_idx = int(page_idx)
    except (TypeError, ValueError):
        return None

    return page_idx + 1 if page_idx >= 0 else None


def is_table_row(line):
    line = line.strip()
    return line.startswith("|")


def split_table_chunk(text):
    """
    Split Sherpa's merged table into logical table sections.

    Preserve the original extracted rows. Do not silently correct
    OCR errors or assume ambiguous cell values are accurate.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Separate table rows from the section context preceding the table.
    table_start = next(
        (i for i, line in enumerate(lines) if is_table_row(line)),
        None,
    )

    if table_start is None:
        return [("table", text)]

    context = "\n".join(lines[:table_start])
    table_lines = [
        clean_table_row(line) for line in lines[table_start:]
        if is_table_row(line)
    ]

    if not table_lines:
        return [("table", text)]

    def normalized(line):
        return re.sub(
            r"[^a-z0-9]+", " ", line.lower()
        ).strip()

    # Detect transitions between logical tables.
    # Keep the boundary row with the table it introduces.
    boundaries = []

    for i, line in enumerate(table_lines):
        row = normalized(line)

        if "how much do i apply" in row:
            boundaries.append((i, "application_quantities"))

        elif (
            "amount required" in row
            or "nutrient source" in row
        ):
            boundaries.append((i, "nutrient_cost_comparison"))

    # Avoid duplicate boundaries for the same table.
    unique_boundaries = []
    for index, label in boundaries:
        if not unique_boundaries or index != unique_boundaries[-1][0]:
            unique_boundaries.append((index, label))

    if not unique_boundaries:
        content = "\n".join(table_lines)
        if context:
            content = f"{context}\n\n{content}"
        return [("table", content)]

    sections = []
    start = 0
    current_label = "nutrient_content"

    for boundary_index, next_label in unique_boundaries:
        if boundary_index > start:
            rows = table_lines[start:boundary_index]
            content = "\n".join(rows)

            if context:
                content = f"{context}\n\n{content}"

            sections.append((current_label, content))

        start = boundary_index
        current_label = next_label

    # Final table section.
    if start < len(table_lines):
        content = "\n".join(table_lines[start:])

        if context:
            content = f"{context}\n\n{content}"

        sections.append((current_label, content))

    return sections


def group_list_chunks(chunks):
    """
    Group consecutive list items from the same page and section.
    Keep separate groups when the section or page changes.
    """
    grouped = []
    pending = []
    pending_key = None

    def flush():
        nonlocal pending, pending_key

        if pending:
            first = pending[0]
            section_prefix = first["section"]
            deduped_lines = []
            for item in pending:
                for line in item["text"].splitlines():
                    if line.strip() == section_prefix:
                        continue
                    deduped_lines.append(line)
            content = section_prefix + "\n" + "\n".join(deduped_lines)

            grouped.append({
                "text": clean_text(content),
                "page": first["page"],
                "type": "list",
                "section": first["section"],
            })

        pending = []
        pending_key = None

    for item in chunks:
        if item["type"] != "list_item":
            flush()
            grouped.append(item)
            continue

        key = (item["page"], item["section"])

        if pending_key is not None and key != pending_key:
            flush()

        pending_key = key
        pending.append(item)

    flush()
    return grouped


def extract_and_chunk(pdf_path, api_url=DEFAULT_API):
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = LayoutPDFReader(api_url)
    sherpa_doc = reader.read_pdf(str(pdf_path))

    # Sherpa performs the initial document parsing and chunking.
    sherpa_chunks = sherpa_doc.chunks()

    extracted = []

    for index, chunk in enumerate(sherpa_chunks):
        text = get_chunk_text(chunk)

        if not text:
            continue

        # Strip section header to check if the actual content is junk.
        lines = text.splitlines()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        if is_junk(body):
            continue

        tag = str(getattr(chunk, "tag", "unknown")).lower()
        page = get_page(chunk)

        # Preserve section context for later retrieval and debugging.
        section = text.splitlines()[0][:250]

        if tag == "table":
            table_sections = split_table_chunk(text)

            for table_index, (table_type, table_text) in enumerate(
                table_sections
            ):
                extracted.append({
                    "text": table_text,
                    "page": page,
                    "type": "table",
                    "table_type": table_type,
                    "section": section,
                    "sherpa_chunk_id": index,
                    "table_part": table_index,
                })

        elif "list_item" in tag:
            extracted.append({
                "text": text,
                "page": page,
                "type": "list_item",
                "section": section,
                "sherpa_chunk_id": index,
            })

        else:
            extracted.append({
                "text": text,
                "page": page,
                "type": tag,
                "section": section,
                "sherpa_chunk_id": index,
            })

    # Group adjacent list items where possible.
    extracted = group_list_chunks(extracted)

    documents = []

    for index, item in enumerate(extracted):
        content = item["text"]

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": pdf_path.name,
                    "page": item["page"],
                    "chunk_id": index,
                    "content_type": item["type"],
                    "section": item["section"],
                    **(
                        {"table_type": item["table_type"]}
                        if "table_type" in item else {}
                    ),
                    **(
                        {"sherpa_chunk_id": item["sherpa_chunk_id"]}
                        if "sherpa_chunk_id" in item else {}
                    ),
                    **(
                        {"table_part": item["table_part"]}
                        if "table_part" in item else {}
                    ),
                },
            )
        )

    return sherpa_doc, documents


def main():
    parser = argparse.ArgumentParser(
        description="LLM Sherpa PDF extraction and RAG chunking"
    )

    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API,
        help="LLM Sherpa API endpoint",
    )
    parser.add_argument(
        "--output",
        default="sherpa_chunks.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    sherpa_doc, documents = extract_and_chunk(
        args.pdf,
        args.api_url,
    )

    output = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]

    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"PDF: {args.pdf}")
    print(f"Sherpa tables detected: {len(sherpa_doc.tables())}")
    print(f"Final RAG chunks: {len(documents)}")

    for doc_type in sorted({
        doc.metadata["content_type"] for doc in documents
    }):
        count = sum(
            doc.metadata["content_type"] == doc_type
            for doc in documents
        )
        print(f"{doc_type}: {count}")

    print(f"Saved to: {args.output}")

    for index, doc in enumerate(documents, start=1):
        print(f"\n--- Chunk {index} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content)


if __name__ == "__main__":
    main()
