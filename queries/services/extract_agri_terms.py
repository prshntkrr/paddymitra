"""
extract_agri_terms.py
=====================
Run this ONE TIME as a Django management command or standalone script to:

  1. Pull every chunk from your ChromaDB vector store.
  2. Extract all English agricultural terms using GPT.
  3. Ask GPT to provide Odia translations for each term.
  4. Save the result to  queries/services/odia_glossary_draft.json

After running, open the JSON, review/correct translations with someone
who knows Odia, then save it as  queries/services/odia_glossary.json
That file is what odia_glossary.py (and your pipeline) will actually use.

Usage (from your Django project root):
    python manage.py shell < queries/services/extract_agri_terms.py
  OR run directly:
    DJANGO_SETTINGS_MODULE=your_project.settings python queries/services/extract_agri_terms.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import django


# ── Django bootstrap (skip if already running inside manage.py shell) ──────
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.dev")

try:
    django.setup()
except RuntimeError:
    pass  # already set up inside shell

from verify_odia_glossary import verify_glossary

# ── Imports that need Django to be ready ──────────────────────────────────
from django.conf import settings
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────

# Where to write the draft glossary
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "odia_glossary_draft.json"
)

# How many chunks to feed per GPT call (stay under context limit)
CHUNK_BATCH_SIZE = 15

# GPT model to use for extraction + translation
GPT_MODEL = getattr(settings, "OPENAI_CHAT_MODEL", "gpt-4.1-mini")

# ── OpenAI client ─────────────────────────────────────────────────────────

def _client():
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _chat(messages: list[dict]) -> str:
    resp = _client().chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


# ── Step 1: Pull all chunks from ChromaDB ─────────────────────────────────

def fetch_all_chunks() -> list[str]:
    """
    Returns a flat list of all chunk texts from every ChromaDB collection.
    Adjust this function if your vector store API differs.
    """
    try:
        import chromadb
        from django.conf import settings as s

        chroma_path = getattr(s, "CHROMA_PERSIST_DIR", "./chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)

        all_texts: list[str] = []
        for collection in client.list_collections():
            print(f"  → Reading collection: {collection.name}")
            # get() with no filter returns everything; limit per page to avoid OOM
            result = collection.get(include=["documents"])
            docs = result.get("documents") or []
            all_texts.extend(d for d in docs if d and d.strip())

        print(f"  Total chunks fetched: {len(all_texts)}")
        return all_texts

    except Exception as exc:
        print(f"[ERROR] Could not fetch chunks from ChromaDB: {exc}")
        print("  Make sure CHROMA_DB_PATH is set correctly in settings.")
        sys.exit(1)


# ── Step 2: Extract English agricultural terms via GPT ────────────────────

EXTRACTION_SYSTEM = """
You are an expert agricultural linguist.
Given excerpts from rice-farming documents, extract every distinct English
agricultural term, crop name, disease name, technique, or agri-chemical name.

Rules:
- Return ONLY a JSON array of strings, no other text, no markdown.
- Each entry is a lowercase term or short phrase (1-4 words).
- Do NOT include: generic words (the, of, in), numbers, units (kg, ha).
- DO include: fertilizers, pesticides, diseases, crop stages, techniques,
  soil types, crop varieties, irrigation methods, organisms, nutrients.

Example output:
["green manure", "blast disease", "sheath blight", "nitrogen fixation",
 "transplanting", "basal dose", "urea", "dhaincha", "kharif season"]
"""

def extract_terms_from_batch(chunks: list[str]) -> list[str]:
    combined = "\n\n---\n\n".join(chunks)
    raw = _chat([
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": combined},
    ])
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        terms = json.loads(raw)
        return [t.strip().lower() for t in terms if isinstance(t, str) and t.strip()]
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse terms batch response: {raw[:200]}")
        return []


def extract_all_terms(chunks: list[str]) -> list[str]:
    print(f"\n[Step 2] Extracting terms from {len(chunks)} chunks "
          f"in batches of {CHUNK_BATCH_SIZE}...")

    all_terms: set[str] = set()
    batches = [chunks[i:i + CHUNK_BATCH_SIZE]
               for i in range(0, len(chunks), CHUNK_BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)}...", end=" ", flush=True)
        terms = extract_terms_from_batch(batch)
        all_terms.update(terms)
        print(f"got {len(terms)} terms  (running total: {len(all_terms)})")

    sorted_terms = sorted(all_terms)
    print(f"  ✓ Unique terms found: {len(sorted_terms)}")
    return sorted_terms


# ── Step 3: Translate terms to Odia via GPT ───────────────────────────────

TRANSLATION_SYSTEM = """
You are an expert Odia agricultural linguist.
Given a JSON array of English agricultural terms, return a JSON object
mapping each English term to its best Odia translation or transliteration.

Rules:
- Return ONLY a valid JSON object, no markdown, no extra text.
- For terms with a natural Odia word: use the Odia word
  (e.g. "green manure" → "ସବୁଜ ସାର", "rice" → "ଧାନ", "soil" → "ମାଟି").
- For terms with no Odia equivalent: transliterate into Odia script
  (e.g. "nitrogen" → "ନାଇଟ୍ରୋଜେନ", "phosphorus" → "ଫସଫରସ").
- Keep fertilizer abbreviations as-is: DAP, MOP, NPK, Urea, N, P, K.
- If genuinely unsure, write the transliteration in Odia script — never
  leave the value in Latin script.

Example:
{
  "green manure": "ସବୁଜ ସାର",
  "blast disease": "ବ୍ଲାଷ୍ଟ ରୋଗ",
  "transplanting": "ରୁଆ",
  "nitrogen fixation": "ନାଇଟ୍ରୋଜେନ ସ୍ଥିରୀକରଣ",
  "DAP": "DAP"
}
"""

TRANSLATION_BATCH_SIZE = 40  # terms per GPT call


def translate_terms_batch(terms: list[str]) -> dict[str, str]:
    raw = _chat([
        {"role": "system", "content": TRANSLATION_SYSTEM},
        {"role": "user", "content": json.dumps(terms, ensure_ascii=False)},
    ])
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        result = json.loads(raw)
        return {k.lower(): v for k, v in result.items()
                if isinstance(k, str) and isinstance(v, str)}
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse translation batch: {raw[:200]}")
        return {}


def translate_all_terms(terms: list[str]) -> dict[str, str]:
    print(f"\n[Step 3] Translating {len(terms)} terms to Odia "
          f"in batches of {TRANSLATION_BATCH_SIZE}...")

    glossary: dict[str, str] = {}
    batches = [terms[i:i + TRANSLATION_BATCH_SIZE]
               for i in range(0, len(terms), TRANSLATION_BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)}...", end=" ", flush=True)
        translated = translate_terms_batch(batch)
        glossary.update(translated)
        print(f"got {len(translated)} translations")

    # Flag any that came back empty or in Latin script
    import re
    flagged = []
    for term, odia in glossary.items():
        if not odia or re.fullmatch(r"[a-zA-Z0-9 /()%.-]+", odia):
            flagged.append(term)

    if flagged:
        print(f"\n  [WARN] {len(flagged)} terms may still be in Latin script "
              f"— review these in the draft JSON:")
        for t in flagged:
            print(f"    '{t}' → '{glossary[t]}'")

    return glossary


# ── Step 4: Save draft JSON ───────────────────────────────────────────────

# def save_draft(glossary: dict[str, str]) -> None:
#     with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
#         json.dump(glossary, fh, ensure_ascii=False, indent=2, sort_keys=True)
#     print(f"\n[Step 4] Draft glossary saved → {OUTPUT_PATH}")
#     print("  Next steps:")
#     print("  1. Open the file and review/correct translations with an Odia speaker.")
#     print("  2. Save the corrected file as  queries/services/odia_glossary.json")
#     print("  3. Your pipeline will pick it up automatically via odia_glossary.py")

def save_draft(glossary: dict[str, str]) -> None:

    draft_path = os.path.join(
        os.path.dirname(__file__),
        "odia_glossary_draft.json"
    )

    final_path = os.path.join(
        os.path.dirname(__file__),
        "odia_glossary.json"
    )

    with open(
        draft_path,
        "w",
        encoding="utf-8"
    ) as fh:
        json.dump(
            glossary,
            fh,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )

    print(
        f"\nDraft glossary saved -> "
        f"{draft_path}"
    )

    print(
        "\nRunning Gemini verification..."
    )

    verify_glossary(
        draft_path,
        final_path
    )

    print(
        f"\nFinal glossary saved -> "
        f"{final_path}"
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Agricultural Term Extractor + Odia Glossary Builder")
    print("=" * 60)

    # Step 1
    print("\n[Step 1] Fetching all chunks from ChromaDB...")
    chunks = fetch_all_chunks()

    if not chunks:
        print("[ERROR] No chunks found. Is your ChromaDB populated?")
        sys.exit(1)

    # Step 2
    terms = extract_all_terms(chunks)

    if not terms:
        print("[ERROR] No terms extracted.")
        sys.exit(1)

    # Step 3
    glossary = translate_all_terms(terms)

    # Step 4
    save_draft(glossary)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
