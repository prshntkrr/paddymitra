"""Quick test: chunk a document with LLM Sherpa's hosted API."""
import sys
from llmsherpa.readers import LayoutPDFReader

API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("PDF path or URL: ").strip()
    reader = LayoutPDFReader(API_URL)
    doc = reader.read_pdf(path)

    print(f"\n=== {len(doc.chunks())} chunks ===\n")
    for i, chunk in enumerate(doc.chunks()):
        print(f"--- Chunk {i+1} ---")
        print(chunk.to_context_text()[:500])
        print()
        if i >= 4:
            print(f"... ({len(doc.chunks()) - 5} more chunks)")
            break

    print(f"\nSections: {len(doc.sections())}")
    print(f"Tables:   {len(doc.tables())}")

if __name__ == "__main__":
    main()
