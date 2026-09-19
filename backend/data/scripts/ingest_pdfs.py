"""
PDF Ingestion Pipeline.
Reads all PDFs from the music_pdfs folder and ingests them into ChromaDB.

Run: python -m data.scripts.ingest_pdfs

Why chunking strategy matters for interview:
  "We use RecursiveCharacterTextSplitter with 1000-char chunks and 150-char overlap.
   The overlap ensures that a concept spanning two chunks is captured in at least one.
   We tag each chunk with its source PDF so the Reviewer Agent can cite it."
"""
import asyncio
import sys
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pypdf import PdfReader
from app.services.vector_store import VectorStoreService

# Path to the user's downloaded PDFs
PDF_DIR = Path(r"C:\Users\USER\Downloads\music_pdfs")

# Map filename prefixes to document types for metadata tagging
DOC_TYPE_MAP = {
    "SC": "counterpoint",
    "SP": "theory_exercises",
    "04": "modern_theory",
    "03": "harmony",
    "02": "part_writing",
    "01": "fundamentals",
    "Fundamentals": "fundamentals",
    "Open": "open_music_theory",
    "Music": "music_theory_comprehensive",
    "pg": "music_history",
}


def get_doc_type(filename: str) -> str:
    """Classify a PDF by its filename prefix."""
    for prefix, doc_type in DOC_TYPE_MAP.items():
        if filename.startswith(prefix):
            return doc_type
    return "music_theory"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  ⚠  Failed to read {pdf_path.name}: {e}")
        return ""


async def ingest_all_pdfs():
    """Main ingestion function — processes all PDFs in the music_pdfs directory."""
    if not PDF_DIR.exists():
        print(f"❌ PDF directory not found: {PDF_DIR}")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"📚 Found {len(pdf_files)} PDFs to ingest\n")

    service = VectorStoreService()
    total_chunks = 0
    failed = []

    for i, pdf_path in enumerate(pdf_files, 1):
        doc_type = get_doc_type(pdf_path.name)
        print(f"[{i:02d}/{len(pdf_files)}] {pdf_path.name} ({doc_type})", end=" ... ")

        text = extract_pdf_text(pdf_path)
        if not text.strip():
            print("EMPTY — skipped")
            continue

        max_retries = 5
        for attempt in range(max_retries):
            try:
                chunks = await service.ingest_text(
                    text=text,
                    source=pdf_path.name,
                    doc_type=doc_type,
                )
                total_chunks += chunks
                print(f"✅ {chunks} chunks")
                await asyncio.sleep(1)  # Small gap
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource exhausted" in error_msg or "Quota exceeded" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"\n      ⏳ Free tier limit (100/min) hit! Pausing 60s for quota reset...", end=" ", flush=True)
                        await asyncio.sleep(60)
                    else:
                        print(f"\n      ❌ Failed after retries: {e}")
                        failed.append(pdf_path.name)
                else:
                    print(f"\n      ❌ Error: {e}")
                    failed.append(pdf_path.name)
                    break

    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Total chunks added: {total_chunks}")
    print(f"   Failed: {len(failed)}")
    stats = service.get_collection_stats()
    print(f"   Total docs in ChromaDB: {stats['document_count']}")
    print(f"   Persist dir: {stats['persist_dir']}")


if __name__ == "__main__":
    asyncio.run(ingest_all_pdfs())
