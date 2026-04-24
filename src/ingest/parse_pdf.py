"""Parse PDFs into per-page text using PyMuPDF."""
from __future__ import annotations
import pymupdf
from pathlib import Path
import json

def save_parsed(result: dict, out_dir:Path) -> Path:
    """Write one parsed PDF dict to out_dir/<arxiv_id>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['arxiv_id']}.json"
    out_path.write_text(json.dumps(result))
    return out_path

def parse_pdf(pdf_path: Path) -> dict:
    """Extract text from a PDF, page by page.

    Returns a dict with arxiv_id (from filename), num_pages,
    and pages (list of {page_num, text}).
    """
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        pages.append({"page_num": page_num, "text": page.get_text()})
    doc.close()

    return {
        "arxiv_id": pdf_path.stem,
        "num_pages": len(pages),
        "pages": pages,
    }

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pdf_dir = project_root / "data" / "pdfs"
    parsed_dir = project_root / "data" / "parsed"

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        result = parse_pdf(pdf_path)
        out_path = parsed_dir / f"{result['arxiv_id']}.json"
        if out_path.exists():
            print(f"{result['arxiv_id']}: already parsed, skipping")
            continue
        save_parsed(result, parsed_dir)
        first_page_preview = result["pages"][0]["text"][:200].replace("\n", " ")
        print(f"{result['arxiv_id']}: {result['num_pages']} pages")
        print(f"  first 200 chars: {first_page_preview}...")
        print()


if __name__ == "__main__":
    main()