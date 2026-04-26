"""Parse PDFs into per-page text using PyMuPDF."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf


def parse_pdf(pdf_path: Path) -> dict:
    """Extract text from a PDF, page by page."""
    doc = pymupdf.open(pdf_path)
    pages = [
        {"page_num": page_num, "text": page.get_text()}
        for page_num, page in enumerate(doc, start=1)
    ]
    doc.close()
    return {
        "arxiv_id": pdf_path.stem,
        "num_pages": len(pages),
        "pages": pages,
    }


def save_parsed(result: dict, out_dir: Path) -> Path:
    """Write one parsed PDF dict to out_dir/<arxiv_id>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['arxiv_id']}.json"
    out_path.write_text(json.dumps(result))
    return out_path
