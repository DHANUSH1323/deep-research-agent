"""Fetch papers from arXiv: download PDFs and append metadata to JSONL."""
from __future__ import annotations

import json

import arxiv

from src.config import METADATA_PATH, PDFS_DIR


def fetch_papers(query: str, max_results: int) -> int:
    """Download PDFs and metadata for papers matching a search query.

    Idempotent: papers already in metadata.jsonl are skipped.
    Returns the number of newly saved papers.
    """
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if METADATA_PATH.exists():
        with METADATA_PATH.open() as f:
            for line in f:
                seen.add(json.loads(line)["arxiv_id"])

    client = arxiv.Client(page_size=25, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    count = 0
    with METADATA_PATH.open("a") as meta_file:
        for result in client.results(search):
            arxiv_id = result.get_short_id()
            title_short = result.title.replace("\n", " ")[:70]

            if arxiv_id in seen:
                print(f"  skip (already have): {arxiv_id}  {title_short}")
                continue

            try:
                result.download_pdf(dirpath=str(PDFS_DIR), filename=f"{arxiv_id}.pdf")
            except Exception as e:
                print(f"  ERROR downloading {arxiv_id}: {e}")
                continue

            metadata = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.isoformat(),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "pdf_path": str(PDFS_DIR / f"{arxiv_id}.pdf"),
            }
            meta_file.write(json.dumps(metadata) + "\n")
            meta_file.flush()
            count += 1
            print(f"  saved ({count}): {arxiv_id}  {title_short}")

    return count


def main() -> None:
    print("Fetching from arXiv...")
    count = fetch_papers(query="transformer attention mechanism", max_results=5)
    print(f"\nDone. Saved {count} new papers.")


if __name__ == "__main__":
    main()
