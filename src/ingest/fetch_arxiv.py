"""Fetch papers from arXiv: download PDFs and append metadata to JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import arxiv


def fetch_papers(
    query: str,
    max_results: int,
    pdf_dir: Path,
    metadata_path: Path,
) -> int:
    """Download PDFs and metadata for papers matching a search query.

    Idempotent: papers already in metadata.jsonl are skipped.
    Returns the number of newly saved papers.
    """
    # Make sure the folders exist. exist_ok=True = don't error if already there.
    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a set of arxiv_ids we already have, so reruns skip them.
    seen: set[str] = set()
    if metadata_path.exists():
        with metadata_path.open() as f:
            for line in f:
                seen.add(json.loads(line)["arxiv_id"])

    # delay_seconds=3 respects arXiv's "1 request per 3 sec" rate limit.
    client = arxiv.Client(page_size=25, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    count = 0
    # "a" = append mode. Past runs' data is preserved.
    with metadata_path.open("a") as meta_file:
        for result in client.results(search):
            arxiv_id = result.get_short_id()
            # Titles can contain newlines — strip them for clean log output.
            title_short = result.title.replace("\n", " ")[:70]

            if arxiv_id in seen:
                print(f"  skip (already have): {arxiv_id}  {title_short}")
                continue

            # Network errors are common. Log and move on — don't crash the batch.
            try:
                result.download_pdf(dirpath=str(pdf_dir), filename=f"{arxiv_id}.pdf")
            except Exception as e:
                print(f"  ERROR downloading {arxiv_id}: {e}")
                continue

            metadata = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                # datetime isn't JSON-serializable — convert to ISO string.
                "published": result.published.isoformat(),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "pdf_path": str(pdf_dir / f"{arxiv_id}.pdf"),
            }
            # One JSON object per line = JSONL format.
            meta_file.write(json.dumps(metadata) + "\n")
            # flush() forces it to disk now, so Ctrl-C won't lose progress.
            meta_file.flush()
            count += 1
            print(f"  saved ({count}): {arxiv_id}  {title_short}")

    return count


def main() -> None:
    # __file__ is src/ingest/fetch_arxiv.py. parents[2] walks up to repo root.
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    print("Fetching from arXiv...")
    count = fetch_papers(
        query="transformer attention mechanism",
        max_results=5,
        pdf_dir=data_dir / "pdfs",
        metadata_path=data_dir / "metadata.jsonl",
    )
    print(f"\nDone. Saved {count} new papers.")


if __name__ == "__main__":
    main()
