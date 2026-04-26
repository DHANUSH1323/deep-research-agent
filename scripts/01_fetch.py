"""Step 1 of ingestion pipeline: fetch papers from arXiv."""
from src.ingest.fetch_arxiv import fetch_papers

QUERY = "transformer attention mechanism"
MAX_RESULTS = 5


def main() -> None:
    print(f"Fetching from arXiv: {QUERY!r} (max {MAX_RESULTS})")
    count = fetch_papers(query=QUERY, max_results=MAX_RESULTS)
    print(f"Done. Saved {count} new papers.")


if __name__ == "__main__":
    main()
