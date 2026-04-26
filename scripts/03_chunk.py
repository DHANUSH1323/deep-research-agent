"""Step 3 of ingestion pipeline: chunk parsed papers."""
import json

from src.config import CHUNKS_DIR, PARSED_DIR
from src.ingest.chunker import chunk_parsed_paper, save_chunks


def main() -> None:
    for parsed_path in sorted(PARSED_DIR.glob("*.json")):
        out_path = CHUNKS_DIR / f"{parsed_path.stem}.jsonl"
        if out_path.exists():
            print(f"{parsed_path.stem}: already chunked, skipping")
            continue

        with parsed_path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        chunks = chunk_parsed_paper(parsed)
        save_chunks(chunks, out_path)
        print(f"{parsed['arxiv_id']}: {parsed['num_pages']} pages -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()
