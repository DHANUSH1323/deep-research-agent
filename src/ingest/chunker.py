"""Chunk parsed papers into ~500-token pieces for embedding."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_DIR, PARSED_DIR


def chunk_parsed_paper(parsed: dict) -> list[dict]:
    """Split a parsed paper into per-page chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[dict] = []
    for page in parsed["pages"]:
        page_number = page["page_num"]
        for i, chunk_text in enumerate(splitter.split_text(page["text"])):
            chunks.append(
                {
                    "chunk_id": f"{parsed['arxiv_id']}::p{page_number}::c{i}",
                    "arxiv_id": parsed["arxiv_id"],
                    "page_num": page_number,
                    "chunk_index": i,
                    "text": chunk_text,
                }
            )
    return chunks


def save_chunks(chunks: list[dict], out_path: Path) -> None:
    """Write a list of chunk dicts as JSONL to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


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
