"""Embed contextualized chunks and upsert them to Qdrant (dense + sparse)."""
from __future__ import annotations

import json
import uuid

from qdrant_client.models import PointStruct

from src.config import COLLECTION, CONTEXTUALIZED_DIR
from src.embeddings import embed_dense, embed_sparse
from src.qdrant_setup import ensure_collection, get_qdrant_client

BATCH_SIZE = 32


def build_point(chunk: dict) -> PointStruct:
    """Construct a Qdrant point (dense + sparse vectors + payload) from a chunk dict."""
    text_to_embed = chunk["context"] + "\n\n" + chunk["text"]
    return PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
        vector={
            "dense": embed_dense(text_to_embed),
            "bm25": embed_sparse(text_to_embed),
        },
        payload={
            "chunk_id": chunk["chunk_id"],
            "arxiv_id": chunk["arxiv_id"],
            "chunk_index": chunk["chunk_index"],
            "page_num": chunk["page_num"],
            "text": chunk["text"],
            "context": chunk["context"],
        },
    )


def main() -> None:
    client = get_qdrant_client()
    ensure_collection(client)

    batch: list[PointStruct] = []
    total = 0

    for chunk_file in sorted(CONTEXTUALIZED_DIR.glob("*.jsonl")):
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                batch.append(build_point(chunk))
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=COLLECTION, points=batch)
                    total += len(batch)
                    print(f"Upserted {total} points")
                    batch = []

    if batch:
        client.upsert(collection_name=COLLECTION, points=batch)
        total += len(batch)
        print(f"Upserted {total} points (final)")


if __name__ == "__main__":
    main()
