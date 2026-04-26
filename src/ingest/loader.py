"""Embed contextualized chunks and upsert them to Qdrant (dense + sparse)."""
from __future__ import annotations

import uuid

from qdrant_client.models import PointStruct

from src.embeddings import embed_dense, embed_sparse


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
