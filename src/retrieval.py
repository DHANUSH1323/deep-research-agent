"""Hybrid retrieval over the papers collection (dense + BM25 fused with RRF)."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, Filter, FieldCondition, MatchValue

from src.config import COLLECTION
from src.embeddings import embed_dense, embed_sparse

PREFETCH_LIMIT_MULTIPLIER = 4


def search(
    client: QdrantClient,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """Run hybrid (dense + BM25) search and return the top_k chunks.

    Each result is {"score": float, "payload": dict}.
    """
    dense_vec = embed_dense(question)
    sparse_vec = embed_sparse(question)
    prefetch_limit = top_k * PREFETCH_LIMIT_MULTIPLIER

    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            Prefetch(query=sparse_vec, using="bm25", limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": p.score, "payload": p.payload} for p in response.points]


def get_paper_chunks(client: QdrantClient, arxiv_id: str) -> list[dict]:
    """Get all chunks for a given paper (identified by arxiv_id)."""
    paper_filter = Filter(
        must=[
            FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))
        ]
    )
    scroll = client.scroll(collection_name=COLLECTION, scroll_filter=paper_filter, with_payload=True, limit=500, with_vectors=False)
    payloads = [p.payload for p in scroll[0] if p.payload is not None]
    return sorted(payloads, key=lambda p: (p["page_num"], p["chunk_index"]))
    