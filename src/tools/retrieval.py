"""Hybrid retrieval over the papers collection (dense + BM25 fused with RRF)."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch

from src.config import COLLECTION
from src.embeddings import embed_dense, embed_sparse
from src.qdrant_setup import get_qdrant_client

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


def main() -> None:
    client = get_qdrant_client()
    question = "How does attention mechanism work in transformers?"
    print(f"Question: {question}\n")

    results = search(client, question, top_k=5)
    for i, hit in enumerate(results, start=1):
        payload = hit["payload"]
        text_preview = payload["text"][:200].replace("\n", " ")
        print(f"--- Result {i} (score={hit['score']:.4f}) ---")
        print(f"  arxiv_id: {payload['arxiv_id']}  page {payload['page_num']}")
        print(f"  chunk_id: {payload['chunk_id']}")
        print(f"  text: {text_preview}...\n")


if __name__ == "__main__":
    main()
