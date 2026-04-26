"""Smoke-test the hybrid retrieval against the local Qdrant collection."""
from src.qdrant_setup import get_qdrant_client
from src.retrieval import search

QUESTION = "How does attention mechanism work in transformers?"
TOP_K = 5


def main() -> None:
    client = get_qdrant_client()
    print(f"Question: {QUESTION}\n")

    results = search(client, QUESTION, top_k=TOP_K)
    for i, hit in enumerate(results, start=1):
        payload = hit["payload"]
        text_preview = payload["text"][:200].replace("\n", " ")
        print(f"--- Result {i} (score={hit['score']:.4f}) ---")
        print(f"  arxiv_id: {payload['arxiv_id']}  page {payload['page_num']}")
        print(f"  chunk_id: {payload['chunk_id']}")
        print(f"  text: {text_preview}...\n")


if __name__ == "__main__":
    main()
