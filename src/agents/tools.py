"""Agent tools — @tool decorated functions usable by LangGraph agents."""
from langchain_core.tools import tool

from src.qdrant_setup import get_qdrant_client
from src.retrieval import search, get_paper_chunks


_qdrant_client = get_qdrant_client()


@tool
def search_corpus(query: str, top_k: int = 5) -> str:
    """Search the academic paper corpus using hybrid retrieval (dense + BM25).

    Returns the top matching chunks with their citations (arxiv_id, page_num,
    chunk_id) and full text. Use this to gather evidence for the research
    sub-question.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return (default 5).
    """
    results = search(_qdrant_client, query, top_k)
    if not results:
        return "No results."

    blocks = []
    for i, hit in enumerate(results, start=1):
        payload = hit["payload"]
        blocks.append(
            f"Result {i} "
            f"[arxiv_id={payload['arxiv_id']}, page={payload['page_num']}, chunk_id={payload['chunk_id']}]:\n"
            f"{payload['text']}"
        )
    return "\n\n".join(blocks)


@tool
def get_paper_full_text(arxiv_id: str) -> str:
    """Fetch all chunks of one specific paper, in reading order.
    
    Use this when you need to deeply analyze one specific paper to summarize
    its contributions, methods, and results.
    
    Args:
        arxiv_id: The paper identifier (e.g., "1809.04281v3").
    """
    results = get_paper_chunks(_qdrant_client, arxiv_id)
    if not results:
        return f"No paper found for arxiv_id: {arxiv_id}."
    
    chunks = []
    for chunk in results:
        chunks.append(
            f"[page_num={chunk['page_num']}, chunk_index={chunk['chunk_index']}, chunk_id={chunk['chunk_id']}]:\n{chunk['text']}"
        )
    return "\n\n".join(chunks)

