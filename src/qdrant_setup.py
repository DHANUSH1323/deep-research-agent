"""Qdrant client factory and collection schema management."""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from src.config import COLLECTION, DENSE_DIM, QDRANT_API_KEY, QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    """Build and return a configured Qdrant client."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient) -> None:
    """Create the papers collection (dense + sparse) if it doesn't already exist."""
    if client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    print(f"Collection '{COLLECTION}' created")
