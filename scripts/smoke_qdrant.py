"""Smoke test: verify we can talk to the local Qdrant instance."""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = "smoke_test"


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant at {QDRANT_URL}")

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION}' (4-dim, cosine distance)")

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"title": "apple"}),
            PointStruct(id=2, vector=[0.9, 0.1, 0.0, 0.0], payload={"title": "banana"}),
            PointStruct(id=3, vector=[0.1, 0.2, 0.31, 0.4], payload={"title": "apricot"}),
        ],
    )
    print("Inserted 3 points")

    results = client.query_points(
        collection_name=COLLECTION,
        query=[0.1, 0.2, 0.3, 0.4],
        limit=2,
    ).points
    print("Top 2 nearest neighbors of query [0.1, 0.2, 0.3, 0.4]:")
    for hit in results:
        print(f"  id={hit.id}  score={hit.score:.4f}  payload={hit.payload}")

    client.delete_collection(COLLECTION)
    print(f"Deleted '{COLLECTION}' — smoke test passed.")


if __name__ == "__main__":
    main()
