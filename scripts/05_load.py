"""Step 5 of ingestion pipeline: embed chunks and upsert to Qdrant."""
import json

from qdrant_client.models import PointStruct

from src.config import COLLECTION, CONTEXTUALIZED_DIR
from src.ingest.loader import build_point
from src.qdrant_setup import ensure_collection, get_qdrant_client

BATCH_SIZE = 32


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
