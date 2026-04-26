from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
import os, json, uuid
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, Modifier, PointStruct, SparseVector
from pathlib import Path

load_dotenv()

dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
COLLECTION = "papers"


def create_collection(client) -> None:
    if client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}
        )
        print(f"Collection '{COLLECTION}' created")

if __name__ == "__main__":
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    create_collection(client)
    project_root = Path(__file__).resolve().parents[2]
    context_chunk = project_root / "data" / "contextualized_chunks"
    batch = []
    total = 0
    for chunk_file in sorted(context_chunk.glob("*.jsonl")):
        with chunk_file.open("r", encoding="utf-8") as f:
            for chunk in f:
                chunk = json.loads(chunk)
                text_to_embed = chunk["context"] + "\n\n" + chunk["text"]
                dense_vectors = list(dense_model.embed([text_to_embed]))[0]
                sparse_vectors = list(sparse_model.embed([text_to_embed]))[0]
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
                point = PointStruct(
                    id = point_id,
                    vector={
                        "dense": dense_vectors.tolist(),
                        "bm25": SparseVector(indices=sparse_vectors.indices.tolist(), values=sparse_vectors.values.tolist())
                    },
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "arxiv_id": chunk["arxiv_id"],
                        "chunk_index": chunk["chunk_index"],
                        "page_num": chunk["page_num"],
                        "text": chunk["text"],
                        "context": chunk["context"]
                    }
                )
                batch.append(point)
                if len(batch) >= 32:
                    client.upsert(collection_name=COLLECTION, points=batch)
                    total += len(batch)
                    print(f"Upserted {total} points")
                    batch = []
    if batch:
        client.upsert(collection_name=COLLECTION, points=batch)