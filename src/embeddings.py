"""Embedding models — single source of truth for dense + sparse vectors."""
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector

from src.config import DENSE_MODEL_NAME, SPARSE_MODEL_NAME

dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_dense(text: str) -> list[float]:
    """Return the dense embedding of one text as a plain list of floats."""
    return list(dense_model.embed([text]))[0].tolist()


def embed_sparse(text: str) -> SparseVector:
    """Return the sparse BM25 embedding of one text as a Qdrant SparseVector."""
    raw = list(sparse_model.embed([text]))[0]
    return SparseVector(
        indices=raw.indices.tolist(),
        values=raw.values.tolist(),
    )
