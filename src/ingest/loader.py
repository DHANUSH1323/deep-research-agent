from fastembed import TextEmbedding, SparseTextEmbedding

dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

if __name__ == "__main__":
    texts = ["This is a test sentence about transformers.", "Vector databases store embeddings."]
    dense_vectors = list(dense_model.embed(texts))
    sparse_vectors = list(sparse_model.embed(texts))
    print("dense_vectors:", dense_vectors, "\n")
    print("sparse_vectors:", sparse_vectors, "\n")
    print("dense_vector shape:", dense_vectors[0].shape, "\n")
    print("sparse_vector index:", sparse_vectors[0].indices, "\n")
    print("sparse_vector value:", sparse_vectors[0].values, "\n")