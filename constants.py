# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# architecture        llama
# parameters          3.2B
# context length      131072
# embedding length    3072
# quantization        Q4_K_M
LLM_MODEL_NAME = "llama3.2"

CHROMA_COLLECTION_METADATA = {
    "hnsw:space": "cosine"
}  # RAG -> use cosine similarity for vector search
