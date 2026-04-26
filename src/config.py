"""Single source of truth for paths, env vars, and constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
CONTEXTUALIZED_DIR = DATA_DIR / "contextualized_chunks"
METADATA_PATH = DATA_DIR / "metadata.jsonl"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = "papers"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384
SPARSE_MODEL_NAME = "Qdrant/bm25"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PLANNER_MODEL = "claude-sonnet-4-6"
CRITIC_MODEL = "claude-sonnet-4-6"
WRITER_MODEL = "claude-sonnet-4-6"
SUBAGENT_MODEL = "claude-haiku-4-5-20251001"

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
