"""Langfuse observability helpers: singleton client + callback handler + flush."""
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse:
    """Initialize and return a langfuse client."""
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


def get_callback_handler() -> CallbackHandler:
    """Get a langfuse CallbackHandler for tracing LLM calls."""
    get_langfuse_client()
    return CallbackHandler()


def flush_traces() -> None:
    """Flush any pending traces to langfuse."""
    get_langfuse_client().flush()
    print("Flushed langfuse traces.")