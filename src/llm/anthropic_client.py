"""Anthropic client factories: raw SDK + LangChain ChatAnthropic wrapper."""
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

from src.config import ANTHROPIC_API_KEY


def get_anthropic_client() -> Anthropic:
    """Build a raw Anthropic SDK client (for direct API access when needed)."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def get_chat_anthropic(model: str, max_tokens: int = 2048) -> ChatAnthropic:
    """Build a LangChain ChatAnthropic client used in LangGraph nodes.

    Supports all Anthropic features (prompt caching, citations, thinking)
    via standard LangChain interfaces.
    """
    return ChatAnthropic(
        model=model,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=max_tokens,
    )
