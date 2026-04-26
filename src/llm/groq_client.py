"""Groq client factory."""
from groq import Groq

from src.config import GROQ_API_KEY


def get_groq_client() -> Groq:
    """Build and return a configured Groq client."""
    return Groq(api_key=GROQ_API_KEY)
