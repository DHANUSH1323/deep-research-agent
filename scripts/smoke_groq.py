"""Smoke test: verify we can call Groq's LLM API."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_key_here":
        raise SystemExit("GROQ_API_KEY is not set in .env")

    client = Groq(api_key=api_key)
    print(f"Calling Groq model: {MODEL}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise technical writer."},
            {"role": "user", "content": "In one sentence, what is a vector database?"},
        ],
    )

    msg = response.choices[0].message.content
    usage = response.usage
    print(f"\nResponse: {msg}")
    print(
        f"\nTokens — prompt: {usage.prompt_tokens}  "
        f"completion: {usage.completion_tokens}  "
        f"total: {usage.total_tokens}"
    )
    print("\nGroq smoke test passed.")


if __name__ == "__main__":
    main()
