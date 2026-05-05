"""Smoke-test the hand-rolled agent loop with a dummy tool."""
from framework.loop import run_agent_loop
from framework.tools import tool


@tool
def search_corpus(query: str, top_k: int = 5) -> str:
    """Search the academic paper corpus and return top matching chunks."""
    return (
        f"Found 3 papers matching '{query}': "
        "(1) Music Transformer (arxiv 1809.04281v3, 2018) — introduces relative "
        "positional encoding with a skewing trick that reduces memory from O(L^2 D) to O(L D); "
        "(2) Attention is All You Need (arxiv 1706.03762, 2017); "
        "(3) Reformer (arxiv 2001.04451, 2020)."
    )


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Search the corpus for 'relative positional encoding' and summarize what you find in 2 sentences.",
        }
    ]

    response = run_agent_loop(
        messages=messages,
        system="You are a research assistant. Use tools when helpful, then summarize findings concisely.",
        max_iterations=5,
    )

    print("=== Final response ===")
    print(response.content[0].text)

    print(f"\n=== Trace ===")
    print(f"Total messages in conversation: {len(messages)}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Usage: {response.usage}")


if __name__ == "__main__":
    main()
