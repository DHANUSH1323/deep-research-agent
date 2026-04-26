"""Smoke-test the research subagent against the local corpus."""
from src.agents.subagent import research

SUB_QUESTION = "How does relative positional encoding work in the Music Transformer?"


def main() -> None:
    print(f"Sub-question: {SUB_QUESTION}\n")
    result = research(SUB_QUESTION)
    print(f"Summary: {result.summary}\n")
    print(f"{len(result.findings)} findings:")
    for i, finding in enumerate(result.findings, start=1):
        print(f"\n[{i}] {finding.claim}")
        for c in finding.citations:
            print(f"    - {c.arxiv_id} p{c.page_num} ({c.chunk_id})")


if __name__ == "__main__":
    main()
