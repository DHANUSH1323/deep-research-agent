"""Smoke-test the orchestrator end-to-end.

Hybrid question that should exercise BOTH dispatch paths:
- search_agent (for the topic concept "relative positional encoding")
- paper_summarizer (for the named paper "Music Transformer")
"""
from src.agents.orchestrator import run_orchestrator
from src.observability import flush_traces


QUESTION = (
    "How does relative positional encoding work in the Music Transformer "
    "(arxiv_id 1809.04281v3), and what are the paper's key contributions?"
)


def main() -> None:
    print(f"Question: {QUESTION}\n")
    result = run_orchestrator(QUESTION)

    print(f"User query echoed: {result.user_query}\n")
    print(f"Executive Summary:\n{result.executive_summary}\n")
    print(f"{len(result.findings)} aggregated findings:")
    for i, finding in enumerate(result.findings, start=1):
        print(f"\n[{i}] {finding.claim}")
        for c in finding.citations:
            print(f"    - {c.arxiv_id} p{c.page_num} ({c.chunk_id})")

    flush_traces()


if __name__ == "__main__":
    main()
