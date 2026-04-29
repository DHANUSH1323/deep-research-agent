"""Smoke-test the paper-summarizer agent against the local corpus."""
from src.agents.paper_summarizer import run_paper_summarizer
from src.agents.schemas import Finding
from src.observability import flush_traces


# Swap this for an arxiv_id actually present in your Qdrant `papers` collection.
# Check data/metadata.jsonl if unsure.
ARXIV_ID = "1809.04281v3"  # Music Transformer


def print_findings(label: str, findings: list[Finding]) -> None:
    print(f"\n=== {label} ({len(findings)}) ===")
    for i, finding in enumerate(findings, start=1):
        print(f"\n[{i}] {finding.claim}")
        for c in finding.citations:
            print(f"    - {c.arxiv_id} p{c.page_num} ({c.chunk_id})")


def main() -> None:
    print(f"Summarizing arxiv_id: {ARXIV_ID}\n")
    result = run_paper_summarizer(ARXIV_ID)

    print(f"Paper: {result.arxiv_id}")
    print(f"\nSummary:\n{result.summary}")

    print_findings("Key contributions", result.key_contributions)
    print_findings("Methodology", result.methodology)
    print_findings("Notable results", result.notable_results)

    flush_traces()


if __name__ == "__main__":
    main()
