"""Add LLM-generated contextual prefixes to chunks (Anthropic-style retrieval)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import CHUNKS_DIR, CONTEXTUALIZED_DIR, METADATA_PATH
from src.llm.client import get_client
from src.llm.prompts import CONTEXTUALIZER_SYSTEM_PROMPT, MODEL


def load_abstracts(metadata_path: Path) -> dict[str, str]:
    """Load all paper abstracts into {arxiv_id: abstract}."""
    result: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result[row["arxiv_id"]] = row["abstract"]
    return result


def contextualize_chunk(client, abstract: str, chunk_text: str) -> str:
    """Call the LLM to produce a 1-2 sentence contextual prefix for one chunk."""
    messages = [
        {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Abstract:\n{abstract}\n\n"
                f"Chunk:\n{chunk_text}\n\n"
                "Write the contextual prefix now:"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    abstracts = load_abstracts(METADATA_PATH)
    client = get_client()

    for chunks_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        out_path = CONTEXTUALIZED_DIR / chunks_path.name
        if out_path.exists():
            print(f"{chunks_path.stem}: already contextualized, skipping")
            continue

        arxiv_id = chunks_path.stem
        abstract = abstracts[arxiv_id]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as out_f, \
             chunks_path.open("r", encoding="utf-8") as in_f:
            for line in in_f:
                chunk = json.loads(line)
                prefix = contextualize_chunk(client, abstract, chunk["text"])
                chunk["context"] = prefix
                out_f.write(json.dumps(chunk) + "\n")
                out_f.flush()
                print(f"  {chunk['chunk_id']}: {prefix[:60]}...")
                time.sleep(4)

        print(f"{arxiv_id}: done")


if __name__ == "__main__":
    main()
