"""Add LLM-generated contextual prefixes to chunks (Anthropic-style retrieval)."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import GROQ_CONTEXTUALIZER_MODEL
from src.llm.prompts import CONTEXTUALIZER_SYSTEM_PROMPT


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
        model=GROQ_CONTEXTUALIZER_MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
