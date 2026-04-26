"""Step 4 of ingestion pipeline: add LLM-generated contextual prefixes to chunks."""
import json
import time

from src.config import CHUNKS_DIR, CONTEXTUALIZED_DIR, METADATA_PATH
from src.ingest.contextualizer import contextualize_chunk, load_abstracts
from src.llm.groq_client import get_groq_client

SLEEP_SECONDS = 4


def main() -> None:
    abstracts = load_abstracts(METADATA_PATH)
    client = get_groq_client()

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
                time.sleep(SLEEP_SECONDS)

        print(f"{arxiv_id}: done")


if __name__ == "__main__":
    main()
