from pathlib import Path
import json, time
from src.llm.client import get_client
from src.llm.prompts import MODEL, CONTEXTUALIZER_SYSTEM_PROMPT

def load_abstracts(metadata_path: Path) -> dict[str, str]:
    result = {}
    with metadata_path.open("r", encoding='utf-8') as f:
        for line in f:
            metadata = json.loads(line)
            result[metadata["arxiv_id"]] = metadata["abstract"]
    return result


def contextualize_chunk(client, abstract, chunk_text) -> str:
    messages = [
        {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
        {"role": "user", "content":f"Abstract:\n{abstract}\n\nChunk:\n{chunk_text}\n\nWrite the contextual prefix now:"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    abstracts = load_abstracts(project_root / "data" / "metadata.jsonl")
    chunks_path = project_root /"data"/ "chunks"
    out_path = project_root / "data" / "contextualized_chunks"
    client = get_client()

    for chunk_file in sorted(chunks_path.glob("*.jsonl")):
        out_filepath = out_path / chunk_file.name
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        if out_filepath.exists():
            print(f"Skipping {chunk_file.name} as it has already been contextualized.")
            continue
        arxiv_id = chunk_file.stem
        abstract = abstracts[arxiv_id]
        with out_filepath.open("w", encoding ='utf-8') as out_f, \
            chunk_file.open("r", encoding='utf-8') as in_f:
            for line in in_f:
                chunk = json.loads(line)
                prefix = contextualize_chunk(client, abstract, chunk["text"])
                chunk["context"] = prefix
                out_f.write(json.dumps(chunk) + "\n")
                out_f.flush()
                print(f"  {chunk['chunk_id']}: {prefix[:60]}...")
                time.sleep(4)

        print(f"{arxiv_id}: done")

