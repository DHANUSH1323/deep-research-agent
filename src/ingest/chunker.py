import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

def chunk_parsed_paper(parsed: dict) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[dict] = []

    for page in parsed["pages"]:
        page_number = page["page_num"]
        page_chunks = text_splitter.split_text(page["text"])

        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"{parsed['arxiv_id']}::p{page_number}::c{i}",
                "arxiv_id": parsed["arxiv_id"],
                "page_num": page_number,
                "chunk_index": i,
                "text": chunk,
            })

    return chunks

def save_chunks(chunks: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parsed_dir = project_root / "data" / "parsed"
    chunks_dir = project_root / "data" / "chunks"

    for parsed_path in sorted(parsed_dir.glob("*.json")):
        out_path = chunks_dir / f"{parsed_path.stem}.jsonl"
        if out_path.exists():
            print(f"{parsed_path.stem}: already chunked, skipping")
            continue

        with parsed_path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        chunks = chunk_parsed_paper(parsed)
        save_chunks(chunks, out_path)
        print(f"{parsed['arxiv_id']}: {parsed['num_pages']} pages -> {len(chunks)} chunks")

if __name__ == "__main__":
    main()