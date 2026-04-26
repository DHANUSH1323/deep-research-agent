"""Step 2 of ingestion pipeline: parse PDFs to per-page text."""
from src.config import PARSED_DIR, PDFS_DIR
from src.ingest.parse_pdf import parse_pdf, save_parsed


def main() -> None:
    for pdf_path in sorted(PDFS_DIR.glob("*.pdf")):
        out_path = PARSED_DIR / f"{pdf_path.stem}.json"
        if out_path.exists():
            print(f"{pdf_path.stem}: already parsed, skipping")
            continue
        result = parse_pdf(pdf_path)
        save_parsed(result, PARSED_DIR)
        print(f"{result['arxiv_id']}: {result['num_pages']} pages")


if __name__ == "__main__":
    main()
