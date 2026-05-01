import boto3
from src.config import S3_BUCKET, PDFS_DIR, PARSED_DIR, CHUNKS_DIR, CONTEXTUALIZED_DIR, METADATA_PATH

s3 = boto3.client("s3")

def sync_paper(arxiv_id: str) -> None:
    upload=[
        (PDFS_DIR / f"{arxiv_id}.pdf", f"papers/{arxiv_id}/raw.pdf"),
        (PARSED_DIR / f"{arxiv_id}.json", f"papers/{arxiv_id}/parsed.json"),
        (CHUNKS_DIR / f"{arxiv_id}.jsonl", f"papers/{arxiv_id}/chunks.jsonl"),
        (CONTEXTUALIZED_DIR / f"{arxiv_id}.jsonl", f"papers/{arxiv_id}/contextualized.jsonl")
    ]

    for local_path, s3_key in upload:
        if local_path.exists():
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
        else:
            print(f"Warning: {local_path} does not exist and will be skipped.")

def sync_metadata() -> None:
    if METADATA_PATH.exists():
        s3.upload_file(str(METADATA_PATH), S3_BUCKET, "metadata/metadata.jsonl")
        print(f"Uploaded {METADATA_PATH} to s3://{S3_BUCKET}/metadata/metadata.jsonl")
    else:
        print(f"Warning: {METADATA_PATH} does not exist and will be skipped.")

def main() -> None:

    arxiv_ids = [f.stem for f in PDFS_DIR.glob("*pdf")]
    for arxiv_id in arxiv_ids:
        sync_paper(arxiv_id)

    sync_metadata()
    print("Done syncing corpus to S3.")

if __name__ == "__main__":
    main()
    