"""Run the full ingestion pipeline: fetch → parse → chunk → contextualize → load."""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
STEPS = [
    "01_fetch.py",
    "02_parse.py",
    "03_chunk.py",
    "04_contextualize.py",
    "05_load.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {step} ===")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / step)], check=True)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
