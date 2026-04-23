# deep-research-agent

A multi-agent research assistant that answers complex questions by reading thousands of ingested documents, reasoning across them in parallel, and returning a cited written report.

Think of it as a junior analyst who has read the whole library.

## Architecture (planned)

- **Planner** — decomposes a complex question into sub-questions
- **Supervisor** — dispatches sub-questions to parallel research subagents
- **Research subagents** — call tools: `vector_search`, `keyword_search`, `fetch_document_section`, `web_search`
- **Critic** — verifies every claim has a supporting citation
- **Writer** — produces the final report with inline citations

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM | [Groq](https://console.groq.com) (Llama 3.3 70B) |
| Vector DB | [Qdrant](https://qdrant.tech) (local via Docker) |
| Embeddings | TBD — planning to use `BAAI/bge-small-en-v1.5` |
| Corpus | ~2,000 arXiv ML papers |

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-user>/deep-research-agent.git
cd deep-research-agent

# 2. Start Qdrant in Docker
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant

# 3. Create venv and install deps
uv venv
source .venv/bin/activate
uv sync

# 4. Configure env vars
cp .env.example .env
# then edit .env with your real GROQ_API_KEY

# 5. Run the smoke tests
python scripts/smoke_qdrant.py
python scripts/smoke_groq.py
```

Qdrant dashboard: <http://localhost:6333/dashboard>

## Project structure

```
deep-research-agent/
├── scripts/        # one-off utilities (smoke tests, ingestion drivers)
├── src/
│   ├── ingest/     # PDF fetch, chunk, embed, load
│   ├── agents/     # planner, supervisor, subagents, critic, writer
│   ├── tools/      # vector_search, keyword_search, etc.
│   └── eval/       # evaluation harness
├── pyproject.toml
└── uv.lock
```

## Status

Early scaffolding. Smoke tests pass for Qdrant and Groq. Next: ingestion pipeline.
