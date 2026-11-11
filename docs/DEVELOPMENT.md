# Development

## Local setup (without Docker)

```bash
# Python 3.12 and uv (https://docs.astral.sh/uv/)
uv sync --frozen            # creates .venv with CPU-only torch and dev tools
uv run pytest -q            # offline unit tests (~10 s)
uv run ruff check .
```

To run the app against a local stack, start only the backing services and point the app
at them:

```bash
docker compose up -d weaviate llm
# Weaviate/LLM ports are private; for local work publish them with an override, e.g.
#   services: {weaviate: {ports: ["127.0.0.1:8080:8080", "127.0.0.1:50051:50051"]},
#              llm: {ports: ["127.0.0.1:8081:8080"]}}
uv run vector-search ingest --shards 143 --limit 100
uv run streamlit run app.py
```

or simply rebuild and use the containers: `docker compose build app && docker compose up -d`.

## Code map (`src/vector_search`)

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` dataclass from environment variables, validated |
| `dataset.py` | read/validate shards, duplicates & conflicts, shard selection |
| `parsing.py` | raw crawler response → parsed product (replacement for the old `product.py`) |
| `embeddings.py` | lazy CLIPfa encoders, single and batched, fingerprint |
| `storage.py` | image download/cache (`ImageStore`) |
| `database.py` | Weaviate connection, `ProductV1` schema, deterministic UUIDs |
| `ingestion.py` | batched, idempotent ingestion + reports |
| `search.py` | text/image → text/image vector search with compatibility checks |
| `chat.py` | assistant: OpenAI-compatible local LLM or offline grounded mode |
| `evaluation.py` | recall@k / MRR on `dataset/eval` |
| `ui.py` | Streamlit page |
| `cli.py` | `vector-search` command |

## Tests

`tests/` are fully offline: model and database boundaries are faked, a socket guard makes
the UI tests fail if anything tries to connect. Integration was verified against the real
Compose stack with the real models (`vector-search ingest/evaluate/ask`); see
`docs/DATA.md` for the measured numbers.

## Configuration reference

All settings are environment variables (defaults in brackets). Compose sets the
service hostnames; `.env` is for the knobs below.

| Variable | Default | Meaning |
|---|---|---|
| `PARSED_DIR` | `dataset/parsed` | input shards |
| `IMAGE_DIR` | `data/images` | image cache |
| `CACHE_DIR` | `data/models` | Hugging Face model cache |
| `REPORTS_DIR` | `data/reports` | ingest reports |
| `WEAVIATE_HOST` / `WEAVIATE_HTTP_PORT` | `localhost` / `8080` | |
| `WEAVIATE_GRPC_HOST` / `WEAVIATE_GRPC_PORT` | `localhost` / `50051` | |
| `WEAVIATE_SECURE` / `WEAVIATE_API_KEY` | `false` / empty | |
| `WEAVIATE_COLLECTION` | `ProductV1` | |
| `CHAT_BACKEND` | `grounded` (Compose: `llm`) | `llm` or `grounded` |
| `LLM_URL` | `http://localhost:8081/v1` (Compose: `http://llm:8080/v1`) | OpenAI-compatible base URL |
| `LLM_MODEL` | `qwen2.5-1.5b-instruct` | model name sent to the endpoint |
| `LLM_API_KEY` | empty | only for hosted OpenAI-compatible endpoints |
| `TEXT_MODEL` / `TEXT_MODEL_REVISION` | `SajjadAyoubi/clip-fa-text` / pinned SHA | |
| `IMAGE_MODEL` / `IMAGE_MODEL_REVISION` | `SajjadAyoubi/clip-fa-vision` / pinned SHA | |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda[:N]` |
| `MAX_IMAGE_BYTES` | `15728640` | upload/download limit |
| `INGEST_BATCH_SIZE` | `32` | products per embedding batch |
| `DOWNLOAD_WORKERS` | `8` | parallel image downloads |

Compose-only knobs (`.env`): `APP_BIND`, `APP_PORT`, `HTTPS_PORT`, `TLS_DIR`,
`LLM_HF_MODEL`, `LLM_MODEL_NAME`, `LLM_THREADS`, `TORCH_THREADS`, `INGEST_THREADS`,
`INGEST_SHARDS`.

## Conventions

* Typed, `ruff`-clean code; comments explain *why*.
* Nothing connects to a service or downloads a model at import time.
* Never commit `.env`, certificates, or data volumes (`.gitignore`).
* The original scripts live in `legacy/` for reference only; nothing imports them.
