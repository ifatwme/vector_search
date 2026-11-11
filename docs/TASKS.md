# Project task list

Status of the "make it work, organise it, document it, deploy it" effort for
`ifatwme/vector_search`. Ticked items are done and verified on the host
(`Testilonia`), unticked ones are still open. Live: https://65.109.162.75:8443/

## 0. Audit (findings that drive the tasks)

| # | Problem found in the original code (`main` @ 0144b4d) | Severity |
|---|---|---|
| A1 | A Google Gemini API key is hard-coded in `app.py` and is in the public Git history | **critical**: revoke the key in Google AI Studio |
| A2 | Uses `google-generativeai` + `gemini-1.5-flash`; that SDK is deprecated and the model is retired, and it needs a paid/keyed cloud API | high |
| A3 | `search_image_image` queries `text_vector` (copy/paste bug) so image→image search never ran | high |
| A4 | Every Streamlit rerun keeps a module-level Weaviate client; models are loaded per session, not shared | high |
| A5 | Chat duplicates history (manual transcript + SDK chat history), streams a finished string char-by-char, and the sidebar sliders are never used | medium |
| A6 | `llm_answer(prompt)` is called with an undefined `prompt` when the chat box is empty | medium |
| A7 | Images are read from `./product/images/<pid>/…` but only 3 images exist; any other result crashes the UI | high |
| A8 | Import script stores vectors twice (as properties *and* named vectors), has no idempotency (re-runs create duplicates), downloads unbounded/untrusted URLs, hard-codes shards `1..9` | high |
| A9 | Dataset has 27 exact duplicate rows and 1 ID with conflicting versions; nothing detects this | medium |
| A10 | `docker-compose.yml` only runs MinIO (default `minioadmin` creds, public ports) and Weaviate (anonymous access, public ports); the app itself is not containerised | high |
| A11 | No README, no tests, unpinned `requirements.txt`, legacy scripts with play.min.io sample credentials | medium |
| A12 | Flat, unorganised layout: scripts in `weaviate/`, `minio/`, `product/` mixed with data | medium |

## 1. Core code (package `src/vector_search`)
- [x] `config.py` – one validated `Settings` object from environment variables, secrets hidden from `repr`
- [x] `dataset.py` – read all 145 shards, validate rows, collapse exact duplicates, quarantine conflicting IDs
- [x] `parsing.py` – fault-tolerant re-implementation of the original Digikala parser
- [x] `embeddings.py` – lazy CLIPfa text + vision encoders, pinned model revisions, 768-d pooler output
- [x] `embeddings.py` – batched encoding for fast ingestion
- [x] `storage.py` – safe image cache (trusted host only, byte limit, JPEG normalisation, atomic writes)
- [x] `database.py` – Weaviate v4 connection, strict `ProductV1` schema with named vectors, deterministic UUIDs
- [x] `ingestion.py` – idempotent, resumable, batched ingestion with content hash + model fingerprint, report file
- [x] `search.py` – text/image query against text/image vectors (fixes A3), compatibility preflight
- [x] `chat.py` – Google Gemini removed; local LLM via OpenAI-compatible API (llama.cpp + Qwen2.5-1.5B) with streaming, + offline "grounded" fallback
- [x] `ui.py` – Streamlit UI: search form, result cards with images, chat panel, settings actually used
- [x] `cli.py` – `vector-search scan | ingest | search | ask | evaluate | doctor`
- [x] `evaluation.py` – recall@k / MRR for all four text/image routes

## 2. Tests
- [x] Offline unit tests for every module (no network, no model download)
- [x] Fix failing/missing tests (`database`, `ui.perform_search`)
- [x] Integration run against the real Compose stack (Weaviate + Ollama + real CLIPfa models)
- [x] Retrieval evaluation on the 100-product hold-out set (`dataset/eval`)

## 3. Organisation
- [x] Standard layout: `src/`, `tests/`, `docker/`, `deploy/`, `scripts/`, `docs/`, `dataset/`, `legacy/`
- [x] Move data to `dataset/` (`parsed/`, `eval/`, `samples/`), old scripts to `legacy/`
- [x] `pyproject.toml` + `uv.lock` pinned dependencies (CPU-only PyTorch), `requirements.txt` exported
- [x] `.gitignore`, `.dockerignore`, `.env.example`, `Makefile`

## 4. Containerisation (Docker Compose, no Google services)
- [x] `app` image (Python 3.12 slim, non-root, CPU torch, healthcheck)
- [x] `weaviate` – private network only, persistent volume, pinned version
- [x] `llm` – llama.cpp server with Qwen2.5-1.5B-Instruct Q4_K_M (≈1 GB), private network only (Ollama was evaluated and rejected: 8.5 GB image)
- [x] `cli` – batch jobs with the app image (`docker compose run --rm cli ingest|evaluate|doctor`)
- [x] Hardening: non-root, `cap_drop: ALL`, `no-new-privileges`, memory/CPU limits, log rotation
- [x] Server override: HTTPS reverse proxy (nginx) on port 8443 using the host's Let's Encrypt IP certificate

## 5. Deployment on this host
- [x] Build + start the stack, ingest a light subset (shards 0–4 + 143 = 2,605 products; re-run verified idempotent)
- [x] Verify health, HTTPS, search and chat end-to-end
- [x] systemd unit so the stack comes back after a reboot

## 6. Documentation
- [x] `README.md` – what it is, quick start, screenshots of commands
- [x] `docs/ARCHITECTURE.md` – components, data flow, schema
- [x] `docs/DEPLOYMENT.md` – server deployment, operations, backups, upgrades
- [x] `docs/DEVELOPMENT.md` – local setup, tests, code map
- [x] `docs/DATA.md` – dataset format, quality report, evaluation set

## 7. GitHub
- [x] Push the work, tag the original code (`v0-original`), merge into `main`, tag deployable `v1.0.0`

## Later (not in this round)
- [ ] GitHub Actions CI (lint + tests + image build) – postponed on request
- [ ] Full 70k-product ingestion (≈ several CPU hours; command documented)
- [ ] Optional MinIO/S3 image store if images must live outside the app host (MinIO was dropped for the light deployment)
- [ ] Fine-tune CLIPfa on (product image, title) pairs to improve cross-modal search (see docs/DATA.md)
