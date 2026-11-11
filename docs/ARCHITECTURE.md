# Architecture

## Components

| Service | Image | Role | Exposed |
|---|---|---|---|
| `app` | `vector-search-app` (built from `docker/app.Dockerfile`) | Streamlit UI; loads CLIPfa in-process on the first search | `127.0.0.1:8501` |
| `cli` | same image, profile `tools` | one-shot jobs: `scan`, `ingest`, `search`, `ask`, `evaluate`, `doctor` | — |
| `weaviate` | `cr.weaviate.io/semitechnologies/weaviate:1.40.5` | vector database (HNSW, cosine) | private network only |
| `llm` | `ghcr.io/ggml-org/llama.cpp:server` (pinned digest) | Qwen2.5-1.5B-Instruct GGUF, OpenAI-compatible `/v1` API | private network only |
| `web` | `nginxinc/nginx-unprivileged:1.31.7-alpine` (server only) | TLS termination, WebSocket proxy | `:8443` |

Volumes: `weaviate-data` (index), `images` (product image cache), `models` (CLIPfa
weights), `llm-models` (GGUF file), `reports` (ingest reports). `docker compose down`
keeps them; `down -v` deletes them.

## Data flow

```
ingest (cli)                                        query (app)
------------                                        -----------
dataset/parsed/*.json                               text or image from the browser
  │ dataset.iter_products: validate, de-duplicate,      │
  │ quarantine conflicting IDs                         ▼
  ▼                                                 search.search
skip if content_hash + model fingerprint unchanged     │ preflight: index model fingerprint
  │                                                    │ and vector size must match
  ▼                                                    ▼
storage.ImageStore: download main image             Embedder.encode_text / encode_image
(trusted CDN only, ≤15 MB, JPEG, atomic write)         │
  ▼                                                    ▼
Embedder.encode_texts(titles) / encode_images       Weaviate near_vector on text_vector
  ▼                                                 or image_vector (user's choice)
Weaviate insert_many (deterministic UUID5)             │
  ▼                                                    ▼
read back + verify hash → report JSON               result cards (image from the cache)
                                                       │
                                                       ▼
                                                    chat.ChatService.stream
                                                    → llm /v1/chat/completions (SSE)
```

## Embeddings

CLIPfa is a Persian CLIP: a RoBERTa text encoder and a CLIP ViT-B/32 vision encoder
trained so that captions and images land in the same 768-d space. We use the published
`pooler_output` convention (no extra projection), so:

* `text_vector` = CLIPfa text embedding of the product title (`title_fa title_en`)
* `image_vector` = CLIPfa vision embedding of the main product image

A text query can therefore be matched against titles (text→text) or photos (text→image),
and an uploaded photo against photos (image→image) or titles (image→text).

Model revisions are pinned to commit SHAs (`TEXT_MODEL_REVISION`, `IMAGE_MODEL_REVISION`).
The embedder exposes a **fingerprint** (model IDs + revisions + preprocessing) that is
stored on every object; search refuses to query an index built with a different model,
and ingestion re-embeds objects whose fingerprint changed.

## Weaviate schema (`ProductV1`)

| Property | Type | Notes |
|---|---|---|
| `pid` | int | Digikala product ID |
| `title` | text | Persian + English title |
| `description` | text | expert review + short review, HTML stripped |
| `url` | text | product page |
| `image` | text | cache reference `<pid>/<pid>_0.jpg` (never an expiring URL) |
| `image_url` | text | original CDN URL (metadata) |
| `attributes` | text | JSON of specifications |
| `sellers` | text | JSON of seller offers at crawl time |
| `content_hash` | text | SHA-256 of the stored properties → idempotent re-runs |
| `model_version` | text | embedder fingerprint |
| `schema_version` | text | `"1"` |

Named vectors `text_vector` and `image_vector`: self-provided, HNSW, cosine. Object UUIDs
are `uuid5(namespace, "product:<pid>")`, so re-ingesting a product overwrites it in place
instead of creating duplicates (the original script appended duplicates on every run).

## Assistant

`CHAT_BACKEND=llm` (default in Compose) calls any OpenAI-compatible endpoint
(`LLM_URL`). The system prompt tells the model to answer in Persian, only from the
products passed as a JSON block, to treat that block as untrusted data, to cite product
IDs, and not to invent prices or availability. Context is bounded (≤5 products, truncated
fields, last 8 messages). Answers stream token by token.

`CHAT_BACKEND=grounded` needs no model at all: it quotes the stored fields of the top
results. The UI also falls back to it automatically if the LLM is unreachable, so search
and a basic assistant keep working.

### Why Qwen2.5-1.5B on llama.cpp

* Free and local: no API key, no quota, nothing leaves the server.
* Light: the Q4_K_M GGUF is ≈1 GB on disk and ≈1 GB RAM; the llama.cpp CPU image is
  ≈1.2 GB (the Ollama image we tried first is 8.5 GB because it bundles GPU runtimes).
* Persian: Qwen2.5 is multilingual and gives usable short Persian answers at 1.5B;
  measured ≈6–25 s per answer on this 4-core VPS.
* Swappable: set `LLM_HF_MODEL` to any GGUF on Hugging Face (e.g. a 3B model for better
  answers if you have the RAM), or point `LLM_URL` at Ollama/vLLM.

## Design decisions

* **No MinIO.** Images live in a Docker volume on the same host; for a single-server
  deployment an object store adds a service and credentials without benefit. The storage
  layer only deals in references (`<pid>/<pid>_0.jpg`), so an S3 backend can be added later.
* **Short-lived DB clients.** The UI opens a Weaviate client per query and always closes
  it; the original kept one per session and leaked them.
* **Models load lazily** on the first search and are shared by all sessions
  (`st.cache_resource`), so the page renders instantly and memory is paid once.
* **Fail loudly on incompatibility, continue on record errors.** A wrong schema or model
  stops ingestion; a broken image only fails that product and is listed in the report.
