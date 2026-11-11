# vector_search — Persian multimodal product search

Find Digikala products by **describing them in Persian** or by **uploading a photo**,
then ask a **local AI shopping assistant** about the results. Undergraduate final
project, rebuilt as a containerised, tested, documented application.

* **Embeddings:** [CLIPfa](https://github.com/sajjjadayobi/CLIPfa) — a Persian CLIP
  (`SajjadAyoubi/clip-fa-text` + `clip-fa-vision`), 768-d, text and images in one space,
  so you can search images with text and vice versa.
* **Vector database:** [Weaviate](https://weaviate.io) with two named vectors per product.
* **Assistant:** [llama.cpp](https://github.com/ggml-org/llama.cpp) server running
  **Qwen2.5-1.5B-Instruct** (Q4_K_M, ≈1 GB, CPU only). Free, runs on the server, needs no
  API key. The Google Gemini API from the original version has been removed.
* **UI:** Streamlit, Persian/RTL. **CLI:** `vector-search` for ingestion, search and evaluation.

```
 browser ──HTTPS:8443──> nginx ──> Streamlit app ──> Weaviate   (vectors + product data)
                                        │      └──> llama.cpp  (Qwen2.5-1.5B, OpenAI API)
                                        └── CLIPfa encoders (in-process, CPU)
 cli ingest: dataset/parsed/*.json ─> images from CDN ─> CLIPfa ─> Weaviate
```

## Quick start (any machine with Docker)

```bash
git clone https://github.com/ifatwme/vector_search.git && cd vector_search
cp .env.example .env
docker compose up -d                       # weaviate + llm + app (first start downloads the LLM, ~1 GB)
docker compose run --rm cli ingest --shards 0-4,143   # ~2.6k products, ~14 min on 4 CPU cores
open http://localhost:8501
```

Or with `make`: `make up`, `make ingest`, `make evaluate`, `make doctor` (`make help` lists all).

Requirements: Docker with Compose v2, ~8 GB disk, ~5 GB RAM free (app 1–2 GB, LLM ~1 GB,
Weaviate <1 GB for the light subset). No GPU and no API keys.

## Using it

| Where | What |
|---|---|
| Search column | Choose **متن** (text) or **تصویر** (image) input, then **جستجو**. The sidebar chooses whether to match against product **titles** or product **images** — all four combinations work. |
| Chat column | Ask about the products currently found, e.g. «کدام یک برای هدیه مناسب‌تر است؟». The assistant answers only from the retrieved products and cites their IDs. |
| Sidebar | Result count, LLM temperature / top_p / answer length, reset. |

Terminal equivalents:

```bash
docker compose run --rm cli search --text "کفش ورزشی مردانه" --limit 5
docker compose run --rm cli search --text "کفش ورزشی" --target image   # text -> product photos
docker compose run --rm cli ask "سیم گیتار" "کدام برای گیتار کلاسیک است؟"
docker compose run --rm cli evaluate        # recall@k on the 100-product hold-out set
docker compose run --rm cli doctor          # health of weaviate / llm / data
```

## Repository layout

```
src/vector_search/   application package (config, dataset, embeddings, storage, database,
                     ingestion, search, chat, evaluation, ui, cli)
app.py               Streamlit entry point
tests/               offline unit tests (pytest) — no network, no model downloads
dataset/parsed/      145 JSON shards, 70,214 crawled Digikala products (input data)
dataset/eval/        100 hold-out products (image + title) for retrieval evaluation
dataset/samples/     raw crawler responses and sample images
docker/              app Dockerfile, Streamlit config, nginx config
deploy/              server override (HTTPS) and systemd unit
docs/                architecture, deployment, development, data, task list
legacy/              the original scripts, kept for reference (not used)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, data flow, schema, design decisions
* [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — production on the VPS, operations, backups, upgrades
* [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — local setup, tests, configuration reference
* [docs/DATA.md](docs/DATA.md) — dataset format, quality report, evaluation results
* [docs/TASKS.md](docs/TASKS.md) — audit findings and the improvement task list

## Security note

The original `app.py` contained a Google API key; it is still visible in the Git history.
**Revoke that key in Google AI Studio** — removing it from the current code does not make it safe.

## License

No license file was provided by the author; all rights reserved unless stated otherwise.
CLIPfa and Qwen2.5-1.5B-Instruct (Apache-2.0) are used under their own licenses; see their model cards.
