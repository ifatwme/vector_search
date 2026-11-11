# legacy/

The original scripts from the first version of the project, kept unchanged for reference
(only the leaked API key was replaced by an environment variable). **Nothing imports
these files and they are not maintained.** Their replacements:

| Legacy file | Replaced by |
|---|---|
| `app_gemini.py` | `app.py` + `src/vector_search/ui.py`, `chat.py`, `search.py` |
| `parse_raw_products.py` | `src/vector_search/parsing.py` |
| `weaviate/import_weaviate.py` | `src/vector_search/ingestion.py`, `database.py`, `embeddings.py`, `storage.py` |
| `weaviate/retriever.py` | `src/vector_search/search.py`, `vector-search search` |
| `minio/*.py` | removed (images are cached in a Docker volume; see docs/ARCHITECTURE.md) |
| `build_eval_set.py` | produced `dataset/eval/`; used by `src/vector_search/evaluation.py` |
