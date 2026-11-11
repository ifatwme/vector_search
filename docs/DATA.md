# Data

## `dataset/parsed/` — the product catalog

145 JSON shards (`products_<n>_parsed.json`, 262 MB) crawled from Digikala. Each file is a
JSON array of products produced by the original parser (now `src/vector_search/parsing.py`):

| Field | Type | Notes |
|---|---|---|
| `id` | int | Digikala product ID (unique key) |
| `title_fa`, `title_en` | str | `title_en` is empty for 87% of products |
| `url` | str | product page |
| `image_main`, `image_list` | str, [str] | CDN URLs (`dkstatics-public`; 71 products use signed `dkstatics-private` URLs) |
| `description`, `short_review` | str | expert review; empty for 46% of products, some contain HTML |
| `attributes` | {str: str} | specifications (material, size, …) |
| `properties` | object | Digikala flags (fast shipping, ad, …) |
| `seller_info`, `seller_count` | object, int | offers at crawl time (prices in Rial) |
| `brand_id`, `brand_fa`, `brand_en` | | |
| `advantages`, `disadvantages` | [str] | pros & cons |

### Quality report (`vector-search scan`)

| Metric | Value |
|---|---|
| shards | 145 |
| records | 70,214 |
| unique IDs | 70,186 |
| exact duplicate rows (collapsed) | 27 |
| IDs with conflicting versions (quarantined) | 1 (`11639370`, twice in shard 0) |
| invalid records | 0 |
| **valid products** | **70,185** |

Conflicting IDs are excluded rather than picking one version arbitrarily; `scan --details`
lists where they occur.

## `dataset/eval/` — hold-out evaluation set

100 products from shard 143 (built by `legacy/build_eval_set.py`): `images/<n>_<pid>.jpg`
(main product image) and `titles/<n>_<pid>.txt` (Persian title). `titles_shard143.txt` is
the full title list of that shard.

## `dataset/samples/`

`example*.json` are raw crawler responses (input of the parser); `images/` holds three
sample product images.

## What gets indexed

The light deployment ingests shards **0–4 and 143** (2,605 products, 14 min on 4 CPU cores
including image downloads). `vector-search ingest` with no `--shards` ingests all 70,185
products (≈4–5 h, resumable). Per product: the title is embedded with the CLIPfa text
encoder, the main image with the CLIPfa vision encoder.

## Retrieval evaluation (`vector-search evaluate`)

Measured on the production index (2,605 products, all 100 evaluation products indexed).
For each evaluation product, its title or image is used as the query and we check the rank
of the product itself.

| Route | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| text → text | 0.95 | 1.00 | 1.00 | 0.973 |
| image → image | 0.98 | 1.00 | 1.00 | 0.990 |
| text → image | 0.02 | 0.08 | 0.12 | 0.046 |
| image → text | 0.01 | 0.07 | 0.10 | 0.033 |

How to read this:

* **text→text and image→image** are sanity checks: the query equals the indexed title or
  image, so ~1.0 shows the pipeline (ingest → storage → search) is consistent. Misses at
  rank 1 are near-identical products (e.g. the same guitar strings in two variants).
* **Cross-modal exact-item retrieval is weak** (random would be 0.004 at recall@10, so it is
  30× better than chance, but low). CLIPfa was trained on 400k general image–caption
  pairs; product titles are dominated by brand and model codes ("T50J", "NF9126") that no
  image can express. At the **category level** cross-modal search works well: «ساعت مچی»
  → 4 of the top 5 product photos are wristwatches, «کفش ورزشی» → 5/5 shoes.
* Practical consequence: use **text→text** for precise queries and **text→image** for
  visual/stylistic browsing; uploading a photo works best against **image** vectors.
  Fine-tuning CLIPfa on (product image, title) pairs from this dataset is the obvious next
  step to improve cross-modal search.
