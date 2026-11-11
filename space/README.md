---
title: Persian Product Search
emoji: 🔎
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Persian text & image product search with a local LLM
---

# جستجوی کالا — Persian multimodal product search (demo)

Search 2,605 Digikala products by **Persian text** or by **uploading a photo**, then ask a
**local AI shopping assistant** about the results.

* **CLIPfa** (Persian CLIP) embeds titles and product photos in one 768-d space
* **Weaviate** stores the vectors (index restored from
  [ifatwme/vector-search-demo-data](https://huggingface.co/datasets/ifatwme/vector-search-demo-data))
* **Qwen2.5-1.5B-Instruct** (llama.cpp, CPU) answers questions using only the products found

Everything runs inside this Space on the free CPU tier: no external API, no keys. The first
search after the Space wakes up loads the models (~20 s); assistant answers take ~30–60 s on
2 vCPUs.

Source code, documentation and the full Docker Compose deployment:
**https://github.com/ifatwme/vector_search** — this Space is deployed automatically from
`main` by GitHub Actions.
