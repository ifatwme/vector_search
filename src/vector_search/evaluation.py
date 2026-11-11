"""Retrieval quality on the hold-out set in ``dataset/eval``.

The set holds 100 products from shard 143: ``images/<n>_<pid>.jpg`` and
``titles/<n>_<pid>.txt``. Each query should return its own product. Text->text and
image->image are sanity checks (the query equals the indexed item); the
cross-modal routes (text->image, image->text) measure real CLIPfa alignment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_NAME = re.compile(r'^(\d+)_(\d+)$')


def load_eval_set(directory: Path) -> list[dict]:
    items = []
    for title_file in sorted((directory / 'titles').glob('*.txt')):
        match = _NAME.match(title_file.stem)
        if not match:
            continue
        image = directory / 'images' / f'{title_file.stem}.jpg'
        items.append({'pid': int(match.group(2)),
                      'title': title_file.read_text(encoding='utf-8').strip(),
                      'image': image if image.is_file() else None})
    if not items:
        raise ValueError(f'No evaluation items found in {directory}')
    return items


def _metrics(ranks: list[int | None], ks) -> dict:
    total = len(ranks)
    out = {f'recall@{k}': round(sum(1 for r in ranks if r is not None and r <= k) / total, 3)
           for k in ks}
    out['mrr'] = round(sum(1 / r for r in ranks if r) / total, 3)
    out['queries'] = total
    return out


def evaluate(settings: Any, directory: Path, *, ks=(1, 5, 10), client: Any = None,
             embedder: Any = None) -> dict:
    from .database import connect, ensure_collection
    from .embeddings import Embedder
    from .search import search

    items = load_eval_set(directory)
    owns = client is None
    client = client or connect(settings)
    embedder = embedder or Embedder(settings)
    depth = max(ks)
    try:
        collection = ensure_collection(client, settings, create=False)
        indexed = {obj.properties['pid'] for obj in collection.query.fetch_objects(
            filters=_pid_filter([i['pid'] for i in items]), limit=len(items),
            return_properties=['pid']).objects}
        present = [item for item in items if item['pid'] in indexed]
        if not present:
            raise RuntimeError('None of the evaluation products are indexed; ingest shard 143 '
                               '(`vector-search ingest --shards 143`).')
        routes: dict[str, list[int | None]] = {}
        for source in ('text', 'image'):
            for target in ('text', 'image'):
                ranks = []
                for item in present:
                    if source == 'image' and item['image'] is None:
                        continue
                    query = ({'text': item['title']} if source == 'text'
                             else {'image': item['image'].read_bytes()})
                    hits = search(client, settings, embedder, target=target, limit=depth, **query)
                    pids = [hit['pid'] for hit in hits]
                    ranks.append(pids.index(item['pid']) + 1 if item['pid'] in pids else None)
                routes[f'{source}->{target}'] = _metrics(ranks, ks)
        index_size = collection.aggregate.over_all(total_count=True).total_count
    finally:
        if owns:
            client.close()
    return {'eval_items': len(items), 'indexed_eval_items': len(present),
            'index_size': index_size,
            'routes': routes}


def _pid_filter(pids: list[int]):
    from weaviate.classes.query import Filter
    return Filter.by_property('pid').contains_any(pids)
