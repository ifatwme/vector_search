"""Idempotent, resumable ingestion: dataset -> images -> CLIPfa vectors -> Weaviate.

Each product has a deterministic UUID and a content hash. Re-running the job
skips products whose hash *and* embedding fingerprint are already stored, so an
interrupted run simply continues where it stopped. Nothing is ever deleted.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from .database import SCHEMA_VERSION, ensure_collection, product_uuid
from .dataset import iter_products
from .storage import image_ref

log = logging.getLogger(__name__)
_TAG = re.compile(r'<[^>]+>')
_SPACE = re.compile(r'[ \t\r\f\v\xa0]+')


def clean_text(value: Any) -> str:
    """Strip the occasional HTML fragment in Digikala reviews and collapse spaces."""
    text = html.unescape(_TAG.sub(' ', value if isinstance(value, str) else ''))
    return '\n'.join(line.strip() for line in _SPACE.sub(' ', text).splitlines() if line.strip())


def to_properties(product: dict, image_ref: str, fingerprint: str) -> dict:
    title = ' '.join(part.strip() for part in (product['title_fa'], product['title_en'])
                     if part.strip())
    description = '\n'.join(part for part in (clean_text(product.get('description')),
                                              clean_text(product.get('short_review'))) if part)
    props = {
        'pid': product['id'],
        'title': title,
        'description': description,
        'url': product.get('url', ''),
        'image': image_ref,
        'image_url': product.get('image_main', ''),
        'attributes': json.dumps(product.get('attributes') or {}, ensure_ascii=False,
                                 sort_keys=True),
        'sellers': json.dumps(product.get('seller_info') or {}, ensure_ascii=False,
                              sort_keys=True),
        'schema_version': SCHEMA_VERSION,
    }
    digest = hashlib.sha256(json.dumps(props, ensure_ascii=False, sort_keys=True).encode())
    props['content_hash'] = digest.hexdigest()
    props['model_version'] = fingerprint
    return props


def _batches(items: Iterable[dict], size: int):
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _existing(collection: Any, uuids: list[str]) -> dict[str, tuple[str, str]]:
    from weaviate.classes.query import Filter
    response = collection.query.fetch_objects(
        filters=Filter.by_id().contains_any(uuids), limit=len(uuids),
        return_properties=['content_hash', 'model_version'])
    return {str(obj.uuid): (obj.properties.get('content_hash', ''),
                            obj.properties.get('model_version', ''))
            for obj in response.objects}


def _write_report(settings: Any, report: dict) -> None:
    directory = settings.reports_dir
    directory.mkdir(parents=True, exist_ok=True)
    for name in (f"ingest-{report['started_at'].replace(':', '')}.json", 'ingest-latest.json'):
        fd, temp = tempfile.mkstemp(dir=directory, suffix='.part')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            os.replace(temp, directory / name)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def ingest(settings: Any, *, limit: int | None = None, shards: Iterable[str] | None = None,
           client: Any = None, embedder: Any = None, store: Any = None,
           progress: Callable[[dict], None] | None = None, write_report: bool = True) -> dict:
    """Ingest products; returns (and persists) a report with per-outcome counts."""
    owns_client = client is None
    if owns_client:
        from .database import connect
        client = connect(settings)
    if embedder is None:
        from .embeddings import Embedder
        embedder = Embedder(settings)
    if store is None:
        from .storage import ImageStore
        store = ImageStore(settings)
    report = dict(started_at=datetime.now(UTC).isoformat(timespec='seconds'), finished_at=None,
                  limit=limit, shards=sorted(shards) if shards else None,
                  collection=settings.collection, model_version=embedder.fingerprint,
                  selected=0, inserted=0, updated=0, unchanged=0, confirmed=0, failed=0,
                  failures=[], seconds=0.0)
    started = time.monotonic()
    try:
        # Schema preflight first: never download models for an incompatible index.
        collection = ensure_collection(client, settings, create=True)
        products = iter_products(settings.parsed_dir, limit=limit, shards=shards)
        with ThreadPoolExecutor(max_workers=settings.download_workers) as pool:
            for batch in _batches(products, settings.ingest_batch_size):
                _ingest_batch(collection, batch, embedder, store, pool, report)
                report['seconds'] = round(time.monotonic() - started, 1)
                log.info('ingest progress: %s', {k: report[k] for k in (
                    'selected', 'inserted', 'updated', 'unchanged', 'failed', 'seconds')})
                if progress:
                    progress(report)
    finally:
        report['finished_at'] = datetime.now(UTC).isoformat(timespec='seconds')
        report['seconds'] = round(time.monotonic() - started, 1)
        if owns_client:
            client.close()
        if write_report:
            _write_report(settings, report)
    return report


def _fail(report: dict, product: dict, reason: str) -> None:
    report['failed'] += 1
    if len(report['failures']) < 500:  # keep the report file bounded
        report['failures'].append({'id': product['id'], 'reason': reason[:300]})


def _ingest_batch(collection, batch, embedder, store, pool, report) -> None:
    from weaviate.classes.data import DataObject

    report['selected'] += len(batch)
    uuids = [product_uuid(p['id']) for p in batch]
    existing = _existing(collection, uuids)
    fingerprint = embedder.fingerprint

    def fetch(product: dict):
        try:
            ref = store.get_or_download(product['image_main'], product['id'])
            return ref, store.read(ref)
        except Exception as exc:  # per-record failure, the batch continues
            return None, f'image: {type(exc).__name__}: {exc}'

    changed = []
    for product, uid in zip(batch, uuids, strict=True):
        # The image reference is deterministic, so unchanged products are skipped
        # before any download or model work.
        props = to_properties(product, image_ref(product['id']), fingerprint)
        if existing.get(uid) == (props['content_hash'], fingerprint):
            report['unchanged'] += 1
        else:
            changed.append((product, uid, props))
    pending = []
    images = pool.map(fetch, [product for product, _, _ in changed])
    for (product, uid, props), (ref, data) in zip(changed, images, strict=True):
        if ref is None:
            _fail(report, product, data)
            continue
        pending.append((product, uid, props, data))
    if not pending:
        return
    try:
        text_vectors = embedder.encode_texts([p[2]['title'] for p in pending])
    except ValueError as exc:
        for product, *_ in pending:
            _fail(report, product, f'text embedding: {exc}')
        return
    image_vectors: list[list[float] | None] = []
    try:
        image_vectors = list(embedder.encode_images([p[3] for p in pending]))
    except ValueError:
        # One undecodable image must not sink the batch: retry individually.
        for _, _, _, data in pending:
            try:
                image_vectors.append(embedder.encode_image(data))
            except ValueError:
                image_vectors.append(None)
    objects, written = [], []
    for (product, uid, props, _), text_vec, image_vec in zip(
            pending, text_vectors, image_vectors, strict=True):
        if image_vec is None:
            _fail(report, product, 'image embedding: undecodable image')
            continue
        objects.append(DataObject(properties=props, uuid=uid,
                                  vector={'text_vector': text_vec, 'image_vector': image_vec}))
        written.append((product, uid, props))
    if not objects:
        return
    result = collection.data.insert_many(objects)
    errors = {int(k): v for k, v in (getattr(result, 'errors', None) or {}).items()}
    # Read back before counting success: a batch "accepted" is not "stored".
    stored = _existing(collection, [uid for _, uid, _ in written])
    for index, (product, uid, props) in enumerate(written):
        if index in errors:
            _fail(report, product, f'weaviate: {getattr(errors[index], "message", errors[index])}')
        elif stored.get(uid) != (props['content_hash'], fingerprint):
            _fail(report, product, 'weaviate: object not readable after write')
        else:
            report['confirmed'] += 1
            report['updated' if uid in existing else 'inserted'] += 1
