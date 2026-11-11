"""Portable index snapshots: export a collection (properties + named vectors + images)
and restore it elsewhere without re-embedding.

Used to ship the production index to the Hugging Face demo Space, where CPU time
is scarce: restoring 2.6k products takes seconds, re-ingesting takes ~15 minutes.

Format: gzip JSON Lines. Line 1 is a header, every other line one object::

    {"format": "vector-search-snapshot", "version": 1, "collection": ..., "model_version": ..., "count": N}
    {"uuid": ..., "properties": {...}, "vectors": {"text_vector": [...], "image_vector": [...]}}
"""
from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .database import PROPERTIES, VECTOR_NAMES, ensure_collection

FORMAT = 'vector-search-snapshot'
VERSION = 1


def export_snapshot(settings: Any, out: Path, *, client: Any = None) -> dict:
    owns = client is None
    if owns:
        from .database import connect
        client = connect(settings)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        collection = ensure_collection(client, settings, create=False)
        objects = []
        versions = set()
        for obj in collection.iterator(include_vector=True):
            vectors = {name: list(obj.vector[name]) for name in VECTOR_NAMES}
            props = {key: obj.properties.get(key) for key in PROPERTIES}
            versions.add(props['model_version'])
            objects.append({'uuid': str(obj.uuid), 'properties': props, 'vectors': vectors})
    finally:
        if owns:
            client.close()
    if len(versions) > 1:
        raise RuntimeError('Collection mixes embedding model versions; re-ingest before exporting')
    objects.sort(key=lambda item: item['properties']['pid'])
    header = {'format': FORMAT, 'version': VERSION, 'collection': settings.collection,
              'model_version': next(iter(versions), ''), 'count': len(objects)}
    fd, temp = tempfile.mkstemp(dir=out.parent, suffix='.part')
    os.close(fd)
    try:
        with gzip.open(temp, 'wt', encoding='utf-8') as handle:
            handle.write(json.dumps(header) + '\n')
            for item in objects:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        os.replace(temp, out)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return header


def read_snapshot(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        header = json.loads(handle.readline())
        if header.get('format') != FORMAT or header.get('version') != VERSION:
            raise ValueError(f'{path} is not a version {VERSION} vector-search snapshot')
        yield header
        for line in handle:
            if line.strip():
                yield json.loads(line)


def import_snapshot(settings: Any, path: Path, *, client: Any = None, batch: int = 200) -> dict:
    """Upsert every snapshot object (deterministic UUIDs make this idempotent)."""
    from weaviate.classes.data import DataObject

    from .embeddings import Embedder
    owns = client is None
    if owns:
        from .database import connect
        client = connect(settings)
    items = read_snapshot(Path(path))
    header = next(items)
    expected = Embedder(settings).fingerprint
    if header['model_version'] and header['model_version'] != expected:
        raise RuntimeError('Snapshot was built with a different embedding model/revision '
                           'than this configuration; search would mix vector spaces')
    written = errors = 0
    try:
        collection = ensure_collection(client, settings, create=True)
        pending: list = []

        def flush() -> None:
            nonlocal written, errors
            if not pending:
                return
            result = collection.data.insert_many(pending)
            failed = len(getattr(result, 'errors', None) or {})
            errors += failed
            written += len(pending) - failed
            pending.clear()

        for item in items:
            pending.append(DataObject(properties=item['properties'], uuid=item['uuid'],
                                      vector=item['vectors']))
            if len(pending) >= batch:
                flush()
        flush()
        total = collection.aggregate.over_all(total_count=True).total_count
    finally:
        if owns:
            client.close()
    if errors:
        raise RuntimeError(f'{errors} snapshot objects failed to import')
    return {'imported': written, 'expected': header['count'], 'collection_total': total}


def export_images(image_dir: Path, out: Path, *, max_side: int = 384, quality: int = 82) -> dict:
    """Pack the image cache as display-size JPEG thumbnails (vectors are already computed
    from the full-size images, so only the UI uses these)."""
    from PIL import Image

    image_dir, out = Path(image_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = size = 0
    fd, temp = tempfile.mkstemp(dir=out.parent, suffix='.part')
    os.close(fd)
    try:
        with tarfile.open(temp, 'w') as tar:
            for path in sorted(image_dir.glob('*/*_0.jpg')):
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    image.thumbnail((max_side, max_side))
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=quality, optimize=True)
                data = buffer.getvalue()
                info = tarfile.TarInfo(str(path.relative_to(image_dir)))
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
                count += 1
                size += len(data)
        os.replace(temp, out)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {'images': count, 'bytes': size}


def extract_images(archive: Path, image_dir: Path) -> int:
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        tar.extractall(image_dir, members=members, filter='data')
    return len(members)
