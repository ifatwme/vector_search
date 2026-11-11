"""Snapshot round trip with in-memory fakes (no Weaviate)."""
import io
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from vector_search.config import Settings
from vector_search.database import PROPERTIES
from vector_search.embeddings import Embedder


def obj(pid, fingerprint):
    props = {key: '' for key in PROPERTIES}
    props.update(pid=pid, title=f'کالا {pid}', model_version=fingerprint)
    return NS(uuid=f'00000000-0000-5000-8000-{pid:012d}', properties=props,
              vector={'text_vector': [0.1] * 768, 'image_vector': [0.2] * 768})


class Collection:
    def __init__(self, objects=()):
        self.objects = list(objects)
        self.inserted = []
        self.data = NS(insert_many=self._insert)
        self.aggregate = NS(over_all=lambda **k: NS(total_count=len(self.inserted)))

    def iterator(self, include_vector):
        assert include_vector
        return iter(self.objects)

    def _insert(self, objects):
        self.inserted.extend(objects)
        return NS(errors={})


def test_export_then_import_round_trip(tmp_path, monkeypatch):
    import vector_search.snapshot as snapshot
    settings = Settings()
    fingerprint = Embedder(settings).fingerprint
    source = Collection([obj(2, fingerprint), obj(1, fingerprint)])
    target = Collection()
    collections = iter([source, target])
    monkeypatch.setattr(snapshot, 'ensure_collection', lambda c, s, create: next(collections))
    client = NS(close=lambda: None)
    header = snapshot.export_snapshot(settings, tmp_path / 's.jsonl.gz', client=client)
    assert header['count'] == 2 and header['model_version'] == fingerprint
    report = snapshot.import_snapshot(settings, tmp_path / 's.jsonl.gz', client=client)
    assert report == {'imported': 2, 'expected': 2, 'collection_total': 2}
    assert [o.properties['pid'] for o in target.inserted] == [1, 2]
    assert set(target.inserted[0].vector) == {'text_vector', 'image_vector'}


def test_import_rejects_snapshot_from_another_model(tmp_path, monkeypatch):
    import vector_search.snapshot as snapshot
    settings = Settings()
    monkeypatch.setattr(snapshot, 'ensure_collection', lambda c, s, create: Collection([obj(1, 'old')]))
    snapshot.export_snapshot(settings, tmp_path / 's.jsonl.gz', client=NS(close=lambda: None))
    with pytest.raises(RuntimeError, match='different embedding model'):
        snapshot.import_snapshot(settings, tmp_path / 's.jsonl.gz', client=NS(close=lambda: None))


def test_images_are_thumbnailed_and_restored(tmp_path):
    from vector_search.snapshot import export_images, extract_images
    source = tmp_path / 'images' / '7'
    source.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new('RGB', (1000, 500), 'red').save(buf, format='JPEG')
    (source / '7_0.jpg').write_bytes(buf.getvalue())
    report = export_images(tmp_path / 'images', tmp_path / 'out' / 'images.tar', max_side=100)
    assert report['images'] == 1
    assert extract_images(tmp_path / 'out' / 'images.tar', tmp_path / 'restored') == 1
    with Image.open(tmp_path / 'restored' / '7' / '7_0.jpg') as image:
        assert max(image.size) == 100
    settings = replace(Settings(), image_dir=tmp_path / 'restored')
    from vector_search.storage import ImageStore
    assert ImageStore(settings).read('7/7_0.jpg')[:2] == b'\xff\xd8'
