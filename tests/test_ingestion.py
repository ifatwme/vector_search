"""Ingestion against in-memory fakes: idempotency, failures, readback, reports."""
import json
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vector_search.config import Settings
from vector_search.database import product_uuid
from vector_search.ingestion import clean_text, ingest, to_properties


def product(pid, title='کفش', image=None):
    return {'id': pid, 'title_fa': title, 'title_en': 'Shoe', 'url': f'https://www.digikala.com/product/dkp-{pid}/',
            'image_main': image or f'https://dkstatics-public.digikala.com/{pid}.jpg',
            'description': '<p>رویه&nbsp;پارچه</p>', 'short_review': '', 'attributes': {'جنس': 'پارچه'},
            'seller_info': {}, 'properties': {}}


class FakeData:
    def __init__(self, store):
        self.store = store
        self.batches = []

    def insert_many(self, objects):
        self.batches.append(objects)
        for obj in objects:
            self.store[str(obj.uuid)] = (obj.properties, obj.vector)
        return NS(errors={})


class FakeQuery:
    def __init__(self, store):
        self.store = store

    def fetch_objects(self, filters=None, limit=None, return_properties=None):
        wanted = {str(v) for v in filters.value}
        return NS(objects=[NS(uuid=uid, properties=props) for uid, (props, _) in self.store.items()
                           if uid in wanted])


class FakeCollection:
    def __init__(self):
        self.objects = {}
        self.data = FakeData(self.objects)
        self.query = FakeQuery(self.objects)


class FakeEmbedder:
    fingerprint = 'fp-1'

    def __init__(self):
        self.texts = self.images = 0

    def encode_texts(self, texts):
        self.texts += len(texts)
        return [[1.0] * 768 for _ in texts]

    def encode_images(self, images):
        if any(data == b'bad' for data in images):
            raise ValueError('undecodable')
        self.images += len(images)
        return [[1.0] * 768 for _ in images]

    def encode_image(self, data):
        if data == b'bad':
            raise ValueError('undecodable')
        return [1.0] * 768


class FakeStore:
    def __init__(self, broken=(), corrupt=()):
        self.broken, self.corrupt, self.downloads = set(broken), set(corrupt), 0

    def get_or_download(self, url, pid):
        self.downloads += 1
        if pid in self.broken:
            raise ValueError('Image server did not return a direct successful response')
        return f'{pid}/{pid}_0.jpg'

    def read(self, ref):
        pid = int(ref.split('/')[0])
        return b'bad' if pid in self.corrupt else b'\xff\xd8jpeg\xff\xd9'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    parsed = tmp_path / 'parsed'
    parsed.mkdir()
    (parsed / 'products_0_parsed.json').write_text(json.dumps([product(1), product(2), product(3)]))
    (parsed / 'products_1_parsed.json').write_text(json.dumps([product(4)]))
    settings = replace(Settings(), parsed_dir=parsed, reports_dir=tmp_path / 'reports',
                       ingest_batch_size=2, download_workers=2)
    collection = FakeCollection()
    import vector_search.ingestion as module
    monkeypatch.setattr(module, 'ensure_collection', lambda client, s, create: collection)
    return settings, collection


def run(settings, store=None, embedder=None, **kwargs):
    client = NS(close=lambda: (_ for _ in ()).throw(AssertionError('borrowed client closed')))
    return ingest(settings, client=client, embedder=embedder or FakeEmbedder(),
                  store=store or FakeStore(), **kwargs)


def test_first_run_inserts_and_second_run_is_a_noop(setup):
    settings, collection = setup
    first = run(settings)
    assert (first['inserted'], first['confirmed'], first['failed']) == (4, 4, 0)
    assert set(collection.objects) == {product_uuid(i) for i in (1, 2, 3, 4)}
    props, vectors = collection.objects[product_uuid(1)]
    assert props['image'] == '1/1_0.jpg' and props['model_version'] == 'fp-1'
    assert set(vectors) == {'text_vector', 'image_vector'}
    assert 'text_vector' not in props  # vectors are not duplicated as properties
    store, embedder = FakeStore(), FakeEmbedder()
    second = run(settings, store=store, embedder=embedder)
    assert (second['unchanged'], second['inserted'], second['updated']) == (4, 0, 0)
    assert store.downloads == 0 and embedder.texts == 0
    report = json.loads((settings.reports_dir / 'ingest-latest.json').read_text())
    assert report['unchanged'] == 4


def test_changed_content_or_model_is_updated_in_place(setup):
    settings, collection = setup
    run(settings)
    class NewModel(FakeEmbedder):
        fingerprint = 'fp-2'
    again = run(settings, embedder=NewModel())
    assert again['updated'] == 4 and len(collection.objects) == 4


def test_record_failures_are_counted_and_do_not_stop_the_batch(setup):
    settings, collection = setup
    report = run(settings, store=FakeStore(broken={2}, corrupt={3}))
    assert report['confirmed'] == 2 and report['failed'] == 2
    reasons = {f['id']: f['reason'] for f in report['failures']}
    assert reasons[2].startswith('image:') and 'embedding' in reasons[3]
    assert product_uuid(2) not in collection.objects


def test_limit_and_shard_selection(setup):
    settings, collection = setup
    assert run(settings, limit=1)['selected'] == 1
    report = run(settings, shards=['products_1_parsed.json'])
    assert report['selected'] == 1 and product_uuid(4) in collection.objects
    with pytest.raises(ValueError, match='Unknown dataset shards'):
        run(settings, shards=['products_9_parsed.json'])


def test_unconfirmed_write_is_a_failure(setup):
    settings, collection = setup
    collection.data.insert_many = lambda objects: NS(errors={0: NS(message='boom')})
    report = run(settings, limit=2)
    assert report['confirmed'] == 0 and report['failed'] == 2


def test_properties_clean_html_and_hash_ignores_model_version():
    a = to_properties(product(5), '5/5_0.jpg', 'fp-1')
    b = to_properties(product(5), '5/5_0.jpg', 'fp-2')
    assert a['description'] == 'رویه پارچه'
    assert a['title'] == 'کفش Shoe'
    assert a['content_hash'] == b['content_hash']
    assert json.loads(a['attributes']) == {'جنس': 'پارچه'}
    assert clean_text(None) == ''
