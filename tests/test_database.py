from types import SimpleNamespace as NS
from uuid import UUID

from vector_search.database import ensure_collection, product_uuid

REQUIRED = {'pid': 'int', **{key: 'text' for key in (
    'title', 'description', 'url', 'image', 'image_url', 'attributes', 'sellers',
    'model_version', 'schema_version', 'content_hash')}}


def config():
    return NS(properties=[NS(name=k, data_type=v) for k, v in REQUIRED.items()],
              vector_config={name: NS(vectorizer=NS(vectorizer='none'),
                                     vector_index_config=NS(distance_metric='cosine'))
                             for name in ('text_vector', 'image_vector')},
              multi_tenancy_config=NS(enabled=False))


class Collections:
    def __init__(self, existing=True):
        self.existing = existing
        self.cfg = config()
        self.collection = NS(config=NS(get=lambda: self.cfg))
        self.created = []
    def exists(self, name):
        return self.existing
    def get(self, name):
        return self.collection
    def create(self, **kwargs):
        self.created.append(kwargs)
        self.existing = True
        return self.collection


def test_create_strict_named_vectors_cosine_schema_and_revalidate():
    collections = Collections(existing=False)
    client, settings = NS(collections=collections), NS(collection='ProductV1')
    assert ensure_collection(client, settings) is collections.collection
    created = collections.created[0]
    assert created['name'] == 'ProductV1'
    assert {p.name: p.dataType.value for p in created['properties']} == REQUIRED
    assert {v.name for v in created['vector_config']} == {'text_vector', 'image_vector'}
    assert all(v.vectorizer.vectorizer.value == 'none' and
               v.vectorIndexConfig.distance.value == 'cosine' for v in created['vector_config'])
    ensure_collection(client, settings)
    assert len(collections.created) == 1
    assert UUID(product_uuid(12)).version == 5
    assert product_uuid(12) == product_uuid(12) != product_uuid(13)
