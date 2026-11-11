"""Weaviate v4 connection and the strict ``ProductV1`` schema.

Vectors are self-provided (CLIPfa runs in our own process) and stored only as
named vectors; the original import script also duplicated them as properties.
"""
from __future__ import annotations

import uuid
from typing import Any

SCHEMA_VERSION = '1'
VECTOR_NAMES = ('text_vector', 'image_vector')
# Stable namespace so a product always maps to the same object (idempotent upserts).
_NAMESPACE = uuid.UUID('6f1c3f7e-2b55-5d0e-9a51-0c8e4e1d7a10')

PROPERTIES: dict[str, str] = {
    'pid': 'int',
    'title': 'text',
    'description': 'text',
    'url': 'text',
    'image': 'text',          # storage reference "<pid>/<pid>_0.jpg"
    'image_url': 'text',      # original catalog URL (metadata only)
    'attributes': 'text',     # JSON
    'sellers': 'text',        # JSON
    'model_version': 'text',  # embedder fingerprint
    'schema_version': 'text',
    'content_hash': 'text',
}


def product_uuid(pid: int) -> str:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError('Product id must be a positive integer')
    return str(uuid.uuid5(_NAMESPACE, f'product:{pid}'))


def connect(settings: Any):
    """Open a client; callers own it and must ``close()`` it."""
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    auth = Auth.api_key(settings.weaviate_api_key) if settings.weaviate_api_key else None
    return weaviate.connect_to_custom(
        http_host=settings.weaviate_host, http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_secure, grpc_host=settings.weaviate_grpc_host,
        grpc_port=settings.weaviate_grpc_port, grpc_secure=settings.weaviate_secure,
        auth_credentials=auth,
        additional_config=AdditionalConfig(timeout=Timeout(init=10, query=60, insert=120)),
    )


def _value(item: Any) -> str:
    return str(getattr(item, 'value', item)).lower()


def _definition():
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances

    types = {'int': DataType.INT, 'text': DataType.TEXT}
    properties = [Property(name=name, data_type=types[kind]) for name, kind in PROPERTIES.items()]
    vectors = [
        Configure.Vectors.self_provided(
            name=name,
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
        )
        for name in VECTOR_NAMES
    ]
    return properties, vectors


def validate_collection(config: Any) -> None:
    """Fail loudly if an existing collection would silently mix incompatible data."""
    found = {p.name: _value(p.data_type) for p in config.properties}
    problems = [f'{name}: expected {kind}, found {found.get(name, "missing")}'
                for name, kind in PROPERTIES.items() if found.get(name) != kind]
    vectors = config.vector_config or {}
    for name in VECTOR_NAMES:
        vector = vectors.get(name)
        if vector is None:
            problems.append(f'named vector {name} is missing')
            continue
        if _value(vector.vectorizer.vectorizer) != 'none':
            problems.append(f'{name} must be self-provided (vectorizer none)')
        if _value(vector.vector_index_config.distance_metric) != 'cosine':
            problems.append(f'{name} must use cosine distance')
    tenancy = getattr(config, 'multi_tenancy_config', None)
    if tenancy is not None and getattr(tenancy, 'enabled', False):
        problems.append('multi-tenancy is not supported')
    if problems:
        raise RuntimeError('Incompatible Weaviate collection: ' + '; '.join(problems))


def ensure_collection(client: Any, settings: Any, *, create: bool = True):
    """Return the collection, creating it if allowed. Never drops existing data."""
    name = settings.collection
    if not client.collections.exists(name):
        if not create:
            raise RuntimeError(
                f'Collection {name} does not exist yet; run `vector-search ingest` first.')
        properties, vectors = _definition()
        client.collections.create(
            name=name,
            description='Digikala products with CLIPfa text and image vectors',
            properties=properties, vector_config=vectors,
        )
    collection = client.collections.get(name)
    validate_collection(collection.config.get())
    return collection
