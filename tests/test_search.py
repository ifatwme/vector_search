"""Offline regression tests: real routing, fake provider boundaries only."""
from types import SimpleNamespace as NS

import pytest


class Embedder:
    fingerprint = "clip-test"

    def __init__(self):
        self.calls = []

    def encode_text(self, value):
        self.calls.append(("text", value))
        return [1.0, 0.0]

    def encode_image(self, value):
        self.calls.append(("image", value))
        return [0.0, 1.0]


class Query:
    def __init__(self):
        self.calls = []
        self.objects = [NS(properties={"pid": 7, "title": "کالا", "description": "توضیح", "url": "https://example.com/p/7", "image": "7/7_0.jpg", "model_version": "clip-test"}, vector={"text_vector": [1., 0.], "image_vector": [0., 1.]}, metadata=NS(distance=0.2))]

    def fetch_objects(self, **kwargs):
        self.calls.append(("fetch", kwargs))
        return NS(objects=self.objects)

    def near_vector(self, **kwargs):
        self.calls.append(("near", kwargs))
        return NS(objects=self.objects)


def dependencies():
    query = Query()
    collection = NS(query=query)
    client = NS(collections=NS(get=lambda name: collection))
    return client, NS(collection="ProductV1", max_image_bytes=1024), Embedder(), query


@pytest.mark.parametrize("source,target", [("text", "text"), ("text", "image"), ("image", "text"), ("image", "image")])
def test_four_search_routes_choose_input_encoder_and_named_target(source, target):
    from vector_search.search import search
    client, settings, embedder, query = dependencies()
    value = "کفش" if source == "text" else b"image"
    result = search(client, settings, embedder, **{source: value}, target=target, limit=3)
    assert embedder.calls == [(source, value)]
    call = query.calls[-1][1]
    assert call["target_vector"] == target + "_vector"
    assert call["limit"] == 3
    assert call["return_metadata"].distance is True
    assert result[0]["pid"] == 7
    assert result[0]["image"] == "7/7_0.jpg"
    assert result[0]["distance"] == 0.2


@pytest.mark.parametrize("kwargs", [
    {}, {"text": "x", "image": b"x"}, {"text": " "}, {"text": 1},
    {"image": b""}, {"image": "path"}, {"image": b"x" * 1025},
    {"text": "x", "target": "wrong"}, {"text": "x", "limit": 0},
    {"text": "x", "limit": 101}, {"text": "x", "limit": True},
])
def test_bad_search_input_never_loads_model_or_queries(kwargs):
    from vector_search.search import search
    client, settings, embedder, query = dependencies()
    with pytest.raises(ValueError):
        search(client, settings, embedder, **kwargs)
    assert embedder.calls == []
    assert query.calls == []


@pytest.mark.parametrize("problem", ["fingerprint", "missing_fingerprint", "missing_vector", "dimension"])
def test_incompatible_index_rejected_instead_of_mixing_spaces(problem):
    from vector_search.search import search
    client, settings, embedder, query = dependencies()
    obj = query.objects[0]
    if problem == "fingerprint":
        obj.properties["model_version"] = "old"
    elif problem == "missing_fingerprint":
        obj.properties.pop("model_version")
    elif problem == "missing_vector":
        obj.vector = {}
    else:
        obj.vector["text_vector"] = [0.0] * 3
    with pytest.raises(ValueError, match="(?i)index|ایندکس"):
        search(client, settings, embedder, text="کالا")
    assert not any(kind == "near" for kind, _ in query.calls)
    if problem != "dimension":
        assert embedder.calls == []


def test_empty_index_does_not_load_models():
    from vector_search.search import search
    client, settings, embedder, query = dependencies()
    query.objects = []
    assert search(client, settings, embedder, text="کالا") == []
    assert embedder.calls == []


@pytest.mark.parametrize("vector", [[float("nan"), 0.], [0., 0.], []])
def test_invalid_embedding_cannot_reach_database_query(vector):
    from vector_search.search import search
    client, settings, embedder, query = dependencies()
    embedder.encode_text = lambda value: vector
    with pytest.raises(ValueError):
        search(client, settings, embedder, text="کالا")
    assert not any(kind == "near" for kind, _ in query.calls)
