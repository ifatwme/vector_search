"""Provider-independent cross-modal retrieval; no connections at import time."""
from __future__ import annotations

from typing import Any

RESULT_FIELDS = ("pid", "title", "description", "url", "image", "attributes")


def search(client: Any, settings: Any, embedder: Any, *, text: str | None = None,
           image: bytes | None = None, target: str = "text", limit: int = 10) -> list[dict]:
    """Encode the input modality and query the independently selected target."""
    if (text is None) == (image is None):
        raise ValueError("دقیقاً یک ورودی متن یا تصویر انتخاب کنید.")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("متن جستجو نباید خالی باشد.")
    if image is not None and (not isinstance(image, bytes) or not image or len(image) > settings.max_image_bytes):
        raise ValueError("تصویر باید بایت معتبر و در محدوده حجم مجاز باشد.")
    if target not in ("text", "image"):
        raise ValueError("مقصد جستجو باید text یا image باشد.")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("تعداد نتایج باید بین ۱ و ۱۰۰ باشد.")
    import math

    from weaviate.classes.query import MetadataQuery

    collection = client.collections.get(settings.collection)
    # Sample preflight avoids downloading weights for an empty/incompatible index.
    # This is not a proof of collection-wide uniformity; check returned hits too.
    sample = collection.query.fetch_objects(limit=1, include_vector=True)
    if not sample.objects:
        return []
    probe = sample.objects[0]
    if probe.properties.get("model_version") != embedder.fingerprint:
        raise ValueError("ایندکس با نسخه مدل سازگار نیست؛ داده‌ها را با تنظیمات فعلی دوباره ingest کنید.")
    indexed_vector = probe.vector.get(f"{target}_vector") if isinstance(probe.vector, dict) else None
    if not indexed_vector:
        raise ValueError("ایندکس فاقد بردار مقصد است؛ schema و ingestion را بررسی کنید.")
    vector = embedder.encode_text(text) if text is not None else embedder.encode_image(image)
    if (not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
            or not any(x != 0 for x in vector)):
        raise ValueError("بردار مدل نامعتبر است؛ تنظیمات مدل را بررسی کنید.")
    if len(vector) != len(indexed_vector):
        raise ValueError("ابعاد مدل و ایندکس متفاوت است؛ ایندکس را با مدل فعلی بسازید.")
    response = collection.query.near_vector(
        near_vector=vector, target_vector=f"{target}_vector", limit=limit,
        return_metadata=MetadataQuery(distance=True),
    )
    results = []
    for obj in response.objects:
        item = {key: obj.properties.get(key, "") for key in RESULT_FIELDS}
        if obj.metadata is not None and obj.metadata.distance is not None:
            item["distance"] = obj.metadata.distance
        results.append(item)
    return results
