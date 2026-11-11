"""Product image cache on the local filesystem (a Docker volume in production).

Only the trusted Digikala CDN host is contacted, downloads are size-bounded and
normalised to JPEG, and files appear atomically so readers never see partial data.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image

from vector_search.config import Settings


def _key(ref: str) -> str:
    match = re.fullmatch(r'([1-9][0-9]*)/\1_0\.jpg', ref)
    if not match:
        raise ValueError('Invalid product image reference')
    return ref


def _jpeg(data: bytes, max_bytes: int) -> bytes:
    if len(data) > max_bytes:
        raise ValueError('Image exceeds MAX_IMAGE_BYTES')
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError('Image pixel count is too large')
        image.load()
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG', quality=90)
        result = buffer.getvalue()
    if len(result) > max_bytes:
        raise ValueError('Converted image exceeds MAX_IMAGE_BYTES')
    return result


# Digikala's CDN; ~0.1% of products use signed URLs on the private host.
TRUSTED_IMAGE_HOSTS = frozenset({'dkstatics-public.digikala.com', 'dkstatics-private.digikala.com'})


def image_ref(product_id: int) -> str:
    """Stable, storage-relative reference for a product's main image."""
    if isinstance(product_id, bool) or not isinstance(product_id, int) or product_id < 1:
        raise ValueError('Product ID must be a positive integer')
    return f'{product_id}/{product_id}_0.jpg'


class ImageStore:
    def __init__(self, settings: Settings, *, session=None):
        self.settings = settings
        self._session = session

    def exists(self, ref: str) -> bool:
        return (self.settings.image_dir / _key(ref)).is_file()

    def read(self, ref: str) -> bytes:
        path = self.settings.image_dir / _key(ref)
        if path.stat().st_size > self.settings.max_image_bytes:
            raise ValueError('Cached image exceeds MAX_IMAGE_BYTES')
        data = path.read_bytes()
        # Files are written only after JPEG normalisation (atomic rename), so a cheap
        # signature check is enough to catch truncated or foreign files.
        if not (data.startswith(b'\xff\xd8') and data.rstrip(b'\0').endswith(b'\xff\xd9')):
            raise ValueError('Cached image is not a complete JPEG')
        return data

    def get_or_download(self, url: str, product_id: int) -> str:
        ref = _key(image_ref(product_id))
        path = self.settings.image_dir / ref
        if path.is_file():
            try:
                self.read(ref)
                return ref
            except (ValueError, OSError):
                pass
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in TRUSTED_IMAGE_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError('Image URL must use the trusted HTTPS catalog image host')
        session = self._session or requests
        response = session.get(url, stream=True, timeout=(5, 30), allow_redirects=False)
        try:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError('Image server did not return a direct successful response')
            buffer = bytearray()
            for chunk in response.iter_content(64 * 1024):
                if len(buffer) + len(chunk) > self.settings.max_image_bytes:
                    raise ValueError('Image exceeds MAX_IMAGE_BYTES')
                buffer.extend(chunk)
            data = _jpeg(bytes(buffer), self.settings.max_image_bytes)
        finally:
            response.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.part')
        try:
            with os.fdopen(fd, 'wb') as target:
                target.write(data)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
        return ref
