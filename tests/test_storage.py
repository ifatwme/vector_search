import io
from dataclasses import replace

import pytest
from PIL import Image

from vector_search.config import Settings


def image_bytes(format='PNG'):
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(buf, format=format)
    return buf.getvalue()


class Response:
    status_code = 200
    headers = {}

    def iter_content(self, chunk_size):
        yield image_bytes()

    def raise_for_status(self):
        pass

    def close(self):
        pass


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()


def test_png_conversion_is_cached_as_jpeg_without_another_download(tmp_path):
    from vector_search.storage import ImageStore

    session = Session()
    store = ImageStore(replace(Settings(), image_dir=tmp_path), session=session)
    ref = store.get_or_download('https://dkstatics-public.digikala.com/image.png', 42)
    assert ref == '42/42_0.jpg'
    assert Image.open(io.BytesIO(store.read(ref))).format == 'JPEG'
    assert store.get_or_download('https://dkstatics-public.digikala.com/image.png', 42) == ref
    assert len(session.calls) == 1
    assert session.calls[0][1]['timeout']
    assert session.calls[0][1]['allow_redirects'] is False


@pytest.mark.parametrize('ref', ['../../etc/passwd', '/etc/passwd', '42/../x.jpg', '42/41_0.jpg'])
def test_read_rejects_path_traversal_and_inconsistent_identity(tmp_path, ref):
    from vector_search.storage import ImageStore

    with pytest.raises(ValueError):
        ImageStore(replace(Settings(), image_dir=tmp_path)).read(ref)


@pytest.mark.parametrize('url', ['http://dkstatics-public.digikala.com/x', 'https://127.0.0.1/x', 'https://evil.example/x', 'file:///etc/passwd', 'https://u:p@dkstatics-public.digikala.com/x'])
def test_untrusted_image_url_is_rejected_before_network(tmp_path, url):
    from vector_search.storage import ImageStore

    session = Session()
    with pytest.raises(ValueError):
        ImageStore(replace(Settings(), image_dir=tmp_path), session=session).get_or_download(url, 42)
    assert not session.calls


def test_redirect_is_rejected_and_no_image_is_cached(tmp_path):
    from vector_search.storage import ImageStore

    class Redirect(Response):
        status_code = 302

    session = Session()
    session.get = lambda *a, **k: Redirect()
    with pytest.raises(ValueError):
        ImageStore(replace(Settings(), image_dir=tmp_path), session=session).get_or_download('https://dkstatics-public.digikala.com/x', 42)
    assert not list(tmp_path.rglob('*.jpg'))


def test_streamed_download_stops_at_configured_byte_limit(tmp_path):
    from vector_search.storage import ImageStore

    consumed = []

    class Large(Response):
        def iter_content(self, chunk_size):
            for _ in range(5):
                consumed.append(1)
                yield b'x' * 1024

    session = Session()
    session.get = lambda *a, **k: Large()
    with pytest.raises(ValueError):
        ImageStore(replace(Settings(), image_dir=tmp_path, max_image_bytes=1500), session=session).get_or_download('https://dkstatics-public.digikala.com/x', 42)
    assert len(consumed) == 2
    assert not list(tmp_path.rglob('*.part'))

