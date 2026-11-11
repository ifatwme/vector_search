import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vector_search.embeddings import Embedder


def test_lazy_pinned_eval_inference_pooler_output_is_deterministic(monkeypatch, tmp_path):
    import transformers
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.9)
        def forward(self, **kwargs):
            assert not self.training
            assert torch.is_inference_mode_enabled()
            return SimpleNamespace(pooler_output=self.dropout(torch.arange(1., 769.).unsqueeze(0)))

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return Model()

    class Processor:
        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.ones(1, 3, dtype=torch.long)}

    monkeypatch.setattr(transformers.RobertaModel, 'from_pretrained', load)
    monkeypatch.setattr(transformers.CLIPVisionModel, 'from_pretrained', load)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Processor())
    monkeypatch.setattr(transformers.CLIPImageProcessor, 'from_pretrained', lambda *a, **kw: Processor())
    settings = SimpleNamespace(cache_dir=tmp_path, text_model='SajjadAyoubi/clip-fa-text',
        image_model='SajjadAyoubi/clip-fa-vision', text_revision='a'*40, image_revision='b'*40,
        device='cpu', max_image_bytes=15000000)
    embedder = Embedder(settings)
    assert embedder.fingerprint == Embedder(settings).fingerprint
    assert calls == []
    expected = list(range(1, 769))
    assert embedder.encode_text('سلام') == expected
    assert embedder.encode_text('سلام') == expected
    image = io.BytesIO()
    Image.new('RGB', (8, 8)).save(image, format='PNG')
    assert embedder.encode_image(image.getvalue()) == expected
    assert len(calls) == 2
    assert {c[1]['revision'] for c in calls} == {settings.text_revision, settings.image_revision}
    assert all(c[1]['cache_dir'] == str(tmp_path) for c in calls)


def settings(**overrides):
    return SimpleNamespace(**dict(dict(cache_dir=Path('/tmp/no-models'), text_model='text',
        image_model='vision', text_revision='a'*40, image_revision='b'*40,
        device='cpu', max_image_bytes=10000), **overrides))


@pytest.mark.parametrize('output', [[1.]*767, [float('nan')]*768, [float('inf')]*768, [0.]*768])
def test_rejects_wrong_or_nonfinite_or_zero_pooler_vectors(output):
    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(pooler_output=torch.tensor([output]))
    embedder = Embedder(settings())
    embedder._text_model = Model().eval()
    embedder._tokenizer = lambda *a, **kw: {'input_ids': torch.ones(1, 1)}
    with pytest.raises(ValueError, match='vector'):
        embedder.encode_text('x')


@pytest.mark.parametrize('text', ['', '   ', None, 1])
def test_invalid_text_rejected_before_loading(text):
    with pytest.raises(ValueError, match='text'):
        Embedder(settings()).encode_text(text)


@pytest.mark.parametrize('image', [b'', b'not an image', b'x'*10001, None])
def test_invalid_images_rejected_before_loading(image):
    with pytest.raises(ValueError, match='[Ii]mage'):
        Embedder(settings()).encode_image(image)


def test_requires_immutable_revisions_and_fingerprint_covers_each_configuration():
    with pytest.raises(ValueError, match='revision'):
        Embedder(settings(text_revision='main'))
    baseline = Embedder(settings()).fingerprint
    for key, value in [('text_model', 'other'), ('image_model', 'other'),
                       ('text_revision', 'c'*40), ('image_revision', 'd'*40)]:
        assert Embedder(settings(**{key: value})).fingerprint != baseline



def test_batched_encoding_returns_one_vector_per_input(monkeypatch):
    import transformers
    seen = {}

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            rows = next(iter(kwargs.values())).shape[0]
            return SimpleNamespace(pooler_output=torch.ones(rows, 768))

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            seen['max_length'] = kwargs.get('max_length')
            return {'input_ids': torch.ones(len(texts) if isinstance(texts, list) else 1, 3, dtype=torch.long)}

    class Processor:
        def __call__(self, images, **kwargs):
            return {'pixel_values': torch.ones(len(images) if isinstance(images, list) else 1, 3, 2, 2)}

    monkeypatch.setattr(transformers.RobertaModel, 'from_pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(transformers.CLIPVisionModel, 'from_pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.CLIPImageProcessor, 'from_pretrained', lambda *a, **k: Processor())
    embedder = Embedder(settings(max_image_bytes=10**6))
    assert len(embedder.encode_texts(['a', 'b', 'c'])) == 3
    assert seen['max_length'] == 128
    png = io.BytesIO()
    Image.new('RGB', (4, 4)).save(png, format='PNG')
    assert len(embedder.encode_images([png.getvalue()] * 2)) == 2
    assert embedder.encode_texts([]) == []
    with pytest.raises(ValueError):
        embedder.encode_images([png.getvalue(), b'not an image'])
