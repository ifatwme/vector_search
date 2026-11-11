"""Lazy, pinned CLIPfa encoders using the published 768-D pooler space."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import warnings
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

DIMENSIONS = 768
# Product titles are short; capping keeps batched padding cheap on CPU.
MAX_TEXT_TOKENS = 128
PREPROCESSING = 'clipfa-pooler-v1:tokenizer-truncation-128-padding;clip-image-processor-rgb;unnormalized'


def validate_vector(vector) -> list[float]:
    """Reject incompatible outputs instead of silently changing embedding space."""
    if not isinstance(vector, (list, tuple)) or len(vector) != DIMENSIONS:
        raise ValueError(f'Embedding vector must have {DIMENSIONS} dimensions')
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) for value in vector):
        raise ValueError('Embedding vector must contain finite numbers')
    if not any(value != 0 for value in vector):
        raise ValueError('Embedding vector must be nonzero for cosine distance')
    return [float(value) for value in vector]


class Embedder:
    """No model imports, downloads or device allocations until explicit encode."""

    def __init__(self, settings: Settings):
        for revision in (settings.text_revision, settings.image_revision):
            if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
                raise ValueError('Model revision must be an immutable 40-character commit SHA')
        self.settings = settings
        self._text_model = self._image_model = None
        self._tokenizer = self._processor = None
        self._lock = RLock()

    @property
    def fingerprint(self) -> str:
        spec = dict(text_model=self.settings.text_model, image_model=self.settings.image_model,
                    text_revision=self.settings.text_revision, image_revision=self.settings.image_revision,
                    dimensions=DIMENSIONS, preprocessing=PREPROCESSING)
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return f'clipfa-pooler-768-v1:{digest}'

    def _load_text(self):
        if self._text_model is None:
            from transformers import AutoTokenizer, RobertaModel
            kwargs = dict(revision=self.settings.text_revision, cache_dir=str(self.settings.cache_dir),
                          trust_remote_code=False)
            tokenizer = AutoTokenizer.from_pretrained(self.settings.text_model, **kwargs)
            model = RobertaModel.from_pretrained(self.settings.text_model, **kwargs)
            model.to(self.settings.device).eval()
            self._tokenizer, self._text_model = tokenizer, model

    def _load_image(self):
        if self._image_model is None:
            from transformers import CLIPImageProcessor, CLIPVisionModel
            kwargs = dict(revision=self.settings.image_revision, cache_dir=str(self.settings.cache_dir),
                          trust_remote_code=False)
            processor = CLIPImageProcessor.from_pretrained(self.settings.image_model, **kwargs)
            model = CLIPVisionModel.from_pretrained(self.settings.image_model, **kwargs)
            model.to(self.settings.device).eval()
            self._processor, self._image_model = processor, model

    def _forward(self, model, inputs) -> list[list[float]]:
        import torch
        inputs = {key: tensor.to(self.settings.device) for key, tensor in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            rows = output.pooler_output.detach().cpu().tolist()
        return [validate_vector(row) for row in rows]

    def _encode(self, model, inputs) -> list[float]:
        rows = self._forward(model, inputs)
        if len(rows) != 1:
            raise ValueError('Expected exactly one embedding')
        return rows[0]

    @staticmethod
    def _check_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Embedding text must be nonempty')
        return text

    def encode_text(self, text: str) -> list[float]:
        self._check_text(text)
        with self._lock:
            self._load_text()
            inputs = self._tokenizer(text, return_tensors='pt', truncation=True, padding=True,
                                     max_length=MAX_TEXT_TOKENS)
            return self._encode(self._text_model, inputs)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Batched variant for ingestion; same pooler space as ``encode_text``."""
        texts = [self._check_text(text) for text in texts]
        if not texts:
            return []
        with self._lock:
            self._load_text()
            inputs = self._tokenizer(texts, return_tensors='pt', truncation=True, padding=True,
                                     max_length=MAX_TEXT_TOKENS)
            return self._forward(self._text_model, inputs)

    def _decode(self, image: bytes | Path):
        from PIL import Image, UnidentifiedImageError
        if isinstance(image, Path):
            if image.stat().st_size > self.settings.max_image_bytes:
                raise ValueError('Image exceeds configured byte limit')
            data = image.read_bytes()
        elif isinstance(image, bytes):
            data = image
        else:
            raise ValueError('Image must be bytes or a pathlib.Path')
        if not data or len(data) > self.settings.max_image_bytes:
            raise ValueError('Image is empty or exceeds configured byte limit')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    return source.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            raise ValueError('Image cannot be safely decoded') from None

    def encode_image(self, image: bytes | Path) -> list[float]:
        rgb = self._decode(image)
        try:
            with self._lock:
                self._load_image()
                inputs = self._processor(images=rgb, return_tensors='pt')
                return self._encode(self._image_model, inputs)
        finally:
            rgb.close()

    def encode_images(self, images: list[bytes | Path]) -> list[list[float]]:
        decoded = [self._decode(image) for image in images]
        if not decoded:
            return []
        try:
            with self._lock:
                self._load_image()
                inputs = self._processor(images=decoded, return_tensors='pt')
                return self._forward(self._image_model, inputs)
        finally:
            for rgb in decoded:
                rgb.close()
