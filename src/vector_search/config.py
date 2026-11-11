"""Validated, portable configuration from environment variables.

Every setting has a safe default for local development; Docker Compose overrides
them (see ``.env.example``). Nothing here connects to a service at import time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit


def quiet_third_party_warnings() -> None:
    """Hide a noisy deprecation warning that weaviate-client's authlib dependency emits."""
    import warnings
    try:  # authlib installs an "always" filter on import; import it first, then override
        import authlib.deprecate  # noqa: F401
    except ImportError:
        pass
    warnings.filterwarnings('ignore', message='The httpx module is deprecated')


def _boolean(value: str, name: str) -> bool:
    if value.lower() in {'true', '1', 'yes'}:
        return True
    if value.lower() in {'false', '0', 'no'}:
        return False
    raise ValueError(f'{name} must be true or false')


@dataclass(frozen=True)
class Settings:
    parsed_dir: Path = Path('dataset/parsed')
    image_dir: Path = Path('data/images')
    cache_dir: Path = Path('data/models')
    reports_dir: Path = Path('data/reports')
    collection: str = 'ProductV1'
    weaviate_host: str = 'localhost'
    weaviate_http_port: int = 8080
    weaviate_grpc_host: str = 'localhost'
    weaviate_grpc_port: int = 50051
    weaviate_secure: bool = False
    weaviate_api_key: str = field(default='', repr=False)
    chat_backend: str = 'grounded'
    llm_url: str = 'http://localhost:8081/v1'
    llm_model: str = 'qwen2.5-1.5b-instruct'
    llm_api_key: str = field(default='', repr=False)
    text_model: str = 'SajjadAyoubi/clip-fa-text'
    image_model: str = 'SajjadAyoubi/clip-fa-vision'
    text_revision: str = 'ad048ce7b7e62cebc7fdbe96f9e745f79cb6823d'
    image_revision: str = 'b78ef94830d17dbb031636b01c4f9cf89b968ed0'
    device: str = 'cpu'
    max_image_bytes: int = 15 * 1024 * 1024
    ingest_batch_size: int = 32
    download_workers: int = 8

    def __post_init__(self) -> None:
        for name in ('weaviate_http_port', 'weaviate_grpc_port'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f'{name} must be an integer between 1 and 65535')
        if not re.fullmatch(r'[A-Z][A-Za-z0-9_]*', self.collection):
            raise ValueError('WEAVIATE_COLLECTION must begin with an uppercase letter and use letters, digits, underscores')
        if self.chat_backend not in {'grounded', 'llm'}:
            raise ValueError('CHAT_BACKEND must be grounded or llm')
        if not isinstance(self.max_image_bytes, int) or isinstance(self.max_image_bytes, bool) or self.max_image_bytes < 1:
            raise ValueError('MAX_IMAGE_BYTES must be positive')
        url = urlsplit(self.llm_url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password:
            raise ValueError('LLM_URL must be an HTTP(S) URL without embedded credentials')
        if not self.weaviate_host or not self.weaviate_grpc_host:
            raise ValueError('Weaviate hostnames cannot be empty')
        for name, upper in (('ingest_batch_size', 256), ('download_workers', 64)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f'{name.upper()} must be an integer between 1 and {upper}')
        if self.device not in {'cpu', 'cuda'} and not self.device.startswith('cuda:'):
            raise ValueError('EMBEDDING_DEVICE must be cpu or cuda[:N]')

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        overrides = {'collection': 'WEAVIATE_COLLECTION', 'text_revision': 'TEXT_MODEL_REVISION', 'image_revision': 'IMAGE_MODEL_REVISION', 'device': 'EMBEDDING_DEVICE'}
        defaults = cls()
        values = {}
        for item in fields(cls):
            key = overrides.get(item.name, item.name.upper())
            if key not in env:
                continue
            value = env[key]
            baseline = getattr(defaults, item.name)
            if isinstance(baseline, Path):
                values[item.name] = Path(value).expanduser()
            elif isinstance(baseline, bool):
                values[item.name] = _boolean(value.strip(), key)
            elif isinstance(baseline, int):
                try:
                    values[item.name] = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{key} must be an integer') from exc
            else:
                values[item.name] = value
        return cls(**values)
