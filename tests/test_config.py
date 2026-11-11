from pathlib import Path

import pytest


def settings(**env):
    from vector_search.config import Settings

    return Settings.from_env(env)


def test_defaults_are_portable_and_optional_chat_is_disabled():
    conf = settings()
    assert conf.parsed_dir == Path('dataset/parsed')
    assert conf.image_dir == Path('data/images')
    assert conf.device == 'cpu'
    assert conf.chat_backend == 'grounded'
    assert conf.llm_url.endswith('/v1')


def test_environment_parses_paths_ports_and_boolean_strings():
    conf = settings(IMAGE_DIR='/tmp/images', WEAVIATE_HTTP_PORT='18080', WEAVIATE_SECURE='true', DOWNLOAD_WORKERS='4')
    assert conf.image_dir == Path('/tmp/images')
    assert conf.weaviate_http_port == 18080
    assert conf.weaviate_secure is True
    assert conf.download_workers == 4


@pytest.mark.parametrize('env', [{'WEAVIATE_HTTP_PORT':'0'}, {'WEAVIATE_GRPC_PORT':'65536'}, {'CHAT_BACKEND':'gemini'}, {'MAX_IMAGE_BYTES':'-1'}, {'WEAVIATE_SECURE':'perhaps'}, {'INGEST_BATCH_SIZE':'0'}, {'EMBEDDING_DEVICE':'tpu'}, {'LLM_URL':'http://user:pw@host'}, {'LLM_URL':'ftp://host'}, {'WEAVIATE_COLLECTION':'bad-name'}])
def test_invalid_environment_is_rejected(env):
    with pytest.raises(ValueError):
        settings(**env)


def test_secret_fields_are_not_exposed_by_repr():
    conf = settings(WEAVIATE_API_KEY='private-db', LLM_API_KEY='private-llm')
    assert 'private-db' not in repr(conf) and 'private-llm' not in repr(conf)


def test_pinned_model_revisions_have_full_commit_ids():
    conf = settings()
    assert len(conf.text_revision) == len(conf.image_revision) == 40
