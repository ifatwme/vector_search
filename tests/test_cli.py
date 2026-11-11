"""CLI parsing and the evaluation metrics, offline."""
import pytest

from vector_search.cli import build_parser, parse_shards
from vector_search.evaluation import _metrics, load_eval_set


def test_parse_shards_ranges_and_all():
    assert parse_shards('0-2,143') == ['products_0_parsed.json', 'products_1_parsed.json',
                                       'products_2_parsed.json', 'products_143_parsed.json']
    assert parse_shards('all') is None
    assert parse_shards(None) is None


def test_parser_requires_exactly_one_search_input():
    parser = build_parser()
    assert parser.parse_args(['search', '--text', 'کفش']).text == 'کفش'
    with pytest.raises(SystemExit):
        parser.parse_args(['search', '--text', 'a', '--image', 'b.jpg'])
    args = parser.parse_args(['ingest', '--limit', '10', '--shards', '0-1'])
    assert (args.limit, args.shards) == (10, '0-1')


def test_metrics():
    m = _metrics([1, 3, None, 12], ks=(1, 5, 10))
    assert m == {'recall@1': 0.25, 'recall@5': 0.5, 'recall@10': 0.5,
                 'mrr': round((1 + 1 / 3 + 1 / 12) / 4, 3), 'queries': 4}


def test_bundled_eval_set_is_complete():
    from pathlib import Path
    items = load_eval_set(Path(__file__).resolve().parents[1] / 'dataset' / 'eval')
    assert len(items) == 100
    assert all(item['image'] is not None and item['title'] for item in items)


def test_evaluate_ranks_each_route_and_closes_nothing_it_borrowed(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    import vector_search.evaluation as evaluation
    for sub in ('titles', 'images'):
        (tmp_path / sub).mkdir()
    for n, pid in ((1, 11), (2, 22)):
        (tmp_path / 'titles' / f'{n}_{pid}.txt').write_text(f'کالا {pid}', encoding='utf-8')
        (tmp_path / 'images' / f'{n}_{pid}.jpg').write_bytes(b'jpeg')
    closed = []
    collection = NS(
        query=NS(fetch_objects=lambda **k: NS(objects=[NS(properties={'pid': 11}),
                                                       NS(properties={'pid': 22})])),
        aggregate=NS(over_all=lambda **k: NS(total_count=2)))
    monkeypatch.setattr('vector_search.database.ensure_collection', lambda c, s, create: collection)
    monkeypatch.setattr(evaluation, '_pid_filter', lambda pids: None)
    def fake_search(client, settings, embedder, *, target, limit, text=None, image=None):
        return [{'pid': 11}, {'pid': 22}] if target == 'text' else [{'pid': 99}]
    monkeypatch.setattr('vector_search.search.search', fake_search)
    client = NS(close=lambda: closed.append(True))
    report = evaluation.evaluate(NS(), tmp_path, ks=(1, 5), client=client, embedder=object())
    assert report['index_size'] == 2 and report['indexed_eval_items'] == 2
    assert report['routes']['text->text'] == {'recall@1': 0.5, 'recall@5': 1.0, 'mrr': 0.75, 'queries': 2}
    assert report['routes']['image->image']['recall@5'] == 0.0
    assert closed == []
