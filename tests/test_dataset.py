import json

from vector_search.dataset import iter_products, scan_dataset


def product(pid=1, **changes):
    return dict(id=pid, title_fa='محصول', title_en='', url='https://www.digikala.com/product/1',
                image_main='https://dkstatics-public.digikala.com/image.jpg',
                attributes={}, seller_info={}, description='', **changes)


def shard(path, name, rows):
    (path / name).write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')


def test_duplicates_and_all_conflict_versions_quarantined_before_limit(tmp_path):
    shard(tmp_path, 'products_2.json', [product(1), product(2), product(2), {'id': True}])
    changed = product(1)
    changed['title_fa'] = 'conflicting title'
    shard(tmp_path, 'products_10.json', [changed, product(3)])
    report = scan_dataset(tmp_path)
    assert {k: report[k] for k in ('shards', 'records', 'unique_ids', 'exact_duplicate_rows',
                                  'conflicting_ids', 'invalid_records')} == {
        'shards': 2, 'records': 6, 'unique_ids': 3, 'exact_duplicate_rows': 1,
        'conflicting_ids': 1, 'invalid_records': 1}
    assert report['conflict_details'][0]['id'] == 1
    assert len(report['conflict_details'][0]['locations']) == 2
    assert [p['id'] for p in iter_products(tmp_path, limit=1)] == [2]
    assert [p['id'] for p in iter_products(tmp_path)] == [2, 3]
    json.dumps(report)
