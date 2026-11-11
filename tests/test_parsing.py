import copy
import json

import pytest

from vector_search.parsing import parse_data, parse_file


def raw():
    return {'product': {'id': 12, 'title_fa': 'محصول', 'title_en': '',
                        'specifications': [{'attributes': [{'title': 'جنس', 'values': ['فلز', 'چوب']}]}],
                        'images': {'main': {'url': []}, 'list': [{'url': ['https://example/image.jpg']}]},
                        'variants': [], 'expert_reviews': {}, 'brand': {}},
            'seo': {'open_graph': {'url': 'https://example/product/12'}}}


def test_missing_review_does_not_hide_specs_or_crash_empty_main_image():
    data = raw()
    before = copy.deepcopy(data)
    result = parse_data(data)
    assert result['attributes'] == {'جنس': 'فلز و چوب'}
    assert result['image_main'] == 'https://example/image.jpg'
    assert result['image_list'] == ['https://example/image.jpg']
    assert data == before


def test_malformed_variant_isolated_counts_real_sellers_and_preserves_pros_lists():
    data = raw()
    data['product']['variants'] = [
        {'seller': {'id': 1, 'title': 'one'}, 'price': {'selling_price': 12}},
        {'seller': None},
        {'seller': {'id': 2, 'title': 'two'}, 'price': {'selling_price': 24}},
    ]
    data['product']['pros_and_cons'] = {'advantages': ['خوب'], 'disadvantages': {'0': 'بد'}}
    result = parse_data(data)
    assert result['seller_count'] == 2
    assert result['seller_info'][1]['selling_price'] == 12
    assert result['seller_info'][2]['seller_name'] == 'two'
    assert result['advantages'] == ['خوب']
    assert result['disadvantages'] == ['بد']


def test_parse_file_counts_only_true_inactive_and_current_rejections(tmp_path):
    active = raw()
    active['product']['is_inactive'] = False
    inactive = raw()
    inactive['product']['is_inactive'] = True
    malformed = {'product': {'id': 99}}
    source, target = tmp_path / 'raw.json', tmp_path / 'parsed.json'
    source.write_text(json.dumps([{'data': active}, {'data': inactive}, {'data': malformed},
                                  {'data': raw()}]))
    original = source.read_bytes()
    report = parse_file(source, target)
    assert (report['parsed'], report['inactive'], report['rejected']) == (2, 1, 1)
    assert report['errors'][0]['index'] == 2
    assert report['errors'][0]['id'] == 99
    assert len(json.loads(target.read_text())) == 2
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match='source'):
        parse_file(source, source)

