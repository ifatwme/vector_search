"""Fault-isolated extraction from Digikala responses; never edits source data."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _text(value) -> str:
    return value if isinstance(value, str) else ''


def _image(value) -> str:
    value = _object(value).get('url')
    if isinstance(value, str):
        return value
    return next((v for v in _list(value) if isinstance(v, str) and v), '')


def parse_data(data: dict) -> dict:
    """Extract independent optional sections without a broad exception boundary."""
    product = _object(_object(data).get('product'))
    pid = product.get('id')
    if type(pid) is not int or pid <= 0:
        raise ValueError('Product requires a positive integer id')
    title_fa, title_en = (_text(product.get(k)) for k in ('title_fa', 'title_en'))
    if not (title_fa.strip() or title_en.strip()):
        raise ValueError('Product requires a title')
    images = _object(product.get('images'))
    image_list = [url for image in _list(images.get('list')) if (url := _image(image))]
    main = _image(images.get('main')) or next(iter(image_list), '')
    review = _object(product.get('review'))
    sections = [review] + [_object(v) for v in _list(product.get('specifications'))]
    attributes = {}
    for section in sections:
        for item in _list(section.get('attributes')):
            item = _object(item)
            title = _text(item.get('title'))
            values = item.get('values')
            value = values if isinstance(values, str) else ' و '.join(v for v in _list(values) if isinstance(v, str))
            if title and value and title not in attributes:
                attributes[title] = value
    sellers = {}
    for variant in _list(product.get('variants')):
        variant = _object(variant)
        seller = _object(variant.get('seller'))
        price = _object(variant.get('price'))
        sid = seller.get('id')
        if type(sid) is not int or sid <= 0 or not price:
            continue
        sellers[sid] = dict(seller_name=_text(seller.get('title')),
                            **{key: price.get(key) for key in ('selling_price', 'rrp_price', 'discount_percent')})
    pros = _object(product.get('pros_and_cons'))
    def points(key):
        value = pros.get(key)
        values = value.values() if isinstance(value, dict) else _list(value)
        return [v for v in values if isinstance(v, str)]
    brand = _object(product.get('brand'))
    expert = _object(product.get('expert_reviews'))
    return dict(url=_text(_object(_object(_object(data).get('seo')).get('open_graph')).get('url')),
                id=pid, title_fa=title_fa, title_en=title_en,
                properties=_object(product.get('properties')).copy(), attributes=attributes,
                seller_count=len(sellers), seller_info=sellers, image_main=main, image_list=image_list,
                brand_id=brand.get('id'), brand_en=_text(brand.get('title_en')),
                brand_fa=_text(brand.get('title_fa')), description=_text(expert.get('description')),
                short_review=_text(expert.get('short_review')), advantages=points('advantages'), disadvantages=points('disadvantages'))


def parse_file(source: Path, target: Path) -> dict:
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('Output target must differ from source')
    rows = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Source must contain a JSON array')
    report = dict(parsed=0, inactive=0, rejected=0, errors=[])
    products = []
    for index, row in enumerate(rows):
        data = _object(row).get('data', row)
        product = _object(_object(data).get('product'))
        if product.get('is_inactive') is True:
            report['inactive'] += 1
            continue
        try:
            products.append(parse_data(data))
            report['parsed'] += 1
        except (ValueError, TypeError) as exc:
            report['rejected'] += 1
            report['errors'].append(dict(index=index, id=product.get('id'), reason=str(exc)))
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, delete=False) as handle:
            name = handle.name
            json.dump(products, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
    return report
