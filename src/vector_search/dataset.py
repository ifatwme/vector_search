"""Read-only validation, deterministic ordering, and whole-dataset quarantine."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path


def _canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _reason(row: object) -> str | None:
    if not isinstance(row, dict):
        return 'record must be an object'
    if type(row.get('id')) is not int or row['id'] <= 0:
        return 'id must be a positive integer'
    for field in ('title_fa', 'title_en', 'url', 'image_main'):
        if not isinstance(row.get(field), str):
            return f'{field} must be text'
    if not (row['title_fa'].strip() or row['title_en'].strip()):
        return 'product title is empty'
    for field in ('description', 'short_review', 'brand_fa', 'brand_en'):
        if field in row and not isinstance(row[field], str):
            return f'{field} must be text'
    for field in ('properties', 'attributes', 'seller_info'):
        if field in row and not isinstance(row[field], dict):
            return f'{field} must be an object'
    for field in ('image_list', 'advantages', 'disadvantages'):
        if field in row and (not isinstance(row[field], list) or
                             not all(isinstance(v, str) for v in row[field])):
            return f'{field} must be a list of text'
    try:
        _canonical(row)
    except (ValueError, TypeError):
        return 'record is not finite JSON'
    return None


def _order(path: Path):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r'(\d+)', path.name))


def _load(path: Path) -> tuple[dict, dict[int, dict], dict[int, str]]:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Dataset directory does not exist: {path}')
    shards = sorted(path.glob('*.json'), key=_order)
    if not shards:
        raise ValueError(f'Dataset directory contains no JSON shards: {path}')
    report = dict(shards=len(shards), records=0, unique_ids=0, exact_duplicate_rows=0,
                  conflicting_ids=0, invalid_records=0, conflict_details=[], invalid_details=[])
    products: dict[int, dict] = {}
    origin: dict[int, str] = {}
    versions: dict[int, set[str]] = {}
    locations: dict[int, list[dict]] = {}
    for shard in shards:
        try:
            rows = json.loads(shard.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read dataset shard {shard.name}: {type(exc).__name__}') from None
        if not isinstance(rows, list):
            raise ValueError(f'Dataset shard {shard.name} must contain a JSON array')
        for index, row in enumerate(rows):
            report['records'] += 1
            location = dict(shard=shard.name, index=index)
            reason = _reason(row)
            if reason:
                report['invalid_records'] += 1
                report['invalid_details'].append(dict(**location, reason=reason))
                continue
            pid = row['id']
            digest = hashlib.sha256(_canonical(row).encode()).hexdigest()
            locations.setdefault(pid, []).append(location)
            if pid not in versions:
                versions[pid] = {digest}
                products[pid] = row
                origin[pid] = shard.name
            elif digest in versions[pid]:
                report['exact_duplicate_rows'] += 1
            else:
                versions[pid].add(digest)
    report['unique_ids'] = len(versions)
    for pid, digests in versions.items():
        if len(digests) > 1:
            products.pop(pid, None)
            report['conflict_details'].append(dict(id=pid, versions=len(digests), locations=locations[pid]))
    report['conflicting_ids'] = len(report['conflict_details'])
    report['valid_products'] = len(products)
    return report, products, origin


def scan_dataset(path: Path) -> dict:
    """Validate every row; invalid shards fail rather than silently disappear."""
    return _load(path)[0]


def shard_name(number: int) -> str:
    return f'products_{number}_parsed.json'


def iter_products(path: Path, *, limit: int | None = None,
                  shards: Iterable[str] | None = None) -> Iterator[dict]:
    """Check all shards before yielding, even when a sample limit is requested.

    ``shards`` restricts the output to products whose first occurrence is in one of
    the named shard files; duplicates/conflicts are still judged dataset-wide.
    """
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be a nonnegative integer or None')
    _, products, origin = _load(path)
    wanted = None if shards is None else set(shards)
    if wanted is not None:
        missing = wanted - set(origin.values()) - {p.name for p in Path(path).glob('*.json')}
        if missing:
            raise ValueError(f'Unknown dataset shards: {sorted(missing)}')
    count = 0
    for pid, product in products.items():
        if wanted is not None and origin[pid] not in wanted:
            continue
        if limit is not None and count >= limit:
            break
        count += 1
        yield product
