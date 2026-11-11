"""Command line interface: ``vector-search <command>`` (also ``python -m vector_search``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings, quiet_third_party_warnings
from .dataset import shard_name


def parse_shards(spec: str | None) -> list[str] | None:
    """``"0-4,143"`` -> shard file names; ``None``/``"all"`` -> every shard."""
    if not spec or spec == 'all':
        return None
    names: list[str] = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            low, high = (int(x) for x in part.split('-', 1))
            if low > high:
                raise argparse.ArgumentTypeError(f'invalid shard range {part}')
            names.extend(shard_name(n) for n in range(low, high + 1))
        elif part:
            names.append(shard_name(int(part)))
    return names


def _print(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_scan(settings: Settings, args) -> int:
    from .dataset import scan_dataset
    report = scan_dataset(settings.parsed_dir)
    if not args.details:
        report.pop('conflict_details')
        report.pop('invalid_details')
    _print(report)
    return 0


def cmd_ingest(settings: Settings, args) -> int:
    from .ingestion import ingest
    report = ingest(settings, limit=args.limit, shards=parse_shards(args.shards))
    summary = {k: v for k, v in report.items() if k != 'failures'}
    summary['first_failures'] = report['failures'][:5]
    _print(summary)
    return 0 if report['confirmed'] + report['unchanged'] > 0 or not report['selected'] else 1


def cmd_search(settings: Settings, args) -> int:
    from .database import connect, ensure_collection
    from .embeddings import Embedder
    from .search import search
    image = Path(args.image).read_bytes() if args.image else None
    client = connect(settings)
    try:
        ensure_collection(client, settings, create=False)
        results = search(client, settings, Embedder(settings), text=args.text, image=image,
                         target=args.target, limit=args.limit)
    finally:
        client.close()
    for item in results:
        item['description'] = item['description'][:160]
        item.pop('attributes', None)
    _print(results)
    return 0


def cmd_ask(settings: Settings, args) -> int:
    """Search, then ask the configured assistant about the results (end-to-end check)."""
    from .chat import ChatService
    from .database import connect, ensure_collection
    from .embeddings import Embedder
    from .search import search
    client = connect(settings)
    try:
        ensure_collection(client, settings, create=False)
        results = search(client, settings, Embedder(settings), text=args.query, limit=args.limit)
    finally:
        client.close()
    messages = [{'role': 'user', 'content': args.question}]
    print('products:', [r['title'] for r in results])
    print(ChatService(settings).answer(messages, results, max_products=min(5, args.limit)))
    return 0


def cmd_evaluate(settings: Settings, args) -> int:
    from .evaluation import evaluate
    report = evaluate(settings, Path(args.eval_dir), ks=(1, 5, 10))
    _print(report)
    return 0


def cmd_export(settings: Settings, args) -> int:
    from .snapshot import export_images, export_snapshot
    out = Path(args.out)
    report = {'snapshot': export_snapshot(settings, out / 'snapshot.jsonl.gz')}
    if not args.no_images:
        report['images'] = export_images(settings.image_dir, out / 'images.tar',
                                         max_side=args.max_side)
    _print(report)
    return 0


def cmd_restore(settings: Settings, args) -> int:
    from .snapshot import extract_images, import_snapshot
    source = Path(args.source)
    report = import_snapshot(settings, source / 'snapshot.jsonl.gz')
    if (source / 'images.tar').is_file():
        report['images'] = extract_images(source / 'images.tar', settings.image_dir)
    _print(report)
    return 0


def cmd_doctor(settings: Settings, args) -> int:
    """Check every dependency the app needs and report what is wrong."""
    import requests
    checks: dict[str, str] = {}
    try:
        from .database import connect, ensure_collection
        client = connect(settings)
        try:
            collection = ensure_collection(client, settings, create=False)
            total = collection.aggregate.over_all(total_count=True).total_count
            checks['weaviate'] = f'ok ({total} products in {settings.collection})'
        finally:
            client.close()
    except Exception as exc:
        checks['weaviate'] = f'error: {type(exc).__name__}: {exc}'[:300]
    if settings.chat_backend == 'llm':
        try:
            response = requests.get(settings.llm_url.rstrip('/') + '/models', timeout=5)
            response.raise_for_status()
            names = [m.get('id') for m in response.json().get('data', [])]
            checks['llm'] = f'ok ({", ".join(map(str, names)) or "model loaded"})'
        except Exception as exc:
            checks['llm'] = f'error: {type(exc).__name__} (is the llm service up and the model loaded?)'
    else:
        checks['chat'] = 'grounded (offline, no LLM)'
    checks['parsed_dir'] = 'ok' if settings.parsed_dir.is_dir() else f'missing: {settings.parsed_dir}'
    checks['image_dir'] = str(settings.image_dir)
    _print(checks)
    return 0 if all(v.startswith('ok') or k in ('image_dir', 'chat') for k, v in checks.items()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='vector-search', description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('scan', help='validate the parsed dataset and print a quality report')
    p.add_argument('--details', action='store_true')
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser('ingest', help='download images, embed, and upsert into Weaviate')
    p.add_argument('--limit', type=int, default=None, help='max products (default: all)')
    p.add_argument('--shards', default=None, help='e.g. "0-4,143" (default: all shards)')
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser('search', help='query the index from the terminal')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--text')
    group.add_argument('--image', help='path to a JPEG/PNG file')
    p.add_argument('--target', choices=('text', 'image'), default='text')
    p.add_argument('--limit', type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser('ask', help='search + assistant answer (end-to-end smoke test)')
    p.add_argument('query')
    p.add_argument('question')
    p.add_argument('--limit', type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser('evaluate', help='recall@k / MRR on the hold-out evaluation set')
    p.add_argument('--eval-dir', default='dataset/eval')
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('export', help='write a portable snapshot (vectors + thumbnails)')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--max-side', type=int, default=384, help='thumbnail size in pixels')
    p.add_argument('--no-images', action='store_true')
    p.set_defaults(func=cmd_export)

    p = sub.add_parser('restore', help='load a snapshot directory written by export')
    p.add_argument('source', help='directory with snapshot.jsonl.gz [+ images.tar]')
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser('doctor', help='check Weaviate, the LLM and data paths')
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_third_party_warnings()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for noisy in ('httpx', 'httpcore', 'urllib3'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return args.func(Settings.from_env(), args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
