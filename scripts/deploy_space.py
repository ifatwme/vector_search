#!/usr/bin/env python3
"""Assemble the Hugging Face Space from this repository and push it.

    HF_TOKEN=... python scripts/deploy_space.py --space ifatwme/vector-search
    python scripts/deploy_space.py --stage-only /tmp/space   # inspect / docker build locally

The Space gets: space/* (Dockerfile, start.sh, Streamlit config, README card) plus the
application code (src/, app.py, pyproject.toml, uv.lock). Remote files that are no longer
part of the Space are deleted, so the Space mirrors `main`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ['src', 'app.py', 'pyproject.toml', 'uv.lock']
SPACE_FILES = ['Dockerfile', 'start.sh', 'streamlit.toml', 'README.md']


def stage(target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / 'space' / name, target / name)
    for name in CODE:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, target / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(source, target / name)
    return target


def revision() -> str:
    sha = os.environ.get('GITHUB_SHA')
    if not sha:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                             text=True, check=False).stdout.strip()
    return sha[:12] or 'unknown'


def deploy(space: str, folder: Path) -> str:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('HF_TOKEN is not set (add it as a GitHub Actions secret)')
    api = HfApi(token=token)
    api.create_repo(space, repo_type='space', space_sdk='docker', exist_ok=True)
    local = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    remote = set(api.list_repo_files(space, repo_type='space'))
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(folder / path))
                  for path in sorted(local)]
    operations += [CommitOperationDelete(path_in_repo=path)
                   for path in sorted(remote - local - {'.gitattributes'})]
    commit = api.create_commit(space, repo_type='space', operations=operations,
                               commit_message=f'Deploy github.com/ifatwme/vector_search@{revision()}')
    return commit.commit_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--space', default='ifatwme/vector-search')
    parser.add_argument('--stage-only', metavar='DIR')
    args = parser.parse_args()
    if args.stage_only:
        print(stage(Path(args.stage_only)))
        return
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        print(deploy(args.space, stage(Path(temp) / 'space')))


if __name__ == '__main__':
    main()
