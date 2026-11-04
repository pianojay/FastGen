"""Restore pinned, hash-checked audit source files; never import upstream code.

Run from the existing EDA environment. No datasets, model weights or packages
are downloaded. Paper PDF URLs are documented separately in SOURCE_REGISTRY.md.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    restored = 0
    for key, repo in manifest.items():
        for name, record in repo['files'].items():
            if record.get('status') != 200:
                continue
            target = (ROOT / 'audit_sources' / key / name).resolve()
            if not target.is_relative_to(ROOT / 'audit_sources'):
                raise ValueError(f'Unexpected source path: {name}')
            if target.exists():
                data = target.read_bytes()
                if hashlib.sha256(data).hexdigest() == record['sha256']:
                    continue
                raise ValueError(f'Existing source hash mismatch: {target}')
            with urllib.request.urlopen(record['url'], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'Download hash mismatch: {record["url"]}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            restored += 1
    (ROOT / 'audit_sources' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Restored {restored} source files; all recorded successful sources verified.')


if __name__ == '__main__':
    main()
