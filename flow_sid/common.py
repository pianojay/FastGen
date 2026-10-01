"""Local campaign bookkeeping; standard library only until GPU qualification."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PARTITION_SHA = '825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_sources():
    p = json.loads((ROOT / 'resources/provenance.json').read_text())
    for path, sha in p['protected_files_sha256'].items():
        assert digest(ROOT / path) == sha, path
    for snapshot in p['snapshots'].values():
        for path, sha in snapshot['files_sha256'].items():
            assert digest(ROOT / snapshot['snapshot'] / path) == sha, path
    source = (ROOT / 'model.py').read_text()
    base = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    assert hashlib.sha256(ast.get_source_segment(source, base).encode()).hexdigest() == p['base_Model']['sha256']
    assert digest(ROOT / 'resources/development_split.json') == PARTITION_SHA
    return {'protected_files': len(p['protected_files_sha256']), 'snapshots': 'verified',
            'base_Model': 'verified', 'partition_sha256': PARTITION_SHA}


def source_identity():
    paths = sorted(list((ROOT / 'flow_sid').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
                   + [ROOT / 'model.py', ROOT / 'scripts/run_gpu.sh'])
    files = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    return {'files': files, 'sha256': object_hash(files),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def host_memory():
    values = {}
    for path in ['/proc/self/status', '/proc/self/smaps_rollup']:
        for line in Path(path).read_text().splitlines():
            k = line.split(':')[0]
            if k in ['VmRSS', 'VmHWM', 'Rss', 'Pss']:
                values[k + '_bytes'] = int(line.split()[1]) * 1024
    return values


class StageCost:
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def __enter__(self):
        append_json(ROOT / 'runs/cost.jsonl', {'event': 'start', 'stage': self.name,
                                             'time': self.start, 'pid': os.getpid()})
        return self

    def __exit__(self, typ, value, tb):
        append_json(ROOT / 'runs/cost.jsonl', {'event': 'end', 'stage': self.name,
                    'time': time.time(), 'seconds': time.time() - self.start,
                    'outcome': 'ok' if typ is None else str(value), 'pid': os.getpid()})
