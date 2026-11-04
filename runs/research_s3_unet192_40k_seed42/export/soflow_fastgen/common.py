import ast
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def verify_protected():
    manifest = json.loads((ROOT / 'configs/protected_files.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'Protected file changed: {name}')
    content = (ROOT / 'model.py').read_text()
    node = next(n for n in ast.parse(content).body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    actual = hashlib.sha256(ast.get_source_segment(content, node).encode()).hexdigest()
    if actual != manifest['base_Model_sha256']:
        raise RuntimeError('Protected base Model changed')
    return manifest


def source_provenance():
    paths = list(ROOT.glob('*.py'))
    for name in ('soflow_fastgen','scripts','tests','configs'):
        paths.extend(path for path in (ROOT/name).rglob('*') if path.suffix in ('.py','.json'))
    record = {'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}}
    for key, command in (('git_commit',['git','rev-parse','HEAD']),
                         ('git_status',['git','status','--short'])):
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        record[key] = result.stdout.strip() if result.returncode == 0 else 'unavailable'
    return record


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def resources():
    import resource
    rss = 0
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss = int(line.split()[1]) * 1024
    result = {'host_rss_bytes': rss, 'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.cuda.is_available():
        result.update(allocated_bytes=torch.cuda.max_memory_allocated(),
                      reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def gpu_telemetry():
    index = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    try:
        result = subprocess.run(['nvidia-smi', '-i', index,
            '--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'


def process_tree_memory():
    """Linux process-tree memory, including loader workers; PSS apportions shared pages."""
    pending, seen = [os.getpid()], set()
    rss, pss = 0, 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text()
            pending.extend(int(child) for child in children.split())
            values = Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()
            for line in values:
                if line.startswith('Rss:'):
                    rss += int(line.split()[1]) * 1024
                elif line.startswith('Pss:'):
                    pss += int(line.split()[1]) * 1024
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return {'process_tree_rss_bytes': rss, 'process_tree_pss_bytes': pss,
            'process_tree_count': len(seen)}


class Budget:
    """Accumulate wall time while a single GPU job is active, across processes."""
    def __init__(self, path, seconds):
        self.path, self.limit = Path(path), seconds
        self.used = json.loads(self.path.read_text())['used_seconds'] if self.path.exists() else 0.
        self.started = time.monotonic()

    def elapsed(self):
        return self.used + time.monotonic() - self.started

    def remaining(self):
        return self.limit - self.elapsed()

    def save(self):
        atomic_json(self.path, {'used_seconds': self.elapsed(), 'limit_seconds': self.limit})
