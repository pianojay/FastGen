import ast
import fcntl
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
GPU_UUID = 'GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_protected():
    p = json.loads((ROOT/'resources/provenance.json').read_text())
    for name, expected in p['protected_files_sha256'].items():
        assert digest(ROOT/name) == expected, name
    source = (ROOT/'model.py').read_text()
    base = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Model')
    assert hashlib.sha256(ast.get_source_segment(source, base).encode()).hexdigest() == p['base_Model']['sha256']
    assert digest(ROOT/p['partition']['path']) == p['partition']['sha256']
    for name, expected in p['afm']['files_sha256'].items():
        assert digest(ROOT/p['afm']['snapshot']/name) == expected, name
    return {'protected': True, 'base_Model': True, 'afm_snapshot': True,
            'partition_sha256': p['partition']['sha256']}


def source_identity():
    paths = sorted([ROOT/'model.py', *ROOT.glob('afm_fastgen/*.py'), *ROOT.glob('scripts/*.py')])
    files = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files': files}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def configure(config, gpu=True):
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('Set PYTHONNOUSERSITE=1')
    torch.set_num_threads(config.get('threads', 4))
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if gpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
            raise RuntimeError('Only physical GPU 0 is allowed')
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if uuid != GPU_UUID:
            raise RuntimeError(f'GPU identity mismatch: {uuid}')


def process_identity(pid=None):
    pid = pid or os.getpid()
    stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19], 'uid': os.getuid(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


class GpuLease:
    def __init__(self):
        self.file = None

    def __enter__(self):
        path = Path('/tmp/fastgen-gpu0.lock')
        self.file = path.open('a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Duplicate GPU 0 job refused')
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if uuid != GPU_UUID:
            self.file.close()
            raise RuntimeError(f'GPU identity mismatch: {uuid}')
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        for row in rows.splitlines():
            uuid, pid = [x.strip() for x in row.split(',')]
            if uuid == GPU_UUID and int(pid) != os.getpid():
                raise RuntimeError(f'GPU 0 occupied by PID {pid}; not terminating it')
        self.file.seek(0); self.file.truncate()
        json.dump({**process_identity(), 'gpu_uuid': GPU_UUID}, self.file); self.file.flush()
        return self

    def __exit__(self, *args):
        self.file.close()


def memory():
    def kb(path, key):
        try:
            return next(int(line.split()[1])*1024 for line in path.read_text().splitlines() if line.startswith(key+':'))
        except (FileNotFoundError, StopIteration, PermissionError):
            return 0
    proc = Path('/proc')/str(os.getpid())
    children = []
    try:
        children = [int(x) for x in (proc/'task'/str(os.getpid())/'children').read_text().split()]
    except FileNotFoundError:
        pass
    device = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_bytes': int(device.strip())*1024**2,
            'host_rss_bytes': kb(proc/'status', 'VmRSS'),
            'tree_pss_bytes': sum(kb(Path('/proc')/str(p)/'smaps_rollup', 'Pss') for p in [os.getpid(), *children])}


def append_json(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n'); f.flush()
