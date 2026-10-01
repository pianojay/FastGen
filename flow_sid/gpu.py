"""Host-wide exclusive physical GPU 2 lease. Never selects another device."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, atomic_json

EXPECTED_UUID = 'GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2'


def process_identity(pid):
    p = Path(f'/proc/{pid}')
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19], 'uid': p.stat().st_uid,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def device_status():
    out = subprocess.check_output(['nvidia-smi', '-i', '2',
        '--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    keys = ['index', 'uuid', 'name', 'capability', 'total_MiB', 'used_MiB', 'utilization_pct']
    return dict(zip(keys, out))


def occupants():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    return [line.split(', ') for line in out.splitlines() if line.startswith(EXPECTED_UUID)]


class GPU2Lease:
    def __init__(self):
        self.fd = None

    def __enter__(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
            raise RuntimeError('Physical GPU 2 requires CUDA_VISIBLE_DEVICES=2 before startup')
        if os.environ.get('PYTHONNOUSERSITE') != '1':
            raise RuntimeError('PYTHONNOUSERSITE=1 required')
        self.fd = os.open('/tmp/fastgen-gpu2.lock', os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError('GPU 2 lease occupied; no fallback device authorized')
        try:
            status = device_status()
            if status['index'] != '2' or status['uuid'] != EXPECTED_UUID:
                raise RuntimeError(f'Physical GPU identity changed: {status}')
            occupied = occupants()
            if occupied or int(status['used_MiB']) > 100:
                raise RuntimeError(f'GPU 2 occupied before qualification: {occupied}, {status}')
            self.identity = process_identity(os.getpid())
            record = {'owner': self.identity, 'device': status, 'acquired_at': time.time(),
                      'project': str(ROOT), 'process_local_device': 'cuda:0'}
            os.ftruncate(self.fd, 0)
            os.write(self.fd, json.dumps(record, sort_keys=True).encode())
            os.fsync(self.fd)
            atomic_json(ROOT / 'runs/last_gpu_lease.json', record)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def verify_ownership(self):
        assert process_identity(os.getpid()) == self.identity
        others = [row for row in occupants() if int(row[1]) != os.getpid()]
        if others:
            raise RuntimeError(f'Unexpected GPU 2 occupancy: {others}; no processes killed')

    def __exit__(self, typ, value, tb):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

