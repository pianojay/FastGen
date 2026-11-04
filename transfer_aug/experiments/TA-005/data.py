"""Frozen A1 sampler adapted from TA-003; no dataset optimization on import."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from engine import SEED
SHARED=Path(__file__).resolve().parents[2]/'shared/round2'
MANIFEST='a15a783934fb636acc7d141d3becdffea049f0082718c0a1e931bc094df18093'
PIXELS='411c174959b83e2521e77ce3a3fa9e5b426c306a46361f8ba0804dd90de31a12'
class FrozenSampler:
    def __init__(self):
        assert hashlib.sha256((SHARED/"manifest.json").read_bytes()).hexdigest()==MANIFEST
        assert hashlib.sha256((SHARED/"pixels_uint8.npy").read_bytes()).hexdigest()==PIXELS
        self.records=json.loads((SHARED/'manifest.json').read_text())['records']; self.pixels=np.load(SHARED/'pixels_uint8.npy',mmap_mode='r')
        assert self.pixels.shape==(8554,64,64,3) and self.pixels.dtype==np.uint8
        self.groups={}; counts={}; opt_hash=set(); diag_hash=set()
        for row in self.records:
            assert row['partition']=='A1' and row['domain'] in ('D0','D1') and row['role'] in ('optimization','diagnostic')
            assert hashlib.sha256(self.pixels[row['pixel_index']].tobytes()).hexdigest()==row['pixel_sha256']
            counts[row['domain'],row['role']]=counts.get((row['domain'],row['role']),0)+1
            (opt_hash if row['role']=='optimization' else diag_hash).add(row['pixel_sha256'])
            if row['role']=='optimization': self.groups.setdefault(row['class_id'],{'D0':[],'D1':[]})[row['domain']].append(row['pixel_index'])
        assert not opt_hash.intersection(diag_hash)
        assert all(counts[d,'optimization']==3975 and counts[d,'diagnostic']==302 for d in ('D0','D1'))
        assert len(self.groups)==151 and all(g[d] for g in self.groups.values() for d in ('D0','D1'))
        self.classes=sorted(self.groups); self.rng=random.Random(SEED); self.draws=0
    def tensor(self,indices):
        return torch.from_numpy(np.array(self.pixels[indices]).transpose(0,3,1,2)).float().div_(127.5).sub_(1)
    def draw(self,count=8):
        indices=[[],[]]; flips=[[],[]]
        for i in range(count):
            c=self.rng.choice(self.classes)
            for j,d in enumerate(('D0','D1')):
                indices[j].append(self.rng.choice(self.groups[c][d])); flips[j].append(self.rng.random()<.5)
        batches=[self.tensor(ix) for ix in indices]
        for batch,ff in zip(batches,flips):
            for i,flip in enumerate(ff):
                if flip: batch[i]=batch[i].flip(-1)
        self.draws+=count
        return *batches,dict(indices=indices,flips=flips)
    def state_dict(self): return dict(manifest_sha256=MANIFEST,pixels_sha256=PIXELS,rng=self.rng.getstate(),draws=self.draws)
    def load_state_dict(self,s):
        assert s['manifest_sha256']==MANIFEST and s['pixels_sha256']==PIXELS
        self.rng.setstate(s['rng']); self.draws=s['draws']
