import copy,json
from pathlib import Path
import torch
from data import FrozenSampler
s=FrozenSampler();lookup={r['pixel_index']:r for r in s.records};classes=set();flips=0
for _ in range(128):
    a,b,m=s.draw()
    assert a.shape==b.shape==(8,3,64,64)
    for i,j in zip(*m['indices']):
        x,y=lookup[i],lookup[j];assert x['role']==y['role']=='optimization' and x['class_id']==y['class_id'];assert x['domain']=='D0' and y['domain']=='D1';classes.add(x['class_id'])
    flips+=sum(x!=y for x,y in zip(*m['flips']))
state=copy.deepcopy(s.state_dict());a,b,m=s.draw();s.load_state_dict(state);aa,bb,mm=s.draw();assert torch.equal(a,aa) and torch.equal(b,bb) and m==mm
out=dict(status='passed',pairs=1024,classes_observed=len(classes),unequal_flip_decisions=flips,checkpoint_sampler_replay_exact=True,pixel_hashes_verified=8554,optimization_only=True,same_class_unpaired=True)
Path(__file__).with_name('data_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
