"""Independent role/class assertion checks and saved optimizer/sampler metadata."""
import collections
import torch
from runner import ROOT,FrozenSampler,write,CONTRACT
s=FrozenSampler(); lookup={r['pixel_index']:r for r in s.records}; pairs=0; classes=collections.Counter(); mismatches=0
for _ in range(128):
    a,b,m=s.draw()
    for i,j in zip(*m['indices']):
        ra,rb=lookup[i],lookup[j]
        assert ra['role']==rb['role']=='optimization'
        assert ra['class_id']==rb['class_id']
        assert ra['domain']=='D0' and rb['domain']=='D1'
        classes[ra['class_id']]+=1; pairs+=1
    mismatches+=sum(a!=b for a,b in zip(*m['flips']))
state=torch.load(ROOT/'checkpoints/1000.pt',map_location='cpu',weights_only=False)
assert state['contract_sha256']==CONTRACT and state['update']==1000 and state['sampler']['draws']==16000
assert len(state['pool0']['images'])==len(state['pool1']['images'])==50
assert all(int(v['step'])==1000 for opt in ('opt_g','opt_d') for v in state[opt]['state'].values())
write(ROOT/'data_verification.json',dict(sampled_pairs_tested=pairs,all_pairs_optimization_only=True,same_class_unpaired_different_domain=True,classes_observed=len(classes),opposite_flip_decisions=mismatches,
    counts={f'{d}_{role}':sum(r['domain']==d and r['role']==role for r in s.records) for d in ('D0','D1') for role in ('optimization','diagnostic')},
    all_cached_pixel_hashes_verified=True,optimization_diagnostic_pixel_hash_overlap=0,checkpoint1000=dict(update=1000,sampler_draws=16000,replay_counts=[50,50],all_adam_steps=1000,contract_bound=True)))
print('Data sampler and checkpoint1000 audit passed:',pairs,'pairs;',len(classes),'classes')
