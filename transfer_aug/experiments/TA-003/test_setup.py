"""CPU-only synthetic correctness checks. Never accesses research cohort images."""
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
from PIL import Image
import torch
from engine import Trainer, ManifestSampler, ReplayPool, SEED, seed_all, sha256, lr_for_update

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'proof'; OUT.mkdir(exist_ok=True)
started=time.time(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_initialized()
checks={}
def require(name,cond):
    assert cond,name
    checks[name]=True

def digest_state(trainer):
    import hashlib
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value): h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value,dict):
            for k,v in value.items(): h.update(str(k).encode()); visit(v)
        elif isinstance(value,(list,tuple)):
            for v in value: visit(v)
        else: h.update(repr(value).encode())
    visit(trainer.net.state_dict()); visit(trainer.opt_g.state_dict()); visit(trainer.opt_d.state_dict())
    visit(trainer.pool0.state_dict()); visit(trainer.pool1.state_dict()); visit(trainer.update)
    return h.hexdigest()

records=[]
rng=np.random.default_rng(SEED)
for cls in range(3):
    for dom in ('D0','D1'):
        for i in range(cls+1):
            p=OUT/f'synthetic_{cls}_{dom}_{i}.png'
            Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(p)
            records.append(dict(path=str(p),sha256=sha256(p),partition='A1',class_id=cls,domain=dom,role='optimization'))
p=OUT/'synthetic_diagnostic.png'; Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)).save(p)
records.append(dict(path=str(p),sha256=sha256(p),partition='A1',class_id=0,domain='D0',role='diagnostic'))
manifest=OUT/'synthetic_manifest.json'; manifest.write_text(json.dumps(dict(records=records),indent=2))
sampler=ManifestSampler(manifest,sha256(manifest)); a,b,provenance=sampler.draw(16)
require('sampler_batch16_rgb64',a.shape==b.shape==(16,3,64,64))
require('sampler_excludes_diagnostic',all('diagnostic' not in pair['path'] for row in provenance for pair in row['pair']))
require('sampler_same_class_unpaired',all(str(row['class_id']) in Path(row['pair'][0]['path']).name and str(row['class_id']) in Path(row['pair'][1]['path']).name for row in provenance))
require('independent_flip_decisions',any(r['pair'][0]['flip']!=r['pair'][1]['flip'] for r in provenance))
# Direct sampler RNG frequency check: class choices ignore unequal within-class counts.
class_counts={c:0 for c in sampler.classes}; clone=random.Random(SEED)
for _ in range(6000):
    c=clone.choice(sampler.classes); class_counts[c]+=1
require('uniform_class_rng',max(class_counts.values())-min(class_counts.values())<200)
# Input tamper/partition protection.
bad=OUT/'invalid_manifest.json'; invalid=[dict(r) for r in records]; invalid[0]['partition']='A2'; bad.write_text(json.dumps(dict(records=invalid)))
try: ManifestSampler(bad,sha256(bad)); raise AssertionError('A2 accepted')
except ValueError: checks['reject_non_A1']=True
try: ManifestSampler(manifest,'0'*64); raise AssertionError('bad hash accepted')
except ValueError: checks['reject_manifest_hash']=True
seed_all(); t=Trainer(); counts=t.net.counts(); require('registered_parameter_cap',counts['total']<100000000)
require('normalization_exact',all(not m.affine and not m.track_running_stats for m in t.net.modules() if isinstance(m,torch.nn.InstanceNorm2d)))
require('no_dropout',not any(isinstance(m,torch.nn.Dropout) for m in t.net.modules()))
# Compare exact pinned upstream network outputs after same weights (independent loaded source).
src=ROOT.parent.parent/'audit_sources/cyclegan/models/networks.py'
spec=importlib.util.spec_from_file_location('pinned_upstream',src); upstream=importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
g=upstream.ResnetGenerator(3,3,64,norm_layer=upstream.get_norm_layer('instance'),n_blocks=6)
d=upstream.NLayerDiscriminator(3,64,n_layers=2,norm_layer=upstream.get_norm_layer('instance'))
g.load_state_dict(t.net.G01.state_dict()); d.load_state_dict(t.net.D0.state_dict())
with torch.no_grad():
    y=t.net.G01(a[:1]); z=t.net.D0(a[:1])
    require('upstream_generator_exact',torch.equal(y,g(a[:1])))
    require('upstream_discriminator_exact',torch.equal(z,d(a[:1])))
require('generator_shape_range_finite',y.shape==(1,3,64,64) and torch.isfinite(y).all().item() and y.min()>=-1 and y.max()<=1)
require('patch_shape_finite',z.shape==(1,1,14,14) and torch.isfinite(z).all().item())
# Convolutional receptive field: IN itself uses image-wide spatial statistics.
rf,jump=1,1
for m in t.net.D0.modules():
    if isinstance(m,torch.nn.Conv2d): rf+=(m.kernel_size[0]-1)*jump; jump*=m.stride[0]
require('convolutional_receptive_field_34',rf==34)
del g,d,y,z
pool=ReplayPool(size=2,seed=1)
x=torch.ones(2,3,4,4,requires_grad=True); got=pool.query(x)
require('replay_fill_detach',len(pool.images)==2 and not got.requires_grad)
s=pool.state_dict(); pool2=ReplayPool(size=2,seed=2); pool2.load_state_dict(s)
next_x=torch.full((20,3,4,4),2.); r1=pool.query(next_x); r2=pool2.query(next_x)
require('replay_rng_restore',torch.equal(r1,r2)); require('replay_returns_historical',bool((r1==1).any()))
pool50=ReplayPool(); pool50.query(torch.zeros(60,3,4,4)); require('replay_default_capacity50',len(pool50.images)==50)
# Full-size networks, CPU batch1 synthetic steps; nominal batch16 separately shape-tested.
loss0=t.step(a[:1],b[:1]); require('all_gradients_finite_nonzero',all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in t.net.parameters()))
require('losses_finite',all(np.isfinite(v) for v in loss0.values()))
checkpoint=OUT/'synthetic_recovery.pt'; contract_hash=sha256(ROOT/'SETUP_CONTRACT_v1.md'); t.save(checkpoint,sampler,contract_hash)
checkpoint_hash=sha256(checkpoint)
expected_rng=[random.random(),float(np.random.random()),float(torch.rand(()))]
a1,b1,meta1=sampler.draw(1); expected=t.step(a1,b1); expected_digest=digest_state(t)
# Destroy optimizer/model and perturb every generator before loading.
del t; seed_all(9); restored=Trainer(); sampler2=ManifestSampler(manifest,sha256(manifest)); restored.load(checkpoint,sampler2,contract_hash)
actual_rng=[random.random(),float(np.random.random()),float(torch.rand(()))]
a2,b2,meta2=sampler2.draw(1); actual=restored.step(a2,b2); actual_digest=digest_state(restored)
require('all_global_rng_exact_recovery',expected_rng==actual_rng)
require('sampler_exact_recovery',meta1==meta2 and torch.equal(a1,a2) and torch.equal(b1,b2))
require('optimizer_model_pool_exact_recovery',expected_digest==actual_digest)
require('resumed_losses_exact',expected==actual)
require('schedule_boundaries',lr_for_update(1)==.0002 and lr_for_update(10000)==.0002 and lr_for_update(10001)<.0002 and lr_for_update(20000)==0)
try: restored.load(checkpoint,sampler2,'bad'); raise AssertionError('bad contract accepted')
except ValueError: checks['reject_checkpoint_contract_mismatch']=True
outputs=restored.evaluate(a[:1],b[:1]); require('evaluation_finite',all(torch.isfinite(x).all() and x.shape==(1,3,64,64) for x in outputs.values()))
require('no_cuda_initialized',not torch.cuda.is_initialized())
result=dict(status='passed',checks=checks,check_count=len(checks),parameters=counts,
            losses_first_synthetic_step=loss0,recovery=dict(expected_state_sha256=expected_digest,actual_state_sha256=actual_digest,checkpoint_sha256=checkpoint_hash),
            synthetic_optimizer_calls=3,nominal_batch_size=16,test_optimizer_batch_size=1,
            class_rng_counts=class_counts,discriminator_convolutional_receptive_field=rf,discriminator_output_shape=[1,1,14,14],
            elapsed_seconds=time.time()-started,peak_host_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpu_seconds=0,cuda_initialized=False,torch_version=torch.__version__,numpy_version=np.__version__)
(OUT/'cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
# Checkpoint is synthetic, reproducible, and large; record hash then remove scratch payload.
checkpoint.unlink()
