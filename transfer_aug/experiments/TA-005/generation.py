"""Fixed one-source batch grouping for reference and sampled MUNIT diagnostics."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent
ROUND3=ROOT.parents[1]/'shared/round3'
def quant(y):return np.clip(np.rint((y.transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8)
@torch.inference_mode()
def infer(t,sampler,r,z):
    src=getattr(t.net,'G'+r['source_domain'][1]);dest=getattr(t.net,'G'+r['target_domain'][1])
    x=sampler.tensor([r['source_index']]).to(t.device);c,own=src.encode(x)
    outputs=[]
    for ix in r['reference_indices']:
        _,style=dest.encode(sampler.tensor([ix]).to(t.device));outputs.append(dest.decode(c,style).cpu().numpy()[0])
    for style in z:
        outputs.append(dest.decode(c,torch.from_numpy(style).reshape(1,8,1,1).to(t.device)).cpu().numpy()[0])
    outputs.append(src.decode(c,own).cpu().numpy()[0]);a=np.asarray(outputs);assert np.isfinite(a).all();return a
@torch.inference_mode()
def generate(t,sampler,ledger):
    from runner import write,status,CONTRACT
    out=ROOT/'outputs';out.mkdir(exist_ok=True);png=out/'png';png.mkdir(exist_ok=True);refs=out/'references';refs.mkdir(exist_ok=True)
    records=json.loads((ROUND3/'evaluation_manifest.json').read_text())['records'];latent=np.load(ROUND3/'munit_latent.npy');t.net.eval()
    arr=np.lib.format.open_memmap(out/'final_float.npy',mode='w+',dtype='float32',shape=(604,9,3,64,64));manifest=[];reproduced=0;last=time.monotonic()
    for i,r in enumerate(records):
        a=infer(t,sampler,r,latent[i]);arr[i]=a
        for j,y in enumerate(a):
            p=png/f'{i:04d}_{j}.png';u=quant(y);Image.fromarray(u).save(p)
            manifest.append(dict(source_eval_index=i,variant=(j if j<4 else j-4) if j<8 else 0,arm='reference' if j<4 else ('latent' if j<8 else 'self'),output_path=str(p),pixel_sha256=hashlib.sha256(u.tobytes()).hexdigest()))
        for j,ix in enumerate(r['reference_indices']):Image.fromarray(sampler.pixels[ix]).save(refs/f'{i:04d}_{j}.png')
        if r['representative']:
            replay=infer(t,sampler,r,latent[i]);assert all(np.array_equal(quant(x),quant(y)) for x,y in zip(a[:8],replay[:8]));reproduced+=8
        ledger.check()
        if time.monotonic()-last>30:status('generation',t,ledger,completed_sources=i+1);last=time.monotonic()
    arr.flush();write(out/'manifest.json',manifest)
    selected={}
    for i,r in enumerate(records):
        if r['representative']:
            key=(r['category'],r['source_domain'])
            if key not in selected or r['source_path']<records[selected[key]]['source_path']:selected[key]=i
    trace_indices=sorted(selected.values());milestones={}
    for label in ('initial','1000','5000','10000','final'):
        p=ROOT/'checkpoints'/f'{label}.pt'
        if not p.exists():continue
        state=torch.load(p,map_location='cpu',weights_only=False);update=state['update']
        if str(update) in milestones:continue
        t.net.load_state_dict(state['net']);del state;t.net.eval();images=[]
        for i in trace_indices:
            # Same fixed grouping as final; retain only declared variant0 per arm.
            a=infer(t,sampler,records[i],latent[i]);images.append(np.stack([quant(a[0]),quant(a[4])]))
            ledger.check()
        np.save(out/f'milestone_{update}.npy',np.asarray(images));milestones[str(update)]=label;status('milestone_generation',t,ledger,checkpoint_update=update)
    write(out/'milestone_manifest.json',dict(source_eval_indices=trace_indices,milestones=milestones))
    write(ROOT/'generation_verification.json',dict(final_outputs=5436,translations=4832,self_controls=604,reference_outputs=2416,latent_outputs=2416,reproduced_outputs=reproduced,quantized_reproduction_exact=True,all_finite=True,milestone_translations=len(milestones)*len(trace_indices)*2,one_source_inference_grouping=True))
