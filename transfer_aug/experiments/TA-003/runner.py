"""Contract-bound CycleGAN run; exclusive physical GPU1 only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, datetime, fcntl, gc, hashlib, json, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from engine import Trainer, seed_all, sha256, SEED

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parents[1]/'shared/round2'
DISCOVERY=ROOT.parents[1]/'domain_discovery/round2'
CONTRACT='046c4419cd02754456f9d94e53e90a1f00fa2b1bc3d541f777ffd1aa5eca6d05'
MANIFEST='a15a783934fb636acc7d141d3becdffea049f0082718c0a1e931bc094df18093'
PIXELS='411c174959b83e2521e77ce3a3fa9e5b426c306a46361f8ba0804dd90de31a12'
UUID='GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888'
LEASE='/home/user/jw_soflow/FastGen-SoFlow/runs/gpu_1.lock'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def preflight():
    assert sha256(ROOT/'CONTRACT_v1.md')==CONTRACT
    assert sha256(SHARED/'TRAINING_PROTOCOL_v1.md')=='ae9323b66e4969f87f000d08d0227cc6770e9f1377bc7f62deb6c92e39409c94'
    assert sha256(ROOT/'SETUP_CONTRACT_v1.md')=='419a8e3372082d6b48ed657f80ec3f5886a5d57ee857bfcda4fc8d657978df3c'
    assert sha256(SHARED/'manifest.json')==MANIFEST and sha256(SHARED/'pixels_uint8.npy')==PIXELS
    prov=json.loads((ROOT/'upstream_provenance.json').read_text())
    for file,meta in prov['files'].items(): assert sha256(ROOT.parents[1]/'audit_sources/cyclegan'/file)==meta['sha256']
    assert json.loads((ROOT/'proof/cpu_checks.json').read_text())['status']=='passed'
    return dict(contract_sha256=CONTRACT,manifest_sha256=MANIFEST,pixels_sha256=PIXELS,utc=utc(),code_sha256={p.name:sha256(p) for p in ROOT.glob('*.py')})

class FrozenSampler:
    def __init__(self):
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
    def draw(self,count=16):
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

class Ledger:
    def __init__(self):
        self.path=ROOT/'gpu_ledger.json'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else dict(experiment='TA-003',limit_seconds=28800,sessions=[])
        # A previous unclosed session is conservatively charged through this restart.
        for s in self.data['sessions']:
            if not s.get('ended_epoch'):
                s['ended_epoch']=time.time(); s['elapsed_seconds']=s['ended_epoch']-s['started_epoch']; s['closure']='conservative_restart'
        self.prior=sum(s['elapsed_seconds'] for s in self.data['sessions'])
        self.session=dict(pid=os.getpid(),uuid=UUID,started_epoch=time.time(),started_utc=utc(),elapsed_seconds=0,ended_epoch=None)
        self.data['sessions'].append(self.session); self.flush()
    def elapsed(self): return self.prior+time.time()-self.session['started_epoch']
    def flush(self,close=False):
        self.session['elapsed_seconds']=time.time()-self.session['started_epoch']; self.session['last_utc']=utc()
        if close: self.session['ended_epoch']=time.time(); self.session['closure']='normal_finally'
        self.data['cumulative_gpu_seconds']=self.prior+self.session['elapsed_seconds']; write(self.path,self.data)
    def check(self):
        if self.elapsed()>=28800: raise RuntimeError('GPU budget exhausted')

def status(stage,t,ledger,**more):
    ledger.flush(); write(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),utc=utc(),completed_updates=t.update if t else 0,expected_updates=20000,
        cumulative_gpu_seconds=ledger.elapsed(),gpu_uuid=UUID,latest_checkpoint=str(ROOT/'checkpoints/latest.pt'),**more))

def state_hash(t):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):h.update(v.detach().cpu().numpy().tobytes())
        elif isinstance(v,dict):
            for k,x in v.items():h.update(str(k).encode());visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(t.net.state_dict());visit(t.opt_g.state_dict());visit(t.opt_d.state_dict());visit(t.pool0.state_dict());visit(t.pool1.state_dict());visit(t.update)
    return h.hexdigest()

def qualify(sampler,ledger):
    seed_all(); sampler.rng.seed(SEED); sampler.draws=0; t=Trainer('cuda'); torch.cuda.reset_peak_memory_stats(); start=time.monotonic(); losses=[]
    for i in range(20):
        a,b,_=sampler.draw(); losses.append(t.step(a,b)); ledger.check()
        status('qualification',t,ledger)
    torch.cuda.synchronize(); elapsed=time.monotonic()-start
    allocated=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved(); assert allocated<20000000000
    path=ROOT/'checkpoints/qualification.pt'; t.save(path,sampler,CONTRACT)
    a,b,meta=sampler.draw(); l1=t.step(a,b); h1=state_hash(t)
    del t,a,b;gc.collect();torch.cuda.empty_cache(); seed_all(5); t=Trainer('cuda'); t.load(path,sampler,CONTRACT)
    a,b,meta2=sampler.draw(); l2=t.step(a,b); h2=state_hash(t)
    assert l1==l2 and h1==h2 and meta==meta2,'GPU checkpoint recovery mismatch'
    result=dict(status='passed',qualification_updates=20,recovery_optimizer_calls=2,seconds_20_updates=elapsed,seconds_per_update=elapsed/20,
        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,parameter_counts=t.net.counts(),gpu_recovery_exact=True,recovery_state_sha256=h1,
        first_loss=losses[0],last_loss=losses[-1],device_name=torch.cuda.get_device_name(),uuid=UUID,torch_version=torch.__version__,cuda_version=torch.version.cuda)
    write(ROOT/'qualification.json',result); path.unlink(); del t,a,b; gc.collect();torch.cuda.empty_cache()
    return result

def checkpoint(t,sampler,name):
    path=ROOT/'checkpoints'/name; t.save(path,sampler,CONTRACT)
    return dict(path=str(path),sha256=sha256(path),update=t.update)

@torch.inference_mode()
def generate(t,sampler,ledger):
    out=ROOT/'outputs'; out.mkdir(exist_ok=True); milestones={}; records=[r for r in sampler.records if r['role']=='diagnostic']
    records.sort(key=lambda r:(r['domain'],r['path']))
    review_cats=set(json.loads((DISCOVERY/'review_selection.json').read_text())['categories'])
    reviews=[r for r in records if r['category'] in review_cats]
    final_update=t.update; final_state={k:v.detach().cpu().clone() for k,v in t.net.state_dict().items()}
    final=[]; reproduction=[]
    t.net.eval()
    for direction in ('D0_D1','D1_D0'):
        rr=[r for r in records if r['domain']==direction[:2]]; G=t.net.G01 if direction=='D0_D1' else t.net.G10; inv=t.net.G10 if direction=='D0_D1' else t.net.G01
        fakes=[];cycles=[];identities=[]
        for start in range(0,len(rr),16):
            x=sampler.tensor([r['pixel_index'] for r in rr[start:start+16]]).cuda(); y=G(x); cy=inv(y); ident=inv(x)
            for arr in (y,cy,ident): assert torch.isfinite(arr).all()
            fakes.extend(y.cpu().numpy());cycles.extend(cy.cpu().numpy());identities.extend(ident.cpu().numpy());ledger.check()
        np.save(out/f'{direction}_float.npy',np.asarray(fakes));np.save(out/f'{direction}_cycle.npy',np.asarray(cycles));np.save(out/f'{direction}_identity.npy',np.asarray(identities))
        dd=out/direction;dd.mkdir(exist_ok=True)
        for i,(r,y) in enumerate(zip(rr,fakes)):
            u=np.clip(np.rint((y.transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8); p=dd/f'{i:04d}.png';Image.fromarray(u).save(p)
            final.append(dict(**r,direction=direction,output_index=i,output_path=str(p),output_sha256=sha256(p),output_pixel_sha256=hashlib.sha256(u.tobytes()).hexdigest()))
        # Repeat exact batches, compare review members: fixed batch grouping prevents convolution rounding changes.
        for start in range(0,len(rr),16):
            batch=rr[start:start+16]
            if not any(r['category'] in review_cats for r in batch):continue
            y=G(sampler.tensor([r['pixel_index'] for r in batch]).cuda()).cpu().numpy()
            for j,r in enumerate(batch):
                if r['category'] in review_cats:
                    u=np.clip(np.rint((y[j].transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8)
                    expected=np.asarray(Image.open(dd/f'{start+j:04d}.png')); assert np.array_equal(u,expected)
                    reproduction.append(r['path'])
        status('generation',t,ledger,direction=direction)
    write(out/'manifest.json',final)
    for label in ('initial','1000','5000','10000','final'):
        path=ROOT/'checkpoints'/f'{label}.pt'
        if not path.exists():continue
        state=torch.load(path,map_location='cpu',weights_only=False);update=state['update']
        if str(update) in milestones: continue
        t.net.load_state_dict(state['net']);t.net.eval();del state
        arrays=[]
        for r in reviews:
            G=t.net.G01 if r['domain']=='D0' else t.net.G10
            y=G(sampler.tensor([r['pixel_index']]).cuda()).cpu().numpy()[0]
            arrays.append(np.clip(np.rint((y.transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8));ledger.check()
        np.save(out/f'review_update_{update}.npy',np.asarray(arrays));milestones[str(update)]=label
        status('milestone_generation',t,ledger,checkpoint_update=update)
    t.net.load_state_dict(final_state)
    write(out/'review_manifest.json',dict(records=reviews,milestones=milestones))
    write(ROOT/'generation_verification.json',dict(final_update=final_update,outputs=len(final),reproduced_review_outputs=len(reproduction),quantized_reproduction_exact=True,all_outputs_finite=True,milestones=milestones))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    before=preflight(); sampler=FrozenSampler();(ROOT/'checkpoints').mkdir(exist_ok=True);write(ROOT/'preflight.json',before)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('1',UUID)
    lease=open(LEASE,'r');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=Ledger();t=None
    try:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        assert any(line.strip()==f'1, {UUID}' for line in inventory.splitlines())
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert not any(UUID in line for line in occupied.splitlines()),'foreign process on assigned GPU'
        assert torch.cuda.device_count()==1
        write(ROOT/'lease_ownership.json',dict(pid=os.getpid(),uuid=UUID,lease=LEASE,inode=os.fstat(lease.fileno()).st_ino,acquired_utc=utc(),inventory=inventory,occupancy=occupied))
        status('qualification_start',None,ledger)
        if not args.resume: qual=qualify(sampler,ledger)
        else:qual=json.loads((ROOT/'qualification.json').read_text());assert qual['status']=='passed'
        seed_all();sampler.rng.seed(SEED);sampler.draws=0;t=Trainer('cuda');torch.cuda.reset_peak_memory_stats()
        if args.resume:t.load(ROOT/'checkpoints/latest.pt',sampler,CONTRACT)
        else:
            checkpoint(t,sampler,'initial.pt');checkpoint(t,sampler,'latest.pt')
        training_start=t.update; tick=time.monotonic();last_status=tick; since_tick=0; checkpoint_records=[]
        log=(ROOT/'training.jsonl').open('a',buffering=1)
        while t.update<20000:
            # Keep a full inference reserve plus conservative save/step margin.
            if ledger.elapsed()+max(30,qual['seconds_per_update']*5)>=28200:break
            a,b,_=sampler.draw(); losses=t.step(a,b);since_tick+=1
            allocated=torch.cuda.max_memory_allocated();assert allocated<20000000000
            if t.update%25==0:log.write(json.dumps(dict(utc=utc(),gpu_seconds=ledger.elapsed(),**losses))+'\n')
            if t.update%1000==0:
                checkpoint_records.append(checkpoint(t,sampler,'latest.pt'))
                if t.update in (1000,5000,10000):checkpoint_records.append(checkpoint(t,sampler,f'{t.update}.pt'))
                write(ROOT/'checkpoint_history.json',checkpoint_records)
            now=time.monotonic()
            if now-last_status>=45 or t.update%1000==0:
                status('training',t,ledger,last_losses=losses,seconds_per_update=(now-tick)/max(since_tick,1),peak_allocated_bytes=allocated,peak_reserved_bytes=torch.cuda.max_memory_reserved());last_status=now
        log.close();checkpoint_records.append(checkpoint(t,sampler,'latest.pt'));checkpoint_records.append(checkpoint(t,sampler,'final.pt'));write(ROOT/'checkpoint_history.json',checkpoint_records)
        write(ROOT/'training_complete.json',dict(status='completed' if t.update==20000 else 'budget_limited',updates=t.update,seed_reset_after_qualification=True,
            production_draws=sampler.draws,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_seconds_before_generation=ledger.elapsed(),parameter_counts=t.net.counts()))
        status('final_generation',t,ledger);generate(t,sampler,ledger);after=preflight()
        write(ROOT/'input_integrity_after.json',dict(manifest_unchanged=before['manifest_sha256']==after['manifest_sha256'],pixels_unchanged=before['pixels_sha256']==after['pixels_sha256'],contract_unchanged=before['contract_sha256']==after['contract_sha256']))
        status('gpu_complete',t,ledger)
    except BaseException as exc:
        write(ROOT/'failure.json',dict(utc=utc(),type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),update=t.update if t else None));raise
    finally:
        ledger.flush(close=True);fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

if __name__=='__main__':main()
