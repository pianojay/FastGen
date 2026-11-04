"""Contract-bound MUNIT run; exclusive physical GPU1 only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, datetime, fcntl, gc, hashlib, json, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from engine import Trainer, seed_all, SEED
from data import FrozenSampler, MANIFEST, PIXELS
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parents[1]/'shared/round2'
DISCOVERY=ROOT.parents[1]/'domain_discovery/round2'
CONTRACT='8ab1115ed4c9554ed47027b76d4c3cba58d6d22f3351a78146e8ddd0fc9cf75c'
ROUND3=ROOT.parents[1]/'shared/round3'
MANIFEST='a15a783934fb636acc7d141d3becdffea049f0082718c0a1e931bc094df18093'
PIXELS='411c174959b83e2521e77ce3a3fa9e5b426c306a46361f8ba0804dd90de31a12'
UUID='GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888'
LEASE='/home/user/jw_soflow/FastGen-SoFlow/runs/gpu_1.lock'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def preflight():
    assert sha256(ROOT/'CONTRACT_v1.md')==CONTRACT
    assert sha256(ROUND3/'TRAINING_PROTOCOL_v1.md')=='75937769254ab5634618d104d97b0e51c4b1bf20128e02c1566eeffa32d6a8f2'
    assert sha256(ROOT/'SETUP_CONTRACT_v1.md')=='045526507a7364844b31b5aefcadf363ee6639b1368286f8c0f97b3ff366f1f9'
    assert sha256(SHARED/'manifest.json')==MANIFEST and sha256(SHARED/'pixels_uint8.npy')==PIXELS
    for name,digest in json.loads((ROUND3/'freeze.json').read_text())['files'].items(): assert sha256(ROUND3/name)==digest
    assert sha256(ROOT/'networks.py')==sha256(ROOT.parents[1]/'audit_sources/munit/networks.py')
    assert json.loads((ROOT/'verification.json').read_text())['status']=='passed'
    return dict(contract_sha256=CONTRACT,manifest_sha256=MANIFEST,pixels_sha256=PIXELS,evaluation_sha256=sha256(ROUND3/'evaluation_manifest.json'),utc=utc(),code_sha256={p.name:sha256(p) for p in ROOT.glob('*.py')})

class Ledger:
    def __init__(self):
        self.path=ROOT/'gpu_ledger.json'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else dict(experiment='TA-005',limit_seconds=28800,sessions=[])
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
    visit(t.net.state_dict());visit(t.opt_g.state_dict());visit(t.opt_d.state_dict());visit(t.sch_g.state_dict());visit(t.sch_d.state_dict());visit(t.update)
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
    del t,a,b;gc.collect();torch.cuda.empty_cache(); seed_all(); t=Trainer('cuda'); t.load(path,sampler,CONTRACT)
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

from generation import generate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--stage',default='all',choices=['all']);args=parser.parse_args()
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
            if ledger.elapsed()+max(30,qual['seconds_per_update']*5)>=27900:break
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
        ownership=json.loads((ROOT/'lease_ownership.json').read_text());ownership['released_utc']=utc();write(ROOT/'lease_ownership.json',ownership)

if __name__=='__main__':main()
