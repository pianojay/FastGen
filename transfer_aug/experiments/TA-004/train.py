"""Frozen TA-004 execution. Run only on explicitly authorized physical GPU 2."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import datetime, fcntl, hashlib, json, random, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cut64 import CUT64,PairSampler,seed_all
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
SHARED=ROOT/'transfer_aug/shared/round2'
UUID='GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2'
CONTRACT='1f296b91bb75055d8604d5f1e6dae9f8897343406eee3c88c5a4408869c2da36'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def preflight():
    assert sha(R/'CONTRACT_v1.md')==CONTRACT
    assert sha(SHARED/'TRAINING_PROTOCOL_v1.md')=='ae9323b66e4969f87f000d08d0227cc6770e9f1377bc7f62deb6c92e39409c94'
    freeze=json.loads((SHARED/'freeze.json').read_text())
    for n,h in freeze['files'].items():assert sha(SHARED/n)==h,n
    setup=json.loads((R/'setup_result.json').read_text());assert setup['status']=='setup_completed'
    provenance=json.loads((ROOT/'transfer_aug/audit_sources/cut/provenance.json').read_text())
    for n,v in provenance['files'].items():assert sha(ROOT/'transfer_aug/audit_sources/cut'/n)==v['sha256']
    records=json.loads((SHARED/'manifest.json').read_text())['records']
    assert all(r['partition']=='A1' for r in records)
    opt={r['pixel_sha256'] for r in records if r['role']=='optimization'}
    diag={r['pixel_sha256'] for r in records if r['role']=='diagnostic'}
    assert not opt&diag
    pixels=np.load(SHARED/'pixels_uint8.npy',mmap_mode='r')
    for r in records:assert hashlib.sha256(pixels[r['pixel_index']].tobytes()).hexdigest()==r['pixel_sha256']
    domains={}
    for r in records:
        if r['role']=='optimization':domains.setdefault(r['category'],{'D0':[],'D1':[]})[r['domain']].append(r['pixel_index'])
    assert len(domains)==151
    write(R/'training_preflight.json',{'timestamp':utc(),'contract_sha256':CONTRACT,'input_hashes':freeze['files'],'all_A1':True,'diagnostic_exclusion':True,'all_pixel_hashes_verified':True,'source_hashes_verified':True,'code_hashes':{p.name:sha(p) for p in R.glob('*.py')}})
    return records,pixels,domains

def main():
    records,pixels,domains=preflight()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    resource_path=R/'gpu_resource_ledger.json'
    ledger=json.loads(resource_path.read_text()) if resource_path.exists() else {'gpu_budget_seconds':28800,'sessions':[]}
    if any(s.get('ended_at') is None for s in ledger['sessions']):raise RuntimeError('Unclosed prior GPU session requires conservative reconciliation')
    previous=sum(s['gpu_seconds'] for s in ledger['sessions'])
    write(resource_path,ledger)
    lock=open('/tmp/fastgen-gpu2.lock','r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    session={'pid':os.getpid(),'started_at':utc(),'ended_at':None,'gpu_seconds':0,'gpu_uuid':UUID,'lease_inode':os.fstat(lock.fileno()).st_ino}
    ledger['sessions'].append(session);write(resource_path,ledger)
    stage='qualification';model=None
    def elapsed():return previous+time.monotonic()-started
    def status(extra=None):
        session['gpu_seconds']=time.monotonic()-started;write(resource_path,ledger)
        payload={'experiment_id':'TA-004','stage':stage,'owned_pid':os.getpid(),'updated_at':utc(),'updates':model.updates if model else 0,'maximum_updates':20000,'gpu_seconds':elapsed(),'gpu_uuid':UUID,'checkpoint':str(R/'latest.pt')}
        if extra:payload.update(extra)
        write(R/'status.json',payload)
    def batch(sampler):
        draws=[sampler.draw() for _ in range(16)];arr=[]
        for d in ('D0','D1'):
            u=np.stack([pixels[q[d]][:,::-1,:] if q['flip_'+d] else pixels[q[d]] for q in draws])
            arr.append(torch.from_numpy(u).permute(0,3,1,2).to('cuda',dtype=torch.float32)/127.5-1)
        return arr
    def model_digest(m):
        h=hashlib.sha256()
        for t in m.state_dict().values():h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    try:
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        assert f'2, {UUID}' in devices
        occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert not any(UUID in line and int(line.split(',')[1])!=os.getpid() for line in occupancy.splitlines()),occupancy
        assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUID.removeprefix('GPU-')
        if (R/'latest.pt').exists():
            seed_all();model=CUT64('cuda');sampler=PairSampler(domains);model.restore(R/'latest.pt',sampler)
        else:
            seed_all();model=CUT64('cuda');sampler=PairSampler(domains)
            assert sum(p.numel() for p in model.parameters())==9061060
            torch.cuda.reset_peak_memory_stats();qstart=time.monotonic();qloss=[]
            for i in range(20):qloss.append(model.step(*batch(sampler)));status({'qualification_updates':i+1})
            torch.cuda.synchronize();qseconds=time.monotonic()-qstart
            allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            assert allocated<20_000_000_000
            model.checkpoint(R/'qualification.pt',sampler)
            reference_loss=model.step(*batch(sampler));reference_digest=model_digest(model)
            model.restore(R/'qualification.pt',sampler)
            restored_loss=model.step(*batch(sampler));restored_digest=model_digest(model)
            assert reference_loss==restored_loss and reference_digest==restored_digest
            write(R/'qualification.json',{'status':'passed','updates':20,'additional_recovery_updates':2,'batch_size':16,'seconds_20_updates':qseconds,'seconds_per_update':qseconds/20,'projected_20000_seconds':qseconds*1000,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'gpu_uuid':UUID,'framework_parameters':9061060,'finite_gradients_losses':True,'exact_gpu_next_step_recovery':True,'reference_loss':reference_loss,'restored_loss':restored_loss,'reference_digest':reference_digest,'restored_digest':restored_digest,'torch_version':torch.__version__,'device_name':torch.cuda.get_device_name(),'reset_before_production':True})
            del model;torch.cuda.empty_cache();seed_all();model=CUT64('cuda');sampler=PairSampler(domains)
            model.checkpoint(R/'initial.pt',sampler);model.checkpoint(R/'latest.pt',sampler)
        stage='training';status();last_status=time.monotonic();times=[]
        with (R/'losses.jsonl').open('a',buffering=1) as log:
            while model.updates<20000:
                # Ensure next block/checkpoint and 600s final reserve fit.
                estimate=max(times[-100:],default=1.)
                if elapsed()+estimate+10>=28200:break
                t=time.monotonic();loss=model.step(*batch(sampler));times.append(time.monotonic()-t)
                if model.updates%20==0:log.write(json.dumps({'update':model.updates,'gpu_seconds':elapsed(),'lr':model.optimizers['G'].param_groups[0]['lr'],**loss})+'\n')
                if model.updates%1000==0:
                    model.checkpoint(R/'latest.pt',sampler)
                    if model.updates in (1000,5000,10000):shutil.copyfile(R/'latest.pt',R/f'update_{model.updates}.pt')
                if time.monotonic()-last_status>30:status({'last_losses':loss,'seconds_per_update_recent':sum(times[-100:])/len(times[-100:])});last_status=time.monotonic()
                if torch.cuda.max_memory_allocated()>=20_000_000_000:raise RuntimeError('Memory cap exceeded')
        model.checkpoint(R/'final.pt',sampler);shutil.copyfile(R/'final.pt',R/'latest.pt')
        production_updates=model.updates
        stage='generation';status()
        out=R/'outputs';out.mkdir(exist_ok=True);(out/'png').mkdir(exist_ok=True)
        diagnostic={d:sorted([r for r in records if r['role']=='diagnostic' and r['domain']==d],key=lambda r:r['path']) for d in ('D0','D1')}
        review_cats=json.loads((ROOT/'transfer_aug/domain_discovery/round2/review_selection.json').read_text())['categories']
        associations=[]
        def generate(rows):
            values=[]
            with torch.no_grad():
                for offset in range(0,len(rows),16):
                    u=np.stack([pixels[r['pixel_index']] for r in rows[offset:offset+16]])
                    x=torch.from_numpy(u).permute(0,3,1,2).to('cuda',dtype=torch.float32)/127.5-1
                    values.append(model.G(x).cpu().permute(0,2,3,1).numpy())
            v=np.concatenate(values);assert np.isfinite(v).all();return v
        def quant(v):return np.rint(np.clip((v+1)*127.5,0,255)).astype(np.uint8)
        final_arrays={}
        for domain,label in [('D0','D0_to_D1'),('D1','D1_identity')]:
            arr=generate(diagnostic[domain]);final_arrays[domain]=arr;np.save(out/f'{label}_float.npy',arr)
            for i,(r,u) in enumerate(zip(diagnostic[domain],quant(arr))):
                file=out/'png'/f'{label}_{i:03d}.png';Image.fromarray(u).save(file)
                associations.append({'direction':label,'array_index':i,'png':str(file),'png_sha256':sha(file),'output_pixel_sha256':hashlib.sha256(u.tobytes()).hexdigest(),**r})
        write(out/'associations.json',associations)
        # Identical batching on rerun prevents batch-size-dependent convolution rounding.
        regenerated=generate(diagnostic['D0']);review_indices=[i for i,r in enumerate(diagnostic['D0']) if r['category'] in review_cats]
        exact=np.array_equal(quant(regenerated)[review_indices],quant(final_arrays['D0'])[review_indices]);assert exact
        milestones=[('initial',R/'initial.pt'),('1000',R/'update_1000.pt'),('5000',R/'update_5000.pt'),('10000',R/'update_10000.pt'),('final',R/'final.pt')]
        for label,path in milestones:
            if not path.exists():continue
            if label=='final' and production_updates in (1000,5000,10000):continue
            state=torch.load(path,map_location='cuda',weights_only=False);model.load_state_dict(state['model']);del state
            for domain in ('D0','D1'):
                rows=[r for r in diagnostic[domain] if r['category'] in review_cats]
                np.save(out/f'review_{domain}_{label}.npy',generate(rows))
                write(out/f'review_{domain}_rows.json',rows)
            status({'generation_milestone':label})
        write(R/'training_complete.json',{'status':'completed' if production_updates==20000 else 'budget_limited','updates':production_updates,'gpu_seconds_before_release':elapsed(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'deterministic_review_rerun_exact':exact,'review_rerun_count':len(review_indices),'translation_count':302,'target_identity_count':302,'final_checkpoint_sha256':sha(R/'final.pt'),'input_hashes_after':{n:sha(SHARED/n) for n in ('manifest.json','pixels_uint8.npy')},'checkpoints':{p.name:sha(p) for p in R.glob('*.pt')},'gpu_uuid':UUID})
        stage='gpu_complete_awaiting_cpu_report';status()
        assert elapsed()<28800
    except BaseException:
        stage='failed';write(R/'failure.json',{'timestamp':utc(),'traceback':traceback.format_exc(),'gpu_seconds':elapsed()});status();raise
    finally:
        session['ended_at']=utc();session['gpu_seconds']=time.monotonic()-started;write(resource_path,ledger)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
