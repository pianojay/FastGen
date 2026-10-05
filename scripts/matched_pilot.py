"""Authorized finite paired learning pilot. No search, sweep, or successor."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from soflow_fastgen.adversarial import AdversarialComponents, endpoint
from soflow_fastgen.audit_support import ClassSampler, job_seconds
from soflow_fastgen.common import ROOT, atomic_json, atomic_save, digest, resources, source_provenance, verify_protected
from soflow_fastgen.data import datasets
from soflow_fastgen.models import Soflow
from soflow_fastgen.pilot import BATCH,HORIZON,LIMITS,TOTAL_LIMIT,D_SHA,D_MANIFEST_SHA,config,rho,index_streams,GSampler,restore_prepared_d
from soflow_fastgen.stylegan2 import verify_vendor
from scripts.smoke_adversarial import SOURCE,SOURCE_SHA,PARTITION_SHA,tensor_hash,recovery_identity
from scripts.pre_audit import GPUS,LOCKS,batch

AUDIT=ROOT/'runs/pre_program_audit_20261004'
D_SOURCE=AUDIT/'d_seed20261004/d_step0512.pt'
REFERENCE=ROOT/'runs/pokemon_unet128_seed42/reference_dev'
REFERENCE_SHA='1c926031c13a015673774e50991e881d5a76a63e5b84e44abbc628f9ecc82f71'


def prepare(run):
    if run.exists():raise FileExistsError('Fresh run required; never reset budget')
    verify_protected();verify_vendor()
    for p,sha in ((SOURCE,SOURCE_SHA),(D_SOURCE,D_SHA),(AUDIT/'frozen.json',D_MANIFEST_SHA),
                  (ROOT/'artifacts/development_split.json',PARTITION_SHA)):
        if digest(p)!=sha:raise ValueError('Source identity mismatch: '+str(p))
    record=dict(format_version=1,authorization='User: Order subagent Soflow matched pilot draft to carry out the pilot.',
        question='one bounded matched generator-learning response pilot',source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA,
        d_checkpoint=str(D_SOURCE),d_sha256=D_SHA,d_manifest_sha256=D_MANIFEST_SHA,partition_sha256=PARTITION_SHA,
        source=source_provenance(),component_config=asdict(config()),horizon=HORIZON,batch=BATCH,
        limits=LIMITS,total_limit=TOTAL_LIMIT,ramp_updates=64,ema_halflife_images=2048,
        evaluation_seeds=[7301,7302,7303],fid_practical_threshold=2.,checkpoint_selection='final EMA step512 only',
        diagnostics='seed7301; all151classes; same-class full14059 optimization neighbors; 16/32/64 RGB MSE with reference horizontal flip',
        protocol_sha256=digest(ROOT/'docs/adversarial/MATCHED_PILOT_DRAFT.md'),qualification_only=False)
    run.mkdir(parents=True);atomic_json(run/'frozen.json',record)
    print(json.dumps({'prepared':str(run),'sha256':digest(run/'frozen.json')}),flush=True)


def setup(run):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    m=json.loads((run/'frozen.json').read_text());verify_protected();verify_vendor()
    for name,expected in m['source']['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    if digest(ROOT/'artifacts/development_split.json')!=PARTITION_SHA:raise RuntimeError('Partition changed')
    source=AdversarialComponents.checkpoint_source(SOURCE,SOURCE_SHA)
    train,_,_=datasets(source['config'])
    return m,source,train


def make_engine(source,arm):
    engine=AdversarialComponents(source,config())
    audit=json.loads((AUDIT/'frozen.json').read_text())
    ds=ClassSampler(audit['fit_groups'],20262004)
    if arm=='A':
        if digest(D_SOURCE)!=D_SHA:raise RuntimeError('Prepared D hash changed')
        state=torch.load(D_SOURCE,map_location='cpu',weights_only=False)
        restore_prepared_d(engine,state,ds)
        if tensor_hash(engine.critic.state_dict())!=tensor_hash(state['critic']):raise AssertionError('D import')
        if tensor_hash(engine.d_optimizer.state_dict())!=tensor_hash(state['optimizer']):raise AssertionError('D optimizer import')
        del state
    return engine,ds


def save_state(engine,gs,ds,path,extra=None):
    engine.save(path,extra=dict(g_sampler=gs.state_dict(),d_sampler=ds.state_dict(),phase_step=engine.g_steps,
                               protocol='matched-pilot-v1',**(extra or {})))


def restore_state(engine,gs,ds,path):
    extra=engine.load(path)
    if extra['protocol']!='matched-pilot-v1' or extra['phase_step']!=engine.g_steps:raise ValueError('Phase recovery mismatch')
    gs.load_state_dict(extra['g_sampler']);ds.load_state_dict(extra['d_sampler'])
    return extra


def pair(engine,gs,ds,train,arm):
    step=engine.g_steps+1;started=time.monotonic();drow=None
    if arm=='A':
        pos,_=ds.draw(BATCH);x,c=batch(train,pos)
        drow=engine.discriminator_update(x,c)
        del x,c
    pos=gs.draw(BATCH);x,c=batch(train,pos)
    seeds=index_streams(engine,step);before=tensor_hash(engine.streams.states['consistency'])
    grow=engine.generator_update(x,c,rho=rho(step) if arm=='A' else 0.)
    torch.cuda.synchronize()
    row=dict(step=step,arm=arm,positions=pos,labels=c.cpu().tolist(),stream_seeds=seeds,
        scm_rng_before=before,scm_rng_after=tensor_hash(engine.streams.states['consistency']),
        rho=rho(step) if arm=='A' else 0.,d=drow,g=grow,seconds=time.monotonic()-started,
        g_images=engine.g_images,d_images=engine.d_images,resources=resources())
    if row['resources']['allocated_bytes']>=20_000_000_000:raise RuntimeError('Memory limit')
    return row


def snapshot(engine,folder,step):
    for kind,model in (('raw',engine.model),('ema',engine.ema)):
        atomic_save(folder/f'{kind}_{step:04d}.pt',dict(weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
            architecture=engine.source_config['architecture'],weight_source=kind,g_steps=step,source_sha256=SOURCE_SHA))


def d_scores(engine,train,step,folder):
    audit=json.loads((AUDIT/'frozen.json').read_text());records=[]
    rng=torch.Generator(device='cuda').manual_seed(6201)
    with torch.no_grad():
        for start in range(0,604,4):
            held,labels=batch(train,audit['hold_positions'][start:start+4])
            fit,_=batch(train,audit['fit_diagnostic_positions'][start:start+4])
            z=torch.randn(4,3,64,64,device='cuda',generator=rng)
            _,fake=endpoint(engine.model,z,labels)
            a=engine.critic(held,labels);b=engine.critic(fit,labels);f=engine.critic(fake,labels)
            w=engine.critic(held,(labels+37)%151)
            records.extend(dict(category=int(c),held=float(x),fit=float(y),fake=float(v),wrong=float(q))
                           for c,x,y,v,q in zip(labels,a,b,f,w))
    atomic_json(folder/f'd_scores_{step:04d}.json',dict(step=step,records=records,
        held_accuracy=float(np.mean([r['held']>r['fake'] for r in records])),
        fit_accuracy=float(np.mean([r['fit']>r['fake'] for r in records]))))


def qualify(run,folder):
    # Only targeted wrapper and unchanged loss contracts; no A0-A4 rerun.
    names=['test_matched_pilot','test_adversarial.SmallGPUContracts',
           'test_adversarial.LossContracts.test_rpgan_signs_and_real_branch',
           'test_adversarial.LossContracts.test_exact_group_penalty_r2_isolation_and_flip']
    code="import sys,unittest;sys.path.insert(0,'tests');unittest.main(module=None,argv=['unittest','-v']+sys.argv[1:])"
    subprocess.run([sys.executable,'-c',code,*names],cwd=ROOT,check=True)
    _,source,train=setup(run);engine,ds=make_engine(source,'A');del source
    gs=GSampler(len(train));gs.draw(63*BATCH)
    engine.g_steps=63;engine.g_images=63*BATCH  # disposable boundary fixture only
    g0=tensor_hash(engine.model.state_dict());e0=tensor_hash(engine.ema.state_dict())
    assert g0==e0 and not engine.g_optimizer.state
    save_state(engine,gs,ds,folder/'replay.pt')
    row1=pair(engine,gs,ds,train,'A');identity=recovery_identity(engine)
    expected_gs=gs.state_dict();expected_ds=ds.state_dict()
    restore_state(engine,gs,ds,folder/'replay.pt')
    row2=pair(engine,gs,ds,train,'A')
    assert recovery_identity(engine)==identity and gs.state_dict()==expected_gs and ds.state_dict()==expected_ds
    for key in ('positions','labels','stream_seeds','scm_rng_before','scm_rng_after','rho','g','d'):
        assert row1[key]==row2[key],key
    snapshot(engine,folder,64)
    architecture=engine.source_config['architecture']
    weights={'backbone.'+key:value.detach().cpu() for key,value in engine.ema.state_dict().items()}
    for mode,nfe in (('one_nfe',1),('few_nfe',4)):
        atomic_save(folder/(mode+'.ckpt'),{'state_dict':dict(weights,sampling_nfe=torch.tensor(nfe))})
    mem=resources();del engine,weights;gc.collect();torch.cuda.empty_cache()
    loads=[]
    for mode in ('one_nfe','few_nfe'):
        p=subprocess.run([sys.executable,'-m','scripts.smoke_adversarial','--load-only',str(folder/(mode+'.ckpt')),
            '--architecture',json.dumps(architecture),'--mode',mode],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            check=True,capture_output=True,text=True,timeout=120)
        loads.append(json.loads(p.stdout.strip().splitlines()[-1]))
    atomic_json(folder/'result.json',dict(state='passed',attempted_d=2,attempted_g=2,exact_recovery=True,
        audited_d_optimizer_restored=True,initial_online_ema_hash=g0,resources=mem,independent_loads=loads,updates=[row1,row2]))


def train_arm(run,folder,arm,resume=False):
    _,source,train=setup(run);engine,ds=make_engine(source,arm);del source
    gs=GSampler(len(train));started=time.monotonic()
    if resume:
        latest=max(folder.glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
        restore_state(engine,gs,ds,latest)
        previous=(folder/'updates.jsonl').read_text().splitlines()
        archive=folder/f'updates_interrupted_{time.time_ns()}.jsonl'
        (folder/'updates.jsonl').rename(archive)
        (folder/'updates.jsonl').write_text(''.join(line+'\n' for line in previous if json.loads(line)['step']<=engine.g_steps))
    else:
        atomic_json(folder/'initial.json',dict(model_sha256=tensor_hash(engine.model.state_dict()),ema_sha256=tensor_hash(engine.ema.state_dict()),
        parameters=engine.counts,inherited_d_updates=engine.d_steps,config=asdict(config())))
        save_state(engine,gs,ds,folder/'step_0000.pt')
        if arm=='A':d_scores(engine,train,0,folder)
    with (folder/'updates.jsonl').open('a',buffering=1) as log:
        while engine.g_steps<HORIZON:
            row=pair(engine,gs,ds,train,arm);log.write(json.dumps(row)+'\n')
            k=engine.g_steps
            if k%16==0:
                atomic_json(folder/'status.json',dict(state='training',arm=arm,g_steps=k,d_steps=engine.d_steps,seconds=time.monotonic()-started))
                print(json.dumps(dict(arm=arm,step=k,seconds=row['seconds'],grad_norm=row['g']['grad_norm'])),flush=True)
            if k%64==0:
                save_state(engine,gs,ds,folder/f'step_{k:04d}.pt')
                previous=folder/f'step_{k-64:04d}.pt'
                if k-64 not in (0,64,128,256,512) and previous.exists():previous.unlink()
            if k in (128,512):
                snapshot(engine,folder,k)
                if arm=='A':d_scores(engine,train,k,folder)
    atomic_json(folder/'result.json',dict(state='completed',arm=arm,g_steps=engine.g_steps,d_steps=engine.d_steps,
        g_images=engine.g_images,d_images=engine.d_images,resources=resources(),seconds=time.monotonic()-started,
        final_checkpoint=str(folder/'step_0512.pt'),final_sha256=digest(folder/'step_0512.pt'),parameters=engine.counts))


def eval_all(run,folder):
    from soflow_fastgen.paired_evaluation import evaluate_schedule,fingerprint,image_manifest
    m,source,train=setup(run);del train
    if fingerprint(image_manifest(REFERENCE))!=REFERENCE_SHA:raise RuntimeError('Development reference identity')
    source_weights=source['ema'];architecture=source['config']['architecture'];del source
    initial=[json.loads((run/a/'initial.json').read_text()) for a in ('A','C')]
    if initial[0]['model_sha256']!=initial[1]['model_sha256'] or initial[0]['ema_sha256']!=initial[1]['ema_sha256']:
        raise AssertionError('Unmatched G initialization')
    rows={a:[json.loads(x) for x in (run/a/'updates.jsonl').read_text().splitlines()] for a in ('A','C')}
    if len(rows['A'])!=512 or len(rows['C'])!=512:raise AssertionError('Incomplete paired horizon')
    for a,c in zip(rows['A'],rows['C']):
        for key in ('step','positions','labels','stream_seeds','scm_rng_before','scm_rng_after'):
            if a[key]!=c[key]:raise AssertionError('SCM pairing failed: '+key)
    atomic_json(folder/'pairing.json',dict(passed=True,steps=512,exact_data_and_scm_stream_hashes=True))
    # Mandatory fixed endpoints first. Intermediate/raw passes cannot consume their reserve.
    tasks=[(label,512,'ema',nfe,seed) for nfe,seeds in ((1,[7301,7302,7303]),(4,[7301]))
           for seed in seeds for label in ('S3','C','A')]
    tasks += [(a,k,w,1,7301) for k in (128,512) for a in ('C','A') for w in ('raw','ema')
              if not(k==512 and w=='ema')]
    assert len(tasks)==18
    results=[]
    for label,step,kind,nfe,seed in tasks:
        model=Soflow(**architecture).cuda().eval().requires_grad_(False)
        if label=='S3':model.load_state_dict(source_weights);sha=SOURCE_SHA
        else:
            p=run/label/f'{kind}_{step:04d}.pt';state=torch.load(p,map_location='cpu',weights_only=False)
            assert state['weight_source']==kind and state['g_steps']==step
            model.load_state_dict(state['weights']);del state;sha=digest(p)
        protocol=dict(run_dir=str(run),checkpoint_sha256=sha,partition_sha256=PARTITION_SHA,nfe=nfe,eval_batch_size=32,
                      weight_source='ema' if label=='S3' else kind)
        schedule=dict(name=f'{label}_{step if label!="S3" else 0}_{kind}_nfe{nfe}',intermediate_times=None)
        result=evaluate_schedule(model,protocol,schedule,seed,REFERENCE,m['source']['git_commit'],REFERENCE_SHA)
        result=dict(result,arm=label,step=step if label!='S3' else 0,weight_source=kind)
        results.append(result);atomic_json(folder/'results.json',results)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic_json(folder/'result.json',dict(state='completed',passes=len(results),paired=True,resources=resources()))


def worker(run,phase,gpu,resume=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or os.environ.get('PYTHONNOUSERSITE')!='1':raise RuntimeError('Environment/device')
    folder=run/phase;folder.mkdir(exist_ok=resume)
    with LOCKS[gpu].open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if uuid!=GPUS[gpu] or any(line.startswith(uuid) for line in occupied.splitlines()):raise RuntimeError('GPU identity/ownership')
        atomic_json(folder/'lease.json',dict(gpu=gpu,uuid=uuid,pid=os.getpid(),started=time.time()))
        try:
            if phase=='qualify':qualify(run,folder)
            elif phase in ('A','C'):train_arm(run,folder,phase,resume)
            elif phase=='evaluate':eval_all(run,folder)
            else:raise ValueError(phase)
        except BaseException as error:
            atomic_json(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise


def supervise(run,resume=False):
    children={};ledger=run/'ledger.json'
    if ledger.exists() and not resume:raise FileExistsError('Existing ledger; never reset budget')
    jobs=json.loads(ledger.read_text())['jobs'] if resume else []
    if resume:
        if not jobs or any(r.get('state')=='running' for r in jobs):raise RuntimeError('Reconcile live/orphan jobs before resume')
        failed=[r for r in jobs if r['returncode']]
        if any(r['name'] not in ('A','C') or r.get('stop_reason') or r['returncode'] not in (-15,-9,143,137) for r in failed):
            raise RuntimeError('Only external interruption permits recovery')
        if any(sum(r['name']==a for r in jobs)>1 for a in ('A','C')):raise RuntimeError('One resume per arm maximum')
    def record():atomic_json(ledger,dict(limit_seconds=TOTAL_LIMIT,used_seconds=job_seconds(jobs,time.time()),jobs=jobs))
    def launch(phase,gpu):
        prior=[r for r in jobs if r['name']==phase]
        recovering=bool(prior)
        log=(run/f'{phase}{"_resume" if recovering else ""}.log').open('w')
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        child=subprocess.Popen([sys.executable,'-m','scripts.matched_pilot','worker','--run-dir',str(run),'--phase',phase,'--gpu',str(gpu)]+(['--resume'] if recovering else []),
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=phase,gpu=gpu,pid=child.pid,started=time.time(),watchdog_seconds=LIMITS[phase]-sum(r['ended']-r['started'] for r in prior),state='running')
        jobs.append(row);children[phase]=(child,log,row);record()
    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def finish():
        while children:
            now=time.time();total=job_seconds(jobs,now)
            for name,(child,log,row) in list(children.items()):
                if total>=TOTAL_LIMIT or now-row['started']>=row['watchdog_seconds']:
                    row['stop_reason']='aggregate_limit' if total>=TOTAL_LIMIT else 'stage_watchdog';stop(child)
                code=child.poll()
                if code is not None:
                    row.update(ended=time.time(),returncode=code,state='completed' if code==0 else 'incomplete')
                    log.close();del children[name];print(json.dumps(row),flush=True)
            record()
            if children:time.sleep(1)
        latest={r['name']:r for r in jobs}
        if any(x['returncode'] for x in latest.values()):raise RuntimeError('Failed stage; no automatic retry or successor')
    try:
        completed={r['name'] for r in jobs if r['returncode']==0}
        if 'qualify' not in completed:launch('qualify',1);finish()
        if 'A' not in completed:launch('A',1)
        if 'C' not in completed:launch('C',2)
        finish()
        if 'evaluate' not in completed:launch('evaluate',1);finish()
    finally:
        for child,log,row in children.values():
            stop(child);row.update(ended=time.time(),returncode=child.returncode,state='interrupted');log.close()
        children.clear();record()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','supervise','worker']);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--phase',choices=list(LIMITS));p.add_argument('--gpu',type=int,choices=[1,2]);args=p.parse_args()
    if args.action=='prepare':prepare(args.run_dir)
    elif args.action=='supervise':supervise(args.run_dir,args.resume)
    else:worker(args.run_dir,args.phase,args.gpu,args.resume)
