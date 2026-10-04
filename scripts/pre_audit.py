"""Fixed diagnostic workload. This module cannot launch autoresearch."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image

from soflow_fastgen.adversarial import AdversarialComponents, ComponentConfig, endpoint, frozen_parameters, generator_objective
from soflow_fastgen.audit_support import ClassSampler, PhaseFixture, add_vector, job_seconds, range_penalty, vector_cosine, vector_norm
from soflow_fastgen.common import ROOT, atomic_json, atomic_save, digest, resources, source_provenance, verify_protected
from soflow_fastgen.data import datasets
from soflow_fastgen.stylegan2 import verify_vendor
from scripts.smoke_adversarial import SOURCE, SOURCE_SHA, PARTITION_SHA, tensor_hash, recovery_identity

GPUS={1:'GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888',2:'GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2'}
LOCKS={1:ROOT/'runs/gpu_1.lock',2:Path('/tmp/fastgen-gpu2.lock')}
STEPS=(0,32,128,512)


def write_status(folder, **values):
    atomic_json(folder/'status.json',dict(timestamp=time.time(),**values))


def prepare(run):
    if run.exists():raise FileExistsError('Use a fresh audit directory')
    verify_protected();verify_vendor()
    if digest(SOURCE)!=SOURCE_SHA or digest(ROOT/'artifacts/development_split.json')!=PARTITION_SHA:
        raise RuntimeError('Source/partition identity mismatch')
    cfg=json.loads((ROOT/'configs/research_s3_train.json').read_text())
    train,_,mapping=datasets(cfg)
    manifest=json.loads((ROOT/'artifacts/development_split.json').read_text())
    duplicate_paths={p for group in manifest['exact_duplicate_groups'] for p in group}
    groups=[[] for _ in range(151)]
    for pos,original in enumerate(train.indices):groups[train.dataset.category_ids[original]].append(pos)
    held=[];fit=[];fit_diag=[]
    for c,group in enumerate(groups):
        order=sorted(group,key=lambda p:hashlib.sha256(f'pre-audit:{c}:{train.dataset.paths[train.indices[p]]}'.encode()).hexdigest())
        eligible=[p for p in order if train.dataset.paths[train.indices[p]] not in duplicate_paths]
        assert len(eligible)>=8
        h=eligible[:4]; held.append(h);fit.append([p for p in group if p not in set(h)])
        fit_diag.append([p for p in order if p not in set(h)][:4])
    # Fixed mixed-class groups, unlike sorted category blocks.
    permutation=np.random.default_rng(6200).permutation(604).tolist()
    flatten=lambda gs:[p for g in gs for p in g]
    labels=np.repeat(np.arange(151),4)[permutation].tolist()
    hold_positions=np.array(flatten(held))[permutation].tolist()
    fit_positions=np.array(flatten(fit_diag))[permutation].tolist()
    record=dict(format_version=1,source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA,
        partition_sha256=PARTITION_SHA,limit_gpu_job_seconds=43200,training_seeds=[20261004,20261005],
        noise_seeds=[6201,6202],checkpoints=list(STEPS),d_updates=512,d_watchdog_seconds=10800,
        d_batch=32,d_lr=2e-5,d_betas=[0,.99],penalty_beta=.1,flip_probability=.25,
        a4_d_update_limit=16,a4_g_update_limit=16,group_size=4,consistency_microbatch=2,
        fit_groups=fit,hold_positions=hold_positions,fit_diagnostic_positions=fit_positions,
        diagnostic_labels=labels,hold_original_indices=[train.indices[p] for p in hold_positions],
        hold_paths=[train.dataset.paths[train.indices[p]] for p in hold_positions],
        category_names=[name for name,_ in sorted(mapping.items(),key=lambda pair:pair[1])],
        qualification_only=True,range_training_coefficient=0,source=source_provenance())
    run.mkdir(parents=True);atomic_json(run/'frozen.json',record)
    print(json.dumps({'frozen_manifest':str(run/'frozen.json'),'sha256':digest(run/'frozen.json')}),flush=True)


def setup(run):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    manifest=json.loads((run/'frozen.json').read_text())
    if digest(ROOT/'artifacts/development_split.json')!=manifest['partition_sha256']:
        raise RuntimeError('Partition identity changed')
    verify_protected();verify_vendor()
    for name,expected in manifest['source']['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    source=AdversarialComponents.checkpoint_source(SOURCE,SOURCE_SHA)
    cfg=source['config'];train,_,_=datasets(cfg)
    return manifest,source,train


def batch(train,positions):
    rows=[train[int(i)] for i in positions]
    return torch.stack([x for x,c in rows]).cuda(),torch.stack([c for x,c in rows]).cuda()


def png(tensor,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    to_pil_image(tensor.detach().cpu().add(1).div(2).clamp(0,1)).save(path)


def saturation(run,folder):
    m,source,train=setup(run)
    engine=AdversarialComponents(source);del source
    engine.model.requires_grad_(False)
    records=[];maps=np.zeros((151,64,64),dtype=np.float64)
    raw_tail=[]
    for seed in m['noise_seeds']:
        rng=torch.Generator(device='cuda').manual_seed(seed)
        for c in range(151):
            with torch.no_grad():
                z=torch.randn(20,3,64,64,device='cuda',generator=rng)
                raw,visible=endpoint(engine.model,z,torch.full((20,),c,device='cuda',dtype=torch.long))
                overflow=raw.abs().sub(1).relu();mask=overflow>0
                maps[c]+=mask.float().sum((0,1)).cpu().numpy()/120
                fractions=mask.float().flatten(1).mean(1).cpu().tolist()
                excess=overflow.flatten(1).mean(1).cpu().tolist()
                maxima=raw.abs().flatten(1).amax(1).cpu().tolist()
                raw_tail.extend(raw.flatten()[::97].cpu().tolist())
            for j in range(20):
                png(visible[j],folder/f'images/{seed}/{c:03d}/{j:02d}.png')
                records.append(dict(seed=seed,category=c,draw=j,clip_fraction=fractions[j],mean_overshoot=excess[j],max_abs=maxima[j]))
            if c%25==0:write_status(folder,phase='A1-generated',seed=seed,category=c)
    real=np.zeros((151,3),dtype=np.float64)
    for start in range(0,len(train),32):
        x,c=batch(train,list(range(start,min(start+32,len(train)))))
        boundary=x.abs().eq(1).float().flatten(1).mean(1).cpu().numpy()
        near=x.abs().ge(.99).float().flatten(1).mean(1).cpu().numpy()
        for label,a,b in zip(c.cpu().tolist(),boundary,near):real[label]+=np.array([a,b,1.])
    summaries=[]
    for c in range(151):
        selected=[row for row in records if row['category']==c]
        values=[row['clip_fraction'] for row in selected]
        summaries.append(dict(category=c,name=m['category_names'][c],clip_mean=float(np.mean(values)),
            image_clip_quantiles=np.quantile(values,[0,.5,.9,.95,1]).tolist(),mean_overshoot=float(np.mean([r['mean_overshoot'] for r in selected])),
            real_exact_boundary=real[c,0]/real[c,2],real_near_boundary=real[c,1]/real[c,2],real_count=int(real[c,2])))
        Image.fromarray(np.uint8(np.clip(maps[c],0,1)*255)).save(folder/f'images/map_{c:03d}.png')
        original=m['hold_positions'][m['diagnostic_labels'].index(c)]
        png(train[original][0],folder/f'images/ref_{c:03d}.png')
    atomic_json(folder/'image_metrics.json',records)
    atomic_json(folder/'result.json',dict(state='completed',generated_count=len(records),optimization_count=len(train),
        mean_clip_fraction=float(np.mean([r['clip_fraction'] for r in records])),categories=summaries,
        raw_pixel_quantiles=np.quantile(raw_tail,[0,.01,.5,.99,1]).tolist(),
        raw_quantile_sampling='every 97th flattened raw coordinate per 20-image class batch',resources=resources()))
    options=''.join(f'<option value="{c:03d}">{c}: {name}</option>' for c,name in enumerate(m['category_names']))
    html='''<!doctype html><meta charset="utf-8"><title>Frozen S3 saturation audit</title><style>body{background:#171a20;color:#eee;font:16px system-ui;max-width:1450px;margin:25px auto}img{width:96px;image-rendering:auto}#samples{display:grid;grid-template-columns:repeat(10,1fr);gap:5px} .bar{position:sticky;top:0;background:#171a20;padding:12px;display:flex;gap:20px;align-items:center}select{font-size:18px}</style><h1>Frozen S3: exact-clamp audit</h1><p>40 fixed noise draws per class. Reference is an optimization image; heatmap is channel-averaged clipping frequency (white = 100%). Generator is unchanged.</p><div class="bar"><select id="category">OPTIONS</select><img id="ref"><img id="map"></div><p id="stats"></p><div id="samples"></div><script>const metrics=METRICS;function show(){const c=document.querySelector('#category').value;document.querySelector('#ref').src=`images/ref_${c}.png`;document.querySelector('#map').src=`images/map_${c}.png`;const r=metrics[Number(c)];document.querySelector('#stats').textContent=`Generated clipping ${(r.clip_mean*100).toFixed(2)}%; real exact-boundary ${(r.real_exact_boundary*100).toFixed(2)}% (different quantities)`;document.querySelector('#samples').innerHTML=[6201,6202].flatMap(s=>Array.from({length:20},(_,i)=>`<img loading="lazy" title="seed ${s}, draw ${i}" src="images/${s}/${c}/${String(i).padStart(2,'0')}.png">`)).join('')}document.querySelector('#category').onchange=show;show()</script>'''
    (folder/'gallery.html').write_text(html.replace('OPTIONS',options).replace('METRICS',json.dumps(summaries)))


def accumulate_current(engine,total,weight):
    return add_vector(total,[p.grad for p in engine.parameters],weight)


def gradient_audit(engine,train,m,folder,step):
    rows=[];labels=torch.tensor(m['diagnostic_labels'],device='cuda'); n=len(labels)
    for noise_seed in m['noise_seeds']:
        rng=torch.Generator(device='cuda').manual_seed(noise_seed)
        original_streams={k:v.clone() for k,v in engine.streams.states.items()}
        engine.streams.states['consistency']=torch.Generator(device='cuda').manual_seed(noise_seed+9000).get_state()
        ga=gb=gr=gcvec=None;energy=masked=clipped=count=0.;train_scores=[];held_scores=[];fake_scores=[];wrong_scores=[];r1s=[];r2s=[]
        # Static G means SCM and range gradients can be reused across D checkpoints.
        cache=folder/f'fixed_gradients_seed{noise_seed}.pt'
        if step!=0:
            saved=torch.load(cache,map_location='cpu',weights_only=False)
            gcvec=saved['consistency'];gr=saved['range']
        for start in range(0,n,4):
            c=labels[start:start+4]
            real,_=batch(train,m['hold_positions'][start:start+4])
            fit,_=batch(train,m['fit_diagnostic_positions'][start:start+4])
            noise=torch.randn(4,3,64,64,device='cuda',generator=rng)
            raw,visible=endpoint(engine.model,noise,c)
            with frozen_parameters(engine.critic):
                # Leaf probes produce exact summed-logit input penalties.
                xr=real.detach().requires_grad_(True);xf=visible.detach().requires_grad_(True)
                dr=engine.critic(xr,c);df=engine.critic(xf,c)
                dx,=torch.autograd.grad(dr.sum(),xr);dy,=torch.autograd.grad(df.sum(),xf)
                r1s.append(float(dx.square().flatten(1).sum(1).mean()));r2s.append(float(dy.square().flatten(1).sum(1).mean()))
                held_scores.extend(dr.detach().cpu().tolist());fake_scores.extend(df.detach().cpu().tolist())
                with torch.no_grad():
                    train_scores.extend(engine.critic(fit,c).cpu().tolist())
                    wrong_scores.extend(engine.critic(real,(c+37)%151).cpu().tolist())
                loss=generator_objective(engine.critic,real,visible,c,torch.zeros(4,device='cuda',dtype=torch.bool))
                u,=torch.autograd.grad(loss,visible,retain_graph=True)
            mask=(raw.detach().abs()>1)
            actual=torch.autograd.grad(visible,engine.parameters,u.detach(),retain_graph=True,allow_unused=True)
            ga=add_vector(ga,actual,4/n);del actual
            blocked=torch.autograd.grad(raw,engine.parameters,u.detach()*mask,retain_graph=(step==0),allow_unused=True)
            gb=add_vector(gb,blocked,4/n);del blocked
            if step==0:
                rg=torch.autograd.grad(range_penalty(raw),engine.parameters,allow_unused=True)
                gr=add_vector(gr,rg,4/n);del rg
                engine.g_optimizer.zero_grad(set_to_none=True)
                # Restore stream after each group: diagnostics never advance training RNG.
                engine.consistency_backward(real,c)
                gcvec=accumulate_current(engine,gcvec,4/n)
                engine.g_optimizer.zero_grad(set_to_none=True)
            energy+=float(u.detach().square().sum());masked+=float((u.detach().square()*mask).sum())
            clipped+=int(mask.sum());count+=mask.numel()
            del raw,visible,loss,u,xr,xf,dr,df,dx,dy
            if start%100==0:write_status(folder,phase='A3',d_step=step,noise_seed=noise_seed,images=start)
            engine._memory()
        # Move summary vectors to CPU to bound GPU memory; no parameter update.
        ga=[x.cpu() if x is not None else None for x in ga];gb=[x.cpu() if x is not None else None for x in gb]
        if step==0:
            gr=[x.cpu() if x is not None else None for x in gr];gcvec=[x.cpu() if x is not None else None for x in gcvec]
            atomic_save(cache,dict(consistency=gcvec,range=gr))
        proxy=[a+b if a is not None else b for a,b in zip(ga,gb)]
        row=dict(d_step=step,noise_seed=noise_seed,count=n,clip_fraction=clipped/count,
            masked_image_gradient_energy=masked/energy if energy else None,
            actual_parameter_grad_norm=vector_norm(ga),blocked_parameter_grad_norm=vector_norm(gb),
            diagnostic_unmasked_parameter_grad_norm=vector_norm(proxy),consistency_grad_norm=vector_norm(gcvec),range_grad_norm=vector_norm(gr),
            actual_vs_blocked_cosine=vector_cosine(ga,gb),actual_vs_unmasked_cosine=vector_cosine(ga,proxy),
            actual_vs_consistency_cosine=vector_cosine(ga,gcvec),actual_vs_range_cosine=vector_cosine(ga,gr),
            range_vs_consistency_cosine=vector_cosine(gr,gcvec),r1_mean=float(np.mean(r1s)),r2_mean=float(np.mean(r2s)),
            held_real_score=float(np.mean(held_scores)),fit_pool_real_score=float(np.mean(train_scores)),fake_score=float(np.mean(fake_scores)),
            correct_minus_wrong_real_score=float(np.mean(np.array(held_scores)-wrong_scores)),
            held_pair_accuracy=float(np.mean(np.array(held_scores)>fake_scores)),fit_pool_pair_accuracy=float(np.mean(np.array(train_scores)>fake_scores)))
        # Group-level logit records preserve category associations for later audits.
        atomic_json(folder/f'diagnostic_scores_step{step:04d}_seed{noise_seed}.json',dict(labels=m['diagnostic_labels'],held=held_scores,fit=train_scores,fake=fake_scores,wrong=wrong_scores))
        engine.streams.states=original_streams
        rows.append(row);del ga,gb,gr,gcvec,proxy;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(event='gradient_audit',**row)),flush=True)
    atomic_json(folder/f'audit_step{step:04d}.json',rows)
    return rows


def discriminator_run(run,folder,seed):
    m,source,train=setup(run)
    engine=AdversarialComponents(source,ComponentConfig(seed=seed));del source
    sampler=ClassSampler(m['fit_groups'],seed+1000)
    source_g=tensor_hash(engine.model.state_dict());source_ema=tensor_hash(engine.ema.state_dict())
    all_rows=[];started=time.monotonic()
    with (folder/'updates.jsonl').open('a',buffering=1) as log:
        for step in range(513):
            if step in STEPS:
                atomic_save(folder/f'd_step{step:04d}.pt',dict(critic=engine.critic.state_dict(),optimizer=engine.d_optimizer.state_dict(),
                    streams=engine.streams.states,sampler=sampler.state_dict(),d_steps=engine.d_steps,d_images=engine.d_images,
                    source_sha256=SOURCE_SHA,manifest_sha256=digest(run/'frozen.json')))
                all_rows.extend(gradient_audit(engine,train,m,folder,step))
                if tensor_hash(engine.model.state_dict())!=source_g or tensor_hash(engine.ema.state_dict())!=source_ema:
                    raise AssertionError('Frozen generator changed')
                assert engine.g_steps==engine.g_images==0 and not engine.g_optimizer.state
            if step==512:break
            pos,_=sampler.draw(32);x,c=batch(train,pos)
            before=time.monotonic();row=engine.discriminator_update(x,c);torch.cuda.synchronize()
            row.update(seconds=time.monotonic()-before,step=step+1)
            log.write(json.dumps(row)+'\n')
            if (step+1)%16==0:
                write_status(folder,phase='A2',d_step=step+1,seconds=time.monotonic()-started)
                print(json.dumps(row),flush=True)
    atomic_json(folder/'result.json',dict(state='completed',seed=seed,d_steps=512,g_steps=0,generator_unchanged=True,
        generator_sha256=source_g,ema_sha256=source_ema,diagnostics=all_rows,resources=resources(),seconds=time.monotonic()-started))


def readiness(run,folder):
    m,source,train=setup(run);report=dict(updates=[],checks={},attempted_d=0,attempted_g=0)
    phase=PhaseFixture()
    for size in (8,32):
        engine=AdversarialComponents(source,ComponentConfig(seed=20261004,ema_decay=.99))
        sampler=ClassSampler(m['fit_groups'],20261004+2000)
        def update(kind):
            key='attempted_'+kind
            if report[key]>=16:raise RuntimeError('A4 attempted-update ceiling')
            report[key]+=1;atomic_json(folder/'progress.json',report)
            before=time.monotonic();pos,_=sampler.draw(size);x,c=batch(train,pos)
            rho=phase.rho(engine.d_steps,engine.g_steps) if kind=='g' else None
            row=engine.generator_update(x,c,rho=rho) if kind=='g' else engine.discriminator_update(x,c)
            torch.cuda.synchronize()
            report['updates'].append(dict(batch=size,kind=kind,rho=rho,seconds=time.monotonic()-before,**row))
            atomic_json(folder/'progress.json',report)
        update('d');update('d')
        update('d');update('g')
        update('d');update('g')
        before_path=folder/f'recovery_b{size}.pt'
        engine.save(before_path,extra=dict(sampler=sampler.state_dict(),phase=asdict(phase)))
        update('d');update('g');expected=recovery_identity(engine);expected_sampler=sampler.state_dict()
        extra=engine.load(before_path)
        assert extra['phase']==asdict(phase);sampler.load_state_dict(extra['sampler'])
        update('d');update('g')
        assert expected==recovery_identity(engine) and expected_sampler==sampler.state_dict()
        report['checks'][f'replay_batch{size}']=True
        report['checks'][f'rho_boundary_batch{size}']=[r['rho'] for r in report['updates'] if r['kind']=='g' and r['batch']==size]==[0.,.05,.1,.1]
        engine._memory();del engine;gc.collect();torch.cuda.empty_cache()
    report.update(state='completed',resources=resources(),phase=asdict(phase));atomic_json(folder/'result.json',report)


def worker(run,phase,gpu,seed):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or os.environ.get('PYTHONNOUSERSITE')!='1':
        raise RuntimeError('Wrong environment/device assignment')
    folder=run/(phase if phase!='discriminator' else f'd_seed{seed}')
    folder.mkdir(exist_ok=False)
    with LOCKS[gpu].open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if uuid!=GPUS[gpu] or any(line.startswith(uuid) for line in occupied.splitlines()):
            raise RuntimeError('GPU identity/ownership check failed')
        atomic_json(folder/'lease.json',dict(gpu=gpu,uuid=uuid,pid=os.getpid(),lock=str(LOCKS[gpu]),started=time.time()))
        try:
            if phase=='contracts':
                setup_data=setup(run);del setup_data;gc.collect();torch.cuda.empty_cache()
                result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
                atomic_json(folder/'result.json',dict(state='completed',returncode=result.returncode))
            elif phase=='saturation':saturation(run,folder)
            elif phase=='readiness':readiness(run,folder)
            elif phase=='discriminator':discriminator_run(run,folder,seed)
            else:raise ValueError(phase)
        except BaseException as error:
            atomic_json(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()))
            raise
        finally:write_status(folder,phase=phase,event='worker_exit')


def supervise(run):
    m=json.loads((run/'frozen.json').read_text());jobs=[];children={};ledger=run/'ledger.json'
    if ledger.exists():raise FileExistsError('Existing ledger: reconcile before recovery; never reset budget')
    def record():atomic_json(ledger,dict(limit_seconds=43200,used_seconds=job_seconds(jobs,time.time()),jobs=jobs))
    def launch(phase,gpu,seed=0):
        name=phase if phase!='discriminator' else f'd_seed{seed}'
        log=(run/f'{name}.log').open('w')
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[sys.executable,'-m','scripts.pre_audit','worker','--run-dir',str(run),'--phase',phase,'--gpu',str(gpu),'--seed',str(seed)]
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=name,gpu=gpu,pid=child.pid,started=time.time(),watchdog_seconds=10800 if phase=='discriminator' else 3600,state='running')
        jobs.append(row);children[name]=(child,log,row);record()
    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def finish_all():
        while children:
            now=time.time();total=job_seconds(jobs,now)
            for name,(child,log,row) in list(children.items()):
                if total>=43200 or now-row['started']>=row['watchdog_seconds']:
                    row['stop_reason']='aggregate_limit' if total>=43200 else 'stage_watchdog';stop(child)
                code=child.poll()
                if code is not None:
                    row.update(ended=time.time(),returncode=code,state='completed' if code==0 else 'incomplete')
                    log.close();del children[name]
                    print(json.dumps(row),flush=True)
            record()
            if children:time.sleep(1)
    try:
        launch('contracts',1);finish_all()
        if jobs[-1]['returncode']:raise RuntimeError('Contracts block dependent stages')
        launch('saturation',1);launch('readiness',2);finish_all()
        if any(j['returncode'] for j in jobs):raise RuntimeError('Diagnostic prerequisite failed')
        launch('discriminator',1,20261004);launch('discriminator',2,20261005);finish_all()
    finally:
        for child,log,row in children.values():
            stop(child);row.update(ended=time.time(),returncode=child.returncode,state='interrupted');log.close()
        children.clear();record()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','supervise','worker']);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase');p.add_argument('--gpu',type=int,choices=[1,2]);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare(a.run_dir)
    elif a.action=='supervise':supervise(a.run_dir)
    else:worker(a.run_dir,a.phase,a.gpu,a.seed)
