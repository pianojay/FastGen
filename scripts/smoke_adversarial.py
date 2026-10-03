"""Finite integration check. Never starts a training/evaluation campaign."""
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

import torch

from soflow_fastgen.adversarial import AdversarialComponents, ComponentConfig, endpoint
from soflow_fastgen.common import ROOT, atomic_json, atomic_save, digest, resources, verify_protected, source_provenance
from soflow_fastgen.data import datasets
from soflow_fastgen.sampling import sample
from soflow_fastgen.stylegan2 import verify_vendor

GPU_UUID = 'GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888'
SOURCE = ROOT/'runs/research_s3_unet192_40k_seed42/checkpoints/step_040000.pt'
SOURCE_SHA = '9b95652fb5e503b2f877b26fd3652129abcce115bdea526db00ffe1f0daefd4a'
PARTITION_SHA = '825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e'


def tensor_hash(state):
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0])):
                h.update(str(key).encode());visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:visit(child)
        else:h.update(repr(value).encode())
    visit(state)
    return h.hexdigest()


def recovery_identity(engine):
    return {key:tensor_hash(getattr(engine,key).state_dict())
            for key in ('model','ema','critic','g_optimizer','d_optimizer')} | {
                'streams':tensor_hash(engine.streams.states),
                'counters':[engine.g_steps,engine.d_steps,engine.g_images,engine.d_images]}


def load_only(path, architecture, mode):
    """Separate process: proves official inference loading needs no critic."""
    from model import Model
    model=Model.load_checkpoint(str(path),mode,device='cpu',**architecture)
    assert not any(name.split('.')[0] in ('training','torch_utils','dnnlib') for name in sys.modules)
    assert model.count_parameters()==89140419
    assert not any('critic' in key or 'optimizer' in key or 'ema' in key for key in model.state_dict())
    calls=[]
    hook=model.backbone.network.register_forward_pre_hook(lambda module,args:calls.append(args))
    with torch.no_grad():
        output=model.sample((1,3,64,64),device='cpu',category=torch.tensor([42]),
                            generator=torch.Generator().manual_seed(101))
    hook.remove()
    assert len(calls)==(1 if mode=='one_nfe' else 4)
    assert all(torch.count_nonzero(args[4])==0 for args in calls)
    assert torch.isfinite(output).all()
    print(json.dumps({'mode':mode,'parameters':model.count_parameters(),'calls':len(calls),
                      'critic_imported':False}),flush=True)


def worker(run):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    verify_protected(); vendor=verify_vendor()
    if digest(ROOT/'artifacts/development_split.json')!=PARTITION_SHA:
        raise RuntimeError('Partition changed')
    source=AdversarialComponents.checkpoint_source(SOURCE,SOURCE_SHA)
    config=ComponentConfig()
    engine=AdversarialComponents(source,config)
    del source
    train, _, _=datasets(engine.source_config)
    positions=[i*len(train)//8 for i in range(8)]
    rows=[train[index] for index in positions]
    images=torch.stack([row[0] for row in rows]);labels=torch.tensor([row[1] for row in rows],dtype=torch.long)
    report={'purpose':'disposable integration smoke; no sample-quality claim',
            'state':'running','source':engine.origin,'partition_sha256':PARTITION_SHA,
            'component_config':asdict(config),'parameters':engine.counts,
            'vendor_commit':vendor['commit'],'optimization_positions':positions,
            'categories':labels.tolist(),'executed_d_updates':0,'executed_g_updates':0,
            'checks':{},'updates':[]}
    def record():atomic_json(run/'result.json',report)
    def update(kind):
        key='executed_'+kind+'_updates';limit=4 if kind=='d' else 3
        if report[key]>=limit:raise RuntimeError('Smoke update ceiling reached')
        report[key]+=1;record()  # charge attempted updates even if they fail
        before=time.monotonic()
        row=(engine.discriminator_update if kind=='d' else engine.generator_update)(images,labels)
        torch.cuda.synchronize()
        report['updates'].append(dict(kind=kind,seconds=time.monotonic()-before,**row))
        record();print(json.dumps(report['updates'][-1]),flush=True)

    # EMA is the origin of both online and new EMA; no old optimizer moments.
    assert tensor_hash(engine.model.state_dict())==tensor_hash(engine.ema.state_dict())
    assert not engine.g_optimizer.state and not engine.d_optimizer.state
    report['checks']['fresh_optimizer_and_initial_ema_identity']=True
    with torch.no_grad():
        c=labels[:4].cuda()
        noise=torch.randn(4,3,64,64,device='cuda',generator=torch.Generator('cuda').manual_seed(99))
        raw,visible=endpoint(engine.model,noise,c)
        with torch.random.fork_rng(devices=[0]):
            engine.ema.eval()
            baseline=sample(engine.ema,noise.shape,device='cuda',category=c,nfe=1,noise=[noise])
        torch.testing.assert_close(visible,baseline,rtol=0,atol=0)
        report['initial_one_call_clipped_fraction']=((raw < -1)|(raw > 1)).float().mean().item()
        report['checks']['source_ema_endpoint_identity']=True
    del raw,visible,baseline,noise
    g_before=tensor_hash(engine.model.state_dict());ema_before=tensor_hash(engine.ema.state_dict())
    update('d')
    assert tensor_hash(engine.model.state_dict())==g_before
    assert tensor_hash(engine.ema.state_dict())==ema_before
    assert engine.g_steps==engine.g_images==0 and not engine.g_optimizer.state
    assert all(p.grad is None for p in engine.model.parameters())
    report['checks']['d_only_preserves_generator_ema_and_clocks']=True
    update('d');update('g')
    assert all(p.grad is None for p in engine.critic.parameters())
    report['checks']['generator_does_not_accumulate_critic_gradients']=True
    engine.save(run/'recovery.pt',extra={'optimization_positions':positions,'next_fixture_batch':'same fixed smoke batch'})
    update('d');update('g')
    expected=recovery_identity(engine)
    engine.load(run/'recovery.pt')
    update('d');update('g')
    actual=recovery_identity(engine)
    if actual!=expected:raise AssertionError('Next-update recovery differs: '+str([k for k in actual if actual[k]!=expected[k]]))
    report['checks']['exact_next_update_recovery']=True
    report['recovery_identity']=actual
    report['resources']=resources()
    assert report['resources']['allocated_bytes']<20_000_000_000
    report['checks']['separate_parameters_and_combined_memory']=True

    architecture=engine.source_config['architecture']
    weights={'backbone.'+key:value.detach().cpu() for key,value in engine.ema.state_dict().items()}
    for mode,nfe in (('one_nfe',1),('few_nfe',4)):
        atomic_save(run/(mode+'.ckpt'),{'state_dict':dict(weights,sampling_nfe=torch.tensor(nfe))})
    del engine,weights
    gc.collect();torch.cuda.empty_cache()
    report['independent_loads']=[]
    for mode in ('one_nfe','few_nfe'):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        loaded=subprocess.run([sys.executable,'-m','scripts.smoke_adversarial','--load-only',
            str(run/(mode+'.ckpt')),'--architecture',json.dumps(architecture),'--mode',mode],
            cwd=ROOT,env=env,check=True,capture_output=True,text=True,timeout=120)
        report['independent_loads'].append(json.loads(loaded.stdout.strip().splitlines()[-1]))
    report['checks']['independent_generator_only_one_and_four_call_loading']=True
    verify_protected()
    report['checks']['protected_files_unchanged']=True
    report['state']='passed';record()


def supervise(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or os.environ.get('PYTHONNOUSERSITE')!='1':
        raise RuntimeError('Launch with CUDA_VISIBLE_DEVICES=1 and PYTHONNOUSERSITE=1')
    if run.exists():raise FileExistsError('Use a fresh smoke directory; existing evidence is immutable')
    with (ROOT/'runs/gpu_1.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        active=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if uuid!=GPU_UUID or active:raise RuntimeError(f'GPU 1 identity/ownership failed: {uuid}, active={active}')
        run.mkdir(parents=True)
        started=time.monotonic()
        record={'state':'running','gpu_uuid':uuid,'watchdog_seconds':900,
                'source':source_provenance(),'python':sys.executable,'torch':torch.__version__}
        atomic_json(run/'supervisor.json',record)
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
        # Test fixtures run on the leased GPU; only the final worker uses S3.
        commands=[['-m','unittest','discover','-s','tests','-v'],
                  ['-m','scripts.smoke_adversarial','--worker','--run-dir',str(run)]]
        try:
            for index,command in enumerate(commands):
                with (run/f'phase_{index}.log').open('w') as log:
                    child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code=child.wait(timeout=max(1,900-(time.monotonic()-started)))
                    except BaseException:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise
                    if code:raise RuntimeError(f'Smoke phase {index} failed with exit {code}; inspect its log')
            record['state']='passed'
        except BaseException as error:
            record.update(state='failed',error=str(error));raise
        finally:
            record['elapsed_seconds']=time.monotonic()-started
            atomic_json(run/'supervisor.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--load-only',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--architecture',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=('one_nfe','few_nfe'),help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.load_only:
        torch.set_num_threads(4);load_only(a.load_only,json.loads(a.architecture),a.mode)
    elif a.run_dir is None:p.error('--run-dir required')
    elif a.worker:worker(a.run_dir)
    else:supervise(a.run_dir)
