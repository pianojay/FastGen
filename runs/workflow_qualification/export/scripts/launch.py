"""Detached, locked training → development evaluation → export workflow."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from soflow_fastgen.common import ROOT, atomic_json


def used_seconds(run):
    path = run/'budget.json'
    return json.loads(path.read_text())['used_seconds'] if path.exists() else 0.


def run_gpu_job(command, config, run):
    """Count process startup too; stop the whole child group when its budget expires."""
    before = used_seconds(run)
    timeout = config['budget_seconds'] - before
    if timeout <= 0:
        return None
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    atomic_json(run/'active_job.json', {'pid':process.pid, 'command':command})
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return None
    finally:
        atomic_json(run/'budget.json', {
            'used_seconds':max(used_seconds(run), before+time.monotonic()-started),
            'limit_seconds':config['budget_seconds']})
        atomic_json(run/'active_job.json', {'pid':None, 'command':command, 'returncode':process.returncode})


def worker(args):
    config = json.loads(Path(args.config).read_text())
    run = Path(config['run_dir'])
    # Locks are acquired before any GPU allocation. They cover this repository's workflows.
    lock = (run/'workflow.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock = (ROOT/'runs'/f'gpu_{args.gpu}.lock').open('w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    active = subprocess.check_output(['nvidia-smi','-i',args.gpu,
        '--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
    if active:
        raise RuntimeError(f'Assigned GPU {args.gpu} already has compute processes: {active}')
    atomic_json(run/'process.json', {
        'pid':os.getpid(),'gpu':args.gpu,'started_unix':time.time(),'config':args.config})
    (run/'exit_status').write_text('running\n')
    status = 'failed'
    try:
        command = [sys.executable,'train.py','--config',args.config]
        if args.resume:
            command += ['--resume',str(run/'checkpoints/latest.pt')]
        result = run_gpu_job(command,config,run)
        if result not in (0,None):
            raise RuntimeError('Training process failed; inspect status.json and console.log')
        progress = json.loads((run/'status.json').read_text())
        if progress.get('reason') == 'signal' and result == 0:
            status = 'stopped'
            return
        checkpoint = run/'checkpoints'/f'step_{progress["step"]:06d}.pt'
        if not checkpoint.exists():
            checkpoint = run/'checkpoints/latest.pt'
        result = run_gpu_job([sys.executable,'-m','scripts.evaluate_dev',
            '--config',args.config,'--checkpoint',str(checkpoint),
            '--seeds',*[str(seed) for seed in config['eval_seeds']]], config,run)
        rows = []
        for path in (run/'evaluations'/checkpoint.stem).glob('nfe*/metrics.json'):
            rows.append(json.loads(path.read_text()))
        eligible = {n:[x['fid'] for x in rows if x['nfe']==n] for n in (2,3,4)}
        eligible = {n:v for n,v in eligible.items() if len(v)==len(config['eval_seeds'])}
        selected = min(eligible,key=lambda n:statistics.mean(eligible[n])) if eligible else 4
        atomic_json(run/'selection.json', {
            'few_nfe':selected,'selected_by_development':bool(eligible),'results':rows,
            'checkpoint':str(checkpoint),'evaluation_returncode':result})
        # Preserve a usable inference checkpoint even if evaluation reaches its time limit.
        export_dir = run/'export'
        if export_dir.exists():
            export_dir = run/f'export_{time.time_ns()}'
        subprocess.run([sys.executable,'-m','scripts.export','--checkpoint',str(checkpoint),
            '--output',str(export_dir),'--few-nfe',str(selected)],check=True)
        with (run/'export_verification.json').open('w') as report:
            subprocess.run([sys.executable,'-m','scripts.verify_export','--package',str(export_dir)],
                           stdout=report,check=True)
        expected = {(n,s) for n in (1,2,3,4) for s in config['eval_seeds']}
        observed = {(row['nfe'],row['seed']) for row in rows}
        status = 'completed' if observed==expected else 'evaluation_failed' if result not in (0,None) else 'budget_limited'
    finally:
        (run/'exit_status').write_text(status+'\n')
        subprocess.run([sys.executable,'-m','scripts.report','--run-dir',str(run)],check=True)
        if status in ('completed','budget_limited','evaluation_failed'):
            subprocess.run([sys.executable,'-m','scripts.bundle','--config',args.config,
                '--output',str(ROOT/'dist'/run.name)],check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pokemon_local.json')
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args)
    else:
        config = json.loads(Path(args.config).read_text())
        run = Path(config['run_dir'])
        run.mkdir(parents=True,exist_ok=True)
        if (run/'checkpoints/latest.pt').exists() and not args.resume:
            raise SystemExit('Existing checkpoint: use --resume or select a fresh run directory')
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLBACKEND='Agg')
        command = [sys.executable,'-m','scripts.launch','--worker','--config',args.config,'--gpu',args.gpu]
        if args.resume:
            command += ['--resume']
        with (run/'console.log').open('a') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,cwd=ROOT,env=env)
        time.sleep(1)
        if process.poll() is not None:
            raise SystemExit(f'Workflow startup failed: inspect {run/"console.log"}')
        print(json.dumps({'workflow_pid':process.pid,'log':str(run/'console.log')}))
