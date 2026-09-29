"""Single-GPU Pokémon training. Official dataset and evaluator stay unchanged."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from soflow_fastgen.common import Budget, atomic_json, digest, seed_all, gpu_telemetry, verify_protected, process_tree_memory, source_provenance
from soflow_fastgen.data import datasets, loader
from soflow_fastgen.training import Trainer
from soflow_fastgen.visualize import demo, sheet


def main(args):
    config=json.loads(Path(args.config).read_text())
    verify_protected()
    if not 0 < config['microbatch_size'] <= config['batch_size']:
        raise ValueError('Invalid microbatch/effective batch')
    run=Path(config['run_dir']); run.mkdir(parents=True,exist_ok=True)
    lock=(run/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'checkpoints/latest.pt').exists() and not args.resume:
        raise ValueError('Existing checkpoint: pass --resume or choose a fresh run directory')
    torch.set_num_threads(config['torch_threads'])
    # A KCloud slice may expose less memory than the challenge's 20 GB ceiling.
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,(config['memory_limit_bytes']-32*1024**2)/capacity))
    seed_all(config['seed'])
    train_set,_,mapping=datasets(config)
    trainer=Trainer(config)
    budget=Budget(run/'budget.json',config['budget_seconds'])
    if args.resume:
        extra=trainer.load(args.resume)
        if extra.get('partition_sha256') != digest(config['partition']):
            raise ValueError('Resume data partition changed')
    atomic_json(run/'config.json',config)
    metadata={'partition_sha256':digest(config['partition']), **source_provenance()}
    atomic_json(run/'provenance.json',metadata)
    stop_requested=False
    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    initial_step=trainer.step
    def save():
        trainer.save(run/'checkpoints/latest.pt',metadata)
        budget.save()
    status={'state':'running','step':trainer.step}
    atomic_json(run/'status.json',status)
    try:
        if trainer.step==0:
            save()
            preview=next(iter(loader(train_set,config,0,0)))
            with torch.random.fork_rng(devices=[0]):
                originals=preview[0][:8].cuda()
                augmented,_=trainer.augment(originals)
                inverse={v:k for k,v in mapping.items()}
                labels=[inverse[int(y)] for y in preview[1][:8]]
                sheet(torch.cat([originals,augmented]),labels+labels,run/'samples/augmentation.png',columns=8)
                demo(trainer.ema,run/'samples/000000',mapping)
        while trainer.step < config['total_steps']:
            if budget.remaining() <= config['final_eval_reserve_seconds'] or stop_requested:
                break
            if args.run_steps and trainer.step-initial_step >= args.run_steps:
                break
            if len(train_set)-trainer.cursor < config['batch_size']:
                trainer.epoch+=1;trainer.cursor=0
            for images,labels in loader(train_set,config,trainer.epoch,trainer.cursor):
                torch.cuda.reset_peak_memory_stats()
                stats=trainer.update(images,labels)
                stats.update(epoch=trainer.epoch,cursor=trainer.cursor,elapsed_budget_seconds=budget.elapsed())
                if trainer.step%config['logging_steps']==0 or trainer.step==initial_step+1:
                    stats.update(process_tree_memory())
                with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
                if trainer.step%config['logging_steps']==0 or trainer.step==initial_step+1:
                    print(json.dumps(stats),flush=True)
                    atomic_json(run/'status.json',dict(state='running',**stats,gpu=gpu_telemetry()))
                    budget.save()
                retain = trainer.step in config.get('checkpoint_retain_steps', [])
                if trainer.step%config['checkpoint_steps']==0 or retain:
                    save()
                if retain:
                    trainer.save(run/'checkpoints'/f'step_{trainer.step:06d}.pt',metadata)
                if trainer.step%config['sample_steps']==0:
                    demo(trainer.ema,run/'samples'/f'{trainer.step:06d}',mapping)
                    subprocess.run([sys.executable,'-m','scripts.report','--run-dir',str(run)],check=True)
                if (trainer.step in config['eval_steps'] and not args.skip_eval and
                    budget.remaining()>config['final_eval_reserve_seconds']+900):
                    save()
                    checkpoint=run/'checkpoints'/f'step_{trainer.step:06d}.pt'
                    trainer.save(checkpoint,metadata)
                    # Parent accounts for all elapsed time; child has no budget writer.
                    command=[sys.executable,'-m','scripts.evaluate_snapshot','--config',args.config,
                        '--checkpoint',str(checkpoint),'--nfes','1','4','--seeds','42']
                    torch.cuda.empty_cache()
                    result=subprocess.run(command,timeout=max(1,int(budget.remaining()-config['final_eval_reserve_seconds'])))
                    if result.returncode:raise RuntimeError('Intermediate evaluation failed')
                    budget.save()
                if (stop_requested or trainer.step>=config['total_steps'] or
                    budget.remaining()<=config['final_eval_reserve_seconds'] or
                    (args.run_steps and trainer.step-initial_step>=args.run_steps)):
                    break
        save()
        trainer.save(run/'checkpoints'/f'step_{trainer.step:06d}.pt',metadata)
        status={'state':'training_stopped' if stop_requested else 'training_complete',
                'step':trainer.step,'schedule_steps':config['total_steps'],
                'reason':'signal' if stop_requested else 'invocation_limit' if args.run_steps else
                         'schedule_complete' if trainer.step>=config['total_steps'] else 'budget',
                'elapsed_budget_seconds':budget.elapsed()}
        atomic_json(run/'status.json',status)
        print(json.dumps(status),flush=True)
    except Exception as error:
        budget.save()
        atomic_json(run/'status.json',{'state':'failed','step':trainer.step,'error':str(error)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pokemon.json')
    parser.add_argument('--resume')
    parser.add_argument('--run-steps',type=int)
    parser.add_argument('--skip-eval',action='store_true')
    main(parser.parse_args())
