"""Benchmark the actual augmented Pokémon update in isolated processes."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from soflow_fastgen.common import Budget, atomic_json, seed_all, gpu_telemetry
from soflow_fastgen.data import datasets, loader
from soflow_fastgen.training import Trainer


def candidate(config,size):
    torch.set_num_threads(config['torch_threads'])
    torch.cuda.set_per_process_memory_fraction(config['memory_target_bytes']/torch.cuda.get_device_properties(0).total_memory)
    config['microbatch_size']=size
    seed_all(config['seed'])
    train_set,_,_=datasets(config)
    iterator=iter(loader(train_set,config,0,0))
    trainer=Trainer(config)
    torch.cuda.reset_peak_memory_stats()
    stats=[trainer.update(*next(iterator)) for _ in range(7)]
    # Exercise the minimum-gap end of the same curriculum as well.
    trainer.step=config['total_steps']-1
    terminal=trainer.update(*next(iterator))
    peak=max(x['allocated_bytes'] for x in stats+[terminal])
    elapsed=statistics.median(x['seconds'] for x in stats[2:])
    return {'microbatch':size,'effective_batch':config['batch_size'],
        'seconds_per_update':elapsed,'images_per_second':config['batch_size']/elapsed,
        'peak_allocated_bytes':peak,'peak_reserved_bytes':max(x['reserved_bytes'] for x in stats+[terminal]),
        'eligible':peak<config['memory_target_bytes'],'gpu':gpu_telemetry(),'updates':stats,
        'terminal_gap_update':terminal}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pokemon.json')
    p.add_argument('--candidate',type=int);p.add_argument('--output');a=p.parse_args()
    config=json.load(open(a.config))
    if a.candidate:
        try: result=candidate(config,a.candidate)
        except torch.cuda.OutOfMemoryError as error:result={'microbatch':a.candidate,'eligible':False,'error':str(error)}
        atomic_json(a.output,result)
    else:
        root=Path('artifacts/pokemon_profile');root.mkdir(parents=True,exist_ok=True)
        budget=Budget(Path(config['run_dir'])/'budget.json',config['budget_seconds'])
        results=[]
        try:
            for size in (16,32,64,128):
                path=root/f'microbatch_{size}.json'
                with (root/f'microbatch_{size}.log').open('w') as stream:
                    result=subprocess.run([sys.executable,'-m','scripts.profile_batch','--config',a.config,
                        '--candidate',str(size),'--output',str(path)],stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'Profile failed; see {stream.name}')
                record=json.loads(path.read_text());results.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in ('updates','terminal_gap_update')}),flush=True)
                budget.save()
                if not record['eligible']:break
            eligible=[x for x in results if x['eligible']]
            if not eligible:raise RuntimeError('No eligible microbatch; investigate before training')
            chosen=min(eligible,key=lambda x:x['seconds_per_update'])
            atomic_json(root/'summary.json',{'selected_microbatch':chosen['microbatch'],'results':results})
            config['microbatch_size']=chosen['microbatch']
            atomic_json('configs/pokemon_local.json',config)
        finally:budget.save()
