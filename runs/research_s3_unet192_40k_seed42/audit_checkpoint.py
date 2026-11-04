import argparse, hashlib, json, math, sys, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path.cwd()))
import torch
from soflow_fastgen.common import verify_protected, digest, atomic_json
from soflow_fastgen.training_study import load_study, check_prepared
from soflow_fastgen.paired_evaluation import fingerprint
from soflow_fastgen.models import Soflow

parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(4)
protocol,config=load_study('configs/research_s3_20261002.json');run=Path(protocol['run_dir'])
record=json.loads((run/'record.json').read_text());verify_protected();check_prepared(protocol,record)
assert record['state']=='running' and record['current_stage']=='training'
trainer=json.loads((run/'active_job.json').read_text())
for proc in (record['worker'],trainer):
 fields=Path(f"/proc/{proc['pid']}/stat").read_text().rpartition(')')[2].split()
 assert fields[19]==proc['proc_start_ticks'] and fields[0]!='Z'
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==proc['boot_id']
assert '--resume' not in trainer['command'] and record['initial_step']==0 and record['starting_checkpoint'] is None
safe_env={k.decode():v.decode() for part in Path(f"/proc/{trainer['pid']}/environ").read_bytes().split(b'\0') if b'=' in part for k,v in [part.split(b'=',1)] if k in (b'CUDA_VISIBLE_DEVICES',b'PYTHONNOUSERSITE')}
assert safe_env=={'CUDA_VISIBLE_DEVICES':'1','PYTHONNOUSERSITE':'1'}
checkpoint=run/'checkpoints'/(f'step_{args.step:06d}.pt' if args.step in (20000,30000,40000) else 'latest.pt')
sha=digest(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['step']==args.step and state['config']==config and config['total_steps']==40000
assert fingerprint(state['extra']['source_sha256'])==record['source_sha256']
assert state['extra']['partition_sha256']==protocol['partition_sha256']==digest(config['partition'])
assert state['rng'] and all(torch.isfinite(x).all() for name in ('model','ema') for x in state[name].values())
if args.step==0:
 assert state['epoch']==state['cursor']==0 and not state['optimizer']['state']
 assert all(torch.equal(v,state['ema'][k]) for k,v in state['model'].items())
else:
 assert len(state['optimizer']['state'])==355
 assert all(int(x['step'])==args.step for x in state['optimizer']['state'].values())
with torch.device('meta'):model=Soflow(**config['architecture'])
model.load_state_dict(state['ema'],assign=True)
parameters=sum(p.numel() for p in model.parameters());assert parameters==89140419<100_000_000
raw=(run/'metrics.jsonl').read_bytes();raw=raw[:raw.rfind(b'\n')+1];lines=raw.splitlines(keepends=True)
if args.step:assert len(lines)>=args.step;lines=lines[:args.step]
rows=[json.loads(x) for x in lines];assert rows and [x['step'] for x in rows]==list(range(1,len(rows)+1))
assert all(all(math.isfinite(x[k]) for k in ('loss_surrogate','consistency_mse','weight','grad_norm','seconds')) for x in rows)
peak=max(x['allocated_bytes'] for x in rows);assert peak<20_000_000_000
control=json.loads((run/'control.json').read_text());previous=json.loads(Path('docs/research/evidence/s2-complete-20261002.json').read_text())
prior={(x['nfe'],x['seed']):x for x in previous['evaluation_passes'] if x['checkpoint_label']=='step_040000'}
reproductions=[]
for row in control['results']:
 old=prior[(row['nfe'],row['seed'])]
 assert all(row[k]==old[k] for k in ('fid','noise_sha256','images_sha256','network_calls','batches'))
 reproductions.append({k:row[k] for k in ('nfe','seed','fid','network_calls','images_sha256')})
assert len(reproductions)==4
budget=json.loads((run/'budget.json').read_text());status=json.loads((run/'status.json').read_text())
training_elapsed=time.time()-trainer['started_unix'];campaign=protocol['prior_research_gpu_job_seconds']+record['gpu_job_seconds']['total']+training_elapsed
result={'timestamp':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'run_dir':str(run),'state':'running','stage':'training','worker':record['worker'],'trainer':trainer,'trainer_environment_verified':safe_env,'prepared_git_commit':record['prepared_git_commit'],'source_sha256':record['source_sha256'],'protocol_sha256':record['protocol_sha256'],'training_config_sha256':record['training_config_sha256'],'checkpoint':{'path':str(checkpoint),'sha256':sha,'step':state['step'],'epoch':state['epoch'],'cursor':state['cursor'],'optimizer_state_entries':len(state['optimizer']['state']),'rng_present':bool(state['rng']),'loaded_ema_parameters':parameters,'model_ema_finite':True,'ema_equals_initial_model':True if args.step==0 else None,'schedule_steps':config['total_steps']},'verified_update_records':len(rows),'live_status_step':status['step'],'image_presentations':128*len(rows),'metrics_prefix_sha256':hashlib.sha256(b''.join(lines)).hexdigest(),'optimizer_seconds':sum(x['seconds'] for x in rows),'mean_update_seconds':sum(x['seconds'] for x in rows)/len(rows),'last_window_mean_seconds':sum(x['seconds'] for x in rows[-min(10000,len(rows)):])/min(10000,len(rows)),'peak_training_allocated_bytes':peak,'peak_training_reserved_bytes':max(x['reserved_bytes'] for x in rows),'peak_host_rss_bytes':max(x['host_peak_rss_bytes'] for x in rows),'peak_sampled_process_tree_pss_bytes':max(x.get('process_tree_pss_bytes',0) for x in rows),'control_reproduction':reproductions,'completed_stage_gpu_job_seconds':record['gpu_job_seconds'],'training_budget_snapshot':budget,'running_training_job_seconds_estimate':training_elapsed,'running_campaign_gpu_job_seconds_estimate':campaign,'prior_research_gpu_job_seconds':protocol['prior_research_gpu_job_seconds'],'pretraining_qualification_gpu_job_seconds':protocol['pretraining_qualification_gpu_job_seconds'],'retries':sum(bool(a['retry']) for a in record['attempts']),'protected_files_verified':True,'candidate_fid':None,'official_score':False,'limitation':'Running snapshot, not final memory/cost/quality. Prior research already includes both S3 qualification probes. The budget file can lag logged updates; the supervisor finalizes complete job cost.'}
atomic_json(args.output,result)
print(json.dumps({k:v for k,v in result.items() if k not in ('control_reproduction','worker','trainer')},indent=2))
