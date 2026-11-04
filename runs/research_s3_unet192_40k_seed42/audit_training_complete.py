"""CPU-only final training audit; run only after the supervisor closes training."""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path.cwd()))
import torch
from PIL import Image
from soflow_fastgen.common import atomic_json, digest, verify_protected
from soflow_fastgen.models import Soflow
from soflow_fastgen.paired_evaluation import fingerprint
from soflow_fastgen.training_study import check_inputs, check_prepared, load_study

torch.set_num_threads(4)
protocol, config = load_study('configs/research_s3_20261002.json')
run = Path(protocol['run_dir'])
record = json.loads((run/'record.json').read_text())
status = json.loads((run/'status.json').read_text())
assert record['state'] in ('running', 'completed')
assert 'training' in record['completed_stages']
assert status['step'] == config['total_steps'] == 40000
assert status['reason'] == 'schedule_complete'
assert record['initial_step'] == 0 and record['starting_checkpoint'] is None
verify_protected()
check_prepared(protocol, record)
check_inputs(protocol, config)

trainer = json.loads((run/'active_job.json').read_text())
stat = Path(f"/proc/{trainer['pid']}/stat")
if stat.exists():
    fields = stat.read_text().rpartition(')')[2].split()
    assert fields[19] != trainer['proc_start_ticks'] or fields[0] == 'Z'

checkpoint = run/'checkpoints/step_040000.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert state['step'] == 40000 and state['config'] == config
assert fingerprint(state['extra']['source_sha256']) == record['source_sha256']
assert state['extra']['partition_sha256'] == protocol['partition_sha256'] == digest(config['partition'])
assert state['rng'] and len(state['optimizer']['state']) == 355
assert all(int(item['step']) == 40000 for item in state['optimizer']['state'].values())
assert all(torch.isfinite(value).all() for name in ('model', 'ema') for value in state[name].values())
assert all(torch.isfinite(value).all() for item in state['optimizer']['state'].values()
           for value in item.values() if isinstance(value, torch.Tensor))
with torch.device('meta'):
    model = Soflow(**config['architecture'])
model.load_state_dict(state['ema'], assign=True)
parameters = sum(p.numel() for p in model.parameters())
assert parameters == 89140419 < 100_000_000

raw = (run/'metrics.jsonl').read_bytes()
assert raw.endswith(b'\n')
lines = raw.splitlines(keepends=True)
rows = [json.loads(line) for line in lines]
assert len(rows) == 40000 and [row['step'] for row in rows] == list(range(1,40001))
assert all(math.isfinite(row[key]) for row in rows
           for key in ('loss_surrogate','consistency_mse','weight','grad_norm','seconds'))
previous = json.loads(Path('docs/research/evidence/s3-30000-20261003.json').read_text())
assert hashlib.sha256(b''.join(lines[:30000])).hexdigest() == previous['metrics_prefix_sha256']
peak = max(row['allocated_bytes'] for row in rows)
assert peak < 20_000_000_000
optimizer_seconds = sum(row['seconds'] for row in rows)
training_seconds = record['gpu_job_seconds']['training']
assert optimizer_seconds < training_seconds < protocol['training_budget_seconds']
panels = []
for nfe in range(1,5):
    path = run/'samples'/'040000'/f'nfe{nfe}.png'
    with Image.open(path) as im:
        size = list(im.size)
        im.verify()
    panels.append({'path':str(path),'sha256':digest(path),'size':size})

result = {
    'timestamp':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    'run_dir':str(run), 'state':'training_complete',
    'workflow_state_at_audit':record['state'], 'workflow_stage_at_audit':record['current_stage'],
    'updates':len(rows), 'image_presentations':128*len(rows),
    'prepared_git_commit':record['prepared_git_commit'],
    **{key:record[key] for key in ('source_sha256','protocol_sha256','training_config_sha256')},
    'checkpoint':{
        'path':str(checkpoint),'sha256':digest(checkpoint),'step':state['step'],
        'epoch':state['epoch'],'cursor':state['cursor'],'schedule_steps':40000,
        'optimizer_state_entries':355,'optimizer_tensors_finite':True,
        'rng_state_present':True,'source_and_partition_match':True,
        'loaded_ema_parameters':parameters,'model_and_ema_weights_finite':True,
    },
    'retained_checkpoint_sha256':{str(step):digest(run/'checkpoints'/f'step_{step:06d}.pt')
                                   for step in (20000,30000,40000)},
    'metrics_sha256':hashlib.sha256(raw).hexdigest(),
    'previous_30000_update_prefix_unchanged':True,
    'optimizer_seconds':optimizer_seconds,'mean_update_seconds':optimizer_seconds/len(rows),
    'last_10000_mean_update_seconds':sum(row['seconds'] for row in rows[-10000:])/10000,
    'peak_training_allocated_bytes':peak,
    'peak_training_reserved_bytes':max(row['reserved_bytes'] for row in rows),
    'peak_host_rss_bytes':max(row['host_peak_rss_bytes'] for row in rows),
    'peak_sampled_process_tree_pss_bytes':max(row.get('process_tree_pss_bytes',0) for row in rows),
    'training_gpu_job_seconds':training_seconds,
    'production_qualification_gpu_job_seconds':record['gpu_job_seconds']['qualification'],
    'pretraining_qualification_gpu_job_seconds':protocol['pretraining_qualification_gpu_job_seconds'],
    'retries_so_far':sum(bool(item['retry']) for item in record['attempts']),
    'protected_files_verified':True,'official_score':False,
    'verified_fixed_panels':panels,
    'limitation':'This audits completed training only. Paired FID decisions and final campaign accounting require the separate completed-workflow audit.',
}
atomic_json(run/'training_verified.json', result)
date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
atomic_json(f'docs/research/evidence/s3-training-complete-{date}.json', result)
print(json.dumps(result, indent=2))
