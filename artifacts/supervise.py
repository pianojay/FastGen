import datetime,json,statistics,time
from pathlib import Path
root=Path('/home/user/jw_soflow/FastGen-AFM/runs/afm_unet128_50k_seed20261001')
while True:
 r=json.loads((root/'record.json').read_text()); s=json.loads((root/'status.json').read_text())
 who=r['attempts'][-1]['process']; p=Path('/proc')/str(who['pid'])/'stat'
 live=p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==who['start_ticks']
 row={'time':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'state':r['state'],'identity_live':live,'step':s.get('step'),'hours_since_start':round((time.time()-r['created_unix'])/3600,4),'status_age_seconds':round(time.time()-(root/'status.json').stat().st_mtime,1),'recoveries':r['recovery_attempts']}
 if (root/'metrics.jsonl').exists():
  with (root/'metrics.jsonl').open('rb') as f:
   f.seek(0,2); size=f.tell(); f.seek(max(0,size-150000)); lines=f.read().decode().splitlines()
  data=[json.loads(x) for x in lines[1:] if x.startswith('{')][-100:]
  if data:
   row.update(latest_step=data[-1]['step'],mean_cycle_seconds=round(statistics.mean(x['cycle_seconds'] for x in data),4),g_adv=round(statistics.mean(x['g_adv'] for x in data),4),g_ot=round(statistics.mean(x['g_ot'] for x in data),4),d_adv=round(statistics.mean(x['d_adv'] for x in data),4),g_grad_max=round(max(x['g_gradient_norm'] for x in data),3),d_grad_max=round(max(x['d_gradient_norm'] for x in data),3))
 if 'memory' in s:row['peak_allocated_gb']=round(s['memory']['peak_allocated_bytes']/1e9,4)
 cp=root/'checkpoints/latest.pt'
 if cp.exists():row['checkpoint_age_seconds']=round(time.time()-cp.stat().st_mtime,1)
 if r.get('error'):row['error']=r['error']
 print(json.dumps(row),flush=True)
 if r['state'] in ['completed','failed'] or not live:break
 time.sleep(45)
