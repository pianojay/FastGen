import fcntl, hashlib, json, os, signal, sys, time
from pathlib import Path
p=Path(__file__).parent
lock=(p/'worker.lock').open('w')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: print('Worker already active'); sys.exit(3)
def read(name): return json.loads((p/name).read_text())
def save(name,data):
 t=p/(name+'.tmp'); t.write_text(json.dumps(data,indent=2)+'\n'); t.replace(p/name)
def event(data):
 with (p/'events.jsonl').open('a') as f: f.write(json.dumps(data)+'\n')
cfg=read('config.json'); digest=hashlib.sha256((p/'config.json').read_bytes()).hexdigest()
s=read('state.json')
if s['config_sha256']!=digest: raise SystemExit('configuration changed')
if s['status']=='completed': raise SystemExit('already completed')
if s['attempts']>=2: raise SystemExit('attempt limit reached')
cp=read('checkpoint.json') if (p/'checkpoint.json').exists() else {'step':0,'sum':0,'config_sha256':digest}
if cp['config_sha256']!=digest: raise SystemExit('checkpoint mismatch')
remaining=cfg['cap_seconds']-s['elapsed_seconds']
if remaining<=0: raise SystemExit('budget exhausted')
s.update(status='running',pid=os.getpid(),attempts=s['attempts']+1,resumed_from=cp['step'])
start=time.monotonic(); s['process_start_ticks']=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]
save('state.json',s); event({'event':'start','attempt':s['attempts'],'from':cp['step'],'remaining_seconds':remaining,'pid':os.getpid(),'config_sha256':digest})
def stop(*args): raise InterruptedError('intentional interruption')
signal.signal(signal.SIGTERM,stop)
try:
 for step in range(cp['step']+1,cfg['steps']+1):
  if time.monotonic()-start+cfg['delay_seconds']>remaining: raise InterruptedError('budget exhausted')
  time.sleep(cfg['delay_seconds']); cp['sum']+=step; cp['step']=step
  event({'event':'progress','attempt':s['attempts'],'step':step})
  if step%cfg['checkpoint_every']==0: save('checkpoint.json',cp)
 save('checkpoint.json',cp); s['status']='completed'
except InterruptedError as e: s['status']='interrupted'; s['reason']=str(e)
finally:
 s['elapsed_seconds']+=time.monotonic()-start; s['pid']=None
 save('state.json',s); event({'event':'stop','status':s['status'],'elapsed_seconds':s['elapsed_seconds']})
print(json.dumps(s))
