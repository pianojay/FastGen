import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import datetime,fcntl,hashlib,json,subprocess,time,traceback,shutil,gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stargan64 import StarGAN64,Sampler,seed_all
R=Path(__file__).resolve().parent;ROOT=R.parents[2];S=ROOT/'transfer_aug/shared/round2';E=ROOT/'transfer_aug/shared/round3'
UUID='GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2';CONTRACT='9eb5c9897b358f0d0c7ab24d724b292752038fa3124e0296856eb8d817ca4813'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(m):
 h=hashlib.sha256()
 for coll in (m.nets,m.ema):
  for n in coll.values():
   for p in n.state_dict().values():h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 assert sha(R/'CONTRACT_v1.md')==CONTRACT
 assert sha(E/'TRAINING_PROTOCOL_v1.md')=='75937769254ab5634618d104d97b0e51c4b1bf20128e02c1566eeffa32d6a8f2'
 frozen={}
 for base in (S,E):
  f=json.loads((base/'freeze.json').read_text())
  for n,h in f['files'].items():assert sha(base/n)==h,n;frozen[str(base/n)]=h
 records=json.loads((S/'manifest.json').read_text())['records'];ev=json.loads((E/'evaluation_manifest.json').read_text())['records'];latent=np.load(E/'stargan_latent.npy');pixels=np.load(S/'pixels_uint8.npy',mmap_mode='r')
 assert all(r['partition']=='A1' for r in records)
 opt={r['pixel_sha256'] for r in records if r['role']=='optimization'};diag={r['pixel_sha256'] for r in records if r['role']=='diagnostic'};assert not opt&diag
 domains={}
 for r in records:
  assert hashlib.sha256(pixels[r['pixel_index']].tobytes()).hexdigest()==r['pixel_sha256']
  if r['role']=='optimization':domains.setdefault(r['category'],{'D0':[],'D1':[]})[r['domain']].append(r['pixel_index'])
 provenance=json.loads((ROOT/'transfer_aug/audit_sources/starganv2/provenance.json').read_text())
 for n,v in provenance['files'].items():assert sha(ROOT/'transfer_aug/audit_sources/starganv2'/n)==v['sha256']
 write(R/'training_preflight.json',{'contract':CONTRACT,'frozen':frozen,'A1_only':True,'all_pixels_verified':True,'source_verified':True,'implementation':{p.name:sha(p) for p in R.glob('*.py')},'timestamp':utc()})
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 lp=R/'gpu_resource_ledger.json';ledger=json.loads(lp.read_text()) if lp.exists() else {'budget_seconds':28800,'sessions':[]};assert all(s['ended_at'] is not None for s in ledger['sessions'])
 prev=sum(s['gpu_seconds'] for s in ledger['sessions']);write(lp,ledger)
 lock=open('/tmp/fastgen-gpu2.lock','r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic();session={'pid':os.getpid(),'started_at':utc(),'ended_at':None,'gpu_seconds':0,'uuid':UUID,'lease_inode':os.fstat(lock.fileno()).st_ino};ledger['sessions'].append(session);write(lp,ledger)
 stage='qualification';m=None
 def elapsed():return prev+time.monotonic()-start
 def status(**kw):
  session['gpu_seconds']=time.monotonic()-start;write(lp,ledger);write(R/'status.json',dict(stage=stage,pid=os.getpid(),timestamp=utc(),updates=m.updates if m else 0,gpu_seconds=elapsed(),**kw))
 def tensor(a):return torch.from_numpy(np.asarray(a).copy()).permute(0,3,1,2).contiguous().to('cuda',dtype=torch.float32)/127.5-1
 def batch(s):
  rows=[s.draw() for _ in range(8)];xs=[]
  for j in range(3):
   imgs=[]
   for q in rows:
    a=pixels[q['source'] if j==0 else q['references'][j-1]]
    imgs.append(a[:,::-1] if q['flips'][j] else a)
   xs.append(tensor(np.stack(imgs)))
  return xs[0],torch.tensor([q['source_domain'] for q in rows],device='cuda'),xs[1:],torch.tensor([q['target_domain'] for q in rows],device='cuda')
 try:
  devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);assert f'2, {UUID}' in devices
  occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(UUID in line and int(line.split(',')[1])!=os.getpid() for line in occupancy.splitlines()),occupancy
  assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUID.removeprefix('GPU-')
  seed_all();m=StarGAN64('cuda');sampler=Sampler(domains)
  if (R/'latest.pt').exists():m.restore(R/'latest.pt',sampler)
  else:
   torch.cuda.reset_peak_memory_stats();t=time.monotonic()
   for i in range(20):loss=m.step(*batch(sampler));status(qualification_iterations=i+1,last_loss=loss)
   torch.cuda.synchronize();dt=time.monotonic()-t;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();assert peak<20_000_000_000
   m.checkpoint(R/'qualification.pt',sampler);l1=m.step(*batch(sampler));h1=digest(m);m.restore(R/'qualification.pt',sampler);l2=m.step(*batch(sampler));h2=digest(m);assert l1==l2 and h1==h2
   write(R/'qualification.json',{'status':'passed','iterations':20,'extra_recovery_iterations':2,'batch_size':8,'seconds':dt,'seconds_per_iteration':dt/20,'projected_20000_seconds':dt*1000,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'parameters':m.counts(),'total_parameters':sum(m.counts().values()),'exact_gpu_recovery':True,'recovery_digest':h1,'gpu_uuid':UUID,'reset_before_production':True})
   del m;gc.collect();torch.cuda.empty_cache();seed_all();m=StarGAN64('cuda');sampler=Sampler(domains);m.checkpoint(R/'initial.pt',sampler);m.checkpoint(R/'latest.pt',sampler)
  stage='training';status();last=time.monotonic();times=[]
  with (R/'losses.jsonl').open('a',buffering=1) as log:
   while m.updates<20000:
    if elapsed()+max(times[-20:],default=10)+20>=27900:break
    t=time.monotonic();loss=m.step(*batch(sampler));times.append(time.monotonic()-t)
    if m.updates%20==0:log.write(json.dumps({'update':m.updates,'gpu_seconds':elapsed(),**loss})+'\n')
    if m.updates%1000==0:
     m.checkpoint(R/'latest.pt',sampler)
     if m.updates in (1000,5000,10000):shutil.copyfile(R/'latest.pt',R/f'update_{m.updates}.pt')
    if time.monotonic()-last>30:status(last_loss=loss,seconds_per_iteration=sum(times[-20:])/len(times[-20:]));last=time.monotonic()
    if torch.cuda.max_memory_allocated()>=20_000_000_000:raise RuntimeError('memory ceiling')
  m.checkpoint(R/'final.pt',sampler);shutil.copyfile(R/'final.pt',R/'latest.pt');updates=m.updates;stage='generation';status()
  out=R/'outputs';out.mkdir(exist_ok=True);png=out/'png';png.mkdir(exist_ok=True);don=out/'references';don.mkdir(exist_ok=True)
  # Four reference and four latent variants per source. Batch grouping remains fixed for replay.
  def gen(indices,arm,variant):
   vals=[]
   with torch.no_grad():
    for offset in range(0,len(indices),8):
     ii=indices[offset:offset+8];rr=[ev[i] for i in ii];x=tensor(np.stack([pixels[q['source_index']] for q in rr]));y=torch.tensor([int(q['source_domain' if arm=='self' else 'target_domain'][1]) for q in rr],device='cuda')
     if arm=='reference':s=m.ema.style_encoder(tensor(np.stack([pixels[q['reference_indices'][variant]] for q in rr])),y)
     elif arm=='latent':s=m.ema.mapping_network(torch.tensor(latent[ii,variant],device='cuda'),y)
     else:s=m.ema.style_encoder(x,y)
     a=m.ema.generator(x,s).cpu().permute(0,2,3,1).numpy();assert np.isfinite(a).all();vals.append(a)
   return np.concatenate(vals)
  quant=lambda a:np.rint(np.clip((a+1)*127.5,0,255)).astype(np.uint8)
  indices=list(range(len(ev)));assoc=[];final={}
  for arm in ('reference','latent','self'):
   arr=np.stack([gen(indices,arm,v) for v in range(1 if arm=='self' else 4)],axis=1);np.save(out/f'{arm}_float.npy',arr);final[arm]=arr
   for i,q in enumerate(ev):
    for v in range(arr.shape[1]):
     p=png/f'{arm}_{i:03d}_{v}.png';u=quant(arr[i,v]);Image.fromarray(u).save(p);assoc.append({'source_id':i,'arm':arm,'variant':v,'png':str(p),'png_sha256':sha(p),'pixel_sha256':hashlib.sha256(u.tobytes()).hexdigest()})
    if arm=='reference':
     for v,idx in enumerate(q['reference_indices']):Image.fromarray(pixels[idx]).save(don/f'{i:03d}_{v}.png')
   status(generated_arm=arm)
  write(out/'associations.json',assoc)
  review=[i for i,q in enumerate(ev) if q['representative']];exact=True
  for arm in ('reference','latent'):
   for v in range(4):
    a=gen(indices,arm,v);exact=exact and np.array_equal(quant(a)[review],quant(final[arm][:,v])[review])
  assert exact
  traces={}
  for i,q in enumerate(ev):
   if q['representative']:traces.setdefault((q['category'],q['source_domain']),i)
  ti=sorted(traces.values());write(out/'trace_indices.json',ti)
  for label,path in [('initial',R/'initial.pt'),('1000',R/'update_1000.pt'),('5000',R/'update_5000.pt'),('10000',R/'update_10000.pt'),('final',R/'final.pt')]:
   if not path.exists():continue
   q=torch.load(path,map_location='cpu',weights_only=False)
   for k,n in m.ema.items():n.load_state_dict(q['ema'][k])
   del q
   for arm in ('reference','latent'):np.save(out/f'trace_{label}_{arm}.npy',gen(ti,arm,0))
   status(trace=label)
  after={p:sha(p) for p in frozen};assert after==frozen
  write(R/'training_complete.json',{'status':'completed' if updates==20000 else 'budget_limited','updates':updates,'final_output_count':len(assoc),'expected_count':5436,'representative_replay_exact':exact,'representative_replay_count':512,'input_hashes_after':after,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_seconds_before_release':elapsed(),'final_checkpoint_sha256':sha(R/'final.pt'),'gpu_uuid':UUID,'trace_source_count':len(ti),'finite_self_controls':True})
  stage='gpu_complete';status();assert elapsed()<28800
 except BaseException:
  write(R/'failure.json',{'timestamp':utc(),'gpu_seconds':elapsed(),'traceback':traceback.format_exc()});stage='failed';status();raise
 finally:
  session['ended_at']=utc();session['gpu_seconds']=time.monotonic()-start;write(lp,ledger);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
