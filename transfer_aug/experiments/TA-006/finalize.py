import fcntl,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
t=read(R/'training_complete.json');ledger=read(R/'gpu_resource_ledger.json');q=read(R/'qualification.json');setup=read(R/'setup_result.json');assoc=read(R/'outputs/associations.json');assert len(assoc)==5436
arrays={a:np.load(R/'outputs'/f'{a}_float.npy',mmap_mode='r') for a in ('reference','latent','self')}
for a,v in arrays.items():assert v.shape==(604,1 if a=='self' else 4,64,64,3) and np.isfinite(v).all()
for r in assoc:
 p=Path(r['png']);u=np.array(Image.open(p));assert sha(p)==r['png_sha256'];assert hashlib.sha256(u.tobytes()).hexdigest()==r['pixel_sha256'];assert np.array_equal(u,np.rint(np.clip((arrays[r['arm']][r['source_id'],r['variant']]+1)*127.5,0,255)).astype(np.uint8))
protected=read(ROOT/'transfer_aug/domain_discovery/round2/verification.json')['protected_inputs'];observed={p:sha(ROOT/p) for p in protected};assert observed==protected
f=open('/tmp/fastgen-gpu2.lock','r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);f.close()
review=read(R/'visual_review.json');assert review['all_required_sheets_inspected']
seconds=sum(s['gpu_seconds'] for s in ledger['sessions']);assert all(s['ended_at'] is not None for s in ledger['sessions']) and seconds<28800
checks={'final_arrays_finite':True,'all_5436_pngs_match_raw_quantization':True,'input_hashes_unchanged':True,'protected_hashes_unchanged':True,'lease_observed_released':True,'gpu_recovery_exact':q['exact_gpu_recovery'],'representative512_replay_exact':t['representative_replay_exact'],'visual_inspection_complete':True}
records=read(ROOT/'transfer_aug/shared/round2/manifest.json')['records'];by={r['path']:r['appearance_score'] for r in records};maxerr=max(abs(r['source_score']-by[r['source_path']]) for r in read(R/'measurements.json'));assert maxerr<1e-10
write(R/'final_verification.json',{'status':'passed','checks':checks,'protected_inputs':observed,'max_frozen_source_score_reproduction_error':maxerr})
proof={n:str(R/n) for n in ('SUMMARY.md','qualification.json','verification.json','final_verification.json','training_preflight.json','training_complete.json','gpu_resource_ledger.json','measurements.json','source_level_metrics.json','aggregate_metrics.json','diversity.json','tails.json','visual_review.json','review_inventory.json','output_hashes.json','gallery.html','commands.txt','metric_correction.json')}
for p in proof.values():assert Path(p).exists(),p
result={'experiment_id':'TA-006','contract_revision':1,'contract_sha256':sha(R/'CONTRACT_v1.md'),'protocol_sha256':sha(ROOT/'transfer_aug/shared/round3/TRAINING_PROTOCOL_v1.md'),'evaluation_manifest_sha256':sha(ROOT/'transfer_aug/shared/round3/evaluation_manifest.json'),'source_revision':setup['upstream_revision'],'status':t['status'],'training_applicable':True,'completed_iterations':t['updates'],'maximum_iterations':20000,'final_outputs':5436,'reference_outputs':2416,'latent_outputs':2416,'self_controls':604,'parameters':q['parameters'],'framework_total_parameters':q['total_parameters'],'parameter_exception':'Authorized StarGAN v2-only exception; outside normal100M cap','resources':{'gpu_uuid':t['gpu_uuid'],'gpu_seconds':seconds,'gpu_hours':seconds/3600,'peak_allocated_bytes':t['peak_allocated_bytes'],'peak_reserved_bytes':t['peak_reserved_bytes'],'lease_released':True},'checks':checks,'proof_paths':proof,'incidents':['Initial GPU qualification stopped before updates because noncontiguous input was incompatible with upstream R1 view; loader made contiguous NCHW. Failed3.8137seconds included.','Report-only correction removed PNG quantization from clipped source RMS and pairwise diversity; tail membership/order unchanged.'],'limitations':['A1 diagnostic subset participated in domain discovery; not independent validation.','Statistical appearance groups are not art-style labels.','Pixel diversity is not semantic diversity or identity accuracy.','No A2/B, external pretrained representation, SiDA training or FID improvement test.','AI visual review is small, unblinded and descriptive.']}
write(R/'result.json',result);write(R/'handoff_hashes.json',{p.name:sha(p) for p in R.iterdir() if p.suffix in ('.py','.json','.md','.html') and p.name!='handoff_hashes.json'});print(json.dumps(result,indent=2))
