import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;T=R/'transfer_aug'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encoded(x):return (json.dumps(x,indent=2,sort_keys=True)+'\n').encode()
def build():
 p=T/'shared/round2';f=json.loads((p/'freeze.json').read_text())
 for name in ['manifest.json','pixels_uint8.npy']:assert sha(p/name)==f['files'][name]
 m=json.loads((p/'manifest.json').read_text());rows=m['records'];rev=json.loads((T/'domain_discovery/round2/review_selection.json').read_text());review={x['path'] for x in rev['rows'] if x['role']=='holdout'}
 opt={}
 for r in rows:
  if r['role']=='optimization':opt.setdefault((r['class_id'],r['domain']),[]).append(r)
 result=[];styles={'munit':[],'stargan':[]}
 for r in sorted([r for r in rows if r['role']=='diagnostic'],key=lambda r:(r['domain'],r['class_id'],r['path'])):
  target='D1' if r['domain']=='D0' else 'D0'
  donors=sorted(opt[r['class_id'],target],key=lambda q:hashlib.sha256(f"20261008|round3-reference|{r['path']}|{q['path']}".encode()).hexdigest())[:4]
  assert len(donors)==4 and len({q['pixel_sha256'] for q in donors})==4
  result.append(dict(source_index=r['pixel_index'],source_path=r['path'],source_pixel_sha256=r['pixel_sha256'],class_id=r['class_id'],category=r['category'],source_domain=r['domain'],target_domain=target,representative=r['path'] in review,reference_indices=[q['pixel_index'] for q in donors],reference_paths=[q['path'] for q in donors]))
  for name,dim in [('munit',8),('stargan',16)]:
   seed=int.from_bytes(hashlib.sha256(f"20261008|round3-latent|{name}|{r['path']}".encode()).digest()[:8],'little');styles[name].append(np.random.default_rng(seed).standard_normal((4,dim)).astype(np.float32))
 assert len(result)==604 and sum(r['representative'] for r in result)==64
 return dict(schema=1,seed=20261008,source_manifest_sha256=f['files']['manifest.json'],source_pixels_sha256=f['files']['pixels_uint8.npy'],reference_policy='Four distinct same-class opposite-domain optimization donors, SHA256 ordered per source; no image selection by output.',style_policy='Independent source/method-hashed NumPy PCG64 standard normals,4 per source;MUNIT8dim,StarGAN16dim.',records=result),{k:np.stack(v) for k,v in styles.items()}
m,z=build();m2,z2=build();assert encoded(m)==encoded(m2)
for k in z:assert np.array_equal(z[k],z2[k]);np.save(O/f'{k}_latent.npy',z[k])
(O/'evaluation_manifest.json').write_bytes(encoded(m))
files=['evaluation_manifest.json','munit_latent.npy','stargan_latent.npy','freeze_evaluation.py'];out=dict(schema=1,files={n:sha(O/n) for n in files},sources=604,representative_sources=64,reference_outputs=2416,latent_outputs=2416,self_controls=604,total_final_outputs_per_method=5436,regeneration_exact=True,all_sources_A1_diagnostic=True,all_references_A1_optimization=True,source_bundle_manifest_sha256=m['source_manifest_sha256'],source_bundle_pixels_sha256=m['source_pixels_sha256'])
(O/'freeze.json').write_bytes(encoded(out));print(json.dumps(out,indent=2))
if __name__=='__main__':pass
