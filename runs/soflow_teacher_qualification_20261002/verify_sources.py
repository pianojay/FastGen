from pathlib import Path
import json,hashlib
SID=Path('/home/user/jw_soflow/FastGen-SiD-DiT');SF=Path('/home/user/jw_soflow/FastGen-SoFlow');OUT=SID/'runs/soflow_teacher_qualification_20261002'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((OUT/'source_manifest.json').read_text())
checks={p:sha(p)==info['sha256']==sha(OUT/info['snapshot']) for p,info in manifest.items()}
assert all(checks.values())
a={p.name:sha(p) for p in sorted((SID/'artifacts/development_reference').glob('*.png'))};b={p.name:sha(p) for p in sorted((SF/'runs/pokemon_unet128_seed42/reference_dev').glob('*.png'))}
ahash=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest();bhash=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert a==b and ahash=='1c926031c13a015673774e50991e881d5a76a63e5b84e44abbc628f9ecc82f71'
cache=SID/'artifacts/data_cache/development.npy';cachemeta=json.loads((SID/'artifacts/data_cache/development.json').read_text());assert sha(cache)==cachemeta['cache_sha256']
record={'source_snapshots_match_originals':checks,'development_reference_identical_between_projects':True,'reference_count':len(a),'reference_image_manifest_sha256':ahash,'development_pixel_cache_sha256':sha(cache),'partition_sha256':sha(OUT/'snapshot/development_split.json')}
(OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
