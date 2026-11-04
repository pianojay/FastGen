import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3]; O=Path(__file__).resolve().parent; D=R/'transfer_aug/domain_discovery/round2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
m=json.loads((D/'domain_manifest.json').read_text()); assert sha(D/'domain_manifest.json')=='d6a99c722a46a873a02cc1cf15bac54ca2e56a5ea4c6395ca6a18f499bcbff7f'
a=(R/'eda/resources/A.txt').read_text().splitlines(); ids={p:i for i,p in enumerate(a)}; a1=set((R/'eda/resources/A1.txt').read_text().splitlines()); src=np.load(R/'eda/features/pixels64.npy',mmap_mode='r'); rows=[]; pixels=[]
for r in m['rows']:
 if r['domain'] not in ('D0','D1'):continue
 assert r['path'] in a1
 u=src[ids[r['path']]]; assert hashlib.sha256(u.tobytes()).hexdigest()==r['pixel_sha256']
 rows.append(dict(path=r['path'],sha256=r['file_sha256'],pixel_sha256=r['pixel_sha256'],partition='A1',class_id=r['category_id'],category=r['category'],domain=r['domain'],role={'optimize':'optimization','holdout':'diagnostic'}[r['role']],pixel_index=len(pixels),appearance_score=r['score']));pixels.append(u)
np.save(O/'pixels_uint8.npy',np.stack(pixels))
write(O/'manifest.json',dict(schema=1,seed=20261008,source_manifest_sha256=sha(D/'domain_manifest.json'),pixels='pixels_uint8.npy',records=rows))
write(O/'freeze.json',dict(files={p.name:sha(p) for p in [O/'manifest.json',O/'pixels_uint8.npy',O/'freeze.py']},domain_manifest_sha256=sha(D/'domain_manifest.json'),domain_artifact_manifest_sha256=sha(D/'artifact_manifest.json'),records=len(rows),counts={d:{role:sum(r['domain']==d and r['role']==role for r in rows) for role in ['optimization','diagnostic']} for d in ['D0','D1']},note='A1 diagnostic images excluded from optimization, but included in domain discovery; not independent validation.'))
print((O/'freeze.json').read_text())
