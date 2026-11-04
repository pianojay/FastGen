import fcntl,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parent
rows=json.loads((r/'outputs/associations.json').read_text());assert len(rows)==604
arrays={n:np.load(r/'outputs'/f'{n}_float.npy') for n in ('D0_to_D1','D1_identity')}
for n,a in arrays.items():assert a.shape==(302,64,64,3) and np.isfinite(a).all()
for item in rows:
    p=Path(item['png']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['png_sha256']
    u=np.array(Image.open(p));assert hashlib.sha256(u.tobytes()).hexdigest()==item['output_pixel_sha256']
    assert np.array_equal(u,np.rint(np.clip((arrays[item['direction']][item['array_index']]+1)*127.5,0,255)).astype(np.uint8))
ms=json.loads((r/'measurements.json').read_text());max_score_error=max(abs(x['source_score']-x['appearance_score']) for x in ms);assert max_score_error<1e-10
lease=open('/tmp/fastgen-gpu2.lock','r');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
result=json.loads((r/'result.json').read_text())
for p in result['proof_paths'].values():assert Path(p).exists(),p
v={'status':'passed','finite_float_arrays':True,'arrays_shape':[302,64,64,3],'translation_count':302,'identity_control_count':302,'all_604_png_hashes_and_quantized_pixels_verified':True,'max_source_score_reproduction_error':max_score_error,'all_handoff_proof_paths_resolve':True,'gpu_lease_observed_free':True}
(r/'final_verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
