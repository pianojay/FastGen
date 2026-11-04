import json,itertools
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
def read(p):return json.loads(Path(p).read_text())
ev=read(ROOT/'transfer_aug/shared/round3/evaluation_manifest.json')['records'];pixels=np.load(ROOT/'transfer_aug/shared/round2/pixels_uint8.npy',mmap_mode='r');rows=read(R/'measurements.json');div=read(R/'diversity.json');by={(q['source_id'],q['arm'],q['variant']):q for q in rows};db={(q['source_id'],q['arm']):q for q in div}
arrays={a:np.load(R/'outputs'/f'{a}_float.npy',mmap_mode='r') for a in ('reference','latent','self')}
maxerr={'clipped_source_rms':0.,'clipped_pairwise_diversity':0.,'clipping_mean_magnitude':0.};zero_outside=0
for arm,ar in arrays.items():
 for i,q in enumerate(ev):
  raw=(ar[i].astype(np.float64)+1)/2;clipped=np.clip(raw,0,1);src=pixels[q['source_index']].astype(np.float64)/255
  for v in range(len(raw)):
   r=by[(i,arm,v)];rms=float(np.sqrt(np.mean((clipped[v]-src)**2)));mag=float(np.mean(np.abs(raw[v]-clipped[v])))
   maxerr['clipped_source_rms']=max(maxerr['clipped_source_rms'],abs(rms-r['rgb_rms_clipped']));maxerr['clipping_mean_magnitude']=max(maxerr['clipping_mean_magnitude'],abs(mag-r['clipping_mean_magnitude']))
   if r['out_of_range_fraction']==0:zero_outside+=1;assert r['rgb_rms_raw']==r['rgb_rms_clipped'] and mag==0
  if arm!='self':
   d=float(np.mean([np.sqrt(np.mean((clipped[a]-clipped[b])**2)) for a,b in itertools.combinations(range(4),2)]));maxerr['clipped_pairwise_diversity']=max(maxerr['clipped_pairwise_diversity'],abs(d-db[(i,arm)]['clipped_pairwise_rms']))
assert all(v<1e-12 for v in maxerr.values())
old=read(R/'metric_correction_before.json');tails=read(R/'tails.json');same=[r['source_id'] for r in old['tails']]==[r['source_id'] for r in tails];assert same
rec={'status':'passed','correction':'Clipped source RMS and pairwise diversity now use floating np.clip((raw.astype(float64)+1)/2,0,1); no PNG rounding.','clipping_magnitude':'Already used mean(abs(raw_affine-np.clip(raw_affine,0,1))); unchanged and exhaustively checked.','descriptor_PCA':'Clipped quantized viewing RGB, unchanged.','measurement_outputs_checked':len(rows),'source_arm_diversity_groups_checked':len(div),'max_absolute_errors':maxerr,'zero_out_of_range_outputs_with_exact_raw_clipped_agreement':zero_outside,'tail_membership_and_order_unchanged':same,'tail_source_ids':[r['source_id'] for r in tails],'GPU_or_model_changes':False,'visual_reinspection_required':False}
(R/'metric_correction.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
