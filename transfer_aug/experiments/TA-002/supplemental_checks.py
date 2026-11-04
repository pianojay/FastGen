"""Explicit synthetic bypass/self-donor checks and substantive clipping audit."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ta002',H/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(725)
fixtures=[('black',np.zeros((64,64,3))),('white',np.ones((64,64,3))),('constant_color',np.broadcast_to([.1,.5,.9],(64,64,3)).copy()),('near_constant',np.full((64,64,3),.5)+rng.normal(0,1e-12,(64,64,3)))]
rows=[]
for name,c in fixtures:
 d=np.broadcast_to([.8,.2,.4],c.shape).copy()
 for r in m.RADII:
  full,_=m.transform(c,d,r,1.)
  zero,_=m.transform(c,d,r,0.)
  assert np.array_equal(zero,c)
  for a in m.ALPHAS:
   own,imag=m.transform(c,c,r,a);x,ximag=m.transform(c,d,r,a)
   err=float(np.max(np.abs(own-c)));lsb=int(np.max(np.abs(m.pixels(own).astype(int)-m.pixels(c).astype(int))))
   linear=float(np.max(np.abs(x-((1-a)*c+a*full))))
   assert err<=1e-6 and lsb<=1 and linear<=1e-10 and max(imag,ximag)<=1e-10
   rows.append({'fixture':name,'radius':r,'alpha':a,'self_donor_max_error':err,'self_donor_export_max_lsb':lsb,'alpha0_byte_exact':True,'linear_error':linear})
metrics=json.loads((H/'metrics.json').read_text());tails=json.loads((H/'tail_selection.json').read_text())
clip_audit={}
for arm in m.ARMS:
 sub=[x for x in metrics if x['arm']==arm]
 clip_audit[arm]={'transformed_outputs':len(sub),'zero_denominator_outputs':sum(x['outside_mask_zero_denominator'] for x in sub),'any_strict_clipping_outputs':sum(x['clipped_fraction']>0 for x in sub),'clipping_only_roundoff_max_excursion_le_1e_12_outputs':sum(x['clipped_fraction']>0 and x['max_excursion']<=1e-12 for x in sub),'substantive_clipping_max_excursion_gt_1e_12_outputs':sum(x['max_excursion']>1e-12 for x in sub),'outside_raw_l2_max':max(x['outside_mask_raw_l2'] for x in sub),'outside_raw_relative_l2_max':max(x['outside_mask_raw_relative_l2'] for x in sub if x['outside_mask_raw_relative_l2'] is not None),'outside_clipped_l2_max':max(x['outside_mask_clipped_l2'] for x in sub),'tail_excursion_min':min(x['max_excursion'] for x in tails if x['arm']==arm),'tail_excursion_max':max(x['max_excursion'] for x in tails if x['arm']==arm)}
result={'pass':True,'synthetic_count':len(rows),'synthetic':rows,'clipping_audit':clip_audit,'caveat':'Strict clipping frequency is not artifact severity. Tail excursions here are substantive, while visually mild white-background examples clip many channels.'}
m.write_json('verification/supplemental_checks.json',result)
print(json.dumps(clip_audit,indent=2))
