"""CPU-only frozen descriptor EDA and complete review sheets."""
import csv,hashlib,importlib.util,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;ROOT=R.parents[2];S=ROOT/'transfer_aug/shared/round2';E=ROOT/'transfer_aug/shared/round3';O=R/'outputs'
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location('frozen_descriptors',ROOT/'transfer_aug/domain_discovery/round2/analyze.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
pca=json.loads((ROOT/'transfer_aug/domain_discovery/round2/pca.json').read_text());means=pca['class_means'];scale=np.array(pca['residual_rms']);keep=np.array(pca['kept']);load=np.array([pca['pc1_loadings'][n] for n,k in zip(mod.NAMES,keep) if k])
def score(u,cat):
 d=np.array(mod.descriptors(u));return float(((d-np.array(means[cat]))/scale)[keep]@load),d
quant=lambda a:np.rint(np.clip((a+1)*127.5,0,255)).astype(np.uint8)
rms=lambda a,b:float(np.sqrt(np.mean((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2)))
ev=json.loads((E/'evaluation_manifest.json').read_text())['records'];records=json.loads((S/'manifest.json').read_text())['records'];pixels=np.load(S/'pixels_uint8.npy',mmap_mode='r');arrays={a:np.load(O/f'{a}_float.npy',mmap_mode='r') for a in ('reference','latent','self')}
assoc=json.loads((O/'associations.json').read_text());lookup={(a['source_id'],a['arm'],a['variant']):a for a in assoc};targets={}
for r in records:
 if r['role']=='optimization':targets.setdefault((r['category'],r['domain']),[]).append(r)
measure=[];diversity=[];source_level=[]
for i,q in enumerate(ev):
 src=pixels[q['source_index']];ss,ds=score(src,q['category']);candidates=targets[(q['category'],q['target_domain'])];candidate_arrays=np.stack([pixels[r['pixel_index']] for r in candidates]).astype(np.float64)/255
 for arm,ar in arrays.items():
  rows=[]
  for v,raw in enumerate(ar[i]):
   f=(raw.astype(np.float64)+1)/2;u=quant(raw);oscore,od=score(u,q['category']);dist=np.sqrt(np.mean((candidate_arrays-f)**2,axis=(1,2,3)));n=int(np.argmin(dist));ts=np.array([r['appearance_score'] for r in candidates])
   row={'source_id':i,'category':q['category'],'source_path':q['source_path'],'direction':q['source_domain']+'_to_'+q['target_domain'],'arm':arm,'variant':v,'source_score':ss,'output_score':oscore,'score_shift':oscore-ss,'rgb_rms_raw':rms(f,src/255),'rgb_rms_clipped':rms(np.clip(f,0,1),src/255),'edge_change':float(od[6]-ds[6]),'raw_min':float(raw.min()),'raw_max':float(raw.max()),'out_of_range_fraction':float(np.mean((raw<-1)|(raw>1))),'clipping_mean_magnitude':float(np.abs(f-np.clip(f,0,1)).mean()),'nearest_target_rms':float(dist[n]),'nearest_target_path':candidates[n]['path'],'target_score_mean':float(ts.mean()),'target_score_std':float(ts.std()),'target_score_percentile':float(np.mean(ts<=oscore)),'descriptor_changes':(od-ds).tolist(),'source_descriptors':ds.tolist(),'output_descriptors':od.tolist(),'pixel_sha256':lookup[(i,arm,v)]['pixel_sha256']}
   rows.append(row);measure.append(row)
  keys=['source_score','output_score','score_shift','rgb_rms_raw','rgb_rms_clipped','edge_change','out_of_range_fraction','clipping_mean_magnitude','nearest_target_rms','target_score_mean','target_score_percentile']
  source_level.append({'source_id':i,'arm':arm,'direction':rows[0]['direction'],**{k:float(np.mean([r[k] for r in rows])) for k in keys}})
  if arm!='self':
   raw=(ar[i].astype(np.float64)+1)/2;cl=np.clip(raw,0,1);pairs=list(itertools.combinations(range(4),2))
   diversity.append({'source_id':i,'source_path':q['source_path'],'arm':arm,'direction':rows[0]['direction'],'raw_pairwise_rms':float(np.mean([rms(raw[a],raw[b]) for a,b in pairs])),'clipped_pairwise_rms':float(np.mean([rms(cl[a],cl[b]) for a,b in pairs])),'appearance_score_std':float(np.std([r['output_score'] for r in rows])),'identical_variant_pair_count':sum(rows[a]['pixel_sha256']==rows[b]['pixel_sha256'] for a,b in pairs)})
 if i%100==0:print('source',i,flush=True)
write(R/'measurements.json',measure);write(R/'source_level_metrics.json',source_level);write(R/'diversity.json',diversity)
with (R/'measurements.csv').open('w') as f:
 keys=[k for k in measure[0] if 'descriptors' not in k and k!='descriptor_changes'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(measure)
def stats(a):
 a=np.asarray(a);return {'mean':float(a.mean()),'median':float(np.median(a)),'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())}
summary={}
for direction in ('D0_to_D1','D1_to_D0'):
 for arm in arrays:
  rows=[r for r in source_level if r['direction']==direction and r['arm']==arm];d=[r for r in diversity if r['direction']==direction and r['arm']==arm];m=[r for r in measure if r['direction']==direction and r['arm']==arm]
  agg={k:stats([r[k] for r in rows]) for k in rows[0] if k not in ('source_id','arm','direction')};agg['source_count']=len(rows);agg['output_count']=len(m);agg['duplicate_excess']=len(m)-len({r['pixel_sha256'] for r in m});agg['raw_min']=min(r['raw_min'] for r in m);agg['raw_max']=max(r['raw_max'] for r in m)
  if d:agg['diversity']={k:stats([r[k] for r in d]) for k in ('raw_pairwise_rms','clipped_pairwise_rms','appearance_score_std','identical_variant_pair_count')}
  summary[direction+'/'+arm]=agg
write(R/'aggregate_metrics.json',summary)
# Unique-source diagnostic tails, exactly as protocol.
tails=[];seen=set()
for r in sorted([r for r in measure if r['arm']!='self'],key=lambda r:(-r['rgb_rms_raw'],r['source_path'],r['arm'],r['variant'])):
 if r['source_id'] in seen:continue
 seen.add(r['source_id']);tails.append({'source_id':r['source_id'],'reason':'largest_raw_source_RMS','arm':r['arm'],'variant':r['variant'],'metric':r['rgb_rms_raw']})
 if len(tails)==4:break
for r in sorted([r for r in diversity if r['arm']=='latent'],key=lambda r:(r['clipped_pairwise_rms'],r['source_path'])):
 if r['source_id'] in seen:continue
 seen.add(r['source_id']);tails.append({'source_id':r['source_id'],'reason':'smallest_clipped_latent_diversity','metric':r['clipped_pairwise_rms']})
 if len(tails)==8:break
for r in tails:r['representative_overlap']=ev[r['source_id']]['representative'];r['source_path']=ev[r['source_id']]['source_path']
write(R/'tails.json',tails)
sheets=R/'sheets';sheets.mkdir(exist_ok=True);inventory=[]
def finalsheet(ids,name,label):
 # Three rows/source: source+four donors; self+four reference; source+four latent.
 im=Image.new('RGB',(880,60+len(ids)*330),'#e8e8e8');draw=ImageDraw.Draw(im);draw.text((8,8),label,fill='black')
 for j,i in enumerate(ids):
  q=ev[i];y=50+j*330;draw.text((8,y),f'{i} {q["category"]} {q["source_domain"]}->{q["target_domain"]}',fill='black')
  groups=[('donors',[pixels[q['source_index']]]+[pixels[k] for k in q['reference_indices']]),('reference',[quant(arrays['self'][i,0])]+list(quant(arrays['reference'][i]))),('latent',[pixels[q['source_index']]]+list(quant(arrays['latent'][i])))]
  for k,(lab,ims) in enumerate(groups):
   yy=y+20+k*100;draw.text((8,yy),lab,fill='black');draw.text((8,yy+20),'first=source' if k!=1 else 'first=self',fill='black')
   for col,u in enumerate(ims):im.paste(Image.fromarray(u).resize((96,96),Image.Resampling.NEAREST),(150+col*140,yy))
  inventory.append({'sheet':name,'source_id':i,'source_path':q['source_path'],'category':q['category'],'direction':q['source_domain']+'_to_'+q['target_domain'],'scope':'representative' if name.startswith('representative') else 'tail','outputs_reviewed':9})
 im.save(sheets/name)
review=[i for i,q in enumerate(ev) if q['representative']]
for p in range(16):finalsheet(review[p*4:p*4+4],f'representative_{p+1:02d}.png','Fixed representative sources; four references and four latent variants')
for p in range(2):finalsheet([r['source_id'] for r in tails[p*4:p*4+4]],f'tails_{p+1}.png','Diagnostic tails; selected observations, not prevalence estimates')
ti=json.loads((O/'trace_indices.json').read_text());labels=[s for s in ('initial','1000','5000','10000','final') if (O/f'trace_{s}_reference.npy').exists()]
trace={s:{a:quant(np.load(O/f'trace_{s}_{a}.npy')) for a in ('reference','latent')} for s in labels}
for p in range(4):
 ids=ti[p*8:p*8+8];im=Image.new('RGB',(180+(1+len(labels))*110,40+len(ids)*215),'#e8e8e8');draw=ImageDraw.Draw(im)
 for j,l in enumerate(['source']+labels):draw.text((180+j*110,8),l,fill='black')
 for j,i in enumerate(ids):
  q=ev[i];y=35+j*215;draw.text((5,y),f'{i} {q["category"]}',fill='black');draw.text((5,y+15),q['source_domain']+' -> '+q['target_domain'],fill='black')
  for k,a in enumerate(('reference','latent')):
   draw.text((5,y+45+k*100),a,fill='black');ims=[pixels[q['source_index']]]+[trace[s][a][p*8+j] for s in labels]
   for col,u in enumerate(ims):im.paste(Image.fromarray(u).resize((96,96),Image.Resampling.NEAREST),(180+col*110,y+k*100))
 im.save(sheets/f'trace_{p+1}.png')
write(R/'review_inventory.json',{'final_sources':inventory,'trace_sources':ti,'trace_milestones':labels,'representative_count':64,'tail_count':8})
write(R/'metric_conventions.json',{'descriptors':'Frozen eleven descriptors, class centering/scales/PC1 on clipped rounded RGB','raw_RMS':'Raw generated values affinely (x+1)/2; may extend outside nominal [0,1]','clipped_RMS_and_diversity':'Floating affine RGB clipped with np.clip((raw.astype(float64)+1)/2,0,1), without PNG quantization','aggregation':'Four variants averaged per source then summarized across302 sources/direction/arm','diversity':'Mean six pairwise pixel RMS; raw and clipped separately; not semantic diversity','tail_selection':'Four largest unique-source raw RMS then four distinct smallest clipped latent diversity','review':'Small unblinded AI review; no class accuracy/FID claims'})
write(R/'output_hashes.json',{'outputs':{str(p.relative_to(R)):sha(p) for p in O.rglob('*') if p.is_file()},'sheets':{str(p.relative_to(R)):sha(p) for p in sheets.glob('*.png')},'frozen_descriptor_sha256':sha(ROOT/'transfer_aug/domain_discovery/round2/analyze.py'),'pca_sha256':sha(ROOT/'transfer_aug/domain_discovery/round2/pca.json')})
html='<html><body><h1>TA-006 StarGAN v2</h1><p>Scratch A1-only exploratory pilot. Statistical appearance groups; no model-selection/FID claim.</p>'
for p in sorted(sheets.glob('*.png')):html+=f'<h2>{p.stem}</h2><img src="sheets/{p.name}" style="max-width:100%">'
(R/'gallery.html').write_text(html+'</body></html>');print('REPORT COMPLETE',flush=True)
