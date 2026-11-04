"""CPU-only frozen descriptor analysis and contact sheets."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;ROOT=R.parents[2];SHARED=ROOT/'transfer_aug/shared/round2';O=R/'outputs'
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_descriptors',ROOT/'transfer_aug/domain_discovery/round2/analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pca=json.loads((ROOT/'transfer_aug/domain_discovery/round2/pca.json').read_text())
means=pca['class_means'];scale=np.array(pca['residual_rms']);keep=np.array(pca['kept']);load=np.array([pca['pc1_loadings'][n] for n,k in zip(module.NAMES,keep) if k])
def score(u,category):
    d=np.array(module.descriptors(u));return float(((d-np.array(means[category]))/scale)[keep]@load),d
records=json.loads((SHARED/'manifest.json').read_text())['records'];pixels=np.load(SHARED/'pixels_uint8.npy',mmap_mode='r');assoc=json.loads((O/'associations.json').read_text());measurements=[]
targets={}
for row in records:
    if row['role']=='optimization' and row['domain']=='D1':targets.setdefault(row['category'],[]).append(row)
arrays={label:np.load(O/f'{label}_float.npy',mmap_mode='r') for label in ('D0_to_D1','D1_identity')}
for r in assoc:
    src=pixels[r['pixel_index']];raw=arrays[r['direction']][r['array_index']];floating=(raw+1)/2
    out=np.rint(np.clip(floating*255,0,255)).astype(np.uint8)
    s,ds=score(src,r['category']);t,dt=score(out,r['category']);candidates=targets[r['category']]
    rmses=[float(np.sqrt(np.mean((floating.astype(np.float64)-pixels[x['pixel_index']].astype(np.float64)/255)**2))) for x in candidates]
    nearest=int(np.argmin(rmses));opt_scores=np.array([x['appearance_score'] for x in candidates])
    m={**r,'source_score':s,'output_score':t,'score_shift':t-s,'rgb_rms':float(np.sqrt(np.mean((floating.astype(np.float64)-src.astype(np.float64)/255)**2))),
       'edge_change':float(dt[6]-ds[6]),'raw_min':float(raw.min()),'raw_max':float(raw.max()),'out_of_range_fraction':float(np.mean((raw < -1)|(raw > 1))),
       'quantized_boundary_fraction':float(np.mean((out==0)|(out==255))),'nearest_target_rms':rmses[nearest],'nearest_target_path':candidates[nearest]['path'],
       'target_optimization_score_mean':float(opt_scores.mean()),'target_optimization_score_std':float(opt_scores.std()),'target_score_percentile':float(np.mean(opt_scores<=t)),
       'source_descriptors':ds.tolist(),'output_descriptors':dt.tolist(),'descriptor_changes':(dt-ds).tolist()}
    measurements.append(m)
write(R/'measurements.json',measurements)
fields=['direction','path','category','source_score','output_score','score_shift','rgb_rms','edge_change','raw_min','raw_max','out_of_range_fraction','quantized_boundary_fraction','nearest_target_rms','nearest_target_path','target_optimization_score_mean','target_optimization_score_std','target_score_percentile']
with (R/'measurements.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(measurements)
summary={}
for direction in arrays:
    rows=[r for r in measurements if r['direction']==direction];stats={}
    for key in ['source_score','output_score','score_shift','rgb_rms','edge_change','out_of_range_fraction','quantized_boundary_fraction','nearest_target_rms','target_optimization_score_mean','target_score_percentile']:
        a=np.array([r[key] for r in rows]);stats[key]={'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95))}
    stats['positive_score_shift_fraction']=float(np.mean([r['score_shift']>0 for r in rows]));stats['exact_duplicate_excess']=len(rows)-len({r['output_pixel_sha256'] for r in rows});stats['count']=len(rows)
    stats['mean_descriptor_changes']=dict(zip(module.NAMES,np.mean([r['descriptor_changes'] for r in rows],axis=0).tolist()))
    summary[direction]=stats
write(R/'aggregate_metrics.json',summary)
write(R/'metric_conventions.json',{'descriptors_PCA':'Quantized uint8 viewing outputs using np.rint(clip((raw+1)*127.5,0,255)); frozen descriptors and class means/scales/PC1 unchanged.','rgb_rms':'Raw generated float mapped to [0,1], compared with source uint8/255, RMS over all RGB channels/pixels.','nearest_target_rms':'Raw generated float mapped to [0,1], minimum RMS to each same-class D1 optimization uint8/255 image.','edge_change':'Difference in frozen neighbor_gradient descriptor on quantized viewing output versus source.','tails':'Eight largest raw-float source/output RGB RMS among D0_to_D1 translations, ties direction then source path. Identity controls excluded from tail-selection population.'})
sheets=R/'sheets';sheets.mkdir(exist_ok=True);review=[]
for domain in ('D0','D1'):
    rows=json.loads((O/f'review_{domain}_rows.json').read_text())
    milestones=[name for name in ('initial','1000','5000','10000','final') if (O/f'review_{domain}_{name}.npy').exists()]
    values={name:np.rint(np.clip((np.load(O/f'review_{domain}_{name}.npy')+1)*127.5,0,255)).astype(np.uint8) for name in milestones}
    for page in range(4):
        sample=rows[page*8:(page+1)*8];im=Image.new('RGB',(190+110*(1+len(milestones)),8*114+44),'#dedede');draw=ImageDraw.Draw(im)
        for j,name in enumerate(['source']+milestones):draw.text((190+j*110,8),name,fill='black')
        for i,row in enumerate(sample):
            y=44+i*114;draw.text((5,y),row['category'],fill='black');draw.text((5,y+16),Path(row['path']).name[:24],fill='black')
            ims=[pixels[row['pixel_index']]]+[values[name][page*8+i] for name in milestones]
            for j,u in enumerate(ims):im.paste(Image.fromarray(u).resize((96,96),Image.Resampling.NEAREST),(190+j*110,y))
            review.append({'sheet':f'representative_{domain}_{page+1}.png','row':i+1,'domain':domain,'path':row['path'],'category':row['category'],'milestones':milestones})
        im.save(sheets/f'representative_{domain}_{page+1}.png')
# Tails are the translated diagnostic population; identity controls are descriptive only.
tails=sorted([r for r in measurements if r['direction']=='D0_to_D1'],key=lambda r:(-r['rgb_rms'],r['direction'],r['path']))[:8]
im=Image.new('RGB',(640,8*118+36),'#dedede');draw=ImageDraw.Draw(im);draw.text((8,8),'Largest source/output RGB RMS tails: source | final | nearest D1 optimization',fill='black')
by_path={r['path']:r for r in records}
for i,r in enumerate(tails):
    y=36+i*118;draw.text((8,y),r['category'],fill='black');draw.text((8,y+18),f'RMS {r["rgb_rms"]:.3f}',fill='black')
    images=[pixels[r['pixel_index']],np.array(Image.open(r['png'])),pixels[by_path[r['nearest_target_path']]['pixel_index']]]
    for j,u in enumerate(images):im.paste(Image.fromarray(u).resize((96,96),Image.Resampling.NEAREST),(220+132*j,y))
    review.append({'sheet':'tails.png','row':i+1,'domain':'D0','path':r['path'],'category':r['category'],'rgb_rms':r['rgb_rms']})
im.save(sheets/'tails.png');write(R/'review_inventory.json',review);write(R/'tails.json',tails)
write(R/'output_hashes.json',{'files':{str(p.relative_to(R)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()},'sheets':{str(p.relative_to(R)):sha(p) for p in sorted(sheets.glob('*.png'))},'descriptor_source_sha256':sha(ROOT/'transfer_aug/domain_discovery/round2/analyze.py'),'pca_sha256':sha(ROOT/'transfer_aug/domain_discovery/round2/pca.json')})
print(json.dumps(summary,indent=2))
