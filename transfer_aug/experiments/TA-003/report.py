"""CPU EDA of completed frozen CycleGAN run, using read-only frozen descriptors."""
import collections, csv, hashlib, importlib.util, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runner import ROOT,SHARED,DISCOVERY,sha256,write,preflight

spec=importlib.util.spec_from_file_location('domain_descriptors',DISCOVERY/'analyze.py');domain=importlib.util.module_from_spec(spec);spec.loader.exec_module(domain)

def main():
    started=time.time();preflight();out=ROOT/'outputs';eda=ROOT/'eda';eda.mkdir(exist_ok=True)
    records=json.loads((SHARED/'manifest.json').read_text())['records'];pix=np.load(SHARED/'pixels_uint8.npy',mmap_mode='r'); outputs=json.loads((out/'manifest.json').read_text())
    assert len(outputs)==604 and all(r['role']=='diagnostic' for r in outputs)
    pca=json.loads((DISCOVERY/'pca.json').read_text());names=pca['descriptors'];scale=np.array(pca['residual_rms']);load=np.array([pca['pc1_loadings'][n] for n in names]);means=pca['class_means']
    def score(desc,category):return float(((np.array(desc)-means[category])/scale)@load)
    targets={}; source_score_error=[]
    for r in records:
        if r['role']=='optimization': targets.setdefault((r['class_id'],r['domain']),[]).append(r)
        else:source_score_error.append(abs(score(domain.descriptors(pix[r['pixel_index']]),r['category'])-r['appearance_score']))
    assert max(source_score_error)<1e-9
    target_descriptors={key:np.asarray([domain.descriptors(pix[r['pixel_index']]) for r in rr]) for key,rr in targets.items()}
    arrays={d:{k:np.load(out/f'{d}_{k}.npy',mmap_mode='r') for k in ('float','cycle','identity')} for d in ('D0_D1','D1_D0')}
    rows=[]
    for r in outputs:
        d=r['direction'];j=r['output_index'];raw=arrays[d]['float'][j].transpose(1,2,0);y=(raw.astype(np.float64)+1)/2;src=pix[r['pixel_index']].astype(np.float64)/255
        u=np.asarray(Image.open(r['output_path']));assert sha256(r['output_path'])==r['output_sha256'];assert np.isfinite(raw).all()
        sd=np.asarray(domain.descriptors(pix[r['pixel_index']]));od=np.asarray(domain.descriptors(u));ss=score(sd,r['category']);os=score(od,r['category'])
        target=targets[r['class_id'],d[-2:]]; td=target_descriptors[r['class_id'],d[-2:]]; tmean=td.mean(0); ts=np.array([z['appearance_score'] for z in target]);timages=pix[[z['pixel_index'] for z in target]].astype(np.float64)/255
        near=np.sqrt(np.mean((timages-y)**2,axis=(1,2,3)));ni=int(near.argmin());cy=(arrays[d]['cycle'][j].transpose(1,2,0).astype(np.float64)+1)/2;ident=(arrays[d]['identity'][j].transpose(1,2,0).astype(np.float64)+1)/2
        row=dict(direction=d,path=r['path'],category=r['category'],class_id=r['class_id'],pixel_index=r['pixel_index'],output_index=j,
            source_sha256=r['sha256'],source_pixel_sha256=r['pixel_sha256'],output_path=r['output_path'],output_sha256=r['output_sha256'],output_pixel_sha256=r['output_pixel_sha256'],
            output_float_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),source_appearance_score=ss,output_appearance_score=os,appearance_shift=os-ss,
            target_score_mean=float(ts.mean()),target_score_std=float(ts.std()),target_score_q10=float(np.quantile(ts,.1)),target_score_q90=float(np.quantile(ts,.9)),
            output_target_score_percentile=float(np.mean(ts<=os)),source_target_score_distance=abs(ss-ts.mean()),output_target_score_distance=abs(os-ts.mean()),
            rgb_rms=float(np.sqrt(np.mean((y-src)**2))),luminance_edge_change=float(od[6]-sd[6]),
            output_min=float(raw.min()),output_max=float(raw.max()),outside_range_fraction=float(np.mean((raw < -1)|(raw > 1))),
            exact_tanh_endpoint_fraction=float(np.mean((raw==-1)|(raw==1))),quantized_endpoint_fraction=float(np.mean((u==0)|(u==255))),
            nearest_target_pixel_rms=float(near[ni]),nearest_target_path=target[ni]['path'],cycle_rgb_rms=float(np.sqrt(np.mean((cy-src)**2))),
            cycle_rgb_mae=float(np.mean(np.abs(cy-src))),identity_rgb_rms=float(np.sqrt(np.mean((ident-src)**2))),identity_rgb_mae=float(np.mean(np.abs(ident-src))))
        row['source_target_descriptor_rms']=float(np.sqrt(np.mean(((sd-tmean)/scale)**2)))
        row['output_target_descriptor_rms']=float(np.sqrt(np.mean(((od-tmean)/scale)**2)))
        for k,name in enumerate(names):
            row['source_'+name]=float(sd[k]);row['output_'+name]=float(od[k]);row['delta_'+name]=float(od[k]-sd[k])
            row['target_mean_'+name]=float(tmean[k]);row['target_std_'+name]=float(td[:,k].std());row['target_q10_'+name]=float(np.quantile(td[:,k],.1));row['target_q90_'+name]=float(np.quantile(td[:,k],.9))
        rows.append(row)
    with (eda/'per_image.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(eda/'per_image.json',rows)
    aggregate={}
    metrics=['source_target_descriptor_rms','output_target_descriptor_rms','source_appearance_score','output_appearance_score','appearance_shift','target_score_mean','source_target_score_distance','output_target_score_distance','rgb_rms','luminance_edge_change','nearest_target_pixel_rms','cycle_rgb_rms','cycle_rgb_mae','identity_rgb_rms','identity_rgb_mae','outside_range_fraction','exact_tanh_endpoint_fraction','quantized_endpoint_fraction']+['delta_'+n for n in names]
    for d in arrays:
        rr=[r for r in rows if r['direction']==d];stats={}
        for key in metrics:
            v=np.array([r[key] for r in rr]);stats[key]=dict(mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max()),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)))
        hashes=collections.Counter(r['output_pixel_sha256'] for r in rr);fh=collections.Counter(r['output_float_sha256'] for r in rr)
        stats['count']=len(rr);stats['exact_duplicate_quantized_excess']=sum(v-1 for v in hashes.values());stats['exact_duplicate_float_excess']=sum(v-1 for v in fh.values())
        stats['toward_target_mean_fraction']=float(np.mean([r['output_target_score_distance']<r['source_target_score_distance'] for r in rr]));aggregate[d]=stats
    write(eda/'aggregate.json',aggregate)
    reviews=json.loads((out/'review_manifest.json').read_text());milestones=sorted(map(int,reviews['milestones']));timeline={i:np.load(out/f'review_update_{i}.npy') for i in milestones}
    review_records=reviews['records'];sheet_records=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
    def paste(im,arr,x,y):im.paste(Image.fromarray(arr).resize((96,96),Image.Resampling.NEAREST),(x,y))
    for d in arrays:
        ix=[i for i,r in enumerate(review_records) if r['domain']==d[:2]];assert len(ix)==32
        for page,start in enumerate(range(0,len(ix),8),1):
            subset=ix[start:start+8];im=Image.new('RGB',(120+(len(milestones)+1)*104,32+len(subset)*130),'#e0e0e0');draw=ImageDraw.Draw(im)
            for col,label in enumerate(['source']+[str(i) for i in milestones]):draw.text((120+col*104,8),label,fill='black',font=font)
            for row,index in enumerate(subset):
                r=review_records[index];yy=32+row*130;draw.text((4,yy+5),f'{r["category"]}\n{d}\nreview {index:02d}',fill='black',font=font);paste(im,pix[r['pixel_index']],120,yy)
                for col,update in enumerate(milestones,1):paste(im,timeline[update][index],120+col*104,yy)
                draw.text((120,yy+100),Path(r['path']).name[:75],fill='black',font=font)
            path=eda/f'representative_{d}_{page}.png';im.save(path);sheet_records.append(dict(sheet=str(path),kind='representative',direction=d,review_indices=subset,paths=[review_records[i]['path'] for i in subset],milestones=milestones))
    tails=sorted(rows,key=lambda r:(-r['rgb_rms'],r['direction'],r['path']))[:8];write(eda/'tail_selection.json',tails)
    im=Image.new('RGB',(610,32+130*8),'#e0e0e0');draw=ImageDraw.Draw(im)
    for col,label in enumerate(['source','final','cycle','identity']):draw.text((190+col*104,8),label,fill='black',font=font)
    for i,r in enumerate(tails):
        yy=32+i*130;d=r['direction'];j=r['output_index'];draw.text((5,yy+5),f'{r["category"]} {d}\nRMS {r["rgb_rms"]:.3f}\nsource index {r["pixel_index"]}',fill='black',font=font)
        imgs=[pix[r['pixel_index']],np.asarray(Image.open(r['output_path']))]
        imgs += [np.clip(np.rint((arrays[d][k][j].transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8) for k in ('cycle','identity')]
        for col,arr in enumerate(imgs):paste(im,arr,190+col*104,yy)
    path=eda/'diagnostic_tails.png';im.save(path);sheet_records.append(dict(sheet=str(path),kind='diagnostic_tail',paths=[r['path'] for r in tails]))
    write(eda/'sheet_manifest.json',sheet_records)
    allhashes=collections.Counter(r['output_pixel_sha256'] for r in rows)
    write(eda/'verification.json',dict(outputs=604,representative_sources=64,tail_sources=8,representative_sheets=8,tail_sheets=1,tail_representative_overlap=len({r['path'] for r in tails}&{r['path'] for r in review_records}),tail_direction_counts=dict(collections.Counter(r['direction'] for r in tails)),
        source_score_max_error=max(source_score_error),diagnostic_only=True,optimization_diagnostic_pixel_overlap=0,
        output_ranges_finite=all(np.isfinite(np.asarray(v['float'])).all() for v in arrays.values()),
        cross_direction_quantized_duplicate_excess=sum(v-1 for v in allhashes.values()),
        clipping_definition='outside [-1,1] before display; quantized endpoints counted separately, not clipping',
        descriptor_space='quantized display RGB uint8 using frozen analyze.py definitions; RGB RMS/cycle/identity measured on raw generated float mapped to [0,1]',
        projection='frozen class-centered residual-RMS-scaled PC1; statistical appearance axis, not style or quality',
        elapsed_cpu_report_seconds=time.time()-started))
    html='<html><head><meta charset="utf-8"><title>TA-003 CycleGAN</title></head><body><h1>TA-003 CycleGAN diagnostic review</h1><p>Columns show source and fixed update checkpoints. Descriptive A1 diagnostics; not independent validation or a quality score.</p>'
    for r in sheet_records:html+=f'<h2>{Path(r["sheet"]).stem}</h2><img style="max-width:100%" src="eda/{Path(r["sheet"]).name}">'
    (ROOT/'gallery.html').write_text(html+'</body></html>')
    print(json.dumps(dict(status='EDA ready for visual review',outputs=604,sheets=len(sheet_records),seconds=time.time()-started)))

if __name__=='__main__':main()
