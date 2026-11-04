"""A1-only fixed multimodal EDA; no learned evaluation representations."""
import collections,csv,hashlib,importlib.util,itertools,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runner import ROOT,SHARED,ROUND3,DISCOVERY,preflight,write
spec=importlib.util.spec_from_file_location('domain_descriptors',DISCOVERY/'analyze.py');domain=importlib.util.module_from_spec(spec);spec.loader.exec_module(domain)
def main():
    cpu_started=time.monotonic();preflight();out=ROOT/'outputs';eda=ROOT/'eda';eda.mkdir(exist_ok=True)
    records=json.loads((ROUND3/'evaluation_manifest.json').read_text())['records'];bundle=json.loads((SHARED/'manifest.json').read_text())['records'];pix=np.load(SHARED/'pixels_uint8.npy',mmap_mode='r');raw=np.load(out/'final_float.npy',mmap_mode='r');assert raw.shape==(604,9,3,64,64)
    pca=json.loads((DISCOVERY/'pca.json').read_text());names=pca['descriptors'];scale=np.asarray(pca['residual_rms']);load=np.asarray([pca['pc1_loadings'][n] for n in names]);means=pca['class_means']
    def score(desc,cat):return float(((desc-means[cat])/scale)@load)
    targets={}
    for r in bundle:
        if r['role']=='optimization':targets.setdefault((r['class_id'],r['domain']),[]).append(r)
    targetcache={key:(pix[[r['pixel_index'] for r in rr]].astype(np.float64)/255,np.asarray([r['appearance_score'] for r in rr])) for key,rr in targets.items()}
    rows=[];source_rows=[];maxsourceerror=0
    for i,r in enumerate(records):
        src=pix[r['source_index']].astype(np.float64)/255;sd=np.asarray(domain.descriptors(pix[r['source_index']]));ss=score(sd,r['category']);source_record=bundle[r['source_index']];maxsourceerror=max(maxsourceerror,abs(ss-source_record['appearance_score']))
        yy=(raw[i].transpose(0,2,3,1).astype(np.float64)+1)/2;uu=np.clip(np.rint((raw[i].transpose(0,2,3,1)+1)*127.5),0,255).astype(np.uint8);oscores=[]
        for j in range(9):
            arm='reference' if j<4 else 'latent' if j<8 else 'self';variant=j%4 if j<8 else 0;y=yy[j];u=uu[j];od=np.asarray(domain.descriptors(u));oscore=score(od,r['category']);oscores.append(oscore)
            p=out/'png'/f'{i:04d}_{j}.png';assert np.array_equal(u,np.asarray(Image.open(p)))
            targetdomain=r['target_domain'] if arm!='self' else r['source_domain'];target,ts=targetcache[r['class_id'],targetdomain];rr=targets[r['class_id'],targetdomain]
            near=np.sqrt(np.mean((target-y)**2,axis=(1,2,3)));ni=int(near.argmin())
            row=dict(source_eval_index=i,source_path=r['source_path'],category=r['category'],source_domain=r['source_domain'],target_domain=targetdomain,arm=arm,variant=variant,output_path=str(p),source_appearance_score=ss,output_appearance_score=oscore,appearance_shift=oscore-ss,target_score_mean=float(ts.mean()),target_score_std=float(ts.std()),target_score_q10=float(np.quantile(ts,.1)),target_score_q90=float(np.quantile(ts,.9)),rgb_rms=float(np.sqrt(np.mean((y-src)**2))),clipped_rgb_rms=float(np.sqrt(np.mean((np.clip(y,0,1)-src)**2))),luminance_edge_change=float(od[6]-sd[6]),output_min=float(raw[i,j].min()),output_max=float(raw[i,j].max()),outside_range_fraction=float(np.mean((y<0)|(y>1))),clipping_mean_absolute=float(np.mean(np.abs(y-np.clip(y,0,1)))),nearest_target_pixel_rms=float(near[ni]),nearest_target_path=rr[ni]['path'],pixel_sha256=hashlib.sha256(u.tobytes()).hexdigest(),float_sha256=hashlib.sha256(raw[i,j].tobytes()).hexdigest())
            for k,name in enumerate(names):row['source_'+name]=float(sd[k]);row['output_'+name]=float(od[k]);row['delta_'+name]=float(od[k]-sd[k])
            rows.append(row)
        for arm,offset in [('reference',0),('latent',4)]:
            subset=rows[-9+offset:-9+offset+4] if offset==0 else rows[-5:-1]
            # Explicit indices avoid Python's negative-zero slicing edge cases.
            subset=rows[i*9+offset:i*9+offset+4]
            pairs=list(itertools.combinations(range(offset,offset+4),2));divraw=np.mean([np.sqrt(np.mean((yy[a]-yy[b])**2)) for a,b in pairs]);divclip=np.mean([np.sqrt(np.mean((np.clip(yy[a],0,1)-np.clip(yy[b],0,1))**2)) for a,b in pairs])
            sr=dict(source_eval_index=i,source_path=r['source_path'],category=r['category'],direction=r['source_domain']+'_'+r['target_domain'],arm=arm,diversity_rgb_rms=float(divraw),diversity_clipped_rgb_rms=float(divclip),appearance_score_std=float(np.std(oscores[offset:offset+4])),exact_identical_variant_pairs=sum(np.array_equal(uu[a],uu[b]) for a,b in pairs),unique_quantized_variants=len({x['pixel_sha256'] for x in subset}))
            for key in ['source_appearance_score','output_appearance_score','appearance_shift','target_score_mean','rgb_rms','clipped_rgb_rms','luminance_edge_change','nearest_target_pixel_rms','outside_range_fraction','clipping_mean_absolute']+['delta_'+n for n in names]:sr[key]=float(np.mean([x[key] for x in subset]))
            source_rows.append(sr)
    assert maxsourceerror<1e-9
    write(eda/'per_output.json',rows);write(eda/'source_aggregates.json',source_rows)
    for filename,data in [('per_output.csv',rows),('source_aggregates.csv',source_rows)]:
        with (eda/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    def stats(v):
        a=np.asarray(v);return dict(mean=float(a.mean()),median=float(np.median(a)),q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))
    aggregate={}
    for direction in ['D0_D1','D1_D0']:
        for arm in ['reference','latent']:
            rr=[x for x in source_rows if x['direction']==direction and x['arm']==arm];assert len(rr)==302
            keys=[k for k,v in rr[0].items() if isinstance(v,(float,int)) and k!='source_eval_index'];aggregate[direction+'_'+arm]={k:stats([x[k] for x in rr]) for k in keys};aggregate[direction+'_'+arm]['source_count']=302
        rr=[x for x in rows if x['source_domain']==direction[:2] and x['arm']=='self'];aggregate[direction[:2]+'_self']={k:stats([x[k] for x in rr]) for k in ['rgb_rms','clipped_rgb_rms','appearance_shift','outside_range_fraction']}
    write(eda/'aggregate.json',aggregate)
    # Largest raw RMS unique sources, then lowest latent diversity, with prescribed tie order.
    tails=[];used=set()
    for row in sorted([x for x in rows if x['arm']!='self'],key=lambda x:(-x['rgb_rms'],x['source_path'],x['arm'],x['variant'])):
        if row['source_eval_index'] in used:continue
        tails.append(dict(source_eval_index=row['source_eval_index'],reason='largest_raw_source_rms',metric=row['rgb_rms'],arm=row['arm'],variant=row['variant']));used.add(row['source_eval_index'])
        if len(tails)==4:break
    for row in sorted([x for x in source_rows if x['arm']=='latent'],key=lambda x:(x['diversity_clipped_rgb_rms'],x['source_path'])):
        if row['source_eval_index'] in used:continue
        tails.append(dict(source_eval_index=row['source_eval_index'],reason='lowest_clipped_latent_diversity',metric=row['diversity_clipped_rgb_rms']));used.add(row['source_eval_index'])
        if len(tails)==8:break
    for row in tails:row['representative_overlap']=records[row['source_eval_index']]['representative'];row['source_path']=records[row['source_eval_index']]['source_path']
    write(eda/'tail_selection.json',tails)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);sheets=[]
    def paste(im,array,x,y):im.paste(Image.fromarray(array).resize((96,96),Image.Resampling.NEAREST),(x,y))
    def sheet(indices,name,kind):
        im=Image.new('RGB',(1150,48+len(indices)*340),'#e8e8e8');draw=ImageDraw.Draw(im)
        for row,i in enumerate(indices):
            r=records[i];top=48+row*340;draw.text((4,top),f'{i}: {r["category"]} {r["source_domain"]} → {r["target_domain"]}',fill='black',font=font)
            paste(im,pix[r['source_index']],4,top+22);draw.text((4,top+122),'source',fill='black',font=font)
            paste(im,np.asarray(Image.open(out/'png'/f'{i:04d}_8.png')),108,top+22);draw.text((108,top+122),'self',fill='black',font=font)
            for j in range(4):
                x=230+j*108;paste(im,pix[r['reference_indices'][j]],x,top+22);draw.text((x,top+122),f'donor {j}',fill='black',font=font)
                paste(im,np.asarray(Image.open(out/'png'/f'{i:04d}_{j}.png')),x,top+144);draw.text((x,top+244),f'reference {j}',fill='black',font=font)
                x=688+j*108;paste(im,np.asarray(Image.open(out/'png'/f'{i:04d}_{j+4}.png')),x,top+144);draw.text((x,top+244),f'latent {j}',fill='black',font=font)
            draw.text((4,top+280),r['source_path'][:125],fill='black',font=font)
        p=eda/name;im.save(p);sheets.append(dict(path=str(p),kind=kind,source_eval_indices=indices))
    reps=[i for i,r in enumerate(records) if r['representative']];assert len(reps)==64
    for page,k in enumerate(range(0,64,4)):sheet(reps[k:k+4],f'representative_{page:02d}.png','representative')
    for page,k in enumerate(range(0,8,4)):sheet([x['source_eval_index'] for x in tails[k:k+4]],f'tails_{page:02d}.png','tail')
    trace=json.loads((out/'milestone_manifest.json').read_text());steps=sorted(map(int,trace['milestones']));arrays={s:np.load(out/f'milestone_{s}.npy') for s in steps}
    for page,start in enumerate(range(0,32,8)):
        im=Image.new('RGB',(1350,48+8*225),'#e8e8e8');draw=ImageDraw.Draw(im)
        for row,k in enumerate(range(start,start+8)):
            i=trace['source_eval_indices'][k];r=records[i];top=48+row*225;draw.text((4,top),f'{i} {r["category"]} {r["source_domain"]}',fill='black',font=font);paste(im,pix[r['source_index']],4,top+25)
            for col,step in enumerate(steps):
                x=140+col*230;draw.text((x,top),f'{step}: reference / latent',fill='black',font=font)
                for arm in range(2):paste(im,arrays[step][k,arm],x+arm*104,top+25)
        p=eda/f'milestones_{page:02d}.png';im.save(p);sheets.append(dict(path=str(p),kind='milestone',source_eval_indices=trace['source_eval_indices'][start:start+8]))
    write(eda/'sheet_manifest.json',sheets)
    counts=collections.Counter(r['pixel_sha256'] for r in rows)
    write(eda/'verification.json',dict(final_outputs=len(rows),source_arm_records=len(source_rows),representative_sources=64,tail_sources=8,milestone_sources=32,sheets=len(sheets),source_score_max_error=maxsourceerror,exact_duplicate_quantized_excess=sum(v-1 for v in counts.values()),all_finite=bool(np.isfinite(raw).all()),out_of_range_fraction=float(np.mean((raw<-1)|(raw>1))),raw_min=float(raw.min()),raw_max=float(raw.max()),cpu_seconds=time.monotonic()-cpu_started,aggregation='four variants averaged per source then302 sources per direction/arm',clipped_metric_definition='np.clip((raw_float64+1)/2,0,1), without uint8 quantization; descriptors use quantized PNG RGB'))
    html='<html><head><meta charset="utf-8"><title>TA-005 MUNIT</title></head><body><h1>MUNIT: fixed A1 diagnostics</h1><p>Unblinded qualitative review; no class-accuracy or augmentation-usefulness claim. Source, self-control, donors, reference variants and sampled variants are separate.</p>'
    for s in sheets:html+=f'<h2>{Path(s["path"]).stem}</h2><img style="max-width:100%" src="eda/{Path(s["path"]).name}">'
    (ROOT/'gallery.html').write_text(html+'</body></html>');print('EDA ready',len(sheets),'sheets')
if __name__=='__main__':main()
