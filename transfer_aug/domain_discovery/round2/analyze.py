import csv, hashlib, json, time, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'transfer_aug'))
from prepare_cohort import verify_integrity, sha, write_json
SEED=20261008
NAMES=['luminance_mean','luminance_std','saturation_mean','saturation_std','near_white','near_black','neighbor_gradient','laplacian_energy','luminance_entropy','high_frequency_fraction','border_difference']
def descriptors(u):
    x=u.astype(np.float64)/255; y=x@np.array([.2126,.7152,.0722]); mx=x.max(2); mn=x.min(2)
    s=np.divide(mx-mn,mx,out=np.zeros_like(mx),where=mx>0)
    lap=y[1:-1,:-2]+y[1:-1,2:]+y[:-2,1:-1]+y[2:,1:-1]-4*y[1:-1,1:-1]
    h=np.histogram(y,bins=32,range=(0,1))[0]/y.size; h=h[h>0]
    f=np.abs(np.fft.fft2(y))**2; f[0,0]=0
    freq=np.fft.fftfreq(64)*64; high=np.maximum(np.abs(freq[:,None]),np.abs(freq[None,:]))>=16
    border=np.ones(y.shape,bool);border[4:-4,4:-4]=False
    return [y.mean(),y.std(),s.mean(),s.std(),np.all(x>.95,2).mean(),np.all(x<.05,2).mean(),(.5*(np.abs(np.diff(y,axis=0)).mean()+np.abs(np.diff(y,axis=1)).mean())),np.mean(lap**2),-np.sum(h*np.log(h))/np.log(32),f[high].sum()/f.sum() if f.sum()>0 else 0,abs(y[border].mean()-y[~border].mean())]
def key(r,label):return hashlib.sha256(f'{SEED}|{label}|{r["category"]}|{r["domain"]}|{r["pixel_sha256"]}|{r["path"]}'.encode()).hexdigest()
def selection(rows):
    groups={}
    for r in rows:groups.setdefault(r['category'],[]).append(r)
    for cat,rs in groups.items():
        ordered=sorted(rs,key=lambda r:(r['score'],r['pixel_sha256'],r['path'])); k=len(rs)//3
        for i,r in enumerate(ordered):r['domain']='D0' if i<k else 'D1' if i>=len(rs)-k else 'middle';r['role']='unused'
        for d in ['D0','D1']:
            subset=sorted([r for r in rs if r['domain']==d],key=lambda r:(key(r,'diagnostic'),r['path']))
            for i,r in enumerate(subset):r['role']='holdout' if i<2 else 'optimize'
    return rows

def main():
    start=time.time(); cpu=time.process_time(); original=verify_integrity()
    a=(ROOT/'eda/resources/A.txt').read_text().splitlines();a1=(ROOT/'eda/resources/A1.txt').read_text().splitlines(); indexes={p:i for i,p in enumerate(a)}
    with (ROOT/'eda/reports/inventory.csv').open() as f: inventory={r['path']:r for r in csv.DictReader(f)}
    pixels=np.load(ROOT/'eda/features/pixels64.npy',mmap_mode='r'); allrows=[]; seen={}; excluded=[]
    for path in sorted(a1):
        inv=inventory[path];assert inv['partition']=='A1';assert sha(ROOT/'data/pokemon-generation-one-22k'/path)==inv['sha256']
        u=pixels[indexes[path]];ph=hashlib.sha256(u.tobytes()).hexdigest(); cat=inv['category']; vals=descriptors(u)
        r=dict(path=path,category=cat,category_id=int(inv['category_id']),pixel_sha256=ph,file_sha256=inv['sha256'],descriptors=vals)
        allrows.append(r)
        if (cat,ph) in seen:excluded.append(dict(path=path,retained_path=seen[cat,ph],category=cat,pixel_sha256=ph))
        else:seen[cat,ph]=path
    rows=[r.copy() for r in allrows if seen[r['category'],r['pixel_sha256']]==r['path']]; cats=sorted(set(r['category'] for r in rows));x=np.array([r['descriptors'] for r in rows]); residual=x.copy(); weights=np.zeros(len(rows));means={}
    for cat in cats:
        idx=np.array([i for i,r in enumerate(rows) if r['category']==cat]);mean=x[idx].mean(0);means[cat]=mean.tolist();residual[idx]-=mean;weights[idx]=1/len(idx)
    scale=np.sqrt(np.sum(weights[:,None]*residual**2,axis=0)/len(cats));keep=scale>1e-14;z=residual[:,keep]/scale[keep]
    _,sv,vt=np.linalg.svd(z*np.sqrt(weights[:,None]),full_matrices=False);load=vt[0];scores=z@load
    target=residual[:,8]/scale[8];corr=np.corrcoef(scores,target)[0,1];orient='luminance_entropy'
    if abs(corr)<1e-12:target=residual[:,6]/scale[6];corr=np.corrcoef(scores,target)[0,1];orient='neighbor_gradient'
    if corr<0:load=-load;scores=-scores;corr=-corr
    for r,s in zip(rows,scores):r['score']=float(s)
    rows=selection(rows); counts={}; allhash={}
    for cat in cats:
        rs=[r for r in rows if r['category']==cat]; counts[cat]={f'{d}_{role}':sum(r['domain']==d and r['role']==role for r in rs) for d in ['D0','D1'] for role in ['optimize','holdout']};counts[cat]['unique']=len(rs);counts[cat]['middle']=sum(r['domain']=='middle' for r in rs)
    for r in rows:allhash.setdefault(r['pixel_sha256'],[]).append(r)
    cross={h:[{k:r[k] for k in ['path','category','domain','role']} for r in rs] for h,rs in allhash.items() if len(set(r['category'] for r in rs))>1}
    leakage={h:rs for h,rs in cross.items() if {'holdout','optimize'}<=set(r['role'] for r in rs)}
    write_json(OUT/'cross_class_duplicates.json',dict(groups=cross,holdout_leakage=leakage))
    write_json(OUT/'duplicate_exclusions.json',excluded)
    write_json(OUT/'counts.json',dict(a1=len(a1),unique=len(rows),excluded=len(excluded),categories=len(cats),totals={role:sum(r['role']==role for r in rows) for role in ['holdout','optimize','unused']},per_class=counts))
    write_json(OUT/'pca.json',dict(descriptors=NAMES,class_means=means,residual_rms=scale.tolist(),kept=keep.tolist(),pc1_loadings=dict(zip(np.array(NAMES)[keep].tolist(),load.tolist())),explained_variance_ratio=(sv**2/(sv**2).sum()).tolist(),orientation=orient,orientation_correlation=float(corr)))
    manifest=dict(schema=1,seed=SEED,partition='A1',contract_sha256=sha(OUT/'CONTRACT_v1.md'),rows=[{k:v for k,v in r.items() if k!='descriptors'} for r in rows])
    write_json(OUT/'domain_manifest.json',manifest)
    with (OUT/'scores.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['path','category','pixel_sha256','dedup_retained',*NAMES,'score','domain','role']); selected={r['path']:r for r in rows}
        for r in allrows:
            q=selected.get(r['path']); w.writerow([r['path'],r['category'],r['pixel_sha256'],q is not None,*r['descriptors'],q['score'] if q else '',q['domain'] if q else '',q['role'] if q else 'duplicate_excluded'])
    reviewcats=sorted(np.random.default_rng(SEED).choice(cats,16,replace=False).tolist());review=[]
    for page in range(4):
        im=Image.new('RGB',(880,4*156+42),'#dedede');draw=ImageDraw.Draw(im);draw.text((8,8),'A1 PC1 appearance domains: D0 opt x2 | D0 holdout x2 | D1 opt x2 | D1 holdout x2',fill='black')
        for row,cat in enumerate(reviewcats[page*4:(page+1)*4]):
            y=42+row*156;draw.text((8,y),cat,fill='black');col=0
            for d in ['D0','D1']:
                for role in ['optimize','holdout']:
                    sample=sorted([r for r in rows if r['category']==cat and r['domain']==d and r['role']==role],key=lambda r:key(r,'review'))[:2]
                    for r in sample:
                        xx=8+col*108;im.paste(Image.fromarray(pixels[indexes[r['path']]]).resize((96,96),Image.Resampling.NEAREST),(xx,y+18));draw.text((xx,y+118),f'{d} {role[:3]} {r["score"]:.2f}',fill='black');review.append(dict(page=page+1,column=col+1,**{k:r[k] for k in ['path','category','domain','role','score']}));col+=1
        im.save(OUT/f'review_{page+1}.png')
    write_json(OUT/'review_selection.json',dict(categories=reviewcats,rows=review))
    regenerated=selection([{k:v for k,v in r.items() if k not in ['domain','role']} for r in rows]); exact=regenerated==rows
    perclass_disjoint=all(len([r['pixel_sha256'] for r in rows if r['category']==c])==len(set(r['pixel_sha256'] for r in rows if r['category']==c)) for c in cats)
    minopt=min(v[k] for v in counts.values() for k in ['D0_optimize','D1_optimize']); after=verify_integrity()
    checks=dict(finite_descriptors=bool(np.isfinite(x).all()),all_a1=all(r['path'] in set(a1) for r in rows),exact_selection_regeneration=exact,per_class_pixel_disjoint=perclass_disjoint,global_train_holdout_disjoint=not leakage,cross_class_duplicate_groups=len(cross),cross_class_leakage_groups=len(leakage),minimum_optimize_per_class_domain=minopt,class_count=len(cats),source_integrity_before_after=original==after,all_a1_source_files_sha256_verified=True,protected_inputs=original)
    write_json(OUT/'verification.json',checks);write_json(OUT/'resource_usage.json',dict(wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,gpu_hours=0,models_loaded=False,threads=2))
    print(json.dumps({k:v for k,v in checks.items() if k!='protected_inputs'},indent=2))
if __name__=='__main__':main()
