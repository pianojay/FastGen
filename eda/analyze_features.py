"""CPU dataset comparisons: FID baselines, PCA, clustering, image properties."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from common import ROOT, RES, OUT, CACHE, SEED, verify, load_features, save_json, save_csv, sheet, quantiles, sha


def statistics(x):
    x=np.asarray(x,dtype=np.float64)
    return x.mean(0),np.cov(x,rowvar=False)


def frechet(s1,s2):
    """FID via symmetric PSD eigendecomposition, float64, full 2048D features."""
    m1,c1=s1; m2,c2=s2
    e,v=eigh(c1,check_finite=False)
    assert e.min()>-1e-7
    root=(v*np.sqrt(np.maximum(e,0)))@v.T
    middle=root@c2@root
    e2=eigh((middle+middle.T)*.5,eigvals_only=True,check_finite=False)
    assert e2.min()>-1e-7
    score=float(np.sum((m1-m2)**2)+np.trace(c1)+np.trace(c2)-2*np.sqrt(np.maximum(e2,0)).sum())
    assert score>-1e-6
    return max(score,0.)


def fid_analysis(x,na,paths,hashes):
    start=time.monotonic(); a=x[:na]; b=x[na:]
    design=json.loads(gzip.decompress((RES/"fid_sampling.json.gz").read_bytes()))
    assert design['A_manifest_sha256']==sha(RES/'A.txt') and design['B_manifest_sha256']==sha(RES/'B.txt')
    sb=statistics(b); sa=statistics(a)
    full=frechet(sa,sb)
    # Independent validation against the actual clean-fid implementation on the full sets.
    from cleanfid.fid import frechet_distance
    official=float(frechet_distance(sa[0],sa[1],sb[0],sb[1]))
    assert np.isclose(full,official,rtol=1e-7,atol=1e-6),(full,official)
    self_fid=frechet(sb,sb); assert self_fid<1e-5,self_fid
    save_json(OUT/"fid_full.json",{"FID_A_B":full,"cleanfid_reference":official,
        "absolute_implementation_difference":abs(full-official),"self_FID_B_B":self_fid,
        "n_A":na,"n_B":len(b),"dimensions":2048})
    print(f"Full FID(A,B)={full:.6f}; clean-fid verification passed",flush=True)
    rows=[]
    for rep in design['repeats']:
        for scheme in ['uniform','balanced']:
            u=np.array(rep[scheme]['U']);v=np.array(rep[scheme]['V'])
            su=statistics(a[u]);sv=statistics(a[v])
            ab=frechet(su,sb);aa=frechet(su,sv)
            hu=set(hashes[u]); hv=set(hashes[v]);hb=set(hashes[na:])
            rows.append({"scheme":scheme,"seed":rep['seed'],"n":len(u),"FID_U_B":ab,
                "FID_U_V":aa,"paired_difference":ab-aa,"U_V_shared_pixel_groups":len(hu&hv),
                "U_B_shared_pixel_groups":len(hu&hb)})
            save_csv(OUT/"fid_repeats.csv",rows)
            print(f"FID {scheme} seed={rep['seed']} U/B={ab:.3f} U/V={aa:.3f} elapsed={time.monotonic()-start:.0f}s",flush=True)
    # One representative per 64px RGB group; remove B groups present anywhere in A.
    _,ai=np.unique(hashes[:na],return_index=True)
    _,bi=np.unique(hashes[na:],return_index=True)
    bi=bi[~np.isin(hashes[na:][bi],hashes[:na])]
    n=len(bi); assert len(ai)>=2*n
    db=statistics(b[bi]);dedup=[];sampling=[]
    for seed in range(SEED,SEED+20):
        chosen=np.random.default_rng(seed).choice(ai,2*n,replace=False);u,v=chosen[:n],chosen[n:]
        su,sv=statistics(a[u]),statistics(a[v]);ab=frechet(su,db);aa=frechet(su,sv)
        dedup.append({"scheme":"unique_pixels_B_excludes_A_matches","seed":seed,"n":n,
                      "FID_U_B":ab,"FID_U_V":aa,"paired_difference":ab-aa})
        sampling.append({"seed":seed,"U":u.tolist(),"V":v.tolist()})
        save_csv(OUT/"fid_duplicate_sensitivity.csv",dedup)
        print(f"FID duplicate sensitivity {seed} U/B={ab:.3f} U/V={aa:.3f}",flush=True)
    (OUT/"fid_duplicate_sampling.json.gz").write_bytes(gzip.compress(json.dumps({"B_indices":bi.tolist(),"repeats":sampling}).encode(),mtime=0))
    summary={"full":full,"counts":{"A":na,"B":len(b)},"schemes":{}}
    for scheme,rr in [('uniform',[r for r in rows if r['scheme']=='uniform']),
                      ('balanced',[r for r in rows if r['scheme']=='balanced']),('unique_pixels',dedup)]:
        summary['schemes'][scheme]={key:quantiles([r[key] for r in rr]) for key in ['FID_U_B','FID_U_V','paired_difference']}
        summary['schemes'][scheme]['n']=rr[0]['n']
    partition=json.loads((RES/"development_split.json").read_text())
    summary['FID_A1_A2']=frechet(statistics(a[partition['optimization_indices']]),statistics(a[partition['development_indices']]))
    summary['duplicate_sensitivity_note']='Both samples unique by 64px RGB; B excludes any A match. Changed population and n; compare within this scheme only.'
    summary['seconds']=time.monotonic()-start
    save_json(OUT/"fid_summary.json",summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(scheme,rr) in zip(axes,[('Observed category mix',[r for r in rows if r['scheme']=='uniform']),('Balanced categories',[r for r in rows if r['scheme']=='balanced']),('Unique pixels; B excludes overlaps',dedup)]):
        ax.boxplot([[r['FID_U_B'] for r in rr],[r['FID_U_V'] for r in rr]],tick_labels=['A subset vs B','Within A'])
        for r in rr:ax.plot([1,2],[r['FID_U_B'],r['FID_U_V']],alpha=.18,color='gray')
        ax.set(title=f'{scheme}\nn={rr[0]["n"]} per set, 20 repeats',ylabel='FID')
    fig.savefig(OUT/"fid_baselines.png",dpi=160);plt.close(fig)


def pca_cluster_analysis(x,na,paths,categories):
    print('Fitting A-only PCA',flush=True)
    # Exact covariance eigensolver; covariance has only 2048 x 2048 entries.
    pca=PCA(n_components=64,svd_solver='covariance_eigh',whiten=False)
    pca.fit(np.asarray(x[:na],dtype=np.float64));z=pca.transform(x)
    np.savez(CACHE/'pca64.npz',coordinates=z,components=pca.components_,mean=pca.mean_,variance=pca.explained_variance_,ratio=pca.explained_variance_ratio_)
    save_json(OUT/'pca_summary.json',{'fit_population':'A','whiten':False,'retained_variance_64':float(pca.explained_variance_ratio_.sum()),'explained_variance_ratio':pca.explained_variance_ratio_})
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.plot(np.arange(1,65),np.cumsum(pca.explained_variance_ratio_));ax.set(xlabel='Number of PCs',ylabel='Cumulative A variance explained',ylim=(0,1));fig.savefig(OUT/'pca_variance.png',dpi=160);plt.close(fig)
    rng=np.random.default_rng(SEED);display=rng.choice(na,len(x)-na,replace=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,(i,j) in zip(axes,[(0,1),(2,3)]):
        ax.scatter(z[display,i],z[display,j],s=5,alpha=.22,label='A: equal display sample',rasterized=True)
        ax.scatter(z[na:,i],z[na:,j],s=5,alpha=.22,label='B: all images',rasterized=True)
        ax.set(xlabel=f'PC{i+1}',ylabel=f'PC{j+1}');ax.legend(markerscale=3)
    fig.savefig(OUT/'pca_scatter.png',dpi=170);plt.close(fig)
    rows=[];fig,axes=plt.subplots(5,2,figsize=(12,16),layout='constrained')
    fig2,axes2=plt.subplots(5,2,figsize=(12,16),layout='constrained')
    for pc,(ax,ax2) in enumerate(zip(axes.flat,axes2.flat)):
        a,b=z[:na,pc],z[na:,pc];bins=np.histogram_bin_edges(z[:,pc],bins=50)
        ax.hist(a,bins=bins,density=True,alpha=.5,label='A');ax.hist(b,bins=bins,density=True,alpha=.5,label='B');ax.set(title=f'PC{pc+1}',ylabel='Density');ax.legend()
        for arr,label in [(a,'A'),(b,'B')]:ax2.plot(np.sort(arr),np.arange(1,len(arr)+1)/len(arr),label=label)
        ax2.set(title=f'PC{pc+1}',ylabel='Empirical CDF');ax2.legend()
        rows.append({'PC':pc+1,'A_variance_fraction':pca.explained_variance_ratio_[pc],
                     'B_minus_A_mean_in_A_SD':(b.mean()-a.mean())/a.std(ddof=1),
                     'Wasserstein_in_A_SD':wasserstein_distance(a,b)/a.std(ddof=1),
                     'KS_distance':ks_2samp(a,b).statistic})
    fig.savefig(OUT/'pca_distributions.png',dpi=140);plt.close(fig);fig2.savefig(OUT/'pca_ecdfs.png',dpi=140);plt.close(fig2)
    save_csv(OUT/'pca_effects.csv',rows)
    endpoints=[]
    for pc in range(4):
        order=np.argsort(z[:,pc]);endpoints.append([(int(i),f'PC{pc+1} {"low" if rank<3 else "high"}; {"A" if i<na else "B"}; {z[i,pc]:.2f}') for rank,i in enumerate(np.r_[order[:3],order[-3:]])])
    sheet(endpoints,'pca_endpoints.png','Extremes of the first four A-fitted principal components')
    classes=sorted(set(categories));counts=Counter(categories[:na])
    weights_a=np.array([1/(len(classes)*counts[c]) for c in categories[:na]])
    counts_b=Counter(categories[na:]);weights_b=np.array([1/(len(classes)*counts_b[c]) for c in categories[na:]])
    summaries={}
    for k in [16,32,64]:
        print(f'Clustering k={k}',flush=True)
        km=KMeans(n_clusters=k,n_init=10,random_state=SEED,algorithm='lloyd').fit(z[:na])
        labels=km.predict(z);dist=np.linalg.norm(z-km.cluster_centers_[labels],axis=1)
        cluster_rows=[]
        for c in range(k):
            ma=labels[:na]==c;mb=labels[na:]==c;pa=float(ma.mean());pb=float(mb.mean())
            ca=Counter(categories[:na][ma]);cb=Counter(categories[na:][mb])
            cluster_rows.append({'cluster':c,'n_A':int(ma.sum()),'n_B':int(mb.sum()),'p_A':pa,'p_B':pb,'p_B_minus_p_A':pb-pa,
                'balanced_p_A':weights_a[ma].sum(),'balanced_p_B':weights_b[mb].sum(),
                'balanced_difference':weights_b[mb].sum()-weights_a[ma].sum(),
                'top_A_categories':json.dumps(ca.most_common(5)),'top_B_categories':json.dumps(cb.most_common(5))})
        save_csv(OUT/f'clusters_k{k}.csv',cluster_rows)
        np.savez(CACHE/f'clusters_k{k}.npz',labels=labels,centers=km.cluster_centers_,distance=dist)
        summaries[str(k)]={'total_variation':sum(abs(r['p_B_minus_p_A']) for r in cluster_rows)/2,
            'category_balanced_total_variation':sum(abs(r['balanced_difference']) for r in cluster_rows)/2,
            'A_distance_quantiles':quantiles(dist[:na]),'B_distance_quantiles':quantiles(dist[na:])}
        if k!=32:continue
        save_csv(OUT/'cluster_assignments.csv',[{'path':p,'split':'A' if i<na else 'B','cluster':int(labels[i]),'distance_to_center':float(dist[i])} for i,p in enumerate(paths)])
        ordered=sorted(cluster_rows,key=lambda r:r['p_B_minus_p_A'])
        fig,ax=plt.subplots(figsize=(9,9),layout='constrained');ax.barh([str(r['cluster']) for r in ordered],[100*r['p_B_minus_p_A'] for r in ordered]);ax.axvline(0,color='black',lw=.6);ax.set(xlabel='B share − A share (percentage points)',ylabel='Cluster',title='A-fitted clusters, k=32');fig.savefig(OUT/'cluster_composition.png',dpi=160);plt.close(fig)
        for rank,r in enumerate(sorted(cluster_rows,key=lambda r:abs(r['p_B_minus_p_A']),reverse=True)[:8]):
            c=r['cluster'];representatives=[]
            for lo,hi,label in [(0,na,'A'),(na,len(x),'B')]:
                candidates=np.flatnonzero(labels[lo:hi]==c)+lo
                if len(candidates):
                    order=candidates[np.argsort(dist[candidates])];chosen=np.r_[order[:3],order[-2:]]
                    representatives.append([(int(i),f'{label}; cluster {c}; distance {dist[i]:.2f}') for i in chosen])
            sheet(representatives,f'cluster_{rank+1:02d}_k32_{c}.png',f'Cluster {c}: A={r["n_A"]}, B={r["n_B"]}; B-A={100*r["p_B_minus_p_A"]:+.2f} pp; 3 central + 2 distant examples')
    save_json(OUT/'cluster_summary.json',summaries)
    # Equal-category PCA shifts make the composition control visible without tiny-sample FIDs.
    conditional=[]
    for c in classes:
        a=z[:na][categories[:na]==c,:10];b=z[na:][categories[na:]==c,:10]
        conditional.append({'category':c,**{f'PC{i+1}_mean_shift_global_A_SD':float((b[:,i].mean()-a[:,i].mean())/np.sqrt(pca.explained_variance_[i])) for i in range(10)}})
    save_csv(OUT/'pca_within_category.csv',conditional)


def image_properties(na,paths):
    pix=np.load(CACHE/'pixels64.npy',mmap_mode='r');rows=[]
    with (ROOT/'eda/reports/inventory.csv').open() as f:inventory={r['path']:r for r in csv.DictReader(f)}
    for i,p in enumerate(paths):
        v=np.asarray(pix[i],dtype=np.float32)/255.;gray=v@np.array([.299,.587,.114]);mx=v.max(2);mn=v.min(2)
        lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
        rows.append({'path':p,'split':'A' if i<na else 'B','width':int(inventory[p]['width']),'height':int(inventory[p]['height']),
            'aspect_ratio':int(inventory[p]['width'])/int(inventory[p]['height']),
            'white_fraction':float((v.min(2)>.95).mean()),'brightness':float(gray.mean()),
            'saturation':float(np.mean((mx-mn)/np.maximum(mx,1e-8))),
            'edge_energy':float((np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2),
            'laplacian_variance':float(lap.var())})
    save_csv(OUT/'image_properties.csv',rows)
    names=['aspect_ratio','white_fraction','brightness','saturation','edge_energy','laplacian_variance']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained');summary={}
    for name,ax in zip(names,axes.flat):
        a=np.array([r[name] for r in rows[:na]]);b=np.array([r[name] for r in rows[na:]])
        for arr,label in [(a,'A'),(b,'B')]:ax.plot(np.sort(arr),np.arange(1,len(arr)+1)/len(arr),label=label)
        ax.set(title=name,ylabel='Empirical CDF');ax.legend()
        summary[name]={'A':quantiles(a),'B':quantiles(b),'KS_distance':float(ks_2samp(a,b).statistic)}
    fig.savefig(OUT/'image_properties.png',dpi=160);plt.close(fig);save_json(OUT/'image_properties_summary.json',summary)


def main():
    verify();OUT.mkdir(parents=True,exist_ok=True);x,paths,na=load_features()
    with (ROOT/'eda/reports/inventory.csv').open() as f:inventory={r['path']:r for r in csv.DictReader(f)}
    hashes=np.array([inventory[p]['rgb64_sha256'] for p in paths]);categories=np.array([Path(p).parent.name for p in paths])
    if not (OUT/'pca_cluster_complete.json').exists():
        pca_cluster_analysis(x,na,paths,categories);image_properties(na,paths)
        save_json(OUT/'pca_cluster_complete.json',{'complete':True,'feature_sha256':sha(CACHE/'inception.npy'),'source_sha256':sha(__file__)})
    if not (OUT/'fid_summary.json').exists():fid_analysis(x,na,paths,hashes)
    verify(); print('CPU feature analyses complete',flush=True)


if __name__=='__main__':main()
