"""Exact candidate-refined nearest-neighbor descriptions of real-image sets."""
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch

from common import ROOT, RES, OUT, CACHE, SEED, verify, gpu_lease, load_features, save_json, save_csv, quantiles, sheet


def search(query,gallery,query_ids=None,gallery_ids=None,query_categories=None,gallery_categories=None,cosine=False):
    """Chunked full gallery search; float64 reranking of 64 candidates per query."""
    q=np.asarray(query,dtype=np.float64);g=np.asarray(gallery,dtype=np.float64)
    if cosine:
        q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
    gf=torch.as_tensor(g,dtype=torch.float32,device='cuda');gd=torch.as_tensor(g,dtype=torch.float64,device='cuda')
    norms=(gf*gf).sum(1)[None,:]; all_indices=[];all_dist=[];same_indices=[];same_dist=[]
    for start in range(0,len(q),128):
        qd=torch.as_tensor(q[start:start+128],device='cuda');qf=qd.float()
        sq=((qf*qf).sum(1)[:,None]+norms-2*(qf@gf.T)).clamp_min_(0)
        if query_ids is not None:
            banned=torch.as_tensor(query_ids[start:start+len(qd),None]==gallery_ids[None,:],device='cuda')
            sq.masked_fill_(banned,float('inf'))
        candidates=sq.topk(min(64,len(g)),largest=False).indices
        dist=((qd[:,None,:]-gd[candidates])**2).sum(2)
        dist,order=dist.sort(1);candidates=candidates.gather(1,order)
        if cosine:dist=dist/2
        else:dist=dist.sqrt()
        all_indices.append(candidates.cpu().numpy());all_dist.append(dist.cpu().numpy())
        if query_categories is not None:
            mask=torch.as_tensor(query_categories[start:start+len(qd),None]!=gallery_categories[None,:],device='cuda')
            sq.masked_fill_(mask,float('inf'))
            candidates=sq.topk(5,largest=False).indices
            dist=((qd[:,None,:]-gd[candidates])**2).sum(2).sqrt()
            dist,order=dist.sort(1);candidates=candidates.gather(1,order)
            same_indices.append(candidates.cpu().numpy());same_dist.append(dist.cpu().numpy())
        if start%2048==0:print(f'Neighbor search {start}/{len(q)} cosine={cosine}',flush=True)
    return (np.concatenate(all_indices),np.concatenate(all_dist),
            np.concatenate(same_indices) if same_indices else None,np.concatenate(same_dist) if same_dist else None)


def first_allowed(indices,distances,allowed):
    mask=allowed[indices];assert mask.any(1).all(),'Increase stored neighbor depth for this gallery'
    rank=mask.argmax(1);return distances[np.arange(len(indices)),rank]


def main():
    verify();x,paths,na=load_features();start=time.monotonic()
    categories=np.array([Path(p).parent.name for p in paths]);_,cats=np.unique(categories,return_inverse=True)
    with (ROOT/'eda/reports/inventory.csv').open() as f:inventory={r['path']:r for r in csv.DictReader(f)}
    hashes=np.array([inventory[p]['rgb64_sha256'] for p in paths])
    target=CACHE/'neighbors.npz'
    if not target.exists():
        with gpu_lease():
            ids,dist,same_ids,same_dist=search(x,x[:na],np.arange(len(x)),np.arange(na),cats,cats[:na])
            cos_ids,cos_dist,_,_=search(x[na:],x[:na],cosine=True)
            reverse_ids,reverse_dist,_,_=search(x[:na],x[na:])
            np.savez(target,indices=ids,distances=dist,same_indices=same_ids,same_distances=same_dist,
                     cosine_indices=cos_ids,cosine_distances=cos_dist,reverse_indices=reverse_ids,reverse_distances=reverse_dist)
            print(f'GPU neighbor searches completed in {time.monotonic()-start:.1f}s',flush=True)
    with np.load(target) as archive:
        data={name:archive[name] for name in archive.files}
    ids=data['indices'];dist=data['distances'];same_ids=data['same_indices'];same_dist=data['same_distances']
    # Independent full-gallery numerical check, including exact duplicates and ordinary queries.
    rng=np.random.default_rng(SEED);probe=rng.choice(len(x)-na,32,replace=False)+na
    truth=cdist(np.asarray(x[probe],dtype=np.float64),np.asarray(x[:na],dtype=np.float64))
    assert np.allclose(np.sort(truth,axis=1)[:,:5],dist[probe,:5],atol=1e-8,rtol=1e-8)
    assert np.all(ids[:na]!=np.arange(na)[:,None]),'A self-match not excluded'
    bdist=dist[na:,0];same=same_dist[na:,0];exact=np.isin(hashes[na:],hashes[:na])
    unique=hashes[ids[:na]]!=hashes[:na,None];assert unique.any(1).all()
    loo_unique=dist[:na][np.arange(na),unique.argmax(1)]
    rows=[]
    for j,i in enumerate(range(na,len(x))):
        rows.append({'path':paths[i],'category':categories[i],'nearest_A_path':paths[ids[i,0]],
            'distance':bdist[j],'same_category_distance':same[j],
            'exact_RGB64_match_in_A':bool(exact[j]),
            'nearest_5_A_paths':json.dumps([paths[a] for a in ids[i,:5]]),
            'nearest_5_distances':json.dumps(dist[i,:5].tolist()),
            'same_category_5_paths':json.dumps([paths[a] for a in same_ids[i,:5]]),
            'cosine_distance':data['cosine_distances'][j,0],
            'cosine_nearest_A':paths[data['cosine_indices'][j,0]]})
    save_csv(OUT/'B_nearest_neighbors.csv',rows)
    save_csv(OUT/'A_reverse_neighbors.csv',[{'path':paths[i],'nearest_B_path':paths[na+data['reverse_indices'][i,0]],'distance':data['reverse_distances'][i,0]} for i in range(na)])
    save_csv(OUT/'A_leave_one_out.csv',[{'path':paths[i],'distance':dist[i,0],'distance_excluding_own_pixel_group':loo_unique[i]} for i in range(na)])
    gallery_rows=[];gallery_sampling=[]
    for seed in range(SEED,SEED+20):
        query=np.random.default_rng(seed).choice(na,len(x)-na,replace=False)
        allowed=np.ones(na,dtype=bool);allowed[query]=False
        ad=first_allowed(ids[query],dist[query],allowed);bd=first_allowed(ids[na:],dist[na:],allowed)
        gallery_rows.append({'seed':seed,'gallery_n':int(allowed.sum()),'A_query_n':len(query),'B_query_n':len(bd),
            'A_query_median':np.median(ad),'B_query_median':np.median(bd),'median_difference':np.median(bd)-np.median(ad),
            'A_query_p95':np.quantile(ad,.95),'B_query_p95':np.quantile(bd,.95)})
        gallery_sampling.append({'seed':seed,'heldout_A_indices':query.tolist()})
    save_json(OUT/'neighbor_gallery_sampling.json',gallery_sampling);save_csv(OUT/'neighbor_gallery_comparisons.csv',gallery_rows)
    partition=json.loads((RES/'development_split.json').read_text());allowed=np.zeros(na,dtype=bool);allowed[partition['optimization_indices']]=True
    fixed_a=first_allowed(ids[partition['development_indices']],dist[partition['development_indices']],allowed)
    fixed_b=first_allowed(ids[na:],dist[na:],allowed)
    cutoff=float(np.quantile(bdist,.95));catrows=[]
    for c in sorted(set(categories)):
        mask=categories[na:]==c
        catrows.append({'category':c,'n_B':int(mask.sum()),'median_distance':np.median(bdist[mask]),
            'same_category_median':np.median(same[mask]),'count_above_global_B_p95':int((bdist[mask]>cutoff).sum()),
            'exact_A_matches':int(exact[mask].sum())})
    save_csv(OUT/'neighbor_category_summary.csv',catrows)
    summary={'B_to_A':quantiles(bdist),'B_to_A_excluding_exact_matched_queries':quantiles(bdist[~exact]),
        'B_to_A_same_category':quantiles(same),'B_to_A_cosine':quantiles(data['cosine_distances'][:,0]),
        'A_to_B':quantiles(data['reverse_distances'][:,0]),'A_leave_one_out':quantiles(dist[:na,0]),
        'A_leave_one_out_excluding_own_duplicate_group':quantiles(loo_unique),
        'B_exact_matched_count':int(exact.sum()),'same_category_first_neighbor_rate':float(np.mean(categories[ids[na:,0]]==categories[na:])),
        'gallery_control_median_difference':quantiles([r['median_difference'] for r in gallery_rows]),
        'fixed_A2_to_A1':quantiles(fixed_a),'fixed_B_to_A1':quantiles(fixed_b),
        'numerical_check_queries':32,'numerical_check_atol':1e-8,
        'tail_B_paths':[paths[na+j] for j in np.argsort(bdist)[-24:][::-1]],
        'seconds':time.monotonic()-start}
    save_json(OUT/'neighbor_summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for values,label in [(bdist,'B → all A'),(dist[:na,0],'A leave-one-out'),(loo_unique,'A excluding own duplicate group')]:
        axes[0].plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=label)
    axes[0].set(xlabel='Euclidean feature distance',ylabel='Empirical CDF');axes[0].legend()
    bins=np.linspace(0,max(bdist.max(),same.max()),50)
    axes[1].hist(bdist,bins=bins,density=True,alpha=.5,label='Any category');axes[1].hist(same,bins=bins,density=True,alpha=.5,label='Same category');axes[1].set(xlabel='B → A distance',ylabel='Density');axes[1].legend()
    fig.savefig(OUT/'neighbor_distributions.png',dpi=160);plt.close(fig)
    order=np.argsort(bdist)[::-1]
    for page in range(3):
        rr=[]
        for j in order[page*8:(page+1)*8]:
            i=na+j;rr.append([(int(i),f'B tail rank {np.where(order==j)[0][0]+1}; d={bdist[j]:.3f}; dup={exact[j]}')]+
                [(int(k),f'A neighbor {r+1}; d={dist[i,r]:.3f}') for r,k in enumerate(ids[i,:3])])
        sheet(rr,f'neighbor_tail_{page+1}.png',f'B samples farthest from A; tail page {page+1}/3')
    middle=np.argsort(np.abs(bdist-np.median(bdist)))[:8]
    sheet([[(na+int(j),f'B central-distance case; d={bdist[j]:.3f}')]+[(int(k),f'A neighbor {r+1}; d={dist[na+j,r]:.3f}') for r,k in enumerate(ids[na+j,:3])] for j in middle],
          'neighbor_central.png','B queries near the median nearest-neighbor distance')
    verify();print('Neighbor analysis complete',flush=True)


if __name__=='__main__':main()
