"""Explain feature differences with metadata and inspect distinct representatives."""
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import spearmanr
from common import ROOT,RES,OUT,CACHE,inputs,save_json,save_csv,quantiles,sheet,verify


def main():
    verify();paths,na=inputs();bset=set(paths[na:]);groups=defaultdict(list)
    for p in paths:groups[Path(p).parent.name].append(p)
    splitrows=[]
    for c,pp in sorted(groups.items()):
        ordered=sorted(pp);expected=set(ordered[:20]);actual=bset&set(pp)
        splitrows.append({'category':c,'total':len(pp),'B_count':len(actual),
            'B_equals_first_20_lexicographic':actual==expected,
            'B_lexicographic_ranks':json.dumps([i for i,p in enumerate(ordered) if p in bset])})
    save_csv(OUT/'split_order_audit.csv',splitrows)
    with (ROOT/'eda/reports/inventory.csv').open() as f:inv={r['path']:r for r in csv.DictReader(f)}
    hashes=np.array([inv[p]['rgb64_sha256'] for p in paths])
    with (OUT/'image_properties.csv').open() as f:props=list(csv.DictReader(f))
    assert [r['path'] for r in props]==paths
    z=np.load(CACHE/'pca64.npz')['coordinates'];features=['white_fraction','brightness','saturation','edge_energy','laplacian_variance']
    correlations=[]
    for pc in range(10):
        for name in features:
            values=np.array([float(r[name]) for r in props])
            correlations.append({'PC':pc+1,'property':name,'A_Spearman':float(spearmanr(z[:na,pc],values[:na]).statistic),
                                 'B_Spearman':float(spearmanr(z[na:,pc],values[na:]).statistic)})
    save_csv(OUT/'pca_property_correlations.csv',correlations)
    white=np.array([float(r['white_fraction']) for r in props]);cats=np.array([Path(p).parent.name for p in paths])
    classrows=[]
    for c in sorted(groups):
        a=white[:na][cats[:na]==c];b=white[na:][cats[na:]==c]
        classrows.append({'category':c,'A_white_median':np.median(a),'B_white_median':np.median(b),'B_minus_A_median':np.median(b)-np.median(a)})
    save_csv(OUT/'white_fraction_by_category.csv',classrows)
    summary={'all_151_categories_B_equals_first_20_lexicographic':all(r['B_equals_first_20_lexicographic'] for r in splitrows),
        'categories':len(splitrows),'selection_code_reviewed':False,
        'inference':'Exact manifest property; filename-based selection is consistent with these manifests. Historical construction code was not examined.',
        'categories_with_higher_B_median_white_fraction':sum(r['B_minus_A_median']>0 for r in classrows)}
    for lo,hi,label in [(0,na,'A'),(na,len(paths),'B')]:
        subset=paths[lo:hi]
        summary[label]={'formats':dict(Counter(inv[p]['format'] for p in subset)),
            'filename_32hex_count':sum(bool(re.fullmatch('[0-9a-f]{32}',Path(p).stem)) for p in subset),
            'count':len(subset),'median_native_width':float(np.median([int(inv[p]['width']) for p in subset])),
            'median_native_height':float(np.median([int(inv[p]['height']) for p in subset]))}
    save_json(OUT/'metadata_shift.json',summary)
    design=json.loads(gzip.decompress((RES/'fid_sampling.json.gz').read_bytes()))
    baselines=[]
    for k in [16,32,64]:
        with np.load(CACHE/f'clusters_k{k}.npz') as f:labels=f['labels'];dist=f['distance']
        for rep in design['repeats']:
            for scheme in ['uniform','balanced']:
                u,v=rep[scheme]['U'],rep[scheme]['V'];pu=np.bincount(labels[u],minlength=k)/len(u);pv=np.bincount(labels[v],minlength=k)/len(v);pb=np.bincount(labels[na:],minlength=k)/(len(paths)-na)
                baselines.append({'k':k,'scheme':scheme,'seed':rep['seed'],'TV_U_B':abs(pu-pb).sum()/2,'TV_U_V':abs(pu-pv).sum()/2})
        if k!=32:continue
        with (OUT/'clusters_k32.csv').open() as f:rows=list(csv.DictReader(f))
        for rank,r in enumerate(sorted(rows,key=lambda r:abs(float(r['p_B_minus_p_A'])),reverse=True)[:8]):
            c=int(r['cluster']);rr=[]
            for lo,hi,label in [(0,na,'A'),(na,len(paths),'B')]:
                candidates=np.flatnonzero(labels[lo:hi]==c)+lo;ordered=candidates[np.argsort(dist[candidates])]
                seen=set();central=[];far=[]
                for i in ordered:
                    if hashes[i] not in seen:central.append(i);seen.add(hashes[i])
                    if len(central)==3:break
                for i in ordered[::-1]:
                    if hashes[i] not in seen:far.append(i);seen.add(hashes[i])
                    if len(far)==2:break
                if central:rr.append([(int(i),f'{label}; cluster {c}; distance {dist[i]:.2f}') for i in central+far])
            sheet(rr,f'cluster_{rank+1:02d}_k32_{c}.png',f'Cluster {c}: A={r["n_A"]}, B={r["n_B"]}; B-A={100*float(r["p_B_minus_p_A"]):+.2f} pp; 3 central + 2 distant, distinct RGB groups')
    save_csv(OUT/'cluster_sampling_baselines.csv',baselines)
    print(json.dumps(summary,indent=2));print('Metadata and representative-image enrichment complete')


if __name__=='__main__':main()
