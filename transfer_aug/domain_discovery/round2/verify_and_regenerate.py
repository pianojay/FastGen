import json, subprocess, time, resource, sys
from pathlib import Path
import numpy as np
from analyze import ROOT, OUT, descriptors, sha, write_json
start=time.time(); names=['domain_manifest.json','scores.csv','pca.json','counts.json','duplicate_exclusions.json','cross_class_duplicates.json','review_selection.json']+[f'review_{i}.png' for i in range(1,5)]
before={n:sha(OUT/n) for n in names}; first=json.loads((OUT/'resource_usage.json').read_text())
black=descriptors(np.zeros((64,64,3),np.uint8));white=descriptors(np.full((64,64,3),255,np.uint8))
assert black[5]==1 and black[4]==0 and all(black[i]==0 for i in [0,1,2,3,6,7,8,9,10])
assert white[4]==1 and white[5]==0 and np.isclose(white[0],1) and all(abs(white[i])<1e-24 for i in [1,2,3,6,7,8,9,10])
checker=np.repeat((((np.indices((64,64)).sum(0)%2)*255).astype(np.uint8))[...,None],3,2);v=descriptors(checker)
assert np.isclose(v[6],1) and np.isclose(v[7],16) and np.isclose(v[8],.2) and np.isclose(v[9],1)
subprocess.run(['bash',str(ROOT/'transfer_aug/run_python.sh'),str(OUT/'analyze.py')],check=True)
after={n:sha(OUT/n) for n in names};assert before==after
with (ROOT/'eda/reports/inventory.csv').open() as f:
    import csv
    inv={r['path']:r for r in csv.DictReader(f)}
a1=(ROOT/'eda/resources/A1.txt').read_text().splitlines()
assert all(sha(ROOT/'data/pokemon-generation-one-22k'/p)==inv[p]['sha256'] for p in a1)
m=json.loads((OUT/'domain_manifest.json').read_text());rows=m['rows'];p=json.loads((OUT/'pca.json').read_text());counts=json.loads((OUT/'counts.json').read_text())
assert len(rows)==len({r['path'] for r in rows}) and len(rows)==counts['unique']
assert all(c['D0_optimize']==c['D1_optimize'] and c['D0_holdout']==c['D1_holdout']==2 for c in counts['per_class'].values())
opt={r['pixel_sha256'] for r in rows if r['role']=='optimize'};hold={r['pixel_sha256'] for r in rows if r['role']=='holdout'};assert not opt&hold
write_json(OUT/'regeneration.json',dict(exact_full_regeneration=True,artifact_sha256=after,a1_source_hashes_rechecked_after_regeneration=len(a1),numerical_checks={'constant_black':True,'constant_white':True,'checkerboard_gradient_laplacian_entropy_fft':True},balanced_domain_counts=True,global_train_holdout_pixel_overlap=0))
second=json.loads((OUT/'resource_usage.json').read_text());write_json(OUT/'resource_usage.json',dict(first_execution=first,second_execution=second,total_analysis_cpu_seconds=first['cpu_seconds']+second['cpu_seconds'],verification_process_cpu_seconds=time.process_time(),regeneration_and_verification_wall_seconds=time.time()-start,gpu_hours=0,models_loaded=False,threads=2,budget_cpu_hours=1))
print(json.dumps(dict(exact_regeneration=True,counts={k:v for k,v in counts.items() if k!='per_class'},pc1_variance=p['explained_variance_ratio'][0])))
