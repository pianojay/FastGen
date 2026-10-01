"""Derive and freeze campaign thresholds from diagnostic controls, before training."""
import json,math
import numpy as np
from pathlib import Path
from afm_fastgen.common import ROOT,atomic_json,digest
p=ROOT/'artifacts/stability/calibration'
rows=json.loads((p/'controls/controls.json').read_text())
real=[r for r in rows if r['control']=='real']
nuisance=[r for r in rows if r['control'] in ['color_copy','shift_copy','noisy_copy','repeat']]
thresholds={'structure_min':1.5*max(r['structure_pair_mse'] for r in nuisance),
 'low_min':.25*float(np.quantile([r['low_pair_mse'] for r in real],.1)),
 'nearest_median_min':1.1*max(float(np.median(r['nearest_pixel_mse'])) for r in rows if r['control']=='noisy_copy')}
def passes(r):
 return r['structure_pair_mse']>=thresholds['structure_min'] and r['low_pair_mse']>=thresholds['low_min'] and np.median(r['nearest_pixel_mse'])>thresholds['nearest_median_min']
counts={n:int(sum(passes(r) for r in rows if r['control']==n)) for n in sorted({r['control'] for r in rows})}
assert all(v==0 for k,v in counts.items() if k!='real'),counts
assert counts['real']>=114,counts
baseline=json.loads((p/'baseline/metrics.json').read_text())['rows']
protocol={'thresholds':thresholds,'calibration_pass_counts':counts,'calibration_classes':151,
 'baseline_pass_counts':{m:int(sum(passes(r) for r in baseline if r['model']==m)) for m in ['raw','ema']},
 'controls_sha256':digest(p/'controls/controls.json'),'historical_audit_sha256':digest(p/'baseline/metrics.json'),
 'acceptance':{'classes_required':114,'classes_total':151,'models_required':['raw','ema'],
 'draws':'eight common seed18001+class and eight fresh seed audit_seed+step+100000+class; quantitative gate on combined16, record separate halves and require visual recognizability and structural variation on fresh draws',
 'visual':'Per class >=6/8 recognizable in each half and >=3 distinct pose/shape/composition variants across16, neither texture/color/translation-only nor near copies. Record uncertain as inconclusive/fail. All151 reviewed for acceptance.',
 'persistence':'same >=114 class intersection in both raw and EMA at two consecutive scheduled audits; confirm at identical two assessment points with independent training/audit seeds',
 'memorization':'flag near optimization copies; nearest pixel distance is no proof of absence; median novelty gate plus visual neighbor review',
 'saturation':'Report unclamped range and saturation; clipped-only apparent variation does not qualify'},
 'early_stop':'After OT decay ends, two consecutive scheduled audits with <=15/151 quantitative-pass classes in BOTH raw and EMA, AND visual review across >=30 classes showing persistent templates or noise-only variation: stop for persistent collapse/nonstructural diversity. Incomplete but improving or ambiguous results continue to full horizon. No stop at first6k audit.',
 'failure_stop':'Numerical/resource failure immediately; at most2 exact infrastructure recoveries per candidate within original watchdog. User stop terminal.',
 'no_FID':True}
atomic_json(ROOT/'resources/stability_audit_protocol.json',protocol)
print(json.dumps(protocol,indent=2))
