import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from analyze import OUT,ROOT,sha,write_json,NAMES
m=json.loads((OUT/'domain_manifest.json').read_text());p=json.loads((OUT/'pca.json').read_text());c=json.loads((OUT/'counts.json').read_text());v=json.loads((OUT/'verification.json').read_text());r=json.loads((OUT/'regeneration.json').read_text())
frozen=json.loads((ROOT/'transfer_aug/shared/cohort_v1/manifest.json').read_text())['protected_inputs'];assert v['protected_inputs']==frozen
v['protected_inputs_match_first_wave_frozen_manifest']=True;write_json(OUT/'verification.json',v)
observations={
'Butterfree':'D1 often has bold contours or black backgrounds; one D1 optimization image remains white-backed. Poses/crops vary.',
'Clefable':'D0 mostly pale/white-backed; D1 includes vivid scenes and black-backed sprites.',
'Clefairy':'D1 outlines and black backgrounds stand out; D0 includes pale renders and lighter pixel art. Similar-looking sprites are not exact duplicates.',
'Cubone':'D0 mainly light/soft; D1 includes darker scenes, black background and strongly outlined colored artwork. Media differ.',
'Dratini':'D1 contains high-contrast black panels and detailed water scenes; D0 includes pale simple depictions and a soft rendered scene.',
'Dugtrio':'Contrast is weaker: both domains contain detailed scenes and outlined sprites, with D1 tending stronger light-dark separation.',
'Exeggcute':'D0 is mostly pale/soft; D1 contains stronger outlines and dark or vivid scenes, plus black-backed line art.',
'Gyarados':'D1 tends detailed high contrast/dark backgrounds; a busy bright D0 card is a counterexample to a simple complexity label.',
'Ivysaur':'D1 often black-backed or stronger-edged; D0 lighter/flatter. Pixel-art/media differences confound appearance.',
'Poliwag':'D1 often stronger outlining/contrast; D0 includes an already dark low-poly rendering. Not a brightness-only split.',
'Primeape':'D0 muted tan/soft backgrounds; D1 black-backed sprites/renders and strong green contours. Pose/crop varies.',
'Seaking':'D0 softer scenic renders; D1 stronger contours and contrasting panels; white-backed drawing shows D1 need not be dark.',
'Shellder':'D1 sharper contrast in several samples but white-backed soft render overlaps D0 appearance; media and crop vary.',
'Tentacool':'D1 repeatedly uses black or high-contrast backgrounds, D0 pale/soft scenes. One D1 white-background example remains.',
'Venusaur':'D1 very heterogeneous: black-backed sprite, white-background line art, dark object and detailed card. D0 often softer color scenes.',
'Weepinbell':'D0 mostly softly lit toy/render depictions; D1 black-backed crisp sprites, vivid scene and one dark render.'}
write_json(OUT/'visual_review.json',dict(inspected_with_image_tool=[f'review_{i}.png' for i in range(1,5)],image_count=128,category_count=16,observations=observations,recurrence='Repeated stronger light-dark contrast/edges and frequent black backgrounds in D1, relative to often paler/softer D0; weaker and heterogeneous in Dugtrio, Shellder, Venusaur.',uncertainty='Unblinded small fixed subset; visual observations are qualitative. Pose, object scale, media, background and fine-detail differences co-vary. No semantic-style or class-correctness inference. Similar non-identical images remain possible across pools.'))
with (OUT/'scores.csv').open() as f:score_rows=[q for q in csv.DictReader(f) if q['dedup_retained']=='True']
metrics={}
for d in ['D0','D1']:
    rs=[q for q in score_rows if q['domain']==d];metrics[d]={'n':len(rs),'score_mean':float(np.mean([float(q['score']) for q in rs])),'class_balanced_descriptor_means':{name:float(np.mean([np.mean([float(q[name]) for q in rs if q['category']==cat]) for cat in c['per_class']])) for name in NAMES}}
write_json(OUT/'domain_measurements.json',metrics)
res=dict(status='complete',recommendation='ready',experiment_id='appearance-domain-round2',revision=1,contract_sha256=m['contract_sha256'],training_applied=False,candidate_manifest_sha256=sha(OUT/'domain_manifest.json'),counts={k:value for k,value in c.items() if k!='per_class'},per_domain_optimization=c['totals']['optimize']//2,per_domain_holdout=c['totals']['holdout']//2,minimum_optimization_per_class_domain=v['minimum_optimize_per_class_domain'],pc1_explained_variance=p['explained_variance_ratio'][0],cross_class_duplicate_groups=v['cross_class_duplicate_groups'],global_train_holdout_pixel_overlap=0,findings='Class-balanced PC1 provides reproducible contrast/edge/background separation with mixed media and content confounding; supports a bounded statistical-domain pilot, not a semantic style claim.',limitations=['A1-only auxiliary holdout; not independent confirmation','128 fixed review images cover 16 of 151 categories','Cross-class duplicate labels persist; five pairs optimize in one domain and one pair unused','No training, pretrained features, GPU, FID or augmentation effectiveness claim','Exact-pixel deduplication does not remove near-duplicate images'],proof_paths={n:str((OUT/n).relative_to(ROOT)) for n in ['domain_manifest.json','scores.csv','counts.json','pca.json','DEFINITIONS.md','duplicate_exclusions.json','cross_class_duplicates.json','verification.json','regeneration.json','visual_review.json','review_selection.json','domain_measurements.json','resource_usage.json']},commands=['bash transfer_aug/run_python.sh transfer_aug/domain_discovery/round2/analyze.py','bash transfer_aug/run_python.sh transfer_aug/domain_discovery/round2/verify_and_regenerate.py','bash transfer_aug/run_python.sh transfer_aug/domain_discovery/round2/finalize.py'])
write_json(OUT/'result.json',res)
summary=f'''# Round 2 appearance-domain candidate — ready

Measured all {c['a1']:,} A1 images and excluded {c['excluded']:,} within-class exact-pixel duplicates, retaining {c['unique']:,} images across all 151 categories. D0 and D1 each contain {res['per_domain_optimization']:,} optimization images and {res['per_domain_holdout']} fixed diagnostic holdout images; {c['totals']['unused']:,} middle images are unused. Every category has equal D0/D1 support and at least {res['minimum_optimization_per_class_domain']} optimization images per domain.

Class-balanced, residual-normalized PC1 explains {100*res['pc1_explained_variance']:.2f}% of weighted variance. Its strongest positive loadings are luminance standard deviation, neighbor gradient, Laplacian energy, border contrast and near-black fraction; mean luminance loads negatively. Entropy sign correlation is {p['orientation_correlation']:.3f}.

All four review sheets were inspected with image tools: 128 images, 16 seeded categories, two optimization plus both holdout examples per domain/category. D1 repeatedly shows stronger light-dark contrast/edges and frequent dark backgrounds; D0 often appears paler/softer. Recurrence is weaker or heterogeneous for Dugtrio, Shellder and Venusaur. Mixed media, poses, scale, object detail and backgrounds remain confounded; these are statistical domains, not established semantic styles.

All descriptors are finite. Full extraction, PCA, membership and image-sheet regeneration were byte-identical. Synthetic constant and checkerboard checks passed. Every A1 source file was rehashed; protected-input hashes match the first-wave frozen manifest. Six cross-class duplicate pairs exist: five entirely optimize within the same domain, one entirely unused. There is zero global optimization/holdout pixel overlap; the prescribed selection rule required no changes.

Candidate: `domain_manifest.json`; exact definitions: `DEFINITIONS.md`; numerical/integrity proofs: `verification.json`, `regeneration.json`; visual evidence: `review_1.png`–`review_4.png`, `visual_review.json`; counts, PCA, measurements and exclusions have dedicated JSON/CSV files. `artifact_manifest.json` pins outputs and implementation.

Recommendation: freeze this candidate for a bounded unpaired-translation pilot, retaining diagnostic holdouts. No training, model loading, GPU use, external holdout evaluation, class-correctness assessment or usefulness/FID claim occurred. Near duplicates may remain. Resource details are in `resource_usage.json`; CPU use is well below one hour.
'''
(OUT/'SUMMARY.md').write_text(summary)
usage=json.loads((OUT/'resource_usage.json').read_text());usage['finalization_process_cpu_seconds']=time.process_time();usage['total_recorded_cpu_seconds']=usage['total_analysis_cpu_seconds']+usage['verification_process_cpu_seconds']+usage['finalization_process_cpu_seconds'];write_json(OUT/'resource_usage.json',usage)
write_json(OUT/'artifact_manifest.json',dict(schema=1,files={q.name:sha(q) for q in sorted(OUT.iterdir()) if q.is_file() and q.name!='artifact_manifest.json'},input_sha256=v['protected_inputs']))
print(json.dumps(res,indent=2))
