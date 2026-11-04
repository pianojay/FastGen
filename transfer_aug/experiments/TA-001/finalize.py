"""Validate the finished finite contract and create a compact, hash-backed handoff."""
from pathlib import Path
import csv
import json
import hashlib
import resource
import time
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import run_experiment as exp

HERE = Path(__file__).resolve().parent
START = time.monotonic()
BOUND = datetime.fromisoformat('2026-10-07T11:51:24.356912+00:00')


def read(path): return json.loads((HERE/path).read_text())
def save(path,value): exp.save_json(HERE/path,value)


exp.status('final_acceptance',generation_process_completed=True,recompute_process_completed=True,
           manual_review_completed=True)
integrity = exp.integrity('final')
numeric = read('verification/numerical_checks.json')
determinism = read('verification/determinism.json')
visual = read('visual_inspection.json')
assert numeric['pass'] and determinism['pass'] and integrity['pass']
cohort = json.loads((exp.BUNDLE/'cohort.json').read_text())
raw = np.load(HERE/'arrays/raw_rgb.npy',mmap_mode='r')
content = np.load(exp.BUNDLE/'content_uint8.npy',mmap_mode='r')
assert raw.shape==(4,604,64,64,3) and raw.dtype==np.float64
assert all(np.isfinite(raw[j]).all() for j in range(4))
assert np.array_equal(raw[0],content.astype(np.float64)/255)
pngs = list((HERE/'png').glob('*/*.png'))
assert len(pngs)==2416
for label in exp.LABELS:
    assert len(list((HERE/'png'/label).glob('*.png')))==604
metrics = list(csv.DictReader((HERE/'per_output_metrics.csv').open()))
assert len(metrics)==2416
assert Counter(r['condition'] for r in metrics)==Counter({label:604 for label in exp.LABELS})
assert len({(r['sample_id'],r['condition']) for r in metrics})==2416
assert determinism['raw_output_arrays_exact']==2416 and determinism['display_png_bytes_exact']==2416
assert determinism['contact_sheet_png_bytes_exact']==24
reviews = list(csv.DictReader((HERE/'visual_review.csv').open()))
assert len(reviews)==56 and Counter(r['review_group'] for r in reviews)=={'representative':48,'tail':8}
assert len({(r['sample_id'],r['condition'],r['review_group']) for r in reviews})==56
expected_reps = {(sid,f'alpha_{a:g}') for sid in cohort['representative_review_ids'] for a in [.25,.5,1.]}
assert {(r['sample_id'],r['condition']) for r in reviews if r['review_group']=='representative'}==expected_reps
tails = read('tail_selection.json')['rows']
assert {(r['sample_id'],float(r['alpha'])) for r in reviews if r['review_group']=='tail'} == {(r['sample_id'],r['alpha']) for r in tails}
for r in reviews:
    for k in ['recognizable_identity','anatomy_markings']:
        assert r[k] in ['retained','disrupted','uncertain']
    for k in ['background_contamination','artifacts']:
        assert r[k] in ['none','visible','uncertain']
    assert r['observation'] and (HERE/r['sheet']).exists()
assert visual['viewed_sheets']==24 and visual['completed_review_rows']==56
for s in visual['sheets']:
    assert s['actual_tool_view_completed'] and exp.sha(HERE/s['path'])==s['sha256']
words = len((HERE/'SUMMARY.md').read_text().split())
assert words<=500
acceptance = {'pass':True,'timestamp_utc':exp.stamp(),'summary_word_count':words,
              'finite_raw_arrays':True,'raw_dtype':str(raw.dtype),'raw_shape':list(raw.shape),
              'exact_alpha0_controls':604,'png_count':2416,'metric_rows':2416,
              'transformed_outputs':1812,'representative_review_rows':48,'tail_review_rows':8,
              'review_conditions_match_frozen_selection':True,'viewed_sheet_hashes_match':24,
              'generation_process_completed':True,'recompute_process_completed':True,
              'tail_ranking_unchanged':True,'contract_scope_complete':True}
save('verification/acceptance.json',acceptance)
generation = read('verification/resources_generate.json')
recompute = read('verification/resources_recompute.json')
now = datetime.now(timezone.utc)
resources = {'device':'cpu','parameters':0,'training_applicable':False,'gpu_seconds':0,
             'gpu_memory_bytes':0,'gpu_reserved_memory_bytes':0,'thread_limit_per_worker':2,
             'elapsed_seconds':(now-BOUND).total_seconds(),
             'elapsed_accounting_start_utc':BOUND.isoformat(),
             'elapsed_accounting_basis':'Conservative contract-freeze bound immediately before dispatch; includes implementation, failed attempt, generation, inspection, recomputation and reporting. Exact spawn timestamp was not persisted.',
             'historical_status_start_note':'Earlier status/config used an approximate 11:51:00 bound; this final record uses the master-provided 11:51:24.356912 freeze bound.',
             'elapsed_budget_seconds':7200,
             'peak_host_memory_bytes':max(generation['peak_host_memory_bytes'],recompute['peak_host_memory_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
             'peak_host_memory_measurement':'Maximum measured per-process ru_maxrss across successful generation, recomputation and finalization. The first test-only failed attempt and short read-only diagnostic processes were not independently RSS-sampled.',
             'successful_generation_process_seconds':generation['process_elapsed_seconds'],
             'successful_recompute_process_seconds':recompute['process_elapsed_seconds'],
             'finalization_process_seconds_at_record':time.monotonic()-START,
             'failed_attempts':1,'failed_attempt_elapsed_seconds':11.001387514173985,
             'failed_attempt_accounted_in_wall_budget':True,
             'stage_timing_proofs':['verification/numerical_checks.json','verification/resources_generate.json',
                                    'verification/resources_recompute.json','verification/integrity_before.json',
                                    'verification/integrity_after_generation.json','verification/integrity_after.json'],
             'recorded_utc':now.isoformat()}
assert resources['elapsed_seconds']<7200
save('verification/resources_final.json',resources)
excluded = {'artifact_manifest.json','handoff_sha256.json','result.json','status.json',
            'verification/artifact_hashes_verified.json'}
files = {}
for path in sorted(HERE.rglob('*')):
    if not path.is_file(): continue
    rel = str(path.relative_to(HERE))
    if rel in excluded: continue
    files[rel] = {'sha256':exp.sha(path),'bytes':path.stat().st_size}
file_list_digest = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
manifest = {'schema':1,'experiment_id':'TA-001','base_directory':str(HERE),
            'file_count':len(files),'total_bytes':sum(r['bytes'] for r in files.values()),
            'entries_canonical_sha256':file_list_digest,'files':files,
            'exclusions':sorted(excluded),
            'exclusion_note':'Avoid recursive hashes and mutable execution state. Handoff hashes separately pin result, summary and this manifest.'}
save('artifact_manifest.json',manifest)
for name,info in files.items():
    assert (HERE/name).stat().st_size==info['bytes']
    assert exp.sha(HERE/name)==info['sha256']
save('verification/artifact_hashes_verified.json',{'pass':True,'file_count':len(files),
      'all_file_hashes_and_sizes_match':True,'artifact_manifest_sha256':exp.sha(HERE/'artifact_manifest.json'),
      'entries_canonical_sha256':file_list_digest,'timestamp_utc':exp.stamp()})

checks = {
 'frozen_and_protected_integrity_before':{'pass':True,'proof':'verification/integrity_before.json'},
 'frozen_and_protected_integrity_after':{'pass':True,'proof':'verification/integrity_final.json'},
 'numerical_acceptance':{'pass':True,'proof':'verification/numerical_checks.json'},
 'zero_variance_exception_documented':{'pass':True,'proof':'verification/degenerate_channels.json'},
 'all_outputs_exact_recomputation':{'pass':True,'proof':'verification/determinism.json'},
 'all_required_sheets_actually_viewed':{'pass':True,'proof':'visual_inspection.json'},
 'counts_review_coverage_and_finite_arrays':{'pass':True,'proof':'verification/acceptance.json'},
 'artifact_hashes':{'pass':True,'proof':'verification/artifact_hashes_verified.json'},
 'resource_budget':{'pass':True,'proof':'verification/resources_final.json'},
}
evidence = {
 'summary':'SUMMARY.md','condition_metrics':'condition_metrics.json','per_output_metrics':'per_output_metrics.csv',
 'review_rows':'visual_review.csv','visual_inspection':'visual_inspection.json','gallery':'gallery.html',
 'representative_sheets':'sheets/representative','tail_sheets':'sheets/tail','tail_selection':'tail_selection.json',
 'raw_numeric_arrays':'arrays/raw_rgb.npy','array_layout':'arrays/layout.json','display_pngs':'png',
 'numerical_verification':'verification/numerical_checks.json','determinism':'verification/determinism.json',
 'input_integrity_before':'verification/integrity_before.json','input_integrity_after':'verification/integrity_final.json',
 'acceptance':'verification/acceptance.json','resources':'verification/resources_final.json',
 'commands':'commands.json','implementation':'run_experiment.py','implementation_notes':'IMPLEMENTATION.md',
 'config':'config.json','environment':'environment.json','hashes':'artifact_manifest.json',
 'hash_verification':'verification/artifact_hashes_verified.json',
 'failed_attempt_diagnosis':'verification/failed_attempt_1_diagnosis.json',
 'failed_attempt_implementation':'verification/failed_attempt_1_implementation.py',
 'failed_attempt_log':'verification/failed_attempt_1.log',
}
for path in evidence.values(): assert (HERE/path).exists()
for check in checks.values(): assert (HERE/check['proof']).exists()
result = {'experiment_id':'TA-001','contract_revision':1,'status':'completed',**exp.PINS,
          'training_applicable':False,'expected_output_count':2416,'completed_output_count':2416,
          'expected_transformed_output_count':1812,'completed_transformed_output_count':1812,
          'expected_control_output_count':604,'completed_control_output_count':604,
          'expected_review_count':56,'completed_review_count':56,
          'expected_representative_review_count':48,'completed_representative_review_count':48,
          'expected_tail_review_count':8,'completed_tail_review_count':8,
          'expected_sheet_count':24,'completed_sheet_count':24,'viewed_sheet_count':24,
          'resources':resources,'checks':checks,'evidence':evidence,
          'implementation_sha256':exp.sha(HERE/'run_experiment.py'),
          'artifact_manifest_sha256':exp.sha(HERE/'artifact_manifest.json'),
          'artifact_entries_canonical_sha256':file_list_digest,
          'key_findings':[
           'Pre-inverse Lab moments follow the contracted interpolation: maximum mean/std errors 5.39e-12/2.59e-12. One near-achromatic b channel (Venusaur 139_3) uses the mean-only fallback, so its donor std is intentionally not matched.',
           'At alpha 0.25/0.5/1.0, median clipped-float RMS changes are 0.06191/0.11564/0.21800 and median strict out-of-gamut scalar fractions are 0.00594/0.02714/0.09363.',
           'At alpha 1, postclip Lab mean-to-donor distance median/p95 is 1.5904/8.2431; std-to-donor distance is 2.2825/11.8132. Mathematical pre-inverse alignment does not imply realizable post-gamut matching.',
           'Strict clipping count alone is not visual severity. Tail Mewtwo 082_3 has clipped fraction 0.61947, RMS 0.00649 and max excursion 0.01239 (small real overshoot, not roundoff); its appearance is nearly unchanged. Dewgong 020_0 has fraction 0.45768, RMS 0.38414, excursion 0.11569 and visible loss of white-body detail. Frozen tail ranking is unchanged.',
           'At alpha 1, maximum-excursion median/p95 is 0.42579/3.33252; the largest raw excursion is 22.45183 on Rhyhorn 117_1. Extreme excursions and broad clipping need not occur on the same outputs.',
           'Unblinded representative identity-retained judgments are 15/16, 15/16 and 12/16 by increasing strength; all other identity labels are uncertain. Alpha-1 anatomy/markings are retained/uncertain/disrupted in 7/7/2 outputs; background shifts are widespread.',
           'Direct inspection finds lost Primeape eye contrast, saturated flat Tentacool head colors, and source-dependent washout/darkening. Some card sources are too blurred for confident identity or fine-anatomy judgment.',
           'All 2416 raw outputs and PNGs, metrics, selected tails, 24 contact sheets and gallery reproduce exactly. Alpha-0 controls are exact; all 1812 self-donor exports match within zero LSB.'
          ],
          'limitations':[
           'A1-only 64-pixel direct transforms; no A2/B image analysis, training, classifier, discriminator test or FID calculation.',
           'Visual labels are unblinded AI comparisons for 48 representative and eight selected outputs, not measured category accuracy or evidence of augmentation utility.',
           'Whole-image statistics mix background and foreground; no segmentation control or causal background ablation was performed.',
           'CIELAB D65 float64 implementation adapts the pinned community method rather than bitwise reproducing OpenCV uint8 Lab or its default inverse ratio.',
           'Source blur, card clutter and nearest-neighbor display limit fine anatomy/marking assessment. Clipping fraction, boundary and high-frequency measures are descriptive, not ground-truth artifact labels.',
           'The 1e-6 degeneracy threshold is sensitive for nearly achromatic channels: Venusaur 139_3 a std is 1.90e-6 and b std 7.61e-7 with the documented matrix. The frozen branch rule was followed without adjusting the threshold.',
           'Peak host RSS is the maximum of measured successful processes; the first test-only failed attempt and short diagnostics were not independently sampled. Conservative elapsed wall time includes the full assignment.',
           'One synthetic fixture initially failed due to mean-summation cancellation; original code/log/diagnosis are preserved. Shifted population-moment reduction fixed it without changing the mathematical contract.',
          ],
          'evidence_notes':{'path_base':str(HERE),'visual_review_method':'All 24 actual PNG sheets opened with tools.view_image, then 56 per-output labels transcribed. Controls and donor columns supply comparison context; only contracted transformed outputs receive review rows.',
                            'recompute_scope':'Deterministic experiment artifacts are recalculated and matched without rewriting successful outputs. Timestamped runtime/provenance and subjective visual judgments are not deterministic image outputs.',
                            'no_recipe_selected':True},
          'completed_utc':exp.stamp()}
save('result.json',result)
save('handoff_sha256.json',{p:exp.sha(HERE/p) for p in ['result.json','SUMMARY.md','artifact_manifest.json']})
save('status.json',{'experiment_id':'TA-001','stage':'completed','status':'completed','updated_utc':exp.stamp(),
                   'active_pid':None,'generation_process_completed':True,'recompute_process_completed':True,
                   'manual_review_completed':True,'completed_outputs':2416,'completed_review_rows':56,
                   'result':'result.json','elapsed_seconds':resources['elapsed_seconds']})
print(json.dumps({'status':'completed','artifact_files':len(files),'summary_words':words,
                  'elapsed_seconds':resources['elapsed_seconds'],'peak_host_memory_bytes':resources['peak_host_memory_bytes'],
                  'result_sha256':exp.sha(HERE/'result.json')},indent=2))
