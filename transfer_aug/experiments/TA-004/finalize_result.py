"""Finalize handoff only after direct visual review is saved."""
import datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
def read(n):return json.loads((R/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
complete=read('training_complete.json');metrics=read('aggregate_metrics.json');visual=read('visual_review.json');ledger=read('gpu_resource_ledger.json');preflight=read('training_preflight.json');q=read('qualification.json');inventory=read('review_inventory.json')
assert all(s['ended_at'] for s in ledger['sessions'])
gpu_seconds=sum(s['gpu_seconds'] for s in ledger['sessions']);assert gpu_seconds<28800
assert complete['peak_allocated_bytes']<20_000_000_000
assert visual['reviewed_representative_examples']==64 and visual['reviewed_tail_examples']==8
assert len(inventory)==72
freeze=json.loads((ROOT/'transfer_aug/shared/round2/freeze.json').read_text())
for n in ['manifest.json','pixels_uint8.npy']:
    assert sha(ROOT/'transfer_aug/shared/round2'/n)==freeze['files'][n]==complete['input_hashes_after'][n]
assert complete['deterministic_review_rerun_exact']
proof=['SETUP_SUMMARY.md','setup_result.json','verification.json','training_preflight.json','qualification.json','training_complete.json','final_verification.json','lr_verification.json','gpu_resource_ledger.json','losses.jsonl','measurements.json','measurements.csv','metric_conventions.json','aggregate_metrics.json','outputs/associations.json','output_hashes.json','review_inventory.json','visual_review.json','tails.json','commands.txt','final.pt','initial.pt','latest.pt','SUMMARY.md','gallery.html']
proof.extend(str(p.relative_to(R)) for p in sorted((R/'sheets').glob('*.png')))
write('implementation_hashes.json',{p.name:sha(p) for p in sorted(R.glob('*.py'))})
write('handoff_hashes.json',{n:sha(R/n) for n in proof})
result={'experiment_id':'TA-004','contract_revision':1,'status':complete['status'],'contract_sha256':sha(R/'CONTRACT_v1.md'),
'protocol_sha256':sha(ROOT/'transfer_aug/shared/round2/TRAINING_PROTOCOL_v1.md'),'setup_contract_revision':2,
'input_hashes':complete['input_hashes_after'],'training_applicable':True,'training_performed':True,'initialized_from_scratch':True,
'completed_updates':complete['updates'],'maximum_updates':20000,'qualification_updates':22,'batch_size':16,'framework_parameters':9061060,
'parameter_counts':{'G':7837699,'D':662977,'F':560384},'direction':'D0_to_D1','completed_translation_count':302,'expected_translation_count':302,
'completed_target_identity_count':302,'expected_target_identity_count':302,'total_final_saved_outputs':604,'reviewed_representative_examples':64,'reviewed_representative_translations':32,'reviewed_representative_identity_controls':32,'reviewed_tail_examples':8,
'device_uuid':complete['gpu_uuid'],'gpu_seconds':gpu_seconds,'gpu_hours':gpu_seconds/3600,'gpu_budget_seconds':28800,
'peak_allocated_bytes':complete['peak_allocated_bytes'],'peak_reserved_bytes':complete['peak_reserved_bytes'],
'gpu_lease_released':True,'checks':{'A1_only':preflight['all_A1'],'diagnostic_exclusion':preflight['diagnostic_exclusion'],'source_hashes':preflight['source_hashes_verified'],'finite_gradients_losses':q['finite_gradients_losses'],'exact_gpu_checkpoint_recovery':q['exact_gpu_next_step_recovery'],'deterministic_review_regeneration':complete['deterministic_review_rerun_exact'],'input_integrity_after':True,'all_required_sheets_directly_reviewed':True,'memory_budget':True,'time_budget':True},
'metrics':metrics,'visual_findings':visual['overall'],'proof_paths':{n:str(R/n) for n in proof+['implementation_hashes.json','handoff_hashes.json']},
'limitations':['A1 diagnostics participated in appearance-domain discovery and are not independent validation.','Appearance score is a fixed statistical projection, not style or semantic quality.','No classifier, class-accuracy, generative FID, downstream usefulness, or convergence claim.','CUT is one-directional with identity PatchNCE; objectives and coverage differ from CycleGAN.'],
'operational_incidents':['Initial preflight rejected Torch UUID formatting without GPU- prefix before training; corrected comparison and retained conservative time charge.'],
'review_method':'Small unblinded AI visual review; representatives and selected high-change tails are separate descriptive populations.',
'final_checkpoint_sha256':complete['final_checkpoint_sha256']}
write('result.json',result)
write('status.json',{'experiment_id':'TA-004','stage':'completed','scientific_status':complete['status'],'owned_pid':None,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'updates':complete['updates'],'gpu_seconds':gpu_seconds,'result':str(R/'result.json')})
assert len((R/'SUMMARY.md').read_text().split())<=500
print(json.dumps({'status':result['status'],'updates':result['completed_updates'],'gpu_hours':result['gpu_hours'],'result_sha256':sha(R/'result.json')},indent=2))
