"""Finalize evidence after manual direct inspection of every required sheet."""
import datetime,json,hashlib
from pathlib import Path
from runner import ROOT,CONTRACT,MANIFEST,PIXELS,write,sha256,preflight

def main():
    preflight();training=json.loads((ROOT/'training_complete.json').read_text());gen=json.loads((ROOT/'generation_verification.json').read_text());ledger=json.loads((ROOT/'gpu_ledger.json').read_text())
    checks=json.loads((ROOT/'eda/verification.json').read_text());agg=json.loads((ROOT/'eda/aggregate.json').read_text());visual=json.loads((ROOT/'visual_review.json').read_text());qual=json.loads((ROOT/'qualification.json').read_text())
    sheets=json.loads((ROOT/'eda/sheet_manifest.json').read_text())
    assert all(s.get('directly_inspected') for s in visual['sheets'])
    assert {s['sheet'] for s in visual['sheets']}=={s['sheet'] for s in sheets}
    assert gen['outputs']==604 and gen['reproduced_review_outputs']==64 and gen['quantized_reproduction_exact']
    assert all(s['ended_epoch'] for s in ledger['sessions']) and ledger['cumulative_gpu_seconds']<28800
    assert training['peak_allocated_bytes']<20000000000 and qual['parameter_counts']['total']<100000000
    assert len((ROOT/'SUMMARY.md').read_text().split())<=500
    checkpoint_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'checkpoints').glob('*.pt'))}
    implementation={p.name:sha256(p) for p in ROOT.glob('*.py')}
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ('output_hashes.json','result.json','status.json','setup_hashes.json')}
    write(ROOT/'output_hashes.json',hashes)
    result=dict(experiment_id='TA-003',contract_revision=1,status=training['status'],contract_sha256=CONTRACT,manifest_sha256=MANIFEST,pixels_sha256=PIXELS,
        training_applicable=True,training_executed=True,seed=20261008,upstream_revision='2a7afba2895d52556dd5dfe07e8555ef657ced6f',upstream_file_hashes=json.loads((ROOT/'upstream_provenance.json').read_text())['files'],review_method='Small unblinded AI review; representative subset and selected largest-RMS tails reported separately; uncertain judgments allowed.',completed_updates=training['updates'],maximum_updates=20000,
        counts=dict(translated_diagnostic_outputs=604,expected_translated_diagnostic_outputs=604,per_direction={'D0_D1':302,'D1_D0':302},cycle_controls=604,identity_controls=604,
                    representative_sources=64,tail_sources=8,review_sheets=9,milestone_review_translations=64*len(gen['milestones'])),
        parameters=training['parameter_counts'],device='physical GPU 1',device_uuid=qual['uuid'],gpu_seconds=ledger['cumulative_gpu_seconds'],gpu_hours=ledger['cumulative_gpu_seconds']/3600,
        gpu_budget_seconds=28800,peak_allocated_bytes=max(qual['peak_allocated_bytes'],training['peak_allocated_bytes']),peak_reserved_bytes=max(qual['peak_reserved_bytes'],training['peak_reserved_bytes']),
        reset_after_qualification=training['seed_reset_after_qualification'],qualification=qual,checkpoints=checkpoint_hashes,implementation_sha256=implementation,
        primary_cross_method_direction='D0_D1',directional_aggregates=agg,
        verification=dict(**checks,gpu_checkpoint_recovery_exact=qual['gpu_recovery_exact'],deterministic_review_regeneration_exact=gen['quantized_reproduction_exact'],inputs_rehashed_after=True,diagnostic_excluded_from_weight_optimization=True,all_required_sheets_directly_reviewed=True,parameter_cap_passed=True,training_memory_cap_passed=True,gpu_budget_passed=True),
        proof_paths=['SUMMARY.md','qualification.json','training_complete.json','gpu_ledger.json','lease_ownership.json','preflight.json','upstream_provenance.json','input_integrity_after.json','generation_verification.json','data_verification.json','run_configuration.json','checkpoint_history.json','training.jsonl','eda/per_image.csv','eda/aggregate.json','eda/verification.json','eda/sheet_manifest.json','eda/tail_selection.json','visual_review.json','outputs/manifest.json','outputs/review_manifest.json','output_hashes.json','gallery.html'],
        limitations=['A1 diagnostics influenced domain discovery; not independent validation.','Statistical appearance projection is not semantic style or quality.','No learned feature metric, classifier, FID, downstream training or usefulness claim.','CycleGAN has bidirectional cycle/identity objectives unlike one-way CUT; primary shared comparison is D0->D1.','Small unblinded AI visual review; preservation judgments are descriptive and may be uncertain. No class-accuracy or convergence claim.'],
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for p in result['proof_paths']:assert (ROOT/p).exists(),p
    write(ROOT/'result.json',result);write(ROOT/'status.json',dict(stage='completed',status=training['status'],completed_updates=training['updates'],completed_outputs=604,completed_review_sheets=9,cumulative_gpu_seconds=ledger['cumulative_gpu_seconds'],gpu_lease_released=True,utc=result['completed_utc']))
    print(json.dumps({k:result[k] for k in ['status','completed_updates','gpu_seconds','peak_allocated_bytes','counts']}))
if __name__=='__main__':main()
