"""Accept complete worker evidence only after direct visual inspection."""
import datetime,json,hashlib
from pathlib import Path
from runner import ROOT,CONTRACT,MANIFEST,PIXELS,write,sha256,preflight,ROUND3

def main():
    preflight();training=json.loads((ROOT/'training_complete.json').read_text());gen=json.loads((ROOT/'generation_verification.json').read_text());ledger=json.loads((ROOT/'gpu_ledger.json').read_text());qual=json.loads((ROOT/'qualification.json').read_text());visual=json.loads((ROOT/'visual_review.json').read_text());sheets=json.loads((ROOT/'eda/sheet_manifest.json').read_text());integrity=json.loads((ROOT/'protected_integrity_after.json').read_text());lease=json.loads((ROOT/'lease_ownership.json').read_text());checks=json.loads((ROOT/'eda/verification.json').read_text())
    assert all(x['directly_inspected'] for x in visual['sheets']) and {x['path'] for x in visual['sheets']}=={x['path'] for x in sheets}
    assert all(sha256(x['path'])==x['sha256'] for x in visual['sheets'])
    assert gen['final_outputs']==5436 and gen['reproduced_outputs']==512 and gen['quantized_reproduction_exact']
    assert all(x['ended_epoch'] for x in ledger['sessions']) and ledger['cumulative_gpu_seconds']<28800 and lease['released_utc']
    assert training['peak_allocated_bytes']<20000000000 and qual['parameter_counts']['total']<100000000 and integrity['all_match']
    assert len((ROOT/'SUMMARY.md').read_text().split())<=500
    hashes={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in ('result.json','output_hashes.json','status.json'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            hashes[str(p.relative_to(ROOT))]=h.hexdigest()
    write(ROOT/'output_hashes.json',hashes)
    result=dict(experiment_id='TA-005',contract_revision=1,status=training['status'],contract_sha256=CONTRACT,protocol_sha256=sha256(ROUND3/'TRAINING_PROTOCOL_v1.md'),manifest_sha256=MANIFEST,pixels_sha256=PIXELS,evaluation_sha256=sha256(ROUND3/'evaluation_manifest.json'),latent_sha256=sha256(ROUND3/'munit_latent.npy'),upstream_revision='a82e222bc359892bd0f522d7a0f1573f3ec4a485',training_applicable=True,training_executed=True,completed_updates=training['updates'],maximum_updates=20000,parameters=training['parameter_counts'],device='physical GPU1',device_uuid=qual['uuid'],gpu_seconds=ledger['cumulative_gpu_seconds'],gpu_hours=ledger['cumulative_gpu_seconds']/3600,gpu_budget_seconds=28800,peak_allocated_bytes=max(training['peak_allocated_bytes'],qual['peak_allocated_bytes']),peak_reserved_bytes=max(training['peak_reserved_bytes'],qual['peak_reserved_bytes']),gpu_lease_released=True,counts=gen,verification=dict(**checks,gpu_next_step_recovery_exact=qual['gpu_recovery_exact'],production_scratch_reset=training['seed_reset_after_qualification'],all_required_sheets_directly_reviewed=True,protected_inputs_unchanged=True,deterministic_reproduction_exact=True),aggregates=json.loads((ROOT/'eda/aggregate.json').read_text()),proof_paths=['SUMMARY.md','setup_result.json','qualification.json','training_complete.json','gpu_ledger.json','lease_ownership.json','preflight.json','input_integrity_after.json','protected_integrity_before.json','protected_integrity_after.json','generation_verification.json','data_verification.json','checkpoint_history.json','training.jsonl','eda/per_output.csv','eda/source_aggregates.csv','eda/aggregate.json','eda/verification.json','eda/sheet_manifest.json','eda/tail_selection.json','visual_review.json','outputs/manifest.json','outputs/milestone_manifest.json','output_hashes.json','gallery.html','COMMANDS.md','incidents.json','report_corrections/clipped_float_v1/verification.json'],limitations=['A1 diagnostic images participated in domain discovery; not independent validation.','Domains encode statistical contrast/edge differences, not semantic style labels.','Pixel/statistical diversity does not establish plausible semantic diversity.','Small unblinded AI review, not class accuracy.','No pretrained metrics, FID, SiDA integration or usefulness/convergence claim.'],completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for p in result['proof_paths']:assert (ROOT/p).exists(),p
    write(ROOT/'result.json',result);write(ROOT/'status.json',dict(stage='completed',status=training['status'],completed_updates=training['updates'],completed_outputs=5436,completed_review_sheets=len(sheets),cumulative_gpu_seconds=ledger['cumulative_gpu_seconds'],gpu_lease_released=True,utc=result['completed_utc']))
    print(json.dumps({k:result[k] for k in ['status','completed_updates','gpu_seconds','peak_allocated_bytes']}))
if __name__=='__main__':main()
