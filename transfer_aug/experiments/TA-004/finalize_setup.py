from pathlib import Path
import hashlib,json,datetime
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((r/'verification.json').read_text())
source=r.parents[1]/'audit_sources/cut'
p=json.loads((source/'provenance.json').read_text())
for name,meta in p['files'].items():
    assert sha(source/name)==meta['sha256'],name
proof=['verification.json','synthetic_checkpoint.pt','cut64.py','networks.py','build_sources.py','verify_setup.py','LICENSE.upstream','SETUP_SUMMARY.md']
hashes={str(r/name):sha(r/name) for name in proof}
(r/'setup_hashes.json').write_text(json.dumps({'artifacts':hashes,'upstream_files':p['files']},indent=2)+'\n')
result={'experiment_id':'TA-004','status':'setup_completed','contract_revision':2,
'contract':str(r/'SETUP_CONTRACT_v2.md'),'contract_sha256':sha(r/'SETUP_CONTRACT_v2.md'),
'upstream_commit':p['revision'],'source_provenance':str(source/'provenance.json'),
'source_integrity_checked':True,'training_applicable':True,'dataset_training_performed':False,
'synthetic_cpu_updates_executed':3,'device':'cpu','gpu_seconds':0,'gpu_allocated_bytes':0,
'parameter_counts':v['parameter_counts'],'framework_parameters':v['framework_parameters'],
'feature_mapping':v['feature_mapping'],'checks':v['checks'],'verification_seconds':v['elapsed_seconds'],
'peak_host_rss_bytes':v['peak_host_rss_bytes'],'proof_paths':{x:str(r/x) for x in proof+['setup_hashes.json']},
'blockers':['Final shared domains and frozen training contract required before profiling/training.'],
'limitations':['CPU synthetic batch2 checks only; batch16 GPU memory and speed unqualified.','No dataset loading or GPU launch runner until final contract.','No learned image quality/convergence claim.'],
'architecture_changes':['Revision 2 corrects antialiased discriminator receptive field to 40px; upstream filters retained.'],
'completed_checks':len(v['checks']),'expected_checks':len(v['checks']),
'reproduce_command':'bash transfer_aug/run_python.sh transfer_aug/experiments/TA-004/verify_setup.py'}
(r/'setup_result.json').write_text(json.dumps(result,indent=2)+'\n')
(r/'status.json').write_text(json.dumps({'experiment_id':'TA-004','stage':'setup_completed_awaiting_training_contract','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owned_pid':None,'gpu_seconds':0,'result':str(r/'setup_result.json')},indent=2)+'\n')
assert len((r/'SETUP_SUMMARY.md').read_text().split())<=500
print(json.dumps({'contract_sha256':result['contract_sha256'],'result_sha256':sha(r/'setup_result.json'),'summary_words':len((r/'SETUP_SUMMARY.md').read_text().split())}))
