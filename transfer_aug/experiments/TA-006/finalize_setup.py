import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
v=json.loads((R/'verification.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.name:sha(p) for p in R.iterdir() if p.suffix in ('.py','.md','.json') and p.name not in ('setup_result.json','setup_hashes.json')}
(R/'setup_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
r={'experiment_id':'TA-006','status':'setup_completed','setup_contract_sha256':sha(R/'SETUP_CONTRACT_v1.md'),'upstream_revision':'875b70a150609e8a678ed8482562e7074cdce7e5','parameters':v['counts'],'framework_total_parameters':v['total_parameters'],'trainable_parameters':v['trainable_parameters'],'outside_normal_parameter_cap':v['outside_normal_100M_limit'],'parameter_exception':'User waived StarGAN v2 100M cap only.','checks':v['checks'],'resources':{'gpu_seconds':0,'cpu_verification_seconds':v['cpu_seconds'],'cpu_batch_size':1,'gpu_batch8_memory_qualified':False},'training_performed':False,'api':'StarGAN64(device); step(x,y,[ref1,ref2],yt,zs=None); checkpoint(path,sampler); restore(path,sampler); .nets/.ema; Sampler(domains)','proof_paths':{'summary':'SETUP_SUMMARY.md','verification':'verification.json','log':'verify_setup.log','source_adaptation':'source_adaptation.json','implementation_hashes':'setup_hashes.json','documentation':'README.md'},'blockers':[],'limitations':['GPU memory/time qualification pending final contract.','Synthetic sensitivity is not learned appearance diversity.','Unbounded RGB outputs require preserved floating outputs and clipping diagnostics.']}
(R/'setup_result.json').write_text(json.dumps(r,indent=2)+'\n')
