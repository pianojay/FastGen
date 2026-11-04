"""Bounded CPU report handoff; no GPU acquisition or successor experiments."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2];deadline=time.monotonic()+28800
while time.monotonic()<deadline:
 status=json.loads((R/'status.json').read_text())
 if status['stage']=='failed':raise SystemExit('GPU stage failed; no report launched')
 if status['stage']=='gpu_complete':
  ledger=json.loads((R/'gpu_resource_ledger.json').read_text())
  if all(s['ended_at'] is not None for s in ledger['sessions']):break
 time.sleep(30)
else:raise SystemExit('CPU handoff wait reached GPU budget duration')
with (R/'report.log').open('w') as f:subprocess.run(['bash',str(root/'transfer_aug/run_python.sh'),str(R/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'cpu_report_complete.json').write_text(json.dumps({'status':'complete','awaiting':'direct visual inspection and summary'})+'\n')
