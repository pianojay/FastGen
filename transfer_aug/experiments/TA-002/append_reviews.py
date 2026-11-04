import json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'visual_judgments_manual.json'
a=json.loads(p.read_text()) if p.exists() else {'reviewer':'TA-002 worker (AI)','basis':'Actual view_image tool inspection; unblinded relative comparison. Codes are manually authored judgments: identity/markings R retained D disrupted U uncertain; background/artifacts N none V visible U uncertain. No metric-to-label rule.','representative':[],'tails':[],'sheets_actually_inspected':[]}
for e in json.loads(sys.stdin.read()):
 assert len(e['flags'].split())==12
 e['flags']=e['flags'].split();e['inspected_unix']=time.time();a['representative'].append(e);a['sheets_actually_inspected'].append(f"sheets/{e['sample_id']}_{e['arm']}.png")
p.write_text(json.dumps(a,indent=2)+'\n')
