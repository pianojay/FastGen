from pathlib import Path
import json,hashlib
p=Path('/tmp/soflow-recovery-test')
s=json.loads((p/'state.json').read_text());cp=json.loads((p/'checkpoint.json').read_text());cfg=json.loads((p/'config.json').read_text());ev=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()];starts=[e for e in ev if e['event']=='start'];stops=[e for e in ev if e['event']=='stop']
checks={
'duplicate_rejected':json.loads((p/'duplicate-check.json').read_text())['exit_code']==3,
'exactly_two_attempts':len(starts)==2 and s['attempts']==2,
'recovered_checkpoint_five':starts[-1]['from']==5,
'configuration_preserved':all(e['config_sha256']==s['config_sha256']==hashlib.sha256((p/'config.json').read_bytes()).hexdigest() for e in starts),
'remaining_allowance_preserved':abs(starts[-1]['remaining_seconds']-(cfg['cap_seconds']-stops[0]['elapsed_seconds']))<1e-6,
'finished_within_cap':s['status']=='completed' and 0<s['elapsed_seconds']<cfg['cap_seconds'],
'correct_final_state':cp['step']==20 and cp['sum']==210,
}
result={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'elapsed_seconds':s['elapsed_seconds'],'first_attempt_seconds':stops[0]['elapsed_seconds'],'recovery_allowance_seconds':starts[-1]['remaining_seconds']}
(p/'check-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert all(checks.values())
