from analyze import OUT, sha, write_json
import json
u=json.loads((OUT/'resource_usage.json').read_text());u['accounting_note']='Measured extraction+regeneration+verification+latest finalization total excludes two earlier brief finalization passes and shell/image-tool overhead. Conservative charged CPU allowance is 120 seconds for all local work; below 3600-second contract cap.';u['conservative_charged_cpu_seconds']=120;write_json(OUT/'resource_usage.json',u)
a=json.loads((OUT/'artifact_manifest.json').read_text());a['files']={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'};write_json(OUT/'artifact_manifest.json',a)
assert all(sha(OUT/n)==h for n,h in a['files'].items())
r=json.loads((OUT/'result.json').read_text());assert sha(OUT/'domain_manifest.json')==r['candidate_manifest_sha256'];assert len((OUT/'SUMMARY.md').read_text().split())<=500
assert all((OUT.parents[2]/path).exists() for path in r['proof_paths'].values())
print(json.dumps({'all_artifact_hashes_verified':True,'artifact_manifest_sha256':sha(OUT/'artifact_manifest.json'),'candidate_manifest_sha256':r['candidate_manifest_sha256'],'summary_word_count':len((OUT/'SUMMARY.md').read_text().split())}))
