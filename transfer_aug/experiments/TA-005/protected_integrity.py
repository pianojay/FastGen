import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;workspace=ROOT.parents[2]
expected=json.loads((ROOT.parents[1]/'shared/cohort_v1/manifest.json').read_text())['protected_inputs'];actual={}
for path,digest in expected.items():
    h=hashlib.sha256()
    with (workspace/path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    actual[path]=h.hexdigest();assert actual[path]==digest,path
(ROOT/('protected_integrity_'+sys.argv[1]+'.json')).write_text(json.dumps(dict(all_match=True,files=actual),indent=2)+'\n')
print('Protected integrity passed:',len(actual))
