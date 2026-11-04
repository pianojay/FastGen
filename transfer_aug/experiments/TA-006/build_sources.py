from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
S=R.parents[1]/'audit_sources/starganv2/core'
a=(S/'model.py').read_text().replace('from core.wing import FAN','')
a=a[:a.index('\ndef build_model(args):')]
(R/'model.py').write_text(a)
b=(S/'solver.py').read_text();b=b[:b.index('import os')]+'from munch import Munch\nimport torch\nimport torch.nn.functional as F\n\n'+b[b.index('def compute_d_loss'):]
(R/'losses.py').write_text(b)
(R/'source_adaptation.json').write_text(json.dumps({'commit':'875b70a150609e8a678ed8482562e7074cdce7e5','changes':['Model classes copied verbatim; removed FAN import and upstream build_model factory.','Loss functions and moving_average copied verbatim; removed runtime imports/Solver which invoke pretrained evaluation.'],'source_sha256':{str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('*.py')}},indent=2))
