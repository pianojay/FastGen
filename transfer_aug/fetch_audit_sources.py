"""Cache small public source files for inspection; never execute upstream code."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent / 'audit_sources'
ROOT.mkdir(exist_ok=True)
REPOS = {
 'color': ('jrosebr1/color_transfer', ['README.md','LICENSE','color_transfer/color_transfer.py']),
 'fda': ('YanchaoYang/FDA', ['README.md','LICENSE','utils/__init__.py','FDA_demo.py','train.py']),
 'mixstyle': ('KaiyangZhou/mixstyle-release', ['README.md','LICENSE','mixstyle.py','imcls/README.md']),
 'adain_original': ('xunhuang1995/AdaIN-style', ['README.md','LICENSE','train.lua','models/decoder.lua']),
 'adain_pytorch': ('naoto0804/pytorch-AdaIN', ['README.md','LICENSE','net.py','train.py']),
 'wct': ('Yijunmaverick/UniversalStyleTransfer', ['README.md','LICENSE','models.py','util.py','WCT.py']),
 'cyclegan': ('junyanz/pytorch-CycleGAN-and-pix2pix', ['README.md','LICENSE','models/networks.py','models/cycle_gan_model.py']),
 'cut': ('taesungp/contrastive-unpaired-translation', ['README.md','LICENSE','models/networks.py','models/cut_model.py','models/patchnce.py']),
 'munit': ('NVlabs/MUNIT', ['README.md','LICENSE.md','networks.py','trainer.py','configs/edges2handbags_folder.yaml','configs/afhq.yaml']),
 'starganv2': ('clovaai/stargan-v2', ['README.md','LICENSE','core/model.py','core/solver.py','main.py']),
 'diffaugment': ('mit-han-lab/data-efficient-gans', ['README.md','LICENSE','DiffAugment_pytorch.py']),
 'gatys': ('leongatys/PytorchNeuralStyleTransfer', ['README.md','LICENSE','NeuralStyleTransfer.ipynb']),
}

def fetch_repo(item):
    key, (repo, files) = item
    folder = ROOT / key
    folder.mkdir(exist_ok=True)
    meta = requests.get(f'https://api.github.com/repos/{repo}', timeout=30)
    meta.raise_for_status()
    m = meta.json()
    head = requests.get(f'https://api.github.com/repos/{repo}/commits/{m["default_branch"]}', timeout=30)
    head.raise_for_status()
    sha = head.json()['sha']
    result = {'repository':m['html_url'], 'revision':sha, 'license':m.get('license'), 'files':{}}
    for name in files:
        url=f'https://raw.githubusercontent.com/{repo}/{sha}/{name}'
        r=requests.get(url, timeout=30)
        info={'url':url,'status':r.status_code}
        if r.ok:
            p=folder/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(r.content)
            info.update(bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        result['files'][name]=info
    (folder/'provenance.json').write_text(json.dumps(result,indent=2))
    return key,result

results={}
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    pending={pool.submit(fetch_repo,item):item[0] for item in REPOS.items()}
    for future in cf.as_completed(pending):
        try:
            key,result=future.result()
            results[key]=result
            print(key,result['revision'],[f for f,v in result['files'].items() if v['status']!=200],flush=True)
        except Exception as e:
            results[pending[future]]={'error':str(e)}
            print(pending[future],str(e),flush=True)
(ROOT/'manifest.json').write_text(json.dumps(results,indent=2))
