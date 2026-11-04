import concurrent.futures as cf
import json
import hashlib
from pathlib import Path
import requests
import subprocess

ROOT=Path(__file__).resolve().parent/'audit_sources'
m=json.loads((ROOT/'manifest.json').read_text())
def tree(key):
    r=m[key]; repo=r['repository'].removeprefix('https://github.com/')
    v=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{r["revision"]}?recursive=1',timeout=30)
    v.raise_for_status()
    paths=[x['path'] for x in v.json()['tree'] if x['type']=='blob']
    (ROOT/key/'tree.json').write_text(json.dumps(paths,indent=2))
    wanted=[]
    for p in paths:
        if p.lower().startswith(('license','licence','copying')): wanted.append(p)
        if key=='mixstyle' and (p.endswith('mixstyle.py') or p.endswith('.yaml')): wanted.append(p)
        if key=='color' and p.endswith('.py'): wanted.append(p)
        if key=='wct' and (p.endswith('.lua') and '/' not in p): wanted.append(p)
        if key=='cut' and p=='options/base_options.py': wanted.append(p)
    for p in sorted(set(wanted)):
        if (ROOT/key/p).exists(): continue
        url=f'https://raw.githubusercontent.com/{repo}/{r["revision"]}/{p}'
        data=requests.get(url,timeout=30);data.raise_for_status()
        dest=ROOT/key/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data.content)
        r['files'][p]={'url':url,'status':200,'sha256':hashlib.sha256(data.content).hexdigest(),'bytes':len(data.content)}
    (ROOT/key/'provenance.json').write_text(json.dumps(r,indent=2))
    return key,len(wanted)

with cf.ThreadPoolExecutor(max_workers=4) as pool:
    for key,n in pool.map(tree,['mixstyle','color','wct','cut','diffaugment','fda']): print(key,n,flush=True)
(ROOT/'manifest.json').write_text(json.dumps(m,indent=2))

papers={
 'adain':'https://arxiv.org/pdf/1703.06868',
 'cyclegan':'https://junyanz.github.io/CycleGAN/paper/CycleGAN.pdf',
 'starganv2':'https://arxiv.org/pdf/1912.01865',
 'cut':'https://arxiv.org/pdf/2007.15651',
 'wct':'https://arxiv.org/pdf/1705.08086',
 'gatys':'https://arxiv.org/pdf/1508.06576',
 'munit':'https://arxiv.org/pdf/1804.04732',
 'diffaugment':'https://arxiv.org/pdf/2006.10738',
 'reinhard':'https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf',
}
def paper(item):
    key,url=item;p=ROOT/'papers';p.mkdir(exist_ok=True)
    r=requests.get(url,timeout=60);r.raise_for_status()
    assert r.content.startswith(b'%PDF'),key
    (p/(key+'.pdf')).write_bytes(r.content)
    subprocess.run(['pdftotext','-layout',str(p/(key+'.pdf')),str(p/(key+'.txt'))],check=True)
    (p/(key+'.json')).write_text(json.dumps({'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content)},indent=2))
    return key,len(r.content)
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    pending={pool.submit(paper,item):item[0] for item in papers.items()}
    for f in cf.as_completed(pending):
        try: print('paper',f.result(),flush=True)
        except Exception as e: print('paper error',pending[f],str(e),flush=True)
