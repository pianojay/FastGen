#!/usr/bin/env python3
"""Build an explicitly selected, immutable archival migration in a new staging area."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

TEMPLATE = '739fd3fcafaebca25e69950876c82de83abc8c3a'
GROUPS = {
    'SoFlow': ('cifar-soflow', 'cc47661a0b0843f8f1b6cff3a17107a052d18b21'),
    'FastGen-SoFlow': ('soflow', 'd6ea8dcda6a3d5fe7914d690525b480bc0c0a075'),
    'FastGen-AFM': ('afm', '0057ef090c96132ca6a8a7be24180f2c403e4489'),
    'FastGen-SiD-DiT': ('sid-dit', '0e116ded7b75251c23438deda98b1a33da8da958'),
    'exploration': ('exploration', '3ccf3261495bd144bf844b7f298d10406b53117a'),
    'fastgen_zip': ('peer-sida', None),
}
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.log', '.txt', '.md', '.yaml', '.yml', '.py', '.sh', '.html', '.toml', '.svg', '.upstream'}
IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
NAMED = {'LICENSE', 'COPYING', 'NOTICE', 'ACKNOWLEDGEMENTS', '.gitignore', '.gitattributes'}
PARAMETERS = [
    ('soflow-s3-one_nfe.ckpt', 'FastGen-SoFlow/runs/research_s3_unet192_40k_seed42/export/checkpoints/one_nfe.ckpt', 356696109, 'd03a959d0117a7d34ce3b82929e7b816becd3cd97199e7423cdc92ba299819c5'),
    ('soflow-s3-few_nfe.ckpt', 'FastGen-SoFlow/runs/research_s3_unet192_40k_seed42/export/checkpoints/few_nfe.ckpt', 356696109, 'fc6337a046f6a1272ffa2f0afe92ca1c7b514f642032ecdb83aec99baf5d400b'),
    ('peer-sida-one_nfe.ckpt', 'fastgen_zip/fastgen_sida_1007/checkpoints/one_nfe.ckpt', 247492073, '7134046f1654da72ad3d7905ef1e2762d87b5eaa0840947aece32eac6be958ee'),
    ('peer-sida-few_nfe.ckpt', 'fastgen_zip/fastgen_sida_1007/checkpoints/few_nfe.ckpt', 386443625, '2246bfc0054026a95f7ecc4b1f54c99f896c75b9d514f9f4a24c5653345a5690'),
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def run(*args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), **kwargs).strip()

def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

def collect(root):
    selected = {}
    def add(p, kind):
        if p.is_symlink():
            raise RuntimeError(f'Unexpected symlink: {p}')
        if p.is_file():
            selected[p] = kind
    def scan(base, cb):
        for dp, dirs, files in os.walk(root/base):
            dirs[:] = [d for d in dirs if d not in {'.git', '.cache', '.conda', '__pycache__', 'papers'}]
            for name in files:
                p = Path(dp)/name
                rel = p.relative_to(root)
                ext = p.suffix.lower()
                if ext in TEXT or name in NAMED or (str(rel).startswith('exploration/eda/') and ext == '.gz'):
                    add(p, 'records')
                if ext not in IMAGES:
                    continue
                parts = rel.parts
                keep = False
                if cb == 'soflow':
                    keep = any(x in parts for x in {'report', 'audit'}) and not any(x in parts for x in {'generated', 'reference_dev', 'reference'})
                elif cb == 'sid':
                    keep = p.name.startswith('gallery_fixed') or 'report' in parts
                elif cb == 'exploration':
                    keep = any(x in parts for x in {'reports', 'eda', 'sheets', 'domain_discovery'})
                if keep:
                    add(p, 'visuals')
    for base, cb in [('SoFlow/runs','cifar'), ('FastGen-SoFlow/runs','soflow'), ('FastGen-SoFlow/artifacts','soflow'), ('FastGen-AFM/runs','afm'), ('FastGen-AFM/artifacts','afm'), ('FastGen-SiD-DiT/runs','sid'), ('FastGen-SiD-DiT/artifacts','sid'), ('exploration/eda','exploration'), ('exploration/transfer_aug','exploration')]:
        scan(base, cb)
    for name in ['pokemon_unet128_seed42', 'research_s1_unet128_40k_seed42', 'research_s2_unet160_40k_seed42', 'research_s3_unet192_40k_seed42']:
        s = root/'FastGen-SoFlow/runs'/name/'samples'
        ds = sorted(p for p in s.iterdir() if p.is_dir() and p.name.isdigit())
        for d in {ds[0], ds[-1]} | {p for p in ds if p.name == '020000'}:
            for p in d.glob('*.png'):
                add(p, 'visuals')
        for p in s.glob('*.png'):
            add(p, 'visuals')
    for c in sorted((root/'FastGen-AFM/runs/stability_campaign').glob('candidate*')):
        if not c.is_dir():
            continue
        audit = sorted(p for p in (c/'audits').glob('*') if p.is_dir())[-1]
        classes = json.loads((audit/'visual_review.json').read_text())['reviewed_classes']
        ends = sorted({min(150, (i//10)*10+9) for i in classes})
        for prefix in ['raw', 'ema']:
            for i in ends:
                p = audit/f'{prefix}_{i:03}.png'
                assert p.exists(), p
                add(p, 'visuals')
        for prefix in ['nearest_raw', 'nearest_ema']:
            add(audit/f'{prefix}_019.png', 'visuals')
        panels = sorted((c/'panels').glob('*.png'))
        for p in ([panels[0], panels[-1]] if panels else []):
            add(p, 'visuals')
    panels = sorted((root/'FastGen-AFM/runs/afm_unet128_50k_seed20261001/panels').glob('*.png'))
    for p in ([panels[0], panels[-1]] if panels else []):
        add(p, 'visuals')
    for repo, (_, expected) in GROUPS.items():
        if expected is None:
            continue
        for name in run('git','-C',root/repo,'ls-files','-z').split('\0'):
            if name:
                add(root/repo/name, 'source')
    for p in [root/'SoFlow/AGENTS.md', root/'SoFlow/cifar_gpu0.yaml', root/'exploration/.gitignore']:
        add(p, 'source')
    for p in (root/'SoFlow/scripts').glob('*'):
        if p.is_file():
            add(p, 'source')
    for p in (root/'fastgen_zip/fastgen_sida_1007').rglob('*'):
        if p.is_file() and 'checkpoints' not in p.parts:
            add(p, 'source')
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    root, stage = args.source.resolve(), args.stage.resolve()
    assert root != stage and root not in stage.parents
    baseline = {}
    for name, (_, expected) in GROUPS.items():
        if expected is None:
            continue
        repo = root/name
        actual = run('git','-C',repo,'rev-parse','HEAD')
        assert actual == expected, (name, actual, expected)
        baseline[name] = {
            'head': actual,
            'refs': run('git','-C',repo,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/tags').splitlines(),
            'status': run('git','-C',repo,'status','--porcelain=v1'),
            'remotes': run('git','-C',repo,'remote','-v').splitlines(),
            'shallow': run('git','-C',repo,'rev-parse','--is-shallow-repository') == 'true',
        }
    dump(stage/'records/source-baseline.json', baseline)
    selected = collect(root)
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    rows = []
    for p, kind in sorted(selected.items()):
        rel = p.relative_to(root)
        group = rel.parts[0]
        slug, base = GROUPS[group]
        if p.suffix.lower() in TEXT or p.name in NAMED:
            if secret.search(p.read_bytes()):
                raise RuntimeError(f'Potential credential in selected file (value withheld): {rel}')
        git_path = str(Path(*rel.parts[1:]))
        if group == 'fastgen_zip':
            git_path = str(Path(*rel.parts[2:]))
        git_eligible = kind == 'source' or p.suffix in {'.py','.sh','.md','.toml','.yaml','.yml','.html','.upstream'} or p.name in NAMED or (kind == 'records' and p.stat().st_size <= 262144)
        row = {
            'original_path': str(rel), 'bytes': p.stat().st_size, 'sha256': sha(p),
            'mode': p.stat().st_mode & 0o777, 'kind': kind,
            'provenance': {'original_codebase':group, 'source_head':base, 'peer_copy':group == 'fastgen_zip'},
            'archive_asset': f'{slug}-evidence.tar.gz', 'restored_path':'legacy/'+str(rel),
            'worker_branch':'archive/'+slug, 'git_path':git_path if git_eligible else None,
        }
        rows.append(row)
    param_rows = []
    for asset, rel, size, expected in PARAMETERS:
        p = root/rel
        assert p.stat().st_size == size and sha(p) == expected, rel
        target = stage/'assets'/asset
        shutil.copy2(p, target)
        param_rows.append({'asset':asset,'original_path':rel,'restored_path':'legacy/'+rel,'bytes':size,'sha256':expected,'purpose':'inference reference; no optimizer/resume state'})
    inventory = {
        'schema':1,'project':'FastGen','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_root':str(root),'template_revision':TEMPLATE,'release':'migration-2026-11-04',
        'files':rows,'parameters':param_rows,'original_repositories':baseline,
        'exclusions':[
            'All parameter/optimizer/resume files except the four named inference exports',
            'Datasets, data caches, features, numpy arrays, and per-image generated collections',
            'Intermediate/unreviewed visual panels except the explicitly selected early/midpoint/final samples',
            'Environments, package/download caches, cached paper PDFs, bytecode, and locks',
            'ZIP/distribution duplicates, Git temporary packs, hidden agent/checkpoint refs and reflog-only objects',
            'claude-audit, codex-session exports, machine-python-user-site-plan.json, and moved NVIDIA incident record',
        ],
        'limits':'Historical source/evidence preservation and inference references; no scientific rerun or full training resumability.',
    }
    dump(stage/'records/inventory.json', inventory)
    assets = []
    for group, (slug, _) in GROUPS.items():
        subset = [r for r in rows if r['provenance']['original_codebase'] == group]
        asset = stage/'assets'/f'{slug}-evidence.tar.gz'
        with asset.open('wb') as dest:
            with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0,compresslevel=6,filename='') as gz:
                with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tf:
                    for row in subset:
                        p = root/row['original_path']
                        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
                        ti = tf.gettarinfo(str(p), arcname=row['original_path'])
                        ti.mtime=0; ti.uid=ti.gid=0; ti.uname=ti.gname=''; ti.pax_headers={}
                        with p.open('rb') as f:
                            tf.addfile(ti, f)
        assets.append({'asset':asset.name,'bytes':asset.stat().st_size,'sha256':sha(asset),'files':len(subset),'raw_bytes':sum(r['bytes'] for r in subset),'kind':'evidence'})
        print(json.dumps(assets[-1]), flush=True)
    # Inventory is itself a release asset; its own digest belongs to the outer release manifest.
    shutil.copy2(stage/'records/inventory.json',stage/'assets/inventory.json')
    for p in [stage/'assets/inventory.json', *[stage/'assets'/r['asset'] for r in param_rows]]:
        assets.append({'asset':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'kind':'parameters' if p.suffix == '.ckpt' else 'inventory'})
    manifest = {'schema':1,'repository':'pianojay/FastGen','tag':'migration-2026-11-04','template_revision':TEMPLATE,'assets':assets,'parameter_bytes':sum(r['bytes'] for r in param_rows),'total_listed_asset_bytes':sum(a['bytes'] for a in assets),'complete':False}
    dump(stage/'assets/release-manifest.json',manifest)
    (stage/'assets/SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['asset']}\n" for a in assets)+f"{sha(stage/'assets/release-manifest.json')}  release-manifest.json\n")
    print(json.dumps({'files':len(rows),'raw_bytes':sum(r['bytes'] for r in rows),'parameter_bytes':manifest['parameter_bytes'],'asset_bytes':manifest['total_listed_asset_bytes']}),flush=True)

if __name__ == '__main__':
    main()
