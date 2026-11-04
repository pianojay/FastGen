#!/usr/bin/env python3
"""Restore and verify the explicitly retained FastGen files from private GitHub releases."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import html
from pathlib import Path, PurePosixPath
import shutil
import re
import subprocess
import tarfile
import time

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def safe(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe archive path: {relative}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes restore directory: {relative}')
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository', default='pianojay/FastGen')
    parser.add_argument('--tag', default='migration-2026-11-04')
    args = parser.parse_args()
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    manifest_bytes = (repo/'migration/release-manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if args.repository != manifest['repository'] or args.tag != manifest['tag']:
        raise ValueError('Repository/tag must match the committed release manifest')
    out = args.output.resolve()
    if out == repo or repo.is_relative_to(out) or out.is_relative_to(repo):
        raise ValueError('Use a restore output separate from the Git checkout')
    downloads = out/'.downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    assets = manifest['assets']
    needed = [a['asset'] for a in assets] + ['release-manifest.json', 'SHA256SUMS']
    missing = [name for name in needed if not (downloads/name).exists()]
    if missing:
        command = ['gh','release','download',args.tag,'--repo',args.repository,'--dir',str(downloads)]
        for name in missing:
            command += ['--pattern', name]
        subprocess.run(command, check=True)
    if (downloads/'release-manifest.json').read_bytes() != manifest_bytes:
        raise ValueError('Downloaded release manifest differs from committed manifest')
    for a in assets:
        p = downloads/a['asset']
        if p.stat().st_size != a['bytes'] or sha(p) != a['sha256']:
            raise ValueError(f"Release asset checksum/size mismatch: {a['asset']}")
    checksums = {}
    for line in (downloads/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        checksums[name] = digest
    for name in needed:
        if name != 'SHA256SUMS' and checksums.get(name) != sha(downloads/name):
            raise ValueError(f'Checksum list mismatch: {name}')
    inventory = json.loads((downloads/'inventory.json').read_text())
    if (repo/'migration/inventory.json').read_bytes() != (downloads/'inventory.json').read_bytes():
        raise ValueError('Downloaded inventory differs from committed inventory')
    expected = {r['original_path']:r for r in inventory['files']}
    extracted = set()
    legacy = out/'legacy'
    legacy.mkdir(exist_ok=True)
    for a in assets:
        if a['kind'] != 'evidence':
            continue
        with tarfile.open(downloads/a['asset'], 'r:gz') as tf:
            for member in tf:
                row = expected.get(member.name)
                if not member.isfile() or row is None or row['archive_asset'] != a['asset']:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                if member.name in extracted or member.size != row['bytes']:
                    raise ValueError(f'Duplicate/incorrect archive member: {member.name}')
                target = safe(legacy, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                        raise ValueError(f'Refusing to overwrite differing restored file: {target}')
                else:
                    with tf.extractfile(member) as src, target.open('xb') as dest:
                        shutil.copyfileobj(src, dest, length=1024*1024)
                    target.chmod(row['mode'])
                extracted.add(member.name)
        print(f"Extracted and checked archive structure: {a['asset']}", flush=True)
    if extracted != set(expected):
        raise ValueError('Archive set does not cover the complete inventory')
    for row in inventory['files']:
        p = safe(out, row['restored_path'])
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError(f"Restored file checksum mismatch: {row['original_path']}")
    for row in inventory['parameters']:
        target = safe(out, row['restored_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != row['sha256']:
                raise ValueError(f'Refusing to overwrite a differing checkpoint: {target}')
        else:
            shutil.copy2(downloads/row['asset'], target)
        if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
            raise ValueError(f"Restored parameter mismatch: {row['asset']}")
    # Keep historical documents byte-identical; provide a separate portable availability index.
    all_rows = inventory['files'] + inventory['parameters']
    available = {r['original_path'] for r in all_rows}
    links = []
    for row in inventory['files']:
        original = PurePosixPath(row['original_path'])
        if original.suffix.lower() not in {'.md','.html'}:
            continue
        text = safe(out,row['restored_path']).read_text(errors='replace')
        targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text) + re.findall(r'(?:src|href)=[\"\x27]([^\"\x27]+)[\"\x27]',text)
        for target in targets:
            target = target.split('#',1)[0].split('?',1)[0].strip('<>')
            if not target or target.startswith(('http:','https:','mailto:','data:','javascript:')):
                continue
            if target.startswith(inventory['source_root']+'/'):
                relative = target[len(inventory['source_root'])+1:]
            elif target.startswith('/'):
                relative = target
            else:
                relative = os.path.normpath(str(original.parent/target))
            links.append({'referrer':row['original_path'],'target':target,'resolved_original_path':relative,'availability':'retained' if relative in available else 'not retained / historical external or omitted reference'})
    (out/'reference-availability.json').write_text(json.dumps({'scope':'Markdown/HTML links only; dynamic/JSON proof references are resolved through inventory','references':links},indent=2)+'\n')
    body = ['<!doctype html><meta charset="utf-8"><title>FastGen restored archive</title>', '<h1>FastGen restored archive</h1><p>Historical files remain unchanged. This index links only retained files. See reference-availability.json for omitted historical links. No training resume state is retained.</p>']
    for group in sorted({r['original_path'].split('/')[0] for r in all_rows}):
        body.append('<h2>'+html.escape(group)+'</h2><ul>')
        for row in all_rows:
            if row['original_path'].split('/')[0] == group:
                body.append('<li><a href="'+html.escape(row['restored_path'],quote=True)+'">'+html.escape(row['original_path'])+'</a> ('+str(row['bytes'])+' bytes)</li>')
        body.append('</ul>')
    (out/'INDEX.html').write_text('\n'.join(body)+'\n')
    report = {
        'state':'pass', 'source':'authenticated GitHub release downloads; no original-directory copies',
        'repository':args.repository,'tag':args.tag,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'restored_root':str(out),'files_verified':len(expected),'parameters_verified':len(inventory['parameters']),
        'downloaded_bytes':sum((downloads/n).stat().st_size for n in needed),
        'evidence_raw_bytes':sum(r['bytes'] for r in inventory['files']),
        'parameter_bytes':sum(r['bytes'] for r in inventory['parameters']),
        'seconds':time.monotonic()-start,
        'limits':'Byte recovery only. Inference verification is separate; no training resume state retained.',
    }
    (out/'restore-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
