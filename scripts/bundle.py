"""Create checked source and artifact archives for SFTP, excluding the raw dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from soflow_fastgen.common import ROOT, atomic_json, digest, verify_protected


def bundle(config, output):
    verify_protected()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    source = sorted({ROOT/name for name in names if name and (ROOT/name).is_file()})
    run = ROOT/config['run_dir']
    artifacts = set()
    for path in (ROOT/'artifacts').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            artifacts.add(path)
    for path in run.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if relative.parts[0] == 'reference_dev' or path.suffix == '.lock':
            continue
        if relative.parts[0] == 'evaluations' and path.suffix != '.json':
            continue
        if relative.parts[0] == 'checkpoints' and path.name != 'latest.pt':
            continue
        if path.name.endswith('.tmp') or '__pycache__' in path.parts:
            continue
        artifacts.add(path)
    metadata = {
        'run_dir': config['run_dir'],
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'source_sha256': {str(p.relative_to(ROOT)):digest(p) for p in source},
        'note': 'A live-run bundle is a progress snapshot. The terminal workflow refreshes it after evaluation and export.',
    }
    atomic_json(output/'bundle_metadata.json',metadata)
    for name, paths in (('source.tar.gz',source),('artifacts.tar.gz',sorted(artifacts))):
        target = output/name
        temporary = target.with_name(name+'.tmp')
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
            for path in paths:
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        temporary.replace(target)
    names=['source.tar.gz','artifacts.tar.gz','bundle_metadata.json']
    (output/'SHA256SUMS').write_text(''.join(f'{digest(output/name)}  {name}\n' for name in names))
    return {'output':str(output),'source_files':len(source),'artifact_files':len(artifacts)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pokemon_local.json')
    parser.add_argument('--output',default='dist/migration')
    args=parser.parse_args()
    print(json.dumps(bundle(json.loads(Path(args.config).read_text()),args.output)))
