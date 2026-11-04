#!/usr/bin/env python3
"""Read-only source stability and optional host process check before retirement readiness."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')).strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--processes',action='store_true');a=p.parse_args();root=a.source.resolve();inv=json.loads(a.inventory.read_text());problems=[]
    for row in inv['files']+inv['parameters']:
        q=root/row['original_path']
        if not q.is_file() or q.stat().st_size!=row['bytes'] or sha(q)!=row['sha256']:problems.append(row['original_path'])
    for name,baseline in inv['original_repositories'].items():
        q=root/name
        if git(q,'rev-parse','HEAD')!=baseline['head'] or git(q,'status','--porcelain=v1')!=baseline['status'] or git(q,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/tags').splitlines()!=baseline['refs']:problems.append('git:'+name)
    processes=[];process_scope='not requested'
    if a.processes:
        process_scope='Read /proc on the execution host; matches original-root cwd/arguments. No signal or GPU operation.'
        for q in Path('/proc').iterdir():
            if not q.name.isdigit() or int(q.name)==os.getpid():continue
            try:
                raw=(q/'cmdline').read_bytes().split(b'\0');argv=[x.decode(errors='replace') for x in raw if x];comm=(q/'comm').read_text().strip();cwd=Path(os.readlink(q/'cwd'))
                relevant=cwd==root or root in cwd.parents or any(x.startswith(str(root)+'/') for x in argv)
                if relevant and comm in {'python','python3','python3.10','torchrun'}:
                    processes.append({'pid':int(q.name),'command':comm,'cwd':str(cwd),'program_arguments':[x for x in argv[:3] if len(x)<300]})
            except (OSError,PermissionError,ProcessLookupError):pass
    result={'state':'pass' if not problems else 'fail','checked_utc':datetime.now(timezone.utc).isoformat(),'retained_files_checked':len(inv['files']),'parameter_files_checked':len(inv['parameters']),'original_git_repositories_checked':len(inv['original_repositories']),'mismatches':problems,'matching_python_processes':processes,'process_scope':process_scope,'limits':'No original files/refs modified. Historical CUDA release records are not treated as a current GPU monitoring result.'}
    a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if problems:raise SystemExit(1)
if __name__=='__main__':main()
