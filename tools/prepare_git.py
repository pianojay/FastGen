#!/usr/bin/env python3
"""Import ordinary histories and selected source/records into isolated archival worktrees."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

TEMPLATE='739fd3fcafaebca25e69950876c82de83abc8c3a'
GROUPS={'SoFlow':'cifar-soflow','FastGen-SoFlow':'soflow','FastGen-AFM':'afm','FastGen-SiD-DiT':'sid-dit','exploration':'exploration','fastgen_zip':'peer-sida'}
REPORTS={
    'cifar-soflow':('Historical CIFAR feasibility experiment','Stopped; finite early optimization, no image-quality evaluation. Last logged step 130; recoverable main checkpoint 5. No CIFAR weights retained.','https://github.com/pianojay/FastGen/blob/archive/soflow/docs/soflow-cifar10/EXPERIMENT_REPORT.md'),
    'soflow':('Historical FastGen SoFlow campaigns','S3 remains the retained local incumbent: confirmation development FID 66.9684 / 25.0072 at one/four calls. The later matched adversarial pilot did not improve S3. No official challenge score or training-seed replication is claimed.','docs/RESEARCH_LOG.md'),
    'afm':('Historical teacher-free AFM studies','Baseline stopped; six-recipe stability campaign completed with no qualifying recipe, no confirmation and no trained-model FID benchmark.','docs/STABILITY_REPORT.md'),
    'sid-dit':('Historical pixel-flow SiD/SiDA pilot','Teacher and eight students completed. SiDA improved the tested one-call arm but degraded the two-/three-/four-call comparisons. One training realization per arm.','docs/RESULTS.md'),
    'peer-sida':('Peer-supplied EDM/SiDA implementation','Externally supplied source/checkpoint copy. README method and training claims remain attributed to the peer; original training logs and write-up were not supplied. Migration validates bytes and CPU inference only.','README.md'),
    'exploration':('Independent EDA and transfer-study session','Related independent master-worker study, not retrospectively subordinate to the FastGen coordinator. EDA and six transfer studies completed; no useful SiDA augmentation or generative FID improvement was established.','transfer_aug/MASTER.md'),
}

def git(*args, input=None):
    return subprocess.check_output(['git',*[str(x) for x in args]],input=input,text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')).strip()

def write(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content)

def dump(path,data):
    write(path,json.dumps(data,indent=2)+'\n')

def stage_files(worktree,files):
    subprocess.run(['git','-C',str(worktree),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input='\0'.join(sorted(set(files)))+'\0',text=True,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--stage',type=Path,required=True);a=parser.parse_args()
    source,stage=a.source.resolve(),a.stage.resolve();master=stage/'master';inventory=json.loads((stage/'records/inventory.json').read_text())
    bare=master/'workers/FastGen/.FastGen.git';bare.parent.mkdir(parents=True,exist_ok=True)
    git('init','--bare','--initial-branch=main',bare)
    git('--git-dir',bare,'config','remote.origin.url','https://github.com/pianojay/FastGen.git')
    git('--git-dir',bare,'config','remote.origin.fetch','+refs/heads/*:refs/remotes/origin/*')
    original_refs={}
    for group in ['FastGen-SoFlow','FastGen-AFM','FastGen-SiD-DiT','exploration']:
        slug=GROUPS[group]
        git('--git-dir',bare,'fetch','--no-tags',source/group,f'refs/heads/*:refs/heads/history/{slug}/*')
        for row in inventory['original_repositories'][group]['refs']:
            ref,sha=row.split()
            if ref.startswith('refs/heads/'):
                mapped='refs/heads/history/'+slug+'/'+ref.removeprefix('refs/heads/')
                assert git('--git-dir',bare,'rev-parse',mapped)==sha
                original_refs[mapped]=sha
    empty_tree=git('--git-dir',bare,'mktree',input='')
    root_commit=git('--git-dir',bare,'commit-tree',empty_tree,'-m','Initialize private FastGen archival navigation')
    git('--git-dir',bare,'update-ref','refs/heads/main',root_commit)
    index=bare.parent/'index';git('--git-dir',bare,'worktree','add',index,'main')
    expected={}
    summaries=[]
    for group,slug in GROUPS.items():
        head=inventory['original_repositories'].get(group,{}).get('head')
        base=head if group not in {'SoFlow','fastgen_zip'} else (inventory['original_repositories']['exploration']['head'] if group=='fastgen_zip' else root_commit)
        worktree=bare.parent/slug;branch='archive/'+slug
        git('--git-dir',bare,'worktree','add','-b',branch,worktree,base)
        files=[];selected=[]
        for row in inventory['files']:
            if row['provenance']['original_codebase']!=group or row['git_path'] is None:
                continue
            src=source/row['original_path'];dest=worktree/row['git_path'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest);files.append(row['git_path']);selected.append({'original_path':row['original_path'],'git_path':row['git_path'],'sha256':row['sha256'],'bytes':row['bytes']})
        title,outcome,report=REPORTS[slug]
        write(worktree/'ARCHIVE.md',f'''# {title}

Archived for migration on November 4, 2026. This branch preserves historical source and compact records; it is not an active assignment.

{outcome}

The historical AGENTS.md, program files, reports, source and configuration retain their original bytes. Their historical resource assignments, launch text and publication permissions do not authorize new work. A future experiment needs a new assignment and host-specific qualification.

- Owning coordination record: [master_root / FastGen](https://github.com/pianojay/master_root/tree/FastGen).
- Historical report: [{report}]({report}).
- Full retained evidence: private release [migration-2026-11-04](https://github.com/pianojay/FastGen/releases/tag/migration-2026-11-04), asset `{slug}-evidence.tar.gz`.
- Restore with `tools/restore.py` from worker main. The inventory maps original paths to restored paths and marks deliberately excluded bulk artifacts.
- Original directory: `{source/group}`. Its deletion is a separate user action.
- Source base: `{head or 'peer source copy; original Git history unavailable'}`. The CIFAR snapshot does not preserve full upstream ancestry.

No dataset images, optimizer/resume states, or unselected parameter files are preserved by this branch. The four inference exports are release assets, not Git blobs.
''')
        dump(worktree/'migration/source-map.json',{'schema':1,'group':group,'archive_branch':branch,'source_head':head,'files':selected,'outcome':outcome,'evidence_review':'Historical report checked; migration hashes/restore verified separately. No scientific rerun.'})
        files+=['ARCHIVE.md','migration/source-map.json']
        if group=='SoFlow':
            patch=git('-C',source/group,'diff','--binary','HEAD')+'\n'
            write(worktree/'migration/original-local.patch',patch);files.append('migration/original-local.patch')
        stage_files(worktree,files)
        git('-C',worktree,'commit','-m',f'Preserve {slug} source and compact historical records for migration')
        commit=git('-C',worktree,'rev-parse','HEAD');expected['refs/heads/'+branch]=commit
        summaries.append({'group':group,'slug':slug,'branch':branch,'commit':commit,'source_head':head,'worktree':str(worktree),'git_files':len(selected),'outcome':outcome,'report':report})
        print(json.dumps(summaries[-1]),flush=True)
    dump(stage/'records/worker-branches.json',summaries)
    dump(stage/'records/expected-refs.json',{'historical':original_refs,'archives':expected})
    # The manifest is a frozen asset inventory, not a claim that verification already happened.
    manifest=json.loads((stage/'assets/release-manifest.json').read_text());manifest.pop('complete',None)
    manifest['state']='frozen asset inventory; publication and restore verification are recorded separately'
    dump(stage/'assets/release-manifest.json',manifest)
    digest=hashlib.sha256((stage/'assets/release-manifest.json').read_bytes()).hexdigest()
    sums=''.join(f"{x['sha256']}  {x['asset']}\n" for x in manifest['assets'])+f'{digest}  release-manifest.json\n'
    write(stage/'assets/SHA256SUMS',sums)
    for name in ['inventory.json','release-manifest.json','SHA256SUMS']:
        dest=index/'migration'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(stage/'assets'/name,dest)
    for name in ['source-baseline.json','expected-refs.json','worker-branches.json']:
        shutil.copy2(stage/'records'/name,index/'migration'/name)
    for name in ['restore.py','verify_inference.py','build_archive.py','prepare_git.py']:
        dest=index/'tools'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(stage/'scripts'/name,dest)
    dump(stage/'records/git-layout.json',{'bare':str(bare),'index':str(index),'master':str(master),'original_refs':original_refs,'archive_refs':expected})

if __name__=='__main__':main()
