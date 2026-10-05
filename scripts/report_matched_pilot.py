"""CPU-only fixed pilot reporting. Consumes completed evidence; never trains."""
import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from soflow_fastgen.common import ROOT,atomic_json,atomic_save,digest
from soflow_fastgen.data import datasets


def features(images):
    vectors=[]
    for size in (16,32,64):
        values=np.stack([np.asarray(x.resize((size,size),Image.Resampling.BOX),dtype=np.float32).reshape(-1)/255 for x in images])
        vectors.append(torch.from_numpy(values)/math.sqrt(values.shape[1]*3))
    return torch.cat(vectors,dim=1)


def distances(a,b):
    return (a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)


def report(run):
    started=time.monotonic();torch.set_num_threads(4)
    rows=json.loads((run/'evaluate/results.json').read_text());ledger=json.loads((run/'ledger.json').read_text())
    if len(rows)!=18 or any(r['state']!='completed' for r in {j['name']:j for j in ledger['jobs']}.values()):raise RuntimeError('Only complete pilot report here')
    out=run/'report';out.mkdir(exist_ok=True)
    exports=[]
    for arm in ('A','C'):
        state=torch.load(run/arm/'ema_0512.pt',map_location='cpu',weights_only=False)
        target=run/arm/'export';target.mkdir(exist_ok=True)
        atomic_json(target/'inference_config.json',state['architecture'])
        weights={'backbone.'+k:v for k,v in state['weights'].items()}
        for mode,nfe in (('one_nfe',1),('few_nfe',4)):
            path=target/(mode+'.ckpt')
            atomic_save(path,{'state_dict':dict(weights,sampling_nfe=torch.tensor(nfe))})
            p=subprocess.run([sys.executable,'-m','scripts.smoke_adversarial','--load-only',str(path),
                '--architecture',json.dumps(state['architecture']),'--mode',mode],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'),check=True,capture_output=True,text=True,timeout=120)
            exports.append(dict(arm=arm,path=str(path),sha256=digest(path),**json.loads(p.stdout.strip().splitlines()[-1])))
        del state,weights
    atomic_json(out/'exports.json',exports)
    cfg=json.loads((ROOT/'configs/research_s3_train.json').read_text());train,_,mapping=datasets(cfg)
    names=[name for name,i in sorted(mapping.items(),key=lambda p:p[1])]
    groups=[[] for _ in range(151)]
    for pos,original in enumerate(train.indices):groups[train.dataset.category_ids[original]].append(pos)
    def row(arm,seed=7301,nfe=1,kind='ema',step=512):
        return next(r for r in rows if r['arm']==arm and r['seed']==seed and r['nfe']==nfe and r['weight_source']==kind and (arm=='S3' or r['step']==step))
    diagnostics=[]
    for c in range(151):
        if time.monotonic()-started>=3600:raise TimeoutError('CPU report cap')
        positions=groups[c]
        # Match actual dataset resize/preprocessing, not an alternative image transform.
        opt=[]
        for p in positions:
            tensor,_=train[p]
            from torchvision.transforms.functional import to_pil_image
            opt.append(to_pil_image(tensor.add(1).div(2).clamp(0,1)).convert('RGB'))
        (out/'references').mkdir(exist_ok=True);opt[0].save(out/'references'/f'{c:03d}.png')
        real=features(opt);realflip=features([x.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for x in opt])
        category={'category':c,'name':names[c],'optimization_count':len(positions),'arms':{}}
        for arm in ('S3','C','A'):
            directory=Path(row(arm)['output_dir']);paths=[directory/f'{c:04d}_{j:02d}.png' for j in range(20)]
            imgs=[Image.open(p).convert('RGB') for p in paths];f=features(imgs)
            pair=distances(f,f);upper=torch.triu_indices(20,20,offset=1);values=pair[upper[0],upper[1]]
            normal=distances(f,real);flipped=distances(f,realflip)
            nearest=torch.minimum(normal,flipped);nv,ni=nearest.min(1)
            order=torch.argsort(values)[:5].tolist();pairs=[dict(left=int(upper[0,k]),right=int(upper[1,k]),mse=float(values[k])) for k in order]
            match=[]
            for j in range(20):
                p=positions[int(ni[j])];original=train.indices[p]
                match.append(dict(draw=j,mse=float(nv[j]),reference_hflip=bool(flipped[j,ni[j]]<normal[j,ni[j]]),optimization_position=p,path=train.dataset.paths[original]))
            nearest_dir=out/'nearest'/arm;nearest_dir.mkdir(parents=True,exist_ok=True)
            closest=int(nv.argmin());neighbor=opt[int(ni[closest])]
            if match[closest]['reference_hflip']:neighbor=neighbor.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            neighbor.save(nearest_dir/f'{c:03d}.png')
            exact=[digest(p) for p in paths]
            category['arms'][arm]=dict(pairwise_mean=float(values.mean()),pairwise_quantiles=torch.quantile(values,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
                exact_duplicate_count=20-len(set(exact)),least_diverse_pairs=pairs,nearest=match,
                nearest_median=float(nv.median()),closest_draw=int(nv.argmin()))
        diagnostics.append(category)
        if c%25==0:print(json.dumps({'report_class':c,'seconds':time.monotonic()-started}),flush=True)
    atomic_json(out/'diversity.json',dict(seed=7301,classes=diagnostics,metric='mean 16/32/64 RGB MSE, pixels0..1; nearest includes reference horizontal flip',
        neighbor_pool='same-class full optimization partition, including D diagnostic holdout; no development/official validation',semantic_quality_established=False))
    # Contact sheets use matched draws; all151 classes are available for review.
    for start in range(0,151,25):
        canvas=Image.new('RGB',(1320,72*min(25,151-start)+30),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),'Class/reference | S3 draws0..5 | C draws0..5 | A draws0..5',fill='black')
        for slot,c in enumerate(range(start,min(start+25,151))):
            y=30+slot*72;draw.text((2,y),f'{c}: {names[c][:13]}',fill='black')
            ref,_=train[groups[c][0]]
            from torchvision.transforms.functional import to_pil_image
            canvas.paste(to_pil_image(ref.add(1).div(2).clamp(0,1)).resize((48,48)),(40,y+15))
            for ai,arm in enumerate(('S3','C','A')):
                directory=Path(row(arm)['output_dir'])
                for j in range(6):canvas.paste(Image.open(directory/f'{c:04d}_{j:02d}.png'),(120+ai*396+j*65,y))
        canvas.save(out/f'overview_{start:03d}.jpg',quality=92)
    relative=lambda p:Path(p).relative_to(run).as_posix()
    records=[dict(r,relative_dir='../'+relative(r['output_dir'])) for r in rows]
    options=''.join(f'<option value="{i}">{i}: {html.escape(name)}</option>' for i,name in enumerate(names))
    template='''<!doctype html><meta charset="utf-8"><title>Matched SoFlow pilot</title><style>body{font:15px system-ui;background:#171a20;color:#eee;max-width:1450px;margin:20px auto}.toolbar{position:sticky;top:0;background:#171a20;padding:10px;display:flex;align-items:center;gap:16px}img{width:64px;height:64px}.samples{display:grid;grid-template-columns:repeat(20,1fr);gap:3px}select{font-size:17px}h3{margin-bottom:5px}a{color:#aaddff}</style><h1>Matched SoFlow adversarial pilot</h1><p><a href="training.png">Training curves: raw traces and trailing32 medians</a></p><p>S3 reference, consistency-only C, adversarial A. For a given seed, each column uses the same noise across model/checkpoint rows; columns are independent draws. No training-seed replication.</p><div class="toolbar"><img id="reference"><select id="category">OPTIONS</select><select id="seed"><option>7301</option><option>7302</option><option>7303</option></select><select id="stage"><option value="final">Final EMA</option><option value="trajectory">Trajectory / raw (seed7301)</option></select></div><div id="rows"></div><script>const records=RECORDS,diag=DIAG;function show(){let c=+document.querySelector('#category').value,s=+document.querySelector('#seed').value,stage=document.querySelector('#stage').value;let rs=records.filter(r=>r.nfe===1&&(stage==='final'?r.seed===s&&r.weight_source==='ema'&&(r.arm==='S3'||r.step===512):r.seed===7301));let d=diag[c];document.querySelector('#reference').src=`references/${String(c).padStart(3,'0')}.png`;document.querySelector('#rows').innerHTML=rs.map(r=>`<h3>${r.arm}, step ${r.step}, ${r.weight_source}, seed ${r.seed}; FID ${r.fid.toFixed(3)}</h3><div class="samples">${Array.from({length:20},(_,j)=>`<img loading="lazy" src="${r.relative_dir}/${String(c).padStart(4,'0')}_${String(j).padStart(2,'0')}.png">`).join('')}</div>`).join('')+['S3','C','A'].map(a=>{let z=d.arms[a],r=records.find(r=>r.arm===a&&r.seed===7301&&r.nfe===1&&r.weight_source==='ema'&&(a==='S3'||r.step===512));return `<h3>${a}: closest same-class optimization neighbor (seed7301; reference flipped if the metric selects it)</h3><img src="${r.relative_dir}/${String(c).padStart(4,'0')}_${String(z.closest_draw).padStart(2,'0')}.png"><img src="nearest/${a}/${String(c).padStart(3,'0')}.png">`}).join('')+`<pre>${JSON.stringify(d.arms,null,2)}</pre>`}document.querySelectorAll('select').forEach(x=>x.onchange=show);show()</script>'''
    (out/'gallery.html').write_text(template.replace('OPTIONS',options).replace('RECORDS',json.dumps(records)).replace('DIAG',json.dumps(diagnostics)))
    # One graph per row; raw traces remain visible beneath rolling medians.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(11,15),constrained_layout=True)
    updates={a:[json.loads(x) for x in (run/a/'updates.jsonl').read_text().splitlines()] for a in ('A','C')}
    specs=[('consistency_mse','Consistency MSE'),('consistency_grad_norm','Consistency gradient norm'),
           ('parameter_update_norm','Actual G parameter displacement'),('clipped_fraction','Adversarial endpoint clipping fraction (A only)'),
           ('adversarial_coefficient','Detached adversarial coefficient (A only)')]
    for axis,(key,title) in zip(axes,specs):
        for arm in ('C','A'):
            if key in ('clipped_fraction','adversarial_coefficient') and arm=='C':continue
            values=np.array([r['g'][key] for r in updates[arm]])
            smooth=np.array([np.median(values[max(0,i-31):i+1]) for i in range(len(values))])
            color='#3366bb' if arm=='C' else '#bb4433'
            axis.plot(np.arange(1,len(values)+1),values,color=color,alpha=.2,linewidth=.6)
            axis.plot(np.arange(1,len(values)+1),smooth,color=color,label=arm+' trailing32 median')
        axis.set_title(title);axis.set_xlabel('Completed G updates');axis.grid(alpha=.2);axis.legend()
    fig.savefig(out/'training.png',dpi=140);plt.close(fig)
    # Numeric outcome kept separate from pending visual interpretation.
    differences=[]
    for seed in (7301,7302,7303):
        f={a:row(a,seed)['fid'] for a in ('S3','C','A')}
        differences.append(dict(seed=seed,**f,A_minus_C=f['A']-f['C'],A_minus_S3=f['A']-f['S3']))
    judgments={}
    for key in ('A_minus_C','A_minus_S3'):
        values=[r[key] for r in differences if r['seed'] in (7302,7303)];mean=float(np.mean(values))
        judgments[key]=dict(mean=mean,per_seed=values,outcome='favorable' if mean<=-2 and all(v<0 for v in values) else 'adverse' if mean>=2 and all(v>0 for v in values) else 'uncertain')
    resources={a:json.loads((run/a/'result.json').read_text()) for a in ('A','C')}
    report=dict(state='numerical reporting complete; visual assessment separate',source_commit=json.loads((run/'frozen.json').read_text())['source']['git_commit'],
        exports=exports,rows=rows,paired_differences=differences,judgments=judgments,ledger=ledger,arms=resources,report_cpu_seconds=time.monotonic()-started,
        diversity_summary={a:dict(mean_class_pairwise=np.mean([d['arms'][a]['pairwise_mean'] for d in diagnostics]),
            exact_duplicates=sum(d['arms'][a]['exact_duplicate_count'] for d in diagnostics),
            median_nearest=np.median([d['arms'][a]['nearest_median'] for d in diagnostics])) for a in ('S3','C','A')})
    atomic_json(out/'summary.json',report)
    print(json.dumps({'complete':str(out),'seconds':report['report_cpu_seconds'],'judgments':judgments}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();report(a.run_dir)
