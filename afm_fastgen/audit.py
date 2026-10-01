"""Evaluation-only multiscale diagnostics. No pretrained features or training feedback."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw
from .common import ROOT, atomic_json, rng_state, restore_rng
from .data import Images


def structure(x):
    x = F.avg_pool2d(x.mean(1, keepdim=True), 4)
    # Per-image color/contrast removal, then translation-tolerant structural distance.
    return (x-x.mean((2,3), keepdim=True))/(x.std((2,3), keepdim=True)+.05)


def aligned_distance(x, y):
    a, b = structure(x), structure(y)
    distances = []
    for dy in range(-2,3):
        for dx in range(-2,3):
            distances.append((a[:,:,2:14,2:14][:,None]-b[:,:,2+dy:14+dy,2+dx:14+dx][None]).square().mean((2,3,4)))
    return torch.stack(distances).amin(0)


def measures(x, references):
    x = x.float(); clipped=x.clamp(-1,1)
    def pair(v):
        d=torch.cdist(v.flatten(1),v.flatten(1)).square()/v[0].numel()
        return float(d[torch.triu_indices(len(v),len(v),1).unbind()].median())
    d=aligned_distance(clipped,clipped)
    tri=torch.triu_indices(len(x),len(x),1)
    nearest=(clipped[:,None]-references[None]).square().mean((2,3,4)).min(1)
    low=F.avg_pool2d(clipped,8)
    high=clipped-F.interpolate(low,size=(64,64),mode='nearest')
    return {'pixel_pair_mse':pair(clipped),'low_pair_mse':pair(low),
            'high_pair_mse':pair(high),'structure_pair_mse':float(d[tri.unbind()].median()),
            'structural_near_duplicate_fraction':float((d[tri.unbind()]<.032).float().mean()),
            'saturation':float((x.abs()>1).float().mean()),'raw_min':float(x.min()),'raw_max':float(x.max()),
            'nearest_pixel_mse':nearest.values.tolist(),'nearest_indices':nearest.indices.tolist()}


def real_groups(config):
    dataset=Images(config,'optimization')
    pixels=np.load(ROOT/'artifacts/data_cache/optimization.npy',mmap_mode='r')
    labels=np.array(dataset.labels)
    return dataset, [torch.from_numpy(np.array(pixels[labels==c],copy=True)).float()/127.5-1 for c in range(151)]


def gallery(rows, output, title):
    canvas=Image.new('RGB',(1280,35+len(rows)*90),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((8,8),title,fill='black')
    for r,(label,refs,samples) in enumerate(rows):
        y=35+r*90; draw.text((4,y),str(label),fill='black')
        for j,x in enumerate(torch.cat([refs[:4],samples[:12]])):
            im=Image.fromarray(((x.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy())
            canvas.paste(im.resize((72,72)),(80+j*74,y))
    canvas.save(output)


@torch.no_grad()
def audit(g,ema,config,out,step,seed=18001,classes=None):
    out.mkdir(parents=True,exist_ok=False)
    saved=rng_state(); dataset,refs=real_groups(config)
    names={v:k for k,v in json.loads((ROOT/'data/pokemon-generation-one-22k/category_to_id.json').read_text()).items()}
    classes=list(range(151)) if classes is None else list(classes)
    rows=[]; pages={'raw':[],'ema':[]}; neighbors={'raw':[],'ema':[]}
    try:
        for c in classes:
            # Common eight plus fresh eight; distinct class seeds avoid batch dependence.
            z=torch.cat([torch.randn(8,3,64,64,generator=torch.Generator().manual_seed(18001+c)),
                         torch.randn(8,3,64,64,generator=torch.Generator().manual_seed(seed+100000+c))]).cuda()
            y=torch.full((16,),c,device='cuda',dtype=torch.long)
            zr=z+.05*torch.randn(z.shape,generator=torch.Generator(device='cuda').manual_seed(seed+200000+c),device='cuda')
            for name,net in [('raw',g),('ema',ema)]:
                x=net(z,y).cpu(); pert=net(zr[:4],y[:4]).cpu()
                torch.save({'noise_cpu':z.cpu(),'raw_outputs':x,'perturbed_outputs':pert},out/f'{name}_class{c:03d}.pt')
                m=measures(x,refs[c]); m.update({'class':c,'name':names[c],'model':name,
                    'noise_sensitivity':float((pert-x[:4]).square().mean())/.0025,
                    'common':measures(x[:8],refs[c]),'fresh':measures(x[8:],refs[c]),
                    'nearest_paths':[dataset.paths[np.where(np.array(dataset.labels)==c)[0][i]] for i in m['nearest_indices']]})
                rows.append(m); pages[name].append((names[c],refs[c],x))
                pairs=torch.stack([v for i in range(6) for v in (x[i],refs[c][m['nearest_indices'][i]])])
                neighbors[name].append((names[c],refs[c],pairs))
            if len(pages['raw'])==10 or c==classes[-1]:
                for name in pages:
                    gallery(pages[name],out/f'{name}_{c:03d}.png',f'{name} step {step}: four real | eight common + four fresh draws')
                    gallery(neighbors[name],out/f'nearest_{name}_{c:03d}.png',f'{name} step {step}: four real | six generated / nearest optimization pairs')
                    pages[name]=[]
                    neighbors[name]=[]
        atomic_json(out/'metrics.json',{'step':step,'fresh_seed':seed,'common_seed':18001,'samples_per_class':16,'rows':rows})
    finally:
        restore_rng(saved)
    return rows


def calibrate(config,out):
    out.mkdir(parents=True,exist_ok=False); _,refs=real_groups(config)
    rng=torch.Generator().manual_seed(77103); records=[]
    for c,r in enumerate(refs):
        real=r[:16]; base=real[:1].expand(16,-1,-1,-1).clone()
        shifted=[]
        for i in range(16):
            dy,dx=(int(v) for v in torch.randint(-8,9,(2,),generator=rng)); tmp=torch.ones_like(base[i])
            ya,yb=max(0,dy),min(64,64+dy); xa,xb=max(0,dx),min(64,64+dx)
            tmp[:,ya:yb,xa:xb]=base[i,:,ya-dy:yb-dy,xa-dx:xb-dx]; shifted.append(tmp)
        controls={'real':real,'repeat':base,'noise_only':torch.rand(base.shape,generator=rng)*2-1,
                  'noisy_copy':(base+.1*torch.randn(base.shape,generator=rng)).clamp(-1,1),
                  'color_copy':(base*torch.rand((16,3,1,1),generator=rng)*.5+torch.rand((16,3,1,1),generator=rng)*.5).clamp(-1,1),
                  'shift_copy':torch.stack(shifted),'small_set_copies':real[:2].repeat(8,1,1,1)}
        for name,x in controls.items():
            # Held-out optimization references for real diversity calibration; all refs for copy detection.
            records.append({'class':c,'control':name,**measures(x,r[16:] if name=='real' else r)})
        if c<3:
            gallery([(name,real,x) for name,x in controls.items()],out/f'controls_{c}.png','Calibration controls; four references then controls')
    atomic_json(out/'controls.json',records)
    return records
