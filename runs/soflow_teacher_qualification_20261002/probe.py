"""Read-only teacher qualification; no optimizer or parameter updates."""
import os,sys,time,json,hashlib,ast,signal,resource
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT/'snapshot'))
from soflow_fastgen.models import Soflow
from soflow_fastgen.sampling import sample as endpoint_sample
from flow_sid.networks import FlowNet
from flow_sid.flow import corrupt,denoise,sid_surrogate,teacher_sample
SF=Path('/home/user/jw_soflow/FastGen-SoFlow');SID=Path('/home/user/jw_soflow/FastGen-SiD-DiT')
MODE=sys.argv[1] if len(sys.argv)>1 else 'smoke'
SEED=2026100207
START=time.monotonic()
torch.set_num_threads(2);torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark=False
def dump(name,data):
 (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def amp():return torch.autocast('cuda',dtype=torch.float16,cache_enabled=False)
def fail_timeout(*args):raise TimeoutError('Declared finite probe watchdog exhausted')
signal.signal(signal.SIGALRM,fail_timeout);signal.alarm(300 if MODE=='smoke' else 2400)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
assert os.environ.get('PYTHONNOUSERSITE')=='1'
assert torch.cuda.get_device_name(0)=='NVIDIA TITAN RTX'
# Frozen, completed sources only.
sfpath=SF/'runs/research_s2_unet160_40k_seed42/checkpoints/step_040000.pt'
rfpath=SID/'runs/campaign/T/best.pt'
assert digest(sfpath)=='32bf7e04d68750aff505546266d97ba9a4077e17fd68ba911254850473e75a79'
assert digest(rfpath)=='8bafcc5cb0e5d1e0f7a644fe6789640ccda9b513813633bc27a40eb81d17b3cf'
c=torch.load(sfpath,map_location='cpu',weights_only=False)
sf=Soflow(**c['config']['architecture']);sf.load_state_dict(c['ema']);del c
rf=FlowNet();rf.load_state_dict(torch.load(rfpath,map_location='cpu',weights_only=False)['weights'])
sf=sf.cuda().eval().requires_grad_(False);rf=rf.cuda().eval().requires_grad_(False)
class Diagonal(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x,t,labels):
  t=torch.as_tensor(t,device=x.device,dtype=torch.float32).reshape(-1)
  if t.numel()==1:t=t.expand(len(x))
  return self.model.predict_velocity(x,t,labels,torch.zeros(len(x),6,device=x.device)).float()
diag=Diagonal(sf)
models={'soflow_diagonal':diag,'rf_teacher':rf}
meta={'seed':SEED,'gpu_physical_index':3,'gpu_uuid':'GPU-c651701d-9ecd-e23e-13a9-2bc13b838b5d','precision':'FP32 parameters and state, FP16 autocast cache disabled','torch':torch.__version__,'python':sys.executable,'parameters':{k:sum(p.numel() for p in v.parameters()) for k,v in models.items()},'soflow_checkpoint_sha256':digest(sfpath),'rf_checkpoint_sha256':digest(rfpath),'no_parameter_updates':True,'mode':MODE}
rng=torch.Generator().manual_seed(SEED)
z=torch.randn(16,3,64,64,generator=rng).cuda();labels=torch.arange(16,device='cuda');t=torch.full((16,),.5,device='cuda')
smoke={}
for key,model in models.items():
 with torch.no_grad():
  with amp():v=model(z,t,labels)
  fp32=model(z,t,labels)
 torch.cuda.synchronize();st=time.monotonic()
 with torch.no_grad(),amp():
  for i in range(8):v=model(z,t,labels)
 torch.cuda.synchronize()
 x=z.detach().clone().requires_grad_(True)
 with amp():den=denoise(model,x,t,labels); val=den.square().mean()
 grad=torch.autograd.grad(val,x)[0]
 smoke[key]={'batch16_forward_seconds':(time.monotonic()-st)/8,'finite_velocity':bool(torch.isfinite(v).all()),'finite_input_gradient':bool(torch.isfinite(grad).all()),'fp16_fp32_relative_rms':float((v-fp32).square().mean().sqrt()/fp32.square().mean().sqrt()),'v_rms':float(v.square().mean().sqrt()),'input_gradient_rms':float(grad.square().mean().sqrt())}
 assert smoke[key]['finite_velocity'] and smoke[key]['finite_input_gradient']
meta['smoke']=smoke;meta['peak_allocated_bytes']=torch.cuda.max_memory_allocated();meta['elapsed_seconds']=time.monotonic()-START
dump(MODE+'_start.json',meta);print(json.dumps(meta),flush=True)
if MODE=='smoke':sys.exit(0)
# Independent fixed images/noise repeated across teachers and bins, four/class.
part=json.loads((OUT/'snapshot/development_split.json').read_text());assert digest(OUT/'snapshot/development_split.json')=='825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e'
source=SID/'data/pokemon-generation-one-22k';paths=(source/'train_split.txt').read_text().splitlines();mapping=json.loads((source/'category_to_id.json').read_text())
all_labels=np.array([mapping[Path(paths[i]).parts[1]] for i in part['development_indices']]);pixels=np.load(SID/'artifacts/data_cache/development.npy',mmap_mode='r')
indices=np.concatenate([np.flatnonzero(all_labels==c)[:4] for c in range(151)])
images=torch.from_numpy(np.array(pixels[indices],copy=True)).float()/127.5-1
labs=torch.tensor(all_labels[indices],dtype=torch.long)
noise=torch.randn(images.shape,generator=torch.Generator().manual_seed(SEED+1))
sid_times=(torch.randn(len(images),generator=torch.Generator().manual_seed(SEED+2))*1.6+np.log(2)).sigmoid().clamp(.01,.99)
times=[.01,.05,.1,.25,.5,.75,.9,.99]
field={};arrays={}
for name,tt in [(str(t),torch.full((len(images),),t)) for t in times]+[('sid_logit_normal',sid_times)]:
 per={}
 for key,model in models.items():
  ve=[];de=[]
  for i in range(0,len(images),16):
   clean=images[i:i+16].cuda();eps=noise[i:i+16].cuda();ti=tt[i:i+16].cuda();yi=labs[i:i+16].cuda();xt=corrupt(clean,eps,ti)
   with torch.no_grad(),amp():v=model(xt,ti,yi);d=xt-ti[:,None,None,None]*v
   ve.extend((v-(eps-clean)).square().flatten(1).mean(1).cpu().tolist());de.extend((d-clean).square().flatten(1).mean(1).cpu().tolist())
  per[key]={'velocity_mse':float(np.mean(ve)),'denoiser_mse':float(np.mean(de))};arrays[name+'_'+key+'_v']=ve;arrays[name+'_'+key+'_d']=de
 delta=np.array(arrays[name+'_soflow_diagonal_v'])-np.array(arrays[name+'_rf_teacher_v'])
 # Bootstrap classes as paired clusters, not correlated pixels.
 bootrng=np.random.default_rng(SEED+3);classmeans=delta.reshape(151,4).mean(1);boot=classmeans[bootrng.integers(151,size=(2000,151))].mean(1)
 per['paired_velocity_delta_soflow_minus_rf']={'mean':float(delta.mean()),'class_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
 ddelta=np.array(arrays[name+'_soflow_diagonal_d'])-np.array(arrays[name+'_rf_teacher_d']);cm=ddelta.reshape(151,4).mean(1);boot=cm[bootrng.integers(151,size=(2000,151))].mean(1)
 per['paired_denoiser_delta']={'mean':float(ddelta.mean()),'class_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
 field[name]=per;print('FIELD',name,json.dumps(per),flush=True);dump('field_metrics.json',field)
np.savez(OUT/'field_per_image.npz',**arrays,indices=indices,labels=labs.numpy(),sid_times=sid_times.numpy())
# Actual SiD surrogate input Jacobians; shared completed S1 generator + fake.
s1path=SID/'runs/campaign/S1/latest.pt';s1hash=digest(s1path)
s1=torch.load(s1path,map_location='cpu',weights_only=False)
assert s1['step']==4000
fake=FlowNet();fake.load_state_dict(s1['fake']);fake=fake.cuda().eval().requires_grad_(False)
gen=FlowNet();gen.load_state_dict(s1['ema']);gen=gen.cuda().eval().requires_grad_(False);del s1
n=32;gl=torch.arange(0,151,5,device='cuda')[:30];n=len(gl)
gz=torch.randn(n,3,64,64,generator=torch.Generator().manual_seed(SEED+4)).cuda()
with torch.no_grad(),amp():generated=denoise(gen,gz,torch.ones(n,device='cuda'),gl)
del gen;torch.cuda.empty_cache()
gnoise=torch.randn(n,3,64,64,generator=torch.Generator().manual_seed(SEED+5)).cuda()
grad_records={};gradarrays={}
for tv in [.05,.25,.5,.75,.95]:
 rec={};grads={}
 for key,model in models.items():
  gs=[];ls=[];jacs=[]
  for i in range(0,n,5):
   yi=gl[i:i+5];ti=torch.full((len(yi),),tv,device='cuda');im=generated[i:i+5].detach().clone().requires_grad_(True)
   xt=corrupt(im,gnoise[i:i+5],ti)
   with amp():
    real=denoise(model,xt,ti,yi);fk=denoise(fake,xt,ti,yi);loss=sid_surrogate(im,real,fk,ti).sum()
   g=torch.autograd.grad(loss,im)[0];gs.append(g.detach().cpu());ls.append(float(loss))
   x=xt.detach().clone().requires_grad_(True)
   # Random normalized vector-Jacobian product, uniform across teachers.
   direction=gnoise[i:i+5]/(3*64*64)**.5
   with amp():out=denoise(model,x,ti,yi);val=(out*direction).sum()
   j=torch.autograd.grad(val,x)[0];jacs.extend(j.flatten(1).norm(dim=1).cpu().tolist())
  g=torch.cat(gs);assert torch.isfinite(g).all();grads[key]=g
  rec[key]={'sid_surrogate_sum':sum(ls),'sid_gradient_rms':float(g.square().mean().sqrt()),'sid_gradient_norm_median':float(g.flatten(1).norm(dim=1).median()),'denoiser_vjp_norm_mean':float(np.mean(jacs)),'finite':True,'nonzero_fraction':float((g!=0).float().mean())}
  gradarrays[str(tv)+'_'+key]=g.flatten(1).norm(dim=1).numpy()
 a=grads['soflow_diagonal'].flatten(1);b=grads['rf_teacher'].flatten(1);rec['paired_gradient_cosine_mean']=float(torch.nn.functional.cosine_similarity(a,b).mean());grad_records[str(tv)]=rec
 dump('gradient_metrics.json',{'fake_generator_source':'completed S1 4000 update EMA generator and raw fake network, fixed for both teachers','source_sha256':s1hash,'results':grad_records});print('GRAD',tv,json.dumps(rec),flush=True)
assert all(p.grad is None for model in [sf,rf,fake] for p in model.parameters())
del fake;torch.cuda.empty_cache()
# Extract the exact official compute_fid function without importing unrelated
# mutable model/dataset modules; function source remains unmodified.
from cleanfid import fid
module=ast.parse((OUT/'snapshot/evaluate.py').read_text());fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='compute_fid')
namespace={'fid':fid};exec(compile(ast.Module(body=[fn],type_ignores=[]),'official_evaluate_compute_fid','exec'),namespace);compute_fid=namespace['compute_fid']
from torchvision.transforms.functional import to_pil_image
categories=torch.arange(151).repeat_interleave(20);genrng=torch.Generator().manual_seed(SEED+6)
# Common CPU initial noise across all models, independent fixed re-noise draws.
noises=[torch.randn(3020,3,64,64,generator=genrng) for _ in range(4)]
noisehash=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in noises]
fids={};model_calls={};ref=SID/'artifacts/development_reference';assert len(list(ref.glob('*.png')))==3020
for name in ['soflow_diagonal32','rf32','soflow_endpoint4']:
 folder=OUT/name/'images';folder.mkdir(parents=True,exist_ok=True);st=time.monotonic();calls=[0];target=sf.network if name.startswith('soflow') else rf.backbone
 hook=target.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
 outliers=0;numel=0;maxabs=0
 for i in range(0,3020,16):
  yi=categories[i:i+16].cuda();zs=[z[i:i+16].cuda() for z in noises]
  with torch.no_grad(),amp():
   if name=='soflow_diagonal32':x=teacher_sample(diag,zs[0],yi,32)
   elif name=='rf32':x=teacher_sample(rf,zs[0],yi,32)
   else:
    # Unclipped endpoint rollout for range audit, identical existing equations.
    x=zs[0]
    for j,tval in enumerate([1.,.75,.5,.25]):
     t=torch.full((len(yi),),tval,device='cuda');x=sf.solution_operator(x,t,torch.zeros_like(t),yi,torch.zeros(len(yi),6,device='cuda'))
     if j<3:x=(1-[.75,.5,.25][j])*x+[.75,.5,.25][j]*zs[j+1]
  assert torch.isfinite(x).all();outliers+=int((x.abs()>1).sum());numel+=x.numel();maxabs=max(maxabs,float(x.abs().max()))
  for j,(im,c) in enumerate(zip(x.cpu(),yi.tolist())):
   to_pil_image(((im+1)/2).clamp(0,1)).convert('RGB').save(folder/f'{c:04d}_{(i+j)%20:02d}.png')
  if i%320==0:print('GENERATE',name,i,'elapsed',round(time.monotonic()-st,1),flush=True)
 hook.remove();assert len(list(folder.glob('*.png')))==3020
 expected=(189 if 3020%16 else 3020//16)*(4 if name.endswith('4') else 32);assert calls[0]==expected,(calls,expected)
 torch.cuda.synchronize();generation=time.monotonic()-st
 # Official extractor only, references are existing readonly development PNGs.
 score=compute_fid(folder,ref,device='cuda:0',batch_size=16)
 rec={'fid':score,'generation_seconds':generation,'total_seconds':time.monotonic()-st,'neural_calls_per_sample':4 if name.endswith('4') else 32,'batched_network_calls':calls[0],'samples':3020,'initial_noise_sha256':noisehash[0],'noise_seed':SEED+6,'raw_absmax':maxabs,'raw_out_of_range_fraction':outliers/numel,'reference_count':3020,'official_evaluate_sha256':digest(OUT/'snapshot/evaluate.py'),'official_score':False}
 fids[name]=rec;dump('fid_metrics.json',fids);print('FID',name,json.dumps(rec),flush=True)
# Readable matched panels: 16 classes, reference and 4 generated samples/method.
names={v:k for k,v in mapping.items()};cs=list(range(0,151,10));canvas=Image.new('RGB',(64*14,25+85*len(cs)),'white');draw=ImageDraw.Draw(canvas);draw.text((5,4),'Reference | SoFlow diagonal32 (4) | RF32 (4) | SoFlow endpoint4 (4)',fill='black')
for r,c in enumerate(cs):
 y=25+85*r;draw.text((0,y),names[c],fill='black');refidx=np.flatnonzero(all_labels==c)[0];canvas.paste(Image.fromarray(np.asarray(pixels[refidx]).transpose(1,2,0)),(64,y))
 for m,key in enumerate(fids):
  for j in range(4):canvas.paste(Image.open(OUT/key/'images'/f'{c:04d}_{j:02d}.png'),(128+(m*4+j)*64,y))
canvas.save(OUT/'comparison_gallery.png')
meta.update({'fids':fids,'elapsed_seconds':time.monotonic()-START,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'noise_hashes':noisehash,'all_parameter_grads_none':all(p.grad is None for m in [sf,rf] for p in m.parameters())})
dump('completed.json',meta);print('COMPLETE',json.dumps(meta),flush=True)
