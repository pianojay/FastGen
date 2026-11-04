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
MODE='gradient_scaled_check'
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
signal.signal(signal.SIGALRM,fail_timeout);signal.alarm(300)
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
   g=torch.autograd.grad(loss*65536.,im)[0]/65536.;gs.append(g.detach().cpu());ls.append(float(loss))
   x=xt.detach().clone().requires_grad_(True)
   # Random normalized vector-Jacobian product, uniform across teachers.
   direction=gnoise[i:i+5]/(3*64*64)**.5
   with amp():out=denoise(model,x,ti,yi);val=(out*direction).sum()
   j=torch.autograd.grad(val,x)[0];jacs.extend(j.flatten(1).norm(dim=1).cpu().tolist())
  g=torch.cat(gs);assert torch.isfinite(g).all();grads[key]=g
  rec[key]={'sid_surrogate_sum':sum(ls),'sid_gradient_rms':float(g.square().mean().sqrt()),'sid_gradient_norm_median':float(g.flatten(1).norm(dim=1).median()),'denoiser_vjp_norm_mean':float(np.mean(jacs)),'finite':True,'nonzero_fraction':float((g!=0).float().mean())}
  gradarrays[str(tv)+'_'+key]=g.flatten(1).norm(dim=1).numpy()
 a=grads['soflow_diagonal'].flatten(1);b=grads['rf_teacher'].flatten(1);rec['paired_gradient_cosine_mean']=float(torch.nn.functional.cosine_similarity(a,b).mean());grad_records[str(tv)]=rec
 dump('gradient_scaled_metrics.json',{'fake_generator_source':'completed S1 4000 update EMA generator and raw fake network, fixed for both teachers','source_sha256':s1hash,'results':grad_records});print('GRAD',tv,json.dumps(rec),flush=True)
assert all(p.grad is None for model in [sf,rf,fake] for p in model.parameters())
del fake;torch.cuda.empty_cache()

meta.update({'elapsed_seconds':time.monotonic()-START,'loss_scale':65536,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});dump('gradient_scaled_completed.json',meta)
