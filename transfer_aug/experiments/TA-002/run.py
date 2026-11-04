"""Frozen TA-002: NumPy FDA, CPU only; all writes are under this directory."""
from pathlib import Path
import argparse, ast, csv, hashlib, io, json, os, platform, resource, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'transfer_aug/shared/cohort_v1'
RADII=[0,1,2,4]; ALPHAS=[.25,.5,1.]; ARMS=['intra','inter']
PINS={'contract_sha256':'366e4ac85810d4b1c0e850f972c9175f4a550ced13f982c1c6c0910f3801e905','common_protocol_sha256':'1be27b900ee25ad95aed707427fb047e2456c122c6a0d2c4cf1feaa02fe51425','cohort_sha256':'da15bf04c3281a170e18ffc6116a298366d0287356c1ba19c6765027be7b5a6c','bundle_manifest_sha256':'46d27966191a14746aa8ec212509aefb0c46b905de0e674947e9b2e094203bda'}
COHORT=json.loads((BUNDLE/'cohort.json').read_text()); ROWS=COHORT['rows']
FREQ=np.fft.fftfreq(64)*64
MASKS={r:(np.abs(FREQ)[:,None]<=r)&(np.abs(FREQ)[None,:]<=r) for r in RADII}
HF=np.maximum(np.abs(FREQ)[:,None],np.abs(FREQ)[None,:])>=16
START=time.time()
if (HERE/'session.json').exists(): START=json.loads((HERE/'session.json').read_text())['start_unix']

def write_json(name,obj):
 p=HERE/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw):
 write_json('status.json',{'experiment_id':'TA-002','stage':stage,'pid':os.getpid(),'updated_unix':time.time(),'elapsed_seconds':time.time()-START,**kw})
def add_command():
 with (HERE/'commands.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),'argv':sys.argv,'command':'bash transfer_aug/run_python.sh '+' '.join(sys.argv),'cwd':str(ROOT)})+'\n')
def integrity(endpoint):
 status('integrity_'+endpoint);t=time.time(); checks={}
 for key,p in [('contract_sha256',HERE/'CONTRACT_v1.md'),('common_protocol_sha256',ROOT/'transfer_aug/shared/PROTOCOL_V1.md'),('cohort_sha256',BUNDLE/'cohort.json'),('bundle_manifest_sha256',BUNDLE/'manifest.json')]:
  checks[str(p.relative_to(ROOT))]={'sha256':sha(p),'expected':PINS[key]};assert checks[str(p.relative_to(ROOT))]['sha256']==PINS[key],p
 manifest=json.loads((BUNDLE/'manifest.json').read_text())
 for rel,digest in manifest['files'].items():
  got=sha(BUNDLE/rel);checks['bundle/'+rel]={'sha256':got,'expected':digest};assert got==digest,rel
 for rel,digest in manifest['protected_inputs'].items():
  got=sha(ROOT/rel);checks[rel]={'sha256':got,'expected':digest};assert got==digest,rel
 snap=json.loads((BUNDLE/'source_stat_snapshot.json').read_text());mismatch=[]
 for rel,expected in snap.items():
  st=(ROOT/'data/pokemon-generation-one-22k'/rel).stat()
  if [st.st_size,st.st_mtime_ns]!=expected:mismatch.append(rel)
 assert not mismatch,mismatch
 a1=set((ROOT/'eda/resources/A1.txt').read_text().splitlines());selected={}
 for row in ROWS:
  for arm in ['content','intra','inter']:
   rel=row[arm+'_path'];assert rel in a1,rel
   if rel not in selected:selected[rel]=sha(ROOT/'data/pokemon-generation-one-22k'/rel)
   assert selected[rel]==row[arm+'_file_sha256'],rel
 upstream=ROOT/'transfer_aug/audit_sources/fda/utils/__init__.py'
 assert sha(upstream)=='165c60e0e96ed4b357550089b499086f562458248f3cec5796d5151f080b8b4d'
 out={'pass':True,'endpoint':endpoint,'hash_checks':checks,'source_stats_checked':len(snap),'source_stat_mismatches':mismatch,'selected_original_A1_hashes':selected,'selected_unique_count':len(selected),'source_reference':{'path':str(upstream.relative_to(ROOT)),'sha256':sha(upstream),'revision':'b9a0cdf0bcc9b787c00e39df73eda5673706f219'},'seconds':time.time()-t}
 write_json('verification/integrity_'+endpoint+'.json',out);return out

def load_inputs():
 return {a:np.load(BUNDLE/(a+'_uint8.npy')) for a in ['content','intra','inter']}
def fft(x):return np.fft.fft2(x,axes=(0,1))
def transform(c,d,r,a):
 if a==0:return c.copy(),0.
 fc,fd=fft(c),fft(d); ac,ad=np.abs(fc),np.abs(fd)
 amp=ac.copy();mask=MASKS[r];amp[mask]=(1-a)*ac[mask]+a*ad[mask]
 z=np.fft.ifft2(amp*np.exp(1j*np.angle(fc)),axes=(0,1))
 return z.real,float(np.max(np.abs(z.imag)))
def pixels(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def png_bytes(x):
 b=io.BytesIO();Image.fromarray(pixels(x)).save(b,format='PNG');return b.getvalue()
def condition(r,a):return f'r{r}_a{a:g}'
def png_path(sid,arm,r,a):return HERE/'png'/arm/condition(r,a)/(sid+'.png')
def energy(x):return .5*(np.abs(np.diff(x,axis=0)).mean()+np.abs(np.diff(x,axis=1)).mean())
def boundary(x):return .5*(np.abs(x[0]-x[-1]).mean()+np.abs(x[:,0]-x[:,-1]).mean())
def hf_fraction(f):
 p=np.abs(f)**2;total=p.sum()-p[0,0].sum()
 return float(p[HF].sum()/total) if total>1e-20 else 0.
def metrics(c,d,raw,r,a,imag):
 z=np.clip(raw,0,1);fc,fd,fr,fz=fft(c),fft(d),fft(raw),fft(z);m=MASKS[r]
 outside=float(np.linalg.norm(fc[~m]));raw_l2=float(np.linalg.norm((fr-fc)[~m]));clip_l2=float(np.linalg.norm((fz-fc)[~m]));denzero=outside==0.
 target=(1-a)*np.abs(fc[m])+a*np.abs(fd[m])
 return {'rms_change':float(np.sqrt(np.mean((z-c)**2))),'clipped_fraction':float(np.mean((raw<0)|(raw>1))),'max_excursion':float(max(0.,-raw.min(),raw.max()-1)),'rgb_mean_shift_l2':float(np.linalg.norm(z.mean((0,1))-c.mean((0,1)))),'edge_energy_delta':float(energy(z)-energy(c)),'boundary_jump_delta':float(boundary(z)-boundary(c)),'hf_fraction_delta':hf_fraction(fz)-hf_fraction(fc),'outside_mask_raw_l2':raw_l2,'outside_mask_clipped_l2':clip_l2,'outside_mask_content_l2':outside,'outside_mask_zero_denominator':denzero,'outside_mask_raw_relative_l2':None if denzero else raw_l2/outside,'outside_mask_clipped_relative_l2':None if denzero else clip_l2/outside,'outside_mask_raw_max_abs':float(np.max(np.abs((fr-fc)[~m]))),'inside_amplitude_max_abs_error':float(np.max(np.abs(np.abs(fr[m])-target))),'max_imaginary_residual':imag}
def numerical_tests(inputs):
 status('numerical_verification');t=time.time();out={'mask_sizes':{},'mask_conjugate_symmetric':{},'self_donor_conditions':604*12,'self_donor_rgb_max_error':0.,'self_donor_export_max_lsb':0,'alpha0_exact_checks':0,'linear_max_error':0.,'radius0_mean_interpolation_max_error':0.,'upstream_alpha1_max_error':0.,'synthetic':[]}
 neg=(-np.arange(64))%64
 for r,m in MASKS.items():
  out['mask_sizes'][str(r)]=int(m.sum());out['mask_conjugate_symmetric'][str(r)]=bool(np.array_equal(m,m[np.ix_(neg,neg)]));assert m.sum()==(2*r+1)**2;assert out['mask_conjugate_symmetric'][str(r)]
 tree=ast.parse((ROOT/'transfer_aug/audit_sources/fda/utils/__init__.py').read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['low_freq_mutate_np','FDA_source_to_target_np']]
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned FDA numpy functions>','exec'),ns)
 for i,cu in enumerate(inputs['content']):
  c=cu.astype(np.float64)/255
  for r in RADII:
   for a in ALPHAS:
    raw,_=transform(c,c,r,a);out['self_donor_rgb_max_error']=max(out['self_donor_rgb_max_error'],float(np.max(np.abs(raw-c))));out['self_donor_export_max_lsb']=max(out['self_donor_export_max_lsb'],int(np.max(np.abs(pixels(raw).astype(int)-cu.astype(int)))))
   for arm in ARMS:
    d=inputs[arm][i].astype(np.float64)/255
    zero,_=transform(c,d,r,0);assert np.array_equal(zero,c);out['alpha0_exact_checks']+=1
    full,_=transform(c,d,r,1.)
    # Direct use of only the audited NumPy AST definitions; torch is never imported.
    ref=ns['FDA_source_to_target_np'](c.transpose(2,0,1),d.transpose(2,0,1),L=r/64).transpose(1,2,0)
    out['upstream_alpha1_max_error']=max(out['upstream_alpha1_max_error'],float(np.max(np.abs(full-ref))))
    for a in ALPHAS:
     x,_=transform(c,d,r,a);out['linear_max_error']=max(out['linear_max_error'],float(np.max(np.abs(x-((1-a)*c+a*full)))))
     if r==0:out['radius0_mean_interpolation_max_error']=max(out['radius0_mean_interpolation_max_error'],float(np.max(np.abs(x-(c+a*(d.mean((0,1))-c.mean((0,1))))))))
  if i%40==0:status('numerical_verification',sources_checked=i+1)
 rng=np.random.default_rng(20261007)
 fixtures=[('constant_black_to_white',np.zeros((64,64,3)),np.ones((64,64,3))),('constant_color',np.broadcast_to([.1,.5,.9],(64,64,3)).copy(),np.broadcast_to([.8,.2,.4],(64,64,3)).copy()),('near_constant',np.full((64,64,3),.5)+rng.normal(0,1e-12,(64,64,3)),np.full((64,64,3),.2)+rng.normal(0,1e-12,(64,64,3)))]
 for name,c,d in fixtures:
  for r in RADII:
   for a in ALPHAS:
    raw,imag=transform(c,d,r,a);m=metrics(c,d,raw,r,a,imag);assert np.isfinite(raw).all();assert m['outside_mask_raw_max_abs']<=1e-8;assert m['inside_amplitude_max_abs_error']<=1e-8;assert imag<=1e-10
    e=float(np.max(np.abs(raw-(c+a*(d.mean((0,1))-c.mean((0,1))))))) if r==0 or name.startswith('constant') else None
    if e is not None:assert e<=1e-10
    out['synthetic'].append({'fixture':name,'radius':r,'alpha':a,'constant_or_dc_error':e,**m})
 assert out['self_donor_rgb_max_error']<=1e-6 and out['self_donor_export_max_lsb']<=1
 assert out['linear_max_error']<=1e-10 and out['radius0_mean_interpolation_max_error']<=1e-10 and out['upstream_alpha1_max_error']<=1e-10
 out['pass']=True;out['seconds']=time.time()-t;write_json('verification/numerical_checks.json',out)

def write_csv(name,rows):
 with (HERE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def condition_summary(rows):
 out=[]
 keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool) and k not in ['radius','alpha','source_index']]
 for arm in ['control']+ARMS:
  for r,a in ([(0,0.)] if arm=='control' else [(r,a) for r in RADII for a in ALPHAS]):
   group=[x for x in rows if x['arm']==arm and x['radius']==r and x['alpha']==a];e={'arm':arm,'radius':r,'alpha':a,'count':len(group),'zero_denominator_count':sum(x['outside_mask_zero_denominator'] for x in group)}
   for k in keys:
    vals=[x[k] for x in group if x[k] is not None];e[k+'_median']=float(np.median(vals)) if vals else None;e[k+'_p95']=float(np.percentile(vals,95)) if vals else None
   out.append(e)
 return out

def generate(inputs):
 status('generation');t=time.time();rawdir=HERE/'raw';rawdir.mkdir(exist_ok=True);rows=[]
 controls=np.lib.format.open_memmap(rawdir/'control.npy',mode='w+',dtype=np.float64,shape=(604,64,64,3))
 stores={arm:np.lib.format.open_memmap(rawdir/(arm+'.npy'),mode='w+',dtype=np.float64,shape=(604,4,3,64,64,3)) for arm in ARMS}
 maxcheck={'finite':True,'max_imaginary_residual':0.,'outside_mask_raw_max_abs':0.,'inside_amplitude_max_abs_error':0.}
 for i,row in enumerate(ROWS):
  c=inputs['content'][i].astype(np.float64)/255;controls[i]=transform(c,c,0,0)[0]
  for arm in ['control']+ARMS:
   d=c if arm=='control' else inputs[arm][i].astype(np.float64)/255
   for ri,r in enumerate([0] if arm=='control' else RADII):
    for ai,a in enumerate([0.] if arm=='control' else ALPHAS):
     raw,imag=transform(c,d,r,a);assert np.isfinite(raw).all()
     if arm!='control':stores[arm][i,ri,ai]=raw
     p=png_path(row['sample_id'],arm,r,a);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(png_bytes(raw))
     m=metrics(c,d,raw,r,a,imag)
     for k in maxcheck:
      if k!='finite':maxcheck[k]=max(maxcheck[k],m[k])
     rows.append({'sample_id':row['sample_id'],'source_index':i,'class':row['category'],'donor_class':row['category'] if arm!='inter' else row['inter_category'],'arm':arm,'radius':r,'alpha':a,'condition':condition(r,a),'png':str(p.relative_to(HERE)),**m})
  if i%20==0:status('generation',sources_completed=i+1,outputs_completed=len(rows))
 controls.flush()
 for store in stores.values():store.flush()
 write_csv('per_output_metrics.csv',rows);write_json('metrics.json',rows);write_csv('condition_metrics.csv',condition_summary(rows));write_json('condition_metrics.json',condition_summary(rows))
 assert maxcheck['max_imaginary_residual']<=1e-10;assert maxcheck['outside_mask_raw_max_abs']<=1e-8;assert maxcheck['inside_amplitude_max_abs_error']<=1e-8
 maxcheck.update({'pass':True,'outputs':len(rows),'transformed_outputs':14496,'controls':604,'seconds':time.time()-t});write_json('verification/generation_checks.json',maxcheck)
 write_json('raw/layout.json',{'dtype':'float64','order':'C','unclipped':True,'source_order':'../../shared/cohort_v1/cohort.json rows','intra.npy':[604,4,3,64,64,3],'inter.npy':[604,4,3,64,64,3],'control.npy':[604,64,64,3],'axis_names':['source','radius','alpha','height','width','RGB'],'radii':RADII,'alphas':ALPHAS,'display':'np.rint(np.clip(raw,0,1)*255).astype(np.uint8)'})
 return rows

FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):
 try:return ImageFont.truetype(FONT_PATH,n)
 except OSError:return ImageFont.load_default()
def sheet_image(sid,arm,inputs):
 i=next(i for i,r in enumerate(ROWS) if r['sample_id']==sid);row=ROWS[i]
 im=Image.new('RGB',(870,780),'#ededed');draw=ImageDraw.Draw(im)
 draw.text((12,9),f"TA-002 | {sid} {row['category']} | {arm.upper()} donor",fill='black',font=font(20))
 draw.text((12,36),f"Donor: {row['category'] if arm=='intra' else row['inter_category']} | Actual 64px images enlarged 2x (nearest)",fill='black',font=font(15))
 for ri,r in enumerate(RADII):
  for col,label in enumerate(['Content','Donor','alpha 0.25','alpha 0.5','alpha 1.0']):
   x=12+col*172;y=70+ri*176
   draw.text((x,y),f'r={r} | {label}',fill='black',font=font(15))
   if col<2:tile=Image.fromarray(inputs['content' if col==0 else arm][i])
   else:tile=Image.open(png_path(sid,arm,r,ALPHAS[col-2]));tile.load()
   im.paste(tile.resize((128,128),Image.Resampling.NEAREST),(x,y+27))
 return im

def tail_image(arm,tails,inputs):
 im=Image.new('RGB',(870,830),'#ededed');draw=ImageDraw.Draw(im)
 draw.text((12,8),f'TA-002 {arm.upper()} four clipping-ranked distinct-ID tails',fill='black',font=font(19))
 draw.text((12,35),'Selection: clipping desc, RMS desc, ID/radius/alpha asc',fill='black',font=font(15))
 for k,m in enumerate([x for x in tails if x['arm']==arm]):
  i=m['source_index'];row=ROWS[i];y=65+k*190
  draw.text((12,y),f"#{k+1} {m['sample_id']} {row['category']} | r={m['radius']} a={m['alpha']:g} | clip={m['clipped_fraction']:.4f} RMS={m['rms_change']:.4f}",fill='black',font=font(16))
  for col,lab in enumerate(['Content','Donor: '+m['donor_class'],'Output']):
   x=12+col*285;draw.text((x,y+24),lab,fill='black',font=font(15))
   tile=Image.fromarray(inputs['content' if col==0 else arm][i]) if col<2 else Image.open(HERE/m['png'])
   im.paste(tile.resize((128,128),Image.Resampling.NEAREST),(x,y+48))
 return im

def image_bytes(im):
 b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()
def sheets(inputs,rows,verify=False):
 status('sheet_recompute' if verify else 'contact_sheets');tails=[]
 for arm in ARMS:
  seen=set()
  for m in sorted([x for x in rows if x['arm']==arm],key=lambda x:(-x['clipped_fraction'],-x['rms_change'],x['sample_id'],x['radius'],x['alpha'])):
   if m['sample_id'] not in seen:tails.append(m);seen.add(m['sample_id'])
   if len(seen)==4:break
 if not verify:write_json('tail_selection.json',tails)
 else:assert tails==json.loads((HERE/'tail_selection.json').read_text())
 entries=[]
 for sid in COHORT['representative_review_ids']:
  for arm in ARMS:entries.append((f'sheets/{sid}_{arm}.png',sheet_image(sid,arm,inputs)))
 for arm in ARMS:entries.append((f'sheets/tails_{arm}.png',tail_image(arm,tails,inputs)))
 for p,im in entries:
  dest=HERE/p
  if verify:assert dest.read_bytes()==image_bytes(im),p
  else:dest.parent.mkdir(exist_ok=True);dest.write_bytes(image_bytes(im))
 gallery='<!doctype html><meta charset="utf-8"><title>TA-002 frozen FDA evidence</title><style>body{font:18px sans-serif;background:#eee}img{max-width:100%;display:block}section{margin:24px 0}</style><h1>TA-002 FDA contact sheets</h1><p>Unblinded AI review; 64×64 pixels enlarged with nearest neighbor. Intra and inter donor arms remain separate.</p>'+''.join(f'<section><h2>{p}</h2><a href="{p}"><img src="{p}" alt="{p}"></a></section>' for p,_ in entries)
 if verify:assert (HERE/'gallery.html').read_text()==gallery
 else:(HERE/'gallery.html').write_text(gallery)
 return len(entries)

def recompute(inputs):
 status('deterministic_recompute');t=time.time();saved=json.loads((HERE/'metrics.json').read_text());stores={a:np.load(HERE/'raw'/f'{a}.npy',mmap_mode='r') for a in ['control']+ARMS};count=0;recomputed=[]
 for i,row in enumerate(ROWS):
  c=inputs['content'][i].astype(np.float64)/255
  for arm in ['control']+ARMS:
   d=c if arm=='control' else inputs[arm][i].astype(np.float64)/255
   for ri,r in enumerate([0] if arm=='control' else RADII):
    for ai,a in enumerate([0.] if arm=='control' else ALPHAS):
     raw,imag=transform(c,d,r,a);stored=stores[arm][i] if arm=='control' else stores[arm][i,ri,ai]
     assert np.array_equal(raw,stored),(i,arm,r,a,'raw')
     assert png_bytes(raw)==png_path(row['sample_id'],arm,r,a).read_bytes(),(i,arm,r,a,'png')
     m=metrics(c,d,raw,r,a,imag);old=saved[count];assert all(old[k]==v for k,v in m.items()),(i,arm,r,a,'metrics');recomputed.append({**old,**m});count+=1
  if i%20==0:status('deterministic_recompute',sources_completed=i+1,outputs_checked=count)
 assert count==15100
 assert condition_summary(recomputed)==json.loads((HERE/'condition_metrics.json').read_text())
 # Regenerate both CSV files in memory, ensuring those declared measurement outputs also agree.
 for file,rs in [('per_output_metrics.csv',recomputed),('condition_metrics.csv',condition_summary(recomputed))]:
  b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs);assert b.getvalue().encode()==(HERE/file).read_bytes(),file
 nsheets=sheets(inputs,recomputed,verify=True)
 write_json('verification/determinism.json',{'pass':True,'raw_arrays_exact_outputs':count,'png_exact_outputs':count,'per_output_metrics_exact_rows':count,'condition_metrics_exact_rows':25,'contact_sheets_exact':nsheets,'tail_selection_exact':True,'gallery_exact':True,'successful_outputs_rewritten':False,'seconds':time.time()-t})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','recompute','integrity_after']);args=parser.parse_args();add_command()
 if not (HERE/'session.json').exists():write_json('session.json',{'start_unix':START,'initial_pid':os.getpid(),'budget_seconds':7200})
 write_json('environment.json',{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'pillow':PIL_VERSION,'platform':platform.platform(),'threads':{x:os.environ.get(x) for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'training_applicable':False,'parameters':0,'GPU_seconds':0,'device':'cpu','reference_sha256':sha(ROOT/'transfer_aug/audit_sources/fda/utils/__init__.py')})
 if args.stage=='all':
  integrity('before');inputs=load_inputs();numerical_tests(inputs);rows=generate(inputs);sheets(inputs,rows);status('awaiting_visual_review',transformed_outputs_completed=14496,control_outputs_completed=604)
 elif args.stage=='recompute':recompute(load_inputs());status('recompute_complete')
 else:integrity('after');status('integrity_complete')
 write_json('resources_'+args.stage+'.json',{'elapsed_seconds':time.time()-START,'process_peak_host_memory_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'user_cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'system_cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime,'device':'cpu','threads':2,'parameters':0,'GPU_seconds':0})
if __name__=='__main__':
 try:main()
 except Exception:
  err=traceback.format_exc();write_json('failures/failure_'+str(time.time_ns())+'.json',{'error':err,'argv':sys.argv,'implementation_sha256':sha(__file__)});status('failed',error=err);raise
