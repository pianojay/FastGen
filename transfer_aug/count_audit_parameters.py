"""Static layer arithmetic from inspected source; no models or weights instantiated.

Counts describe the configurations named below, not measured training memory.
Biases and affine normalization parameters count; buffers and optimizer state do not.
"""
import ast
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'audit_sources'
manifest=json.loads((SOURCES/'manifest.json').read_text())

def conv(i,o,k,bias=True): return i*o*k*k+(o if bias else 0)
def linear(i,o): return i*o+o

# AdaIN's standalone sequential definitions are parsed as syntax, never imported.
tree=ast.parse((SOURCES/'adain_pytorch/net.py').read_text())
seq={}
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('vgg','decoder'):
        entries=[]
        for child in node.value.args:
            n=0
            if isinstance(child,ast.Call) and isinstance(child.func,ast.Attribute) and child.func.attr=='Conv2d':
                i,o,k=[ast.literal_eval(x) for x in child.args[:3]]
                n=conv(i,o,k[0] if isinstance(k,tuple) else k)
            entries.append(n)
        seq[node.targets[0].id]=entries
adain={'encoder_through_relu4_1':sum(seq['vgg'][:31]),'decoder':sum(seq['decoder']),
       'full_vgg_conv_object_in_file':sum(seq['vgg'])}
adain['active_encoder_plus_decoder']=adain['encoder_through_relu4_1']+adain['decoder']

# WCT community port exposes all five encoder/decoder definitions explicitly.
wct={}
for cls in ast.parse((SOURCES/'wct_pytorch/modelsNIPS.py').read_text()).body:
    if not isinstance(cls,ast.ClassDef): continue
    layers={}
    for method in cls.body:
        if isinstance(method,ast.FunctionDef) and method.name=='__init__':
            for node in ast.walk(method):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='Conv2d':
                    i,o,k=[ast.literal_eval(a) for a in node.value.args[:3]]
                    layers[ast.unparse(node.targets[0])]=conv(i,o,k)
    wct[cls.name]=sum(layers.values())

def resnet_g(width=64,blocks=9):
    return (conv(3,width,7)+conv(width,width*2,3)+conv(width*2,width*4,3)
            +blocks*2*conv(width*4,width*4,3)+conv(width*4,width*2,3)
            +conv(width*2,width,3)+conv(width,3,7))

def patch_d(width=64):
    return sum(conv(i,o,4) for i,o in [(3,width),(width,2*width),(2*width,4*width),(4*width,8*width),(8*width,1)])
cg=resnet_g();cd=patch_d()
cut_f=sum(linear(c,256)+linear(256,256) for c in [3,128,256,256,256])

# MUNIT edges2handbags_folder.yaml: dim64, 4 resblocks, style8, MLP256,
# 2 content downsamplings, 4 style downsamplings, 3 discriminator scales, vgg_w0.
style=(conv(3,64,7)+conv(64,128,4)+conv(128,256,4)+2*conv(256,256,4)+conv(256,8,1))
content=conv(3,64,7)+conv(64,128,4)+conv(128,256,4)+8*conv(256,256,3)
decoder=8*conv(256,256,3)+conv(256,128,5)+2*128+conv(128,64,5)+2*64+conv(64,3,7)
mlp=linear(8,256)+linear(256,256)+linear(256,4096)
mg=style+content+decoder+mlp
md=3*(conv(3,64,4)+conv(64,128,4)+conv(128,256,4)+conv(256,512,4)+conv(512,1,1))

def sg_res(i,o,norm=False):
    return conv(i,i,3)+conv(i,o,3)+(conv(i,o,1,False) if i!=o else 0)+(4*i if norm else 0)
def sg_ada(i,o,s=64):
    return conv(i,o,3)+conv(o,o,3)+(conv(i,o,1,False) if i!=o else 0)+linear(s,2*i)+linear(s,2*o)
def stargan(size=256,domains=3):
    # AFHQ path: w_hpf=0, max_conv_dim512, style_dim64, latent_dim16.
    start=2**14//size
    g=conv(3,start,3)+2*start+conv(start,3,1)
    dim=start
    for _ in range(int(math.log2(size))-4):
        out=min(dim*2,512)
        g+=sg_res(dim,out,True)+sg_ada(out,dim)
        dim=out
    g+=2*(sg_res(dim,dim,True)+sg_ada(dim,dim))
    shared=conv(3,start,3);dim=start
    for _ in range(int(math.log2(size))-2):
        out=min(dim*2,512);shared+=sg_res(dim,out);dim=out
    shared+=conv(dim,dim,4)
    s=shared+domains*linear(dim,64)
    d=shared+conv(dim,domains,1)
    f=linear(16,512)+3*linear(512,512)+domains*(3*linear(512,512)+linear(512,64))
    return {'generator':g,'mapping':f,'style_encoder':s,'discriminator':d,
            'four_components_sum_no_ema':g+f+s+d,
            'released_solver_with_registered_EMA_copies':2*(g+f+s)+d,
            'reference_inference_generator_plus_style_encoder':g+s}

def swapping():
    # Released church/default capacity, unconditional (num_classes=0), 256px,
    # structure8, global2048, patch128. StyleGAN2 equalized weights count normally.
    def res(i,o): return conv(i,i,3)+conv(i,o,3)+conv(i,o,1,False)
    def styled(i,o): return conv(i,o,3,False)+linear(2048,i)+1+o
    e=conv(3,32,1)
    for i,o in [(32,64),(64,128),(128,256),(256,512)]: e+=res(i,o)
    e+=conv(512,512,1)+conv(512,8,1)+conv(512,1024,3)+conv(1024,2048,3)+linear(2048,2048)
    g=2*linear(2048,8)
    for i,o in [(8,256),(256,512)]:
        g+=styled(i,o)+styled(o,o)+(conv(i,o,1,False) if i!=o else 0)
    for i,o in [(512,512),(512,512),(512,256),(256,128)]:
        g+=styled(i,o)+styled(o,o)+(conv(i,o,1) if i!=o else 0)
    g+=conv(128,3,1,False)+linear(2048,128)+3
    d=conv(3,128,1)
    for i,o in [(128,256),(256,512),(512,512),(512,512),(512,512),(512,512)]: d+=res(i,o)
    d+=conv(512,512,3)+linear(512*4*4,512)+linear(512,1)
    p=conv(3,32,3)
    for i,o in [(32,64),(64,128),(128,256),(256,384),(384,384)]: p+=res(i,o)
    p+=res(384,768)+conv(768,384,3)
    p+=linear(384*2*2*2,2048)+linear(2048,2048)+linear(2048,1024)+linear(1024,1)
    return {'encoder':e,'generator':g,'image_discriminator':d,'patch_discriminator':p,
            'inference_E_plus_G':e+g,'four_components_sum_no_ema':e+g+d+p}

# Standard VGG19 convolutions, with no classification head.
vgg=[];channels=3
for width,n in [(64,2),(128,2),(256,4),(512,4),(512,4)]:
    for _ in range(n): vgg.append(conv(channels,width,3));channels=width

result={
 'scope':'Static source-derived counts. No upstream execution, model creation, weights, forward passes, or GPU use.',
 'revisions':{k:v['revision'] for k,v in manifest.items() if 'revision' in v},
 'adain_pytorch':adain,
 'wct_pytorch_separate_five_encoders_and_decoders':dict(wct,total=sum(wct.values())),
 'vgg19':{'through_conv5_1':sum(vgg[:13]),'all_convolution_layers':sum(vgg),'classification_head':linear(512*7*7,4096)+linear(4096,4096)+linear(4096,1000)},
 'cyclegan_rgb_ngf64_instance_norm':{'generator_9blocks':cg,'generator_6blocks':resnet_g(blocks=6),'discriminator_3layers':cd,'two_G_two_D_sum':2*cg+2*cd},
 'cut_rgb_ngf64_instance_norm_antialias':{'generator_9blocks':cg,'discriminator_3layers':cd,'five_patch_MLPs':cut_f,'sum':cg+cd+cut_f,'MLP_input_channels':[3,128,256,256,256]},
 'munit_edges2handbags_config':{'style_encoder':style,'content_encoder':content,'decoder':decoder,'adain_parameter_MLP':mlp,'one_generator':mg,'one_multiscale_discriminator':md,'two_G_two_D_sum':2*mg+2*md,'vgg_w':0},
 'starganv2_AFHQ_256_3domains_hpf0':stargan(),
 'starganv2_mechanical_64_3domains_hpf0':stargan(64,3),
 'starganv2_mapping_if_151domains':stargan(256,151)['mapping'],
 'swapping_autoencoder_church_256_default_capacity':swapping(),
 'mixstyle_incremental_parameters':0,
 'mixstyle_ResNet18_7class_standard_count':11689512-513000+513*7,
 'fda_transfer_parameters':0,'reinhard_transfer_parameters':0,'diffaugment_incremental_parameters':0,
}
(ROOT/'parameter_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
