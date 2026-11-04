import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from model import Model, ModelOneNFE, ModelFewNFE
from soflow_fastgen.common import verify_protected, seed_all
from soflow_fastgen.models import Soflow
from soflow_fastgen.sampling import sample
from soflow_fastgen.training import Trainer


class Contracts(unittest.TestCase):
    def test_protected(self):
        self.assertEqual(len(verify_protected()['files']),7)

    def test_parameters(self):
        with torch.device('meta'):
            net=Soflow()
        self.assertEqual(sum(p.numel() for p in net.parameters()),39667843)

    def test_official_checkpoint(self):
        for cls,mode in ((ModelOneNFE,'one_nfe'),(ModelFewNFE,'few_nfe')):
            model=cls(model_channels=32,num_blocks=1)
            with tempfile.TemporaryDirectory() as directory:
                file=Path(directory)/'model.ckpt'
                torch.save({'state_dict':model.state_dict()},file)
                loaded=Model.load_checkpoint(str(file),evaluate_mode=mode,model_channels=32,num_blocks=1)
                self.assertEqual(model.count_parameters(),loaded.count_parameters())
                self.assertFalse(any('ema' in k or 'teacher' in k for k in loaded.state_dict()))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GPUBehavior(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        seed_all(42)
        self.net=Soflow(model_channels=32,num_blocks=1,dropout=0).cuda()
        self.images=torch.randn(3,3,64,64,device='cuda')
        self.labels=torch.tensor([0,30,150],device='cuda')

    def test_identity_and_conditioning(self):
        t=torch.full((3,),.5,device='cuda')
        self.assertTrue(torch.equal(self.net.solution_operator(self.images,t,t,self.labels),self.images))
        a=self.net.network(self.images,t,t,self.labels)
        b=self.net.network(self.images,t,t,(self.labels+1)%151)
        self.assertFalse(torch.equal(a,b))
        a.square().mean().backward()
        self.assertGreater(float(self.net.network.map_label.weight.grad.abs().sum()),0)

    def test_calls_and_zero_augmentation(self):
        calls=[]
        def inspect(module,args):
            calls.append(args)
            self.assertEqual(tuple(args[4].shape),(3,6))
            self.assertEqual(float(args[4].abs().sum()),0)
        hook=self.net.network.register_forward_pre_hook(inspect)
        self.net.eval()
        for nfe in (1,2,3,4):
            calls.clear()
            output=sample(self.net,(3,3,64,64),device='cuda',category=self.labels,nfe=nfe)
            self.assertEqual(len(calls),nfe)
            self.assertTrue(torch.isfinite(output).all())
            self.assertLessEqual(float(output.abs().max()),1)
        hook.remove()
        with self.assertRaises(ValueError):
            sample(self.net,(3,3,64,64),device='cuda',category=self.labels+151)

    def test_consistency_branch_conditioning_and_small_gap(self):
        seen=[]
        hook=self.net.network.register_forward_pre_hook(lambda module,args:seen.append((args[3].clone(),args[4].clone(),torch.is_grad_enabled())))
        t=torch.full((3,),.7,device='cuda');s=t-.5;l=t-1e-4
        aug=torch.randn(3,6,device='cuda')
        loss,_,_=self.net.get_consistency_loss(self.images,t,l,s,self.labels,None,1,1,.75,.001,1,aug)
        loss.backward();hook.remove()
        self.assertEqual([row[2] for row in seen],[True,False])
        self.assertTrue(torch.equal(seen[0][0],seen[1][0]))
        self.assertTrue(torch.equal(seen[0][1],seen[1][1]))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.net.parameters() if p.grad is not None))

    def test_accumulation_of_fixed_objective(self):
        # Same per-example randomness isolates accumulation from RNG ordering.
        noise=torch.randn_like(self.images)
        t=torch.full((3,),.7,device='cuda');s=t-.5;l=t-.1
        original=self.net.scheduler.get_x_t_v_t
        def objective(start,end):
            clean=self.images[start:end];epsilon=noise[start:end]
            self.net.scheduler.get_x_t_v_t=lambda samples,t:((1-t)*samples+t*epsilon,epsilon-samples)
            return self.net.get_consistency_loss(clean,t[start:end],l[start:end],s[start:end],self.labels[start:end],None,1,1,.75,.001)[0]
        objective(0,3).backward()
        expected={k:p.grad.clone() for k,p in self.net.named_parameters() if p.grad is not None}
        self.net.zero_grad(set_to_none=True)
        (objective(0,2)*2/3).backward();(objective(2,3)/3).backward()
        self.net.scheduler.get_x_t_v_t=original
        for k,p in self.net.named_parameters():
            if k in expected:
                torch.testing.assert_close(p.grad,expected[k],atol=2e-5,rtol=2e-3)

    def test_resume_next_update(self):
        config=json.loads(Path('configs/pokemon.json').read_text())
        config.update(architecture={'model_channels':32,'num_blocks':1,'dropout':.1},batch_size=3,microbatch_size=2)
        first=Trainer(config)
        first.update(self.images.cpu(),self.labels.cpu())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';first.save(path)
            first.update(self.images.cpu(),self.labels.cpu())
            second=Trainer(config);second.load(path)
            second.update(self.images.cpu(),self.labels.cpu())
            for p,q in zip(first.model.parameters(),second.model.parameters()):
                torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
            for p,q in zip(first.ema.parameters(),second.ema.parameters()):
                torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)


if __name__=='__main__':unittest.main()
