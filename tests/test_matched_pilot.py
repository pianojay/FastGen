import copy
import unittest
from types import SimpleNamespace
import torch
from soflow_fastgen.adversarial import RandomStreams
from soflow_fastgen.pilot import GSampler, index_streams, step_seed, rho, config, TOTAL_LIMIT

class PilotContracts(unittest.TestCase):
    def test_permutation_wrap_recovery(self):
        sampler=GSampler(35);x=sampler.draw(32);state=copy.deepcopy(sampler.state_dict());y=sampler.draw(32)
        sampler.load_state_dict(state);self.assertEqual(y,sampler.draw(32));self.assertEqual(len(set(x+y[:3])),35)
        with self.assertRaises(ValueError):GSampler(34).load_state_dict(state)
    def test_common_draws_ignore_extra_d_calls_and_zero_rho(self):
        a=SimpleNamespace(device='cpu',streams=RandomStreams('cpu',11))
        c=SimpleNamespace(device='cpu',streams=RandomStreams('cpu',11))
        for step in (1,63,64,65):
            with a.streams.use('d_noise'):torch.randn(123)
            index_streams(a,step);index_streams(c,step)
            for name in ('consistency','g_noise','g_flip'):
                with a.streams.use(name):left=torch.randn(32)
                with c.streams.use(name):right=torch.randn(32)
                torch.testing.assert_close(left,right,rtol=0,atol=0)
        index_streams(a,64)
        with a.streams.use('g_noise'):first=torch.randn(32)
        index_streams(a,65)
        with a.streams.use('g_noise'):expected=torch.randn(32)
        # Skipping the G adversarial forward at step64 does not shift step65.
        index_streams(c,64);index_streams(c,65)
        with c.streams.use('g_noise'):actual=torch.randn(32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(first,actual))
    def test_ramp_and_ema_clocks(self):
        self.assertEqual([rho(k) for k in (1,63,64,65,512)],[.1/64,.1*63/64,.1,.1,.1])
        self.assertAlmostEqual(config().ema_decay**64,.5)
        self.assertEqual(TOTAL_LIMIT,16200)
        with self.assertRaises(ValueError):rho(513)
        with self.assertRaises(ValueError):step_seed('d_noise',1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class PilotGPUContracts(unittest.TestCase):
    def test_real_pairing_and_boundary_recovery(self):
        import json,tempfile
        from pathlib import Path
        from soflow_fastgen.adversarial import AdversarialComponents
        from soflow_fastgen.models import Soflow
        from soflow_fastgen.stylegan2 import StyleGAN2Critic
        from scripts.smoke_adversarial import recovery_identity
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        cfg=json.loads(Path('configs/research_s3_train.json').read_text())
        cfg['architecture']={'model_channels':32,'num_blocks':1,'dropout':.1}
        source={'config':cfg,'ema':Soflow(**cfg['architecture']).state_dict(),'origin':{'fixture':'pilot boundary'}}
        make=lambda:AdversarialComponents(source,config(),critic=StyleGAN2Critic(channel_base=256,channel_max=16,cmap_dim=16))
        a,c=make(),make();x=torch.randn(4,3,64,64,device='cuda').clamp(-1,1);labels=torch.arange(4,device='cuda')
        a.discriminator_update(x,labels)
        index_streams(a,63);index_streams(c,63)
        a.generator_update(x,labels,rho=0);c.generator_update(x,labels,rho=0)
        for p,q in zip(a.model.parameters(),c.model.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a.streams.states['consistency'],c.streams.states['consistency'],atol=0,rtol=0)
        del c
        index_streams(a,64);a.discriminator_update(x,labels);a.generator_update(x,labels,rho=rho(64))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';a.save(path)
            index_streams(a,65);a.discriminator_update(x,labels);a.generator_update(x,labels,rho=rho(65));expected=recovery_identity(a)
            a.load(path);index_streams(a,65);a.discriminator_update(x,labels);a.generator_update(x,labels,rho=rho(65))
            self.assertEqual(recovery_identity(a),expected)

if __name__=='__main__':unittest.main()
