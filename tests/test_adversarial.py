"""CPU contracts for RpGAN/StyleGAN2; S3 integration has its own bounded smoke."""
import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from soflow_fastgen.adversarial import (ComponentConfig, RandomStreams,
    balanced_gradients, discriminator_objective, endpoint, evaluation_mode,
    generator_objective, input_penalty, paired_flip)
from soflow_fastgen.models import Soflow
from soflow_fastgen.stylegan2 import DEFAULTS, StyleGAN2Critic, reference_module, verify_vendor


class CoupledCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(.2))
        self.b = nn.Parameter(torch.tensor(.3))

    def forward(self, x, c):
        scalar = x.flatten(1).sum(1)
        return self.a * scalar + self.b * scalar.mean()


class LossContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_rpgan_signs_and_real_branch(self):
        class TwoLogits(nn.Module):
            def __init__(self):
                super().__init__()
                self.real = nn.Parameter(torch.tensor(0.))
                self.fake = nn.Parameter(torch.tensor(0.))

            def forward(self, x, c):
                return torch.where(x[:, 0, 0, 0] > 0, self.real, self.fake) + x.flatten(1).sum(1) * 0
        d = TwoLogits()
        real = torch.ones(4, 3, 2, 2)
        loss = discriminator_objective(d, real, -real, torch.arange(4), torch.zeros(4, dtype=torch.bool), .1)
        loss['loss'].backward()
        self.assertAlmostEqual(d.real.grad.item(), -.5)
        self.assertAlmostEqual(d.fake.grad.item(), .5)
        d = CoupledCritic()
        visible = torch.zeros(4, 3, 2, 2, requires_grad=True)
        loss = generator_objective(d, visible.detach(), visible, torch.arange(4), torch.zeros(4, dtype=torch.bool))
        loss.backward()
        torch.testing.assert_close(visible.grad, torch.full_like(visible, -.5 * (.2 + .3) / 4))
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_exact_group_penalty_r2_isolation_and_flip(self):
        d = CoupledCritic()
        real = torch.randn(4, 3, 2, 2)
        g = nn.Parameter(torch.randn(4, 3, 2, 2))
        row = discriminator_objective(d, real, g.clamp(-1, 1), torch.arange(4),
                                       torch.tensor([True, False, True, False]), .1)
        self.assertAlmostEqual(row['r1'].item(), 12 * .5 ** 2, places=6)
        self.assertAlmostEqual(row['r2'].item(), 12 * .5 ** 2, places=6)
        row['r2'].backward()
        self.assertIsNone(g.grad)
        self.assertAlmostEqual(d.a.grad.item(), 12.)
        self.assertAlmostEqual(d.b.grad.item(), 12.)

    def test_full_batch_gradient_balance_and_zero(self):
        c, a = [torch.tensor([3., 4.])], [torch.tensor([0., 2.])]
        result, row = balanced_gradients(c, a, .1)
        torch.testing.assert_close(result[0], torch.tensor([3., 4.5]))
        self.assertAlmostEqual(row['adversarial_coefficient'], .25)
        result, row = balanced_gradients(c, [torch.zeros(2)], .1)
        torch.testing.assert_close(result[0], c[0], rtol=0, atol=0)
        self.assertTrue(row['adversarial_gradient_zero'])
        result, row = balanced_gradients(c, a, 0)
        torch.testing.assert_close(result[0], c[0], rtol=0, atol=0)
        with self.assertRaises(FloatingPointError):
            balanced_gradients(c, [torch.tensor([float('nan'), 0.])], .1)
        # Loss offsets cannot affect component gradients or balancing.
        p = torch.tensor([2., -1.], requires_grad=True)
        g1, = torch.autograd.grad(p.square().sum(), p)
        g2, = torch.autograd.grad(p.square().sum() + 1e6, p)
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)

    def test_rng_stream_isolation_and_replay(self):
        streams = RandomStreams('cpu', 11)
        saved = copy.deepcopy(streams.states)
        with streams.use('consistency'):
            expected = torch.randn(8)
        streams.states = saved
        with streams.use('d_noise'):
            torch.randn(200)
        with streams.use('consistency'):
            actual = torch.randn(8)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_endpoint_modes_clipping_and_gradients(self):
        class Map(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = nn.Parameter(torch.tensor(.2))

            def solution_operator(self, x, t, s, c, a):
                assert not self.training and torch.is_grad_enabled()
                assert torch.equal(t, torch.ones_like(t)) and torch.count_nonzero(s) == 0
                assert torch.count_nonzero(a) == 0
                return x + self.offset
        model = Map().train()
        noise = torch.tensor([[[[-2., 0., 2.]]]])
        raw, visible = endpoint(model, noise, torch.tensor([0]))
        visible.sum().backward()
        self.assertTrue(model.training)
        self.assertAlmostEqual(model.offset.grad.item(), 1.)
        self.assertGreater(raw.abs().max().item(), 1)

    def test_completed_generator_counts(self):
        for channels, expected in ((128,39667843),(160,61934243),(192,89140419)):
            with torch.device('meta'):
                model = Soflow(model_channels=channels)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)


class StyleGANContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def fixture(self):
        torch.manual_seed(73)
        return StyleGAN2Critic(channel_base=256, channel_max=16, cmap_dim=16)

    def test_source_hashes_and_production_size(self):
        self.assertEqual(verify_vendor()['commit'], 'd72cc7d041b42ec8e806021a205ed9349f87c6a4')
        model = StyleGAN2Critic()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 28930048)

    def test_forward_first_and_second_derivative_reference_parity(self):
        adapter = self.fixture()
        reference = reference_module().Discriminator(**dict(DEFAULTS,
            channel_base=256, channel_max=16, cmap_dim=16))
        reference.load_state_dict(adapter.network.state_dict())
        x = torch.randn(4,3,64,64,requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        labels = torch.tensor([0, 42, 99, 150])
        a = adapter(x, labels)
        b = reference(y, F.one_hot(labels,151).float(), force_fp32=True).flatten()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga, = torch.autograd.grad(a.sum(),x,create_graph=True)
        gb, = torch.autograd.grad(b.sum(),y,create_graph=True)
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
        ha = torch.autograd.grad(ga.square().sum(),tuple(adapter.parameters()),allow_unused=True)
        hb = torch.autograd.grad(gb.square().sum(),tuple(reference.parameters()),allow_unused=True)
        for left,right in zip(ha,hb):
            if left is None or right is None:
                self.assertIs(left,right)
            else:
                torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_real_second_order_path_and_label_conditioning(self):
        d = self.fixture()
        images = torch.randn(4,3,64,64)
        labels = torch.tensor([0,20,100,150])
        self.assertFalse(torch.equal(d(images,labels), d(images,(labels+1)%151)))
        row = discriminator_objective(d,images,images.flip(-2),labels,
                                       torch.tensor([True,False,True,False]),.1)
        row['loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None))
        self.assertGreater(d.network.mapping.embed.weight.grad.norm().item(), 0)
        with self.assertRaises(ValueError):
            d(images[:3],labels[:3])
        with self.assertRaises(ValueError):
            d(images.repeat(2,1,1,1),labels.repeat(2))

    def test_accumulation_keeps_logical_groups(self):
        first, second = self.fixture(), self.fixture()
        real, fake = torch.randn(8,3,64,64), torch.randn(8,3,64,64)
        labels, mask = torch.arange(8), torch.tensor([True,False]*4)
        losses=[]
        for start in (0,4):
            sl=slice(start,start+4)
            losses.append(discriminator_objective(first,real[sl],fake[sl],labels[sl],mask[sl],.1)['loss']/2)
        sum(losses).backward()
        for start in (0,4):
            sl=slice(start,start+4)
            (discriminator_objective(second,real[sl],fake[sl],labels[sl],mask[sl],.1)['loss']/2).backward()
        for a,b in zip(first.parameters(),second.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SmallGPUContracts(unittest.TestCase):
    def test_zero_adversarial_is_actual_consistency_control(self):
        from soflow_fastgen.adversarial import AdversarialComponents
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        config=json.loads(Path('configs/research_s3_train.json').read_text())
        config['architecture']={'model_channels':32,'num_blocks':1,'dropout':.1}
        source={'config':config,'ema':Soflow(**config['architecture']).state_dict(),
                'origin':{'fixture':'small randomly initialized unit-test model'}}
        fixture=ComponentConfig(rho=0)
        first=AdversarialComponents(source,fixture,critic=StyleGAN2Critic(channel_base=256,channel_max=16,cmap_dim=16))
        control=AdversarialComponents(source,fixture,critic=StyleGAN2Critic(channel_base=256,channel_max=16,cmap_dim=16))
        x=torch.randn(4,3,64,64,device='cuda').clamp(-1,1);labels=torch.arange(4,device='cuda')
        d_streams=copy.deepcopy(first.streams.states)
        captured=[]
        hook=first.g_optimizer.register_step_pre_hook(
            lambda optimizer,args,kwargs:captured.extend(
                [None if p.grad is None else p.grad.clone() for p in first.model.parameters()]))
        first.generator_update(x,labels)
        hook.remove()
        control.g_optimizer.zero_grad(set_to_none=True)
        control.consistency_backward(x,labels)
        grad_max=0.
        for observed,p in zip(captured,control.model.parameters()):
            expected=torch.zeros_like(p) if p.grad is None else p.grad
            torch.testing.assert_close(observed,expected,rtol=2e-5,atol=2e-6)
            grad_max=max(grad_max,(observed-expected).abs().max().item())
        control.g_optimizer.step()
        update_max=0.
        for a,b in zip(first.model.parameters(),control.model.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
            update_max=max(update_max,(a-b).abs().max().item())
        print(f'zero-rho control: max gradient difference={grad_max:.3g}; max weight difference={update_max:.3g}',flush=True)
        for name in ('d_noise','d_flip','g_noise','g_flip'):
            torch.testing.assert_close(d_streams[name],first.streams.states[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in first.critic.parameters()))


if __name__ == '__main__':
    unittest.main()
