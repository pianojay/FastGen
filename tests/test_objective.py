import copy
import importlib.util
import unittest
import torch
from torch import nn
from afm_fastgen.losses import (GradientNormalization, augmentation_parameters,
    cosine, discriminator_losses, generator_losses, paired_augment)
from afm_fastgen.models import Discriminator


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14); torch.set_num_threads(2)

    def test_signs_ot_and_penalties(self):
        real = torch.tensor([1., -2.], requires_grad=True)
        fake = torch.tensor([-.5, .2], requires_grad=True)
        rg, fg = real+.01, fake-.02
        losses = discriminator_losses(real, fake, rg, fg, torch.arange(2), .01, .25, .01)
        self.assertTrue(torch.allclose(losses['d_adv'], -torch.log(torch.sigmoid(real-fake)).mean()))
        self.assertAlmostEqual(float(losses['d_r1']), .25, places=4)
        self.assertAlmostEqual(float(losses['d_r2']), 1., places=4)
        self.assertTrue(torch.allclose(losses['d_center'], .01*(real+fake).square().mean()))
        dr, df = torch.autograd.grad(losses['d_adv'], (real, fake))
        self.assertTrue((dr < 0).all() and (df > 0).all())
        z = torch.randn(2, 3, 4, 4); prediction = torch.randn_like(z).requires_grad_()
        gl = generator_losses(real, fake, prediction, z, .2)
        self.assertTrue(torch.allclose(gl['g_ot'], .2*(prediction-z).square().mean()))
        self.assertTrue((torch.autograd.grad(gl['g_adv'], fake)[0] < 0).all())
        self.assertTrue(torch.allclose(torch.autograd.grad(gl['g_ot'], prediction)[0], .4*(prediction-z)/prediction.numel()))

    def test_gradient_normalizer_reference_and_accumulation(self):
        spec = importlib.util.spec_from_file_location('reference_grad_norm', 'vendor/adversarial-flow-models/grad_norm.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        reference = module.GradientNormalization(); port = GradientNormalization()
        for _ in range(3):
            x = torch.randn(5, 3, 4, 4, requires_grad=True); y = x.detach().clone().requires_grad_()
            (reference(x).square().mean()*3).backward(); (port(y).square().mean()*3).backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            torch.testing.assert_close(reference.square_avg, port.square_avg, rtol=0, atol=0)
        port = GradientNormalization(); avg = torch.tensor(0.)
        for size in [4, 4, 3]:
            x = torch.randn(size, 3, 4, 4, requires_grad=True); weight = size/11
            local = 2*x.detach()/x.numel()
            avg.lerp_(local.square().sum()*local.numel(), .1)
            (port(x, weight).square().mean()*weight).backward()
            torch.testing.assert_close(x.grad, local*weight/(avg.sqrt()+1e-8))
            torch.testing.assert_close(port.square_avg, avg)
        self.assertEqual(int(port.backwards), 3)

    def test_uneven_discriminator_accumulation(self):
        # Exact global 25% sample selection, independent of microbatch boundaries.
        x = torch.randn(11, 5); fake = torch.randn(11, 5); ix = torch.tensor([1, 5, 9])
        rg, fg = x[ix]+.01, fake[ix]-.01
        full = nn.Linear(5, 1); micro = copy.deepcopy(full)
        r, f = full(x).flatten(), full(fake).flatten()
        loss = discriminator_losses(r, f, full(rg).flatten(), full(fg).flatten(), ix, .01, .25, .01)
        sum(loss.values()).backward()
        for lo, hi in [(0, 4), (4, 8), (8, 11)]:
            chosen = (ix >= lo) & (ix < hi); local_ix = ix[chosen]-lo
            r, f = micro(x[lo:hi]).flatten(), micro(fake[lo:hi]).flatten()
            loss = discriminator_losses(r, f, micro(rg[chosen]).flatten(), micro(fg[chosen]).flatten(), local_ix, .01, .25, .01)
            ((loss['d_adv']+loss['d_center'])*((hi-lo)/11)+(loss['d_r1']+loss['d_r2'])*(len(local_ix)/3)).backward()
        for a, b in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)

    def test_post_normalization_target_scale(self):
        # A loss coefficient is canceled at equilibrium; the target is not.
        x = torch.randn(7, 3, 8, 8)
        gradients = []
        for coefficient, target in [(1., 1.), (.1, 1.), (1., .25)]:
            z = x.clone().requires_grad_()
            n = GradientNormalization(decay=0., target_scale=target)
            (n(z).square().mean()*coefficient).backward()
            gradients.append(z.grad)
        torch.testing.assert_close(gradients[0], gradients[1])
        torch.testing.assert_close(gradients[0]*.25, gradients[2])

    def test_routing_and_classes(self):
        g = nn.Linear(5, 5); d = Discriminator(channels=8)
        z = torch.randn(2, 5); x = torch.randn(2, 3, 64, 64); y = torch.tensor([0, 1])
        with torch.no_grad():
            generated = g(z)
        self.assertFalse(generated.requires_grad)
        self.assertFalse(torch.allclose(d(x, y), d(x, y.flip(0))))
        d.requires_grad_(False); image = x.clone().requires_grad_()
        d(image, y).sum().backward()
        self.assertGreater(float(image.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_paired_augmentation(self):
        x = torch.randn(3, 3, 64, 64, requires_grad=True)
        p = augmentation_parameters(3, x.device, probability=1.)
        a, b = paired_augment(x, x, p)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward(); self.assertGreater(float(x.grad.abs().sum()), 0.)
        self.assertLess(float(x.grad.abs().sum()), x.numel())

    def test_schedule_units(self):
        self.assertEqual(cosine(0, 9, .2, .005), .2)
        self.assertEqual(cosine(8, 9, .2, .005), .005)
        self.assertEqual(cosine(20, 9, .2, .005), .005)
        self.assertAlmostEqual(cosine(4, 9, .2, .005), .1025)


if __name__ == '__main__':
    unittest.main(verbosity=2)
