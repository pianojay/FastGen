"""Algorithmic checks independent of full-network fitting."""
import unittest
import torch
from flow_sid.flow import (corrupt, denoise, velocity_score, teacher_sample, student_rollout,
    ladder, sid_surrogate, discriminator_loss, generator_adversarial, adversarial_scale)


class Toy(torch.nn.Module):
    def __init__(self, slope=.3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(slope, dtype=torch.float64))
        self.calls = []

    def forward(self, x, t, labels, **kwargs):
        self.calls.append((torch.is_grad_enabled(), x.requires_grad, float(torch.as_tensor(t).flatten()[0])))
        return self.weight * x + labels[:, None, None, None] / 10


class Mathematics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(3, 2, 4, 4, dtype=torch.float64)
        self.z = torch.randn_like(self.x)
        self.labels = torch.tensor([0, 1, 2])

    def test_flow_direction_and_conversion(self):
        for time in [0., .01, .3, .99, 1.]:
            xt = corrupt(self.x, self.z, time)
            self.assertTrue(torch.allclose(xt - time * (self.z - self.x), self.x))
        class Oracle:
            def __call__(this, x, t, labels):
                return self.z - self.x
        self.assertTrue(torch.allclose(teacher_sample(Oracle(), self.z, self.labels, 32), self.x))
        # X0~N(mu,sigma^2), eps~N(0,1): exact conditional mean and score.
        mu, var = .4, .7
        for t in [.01, .3, .8, 1.]:
            total_var = (1 - t)**2 * var + t*t
            mean = mu + (1 - t) * var / total_var * (self.x - (1 - t) * mu)
            v = (self.x - mean) / t
            score = -(self.x - (1 - t) * mu) / total_var
            self.assertTrue(torch.allclose(velocity_score(self.x, t, v), score))
        with self.assertRaises(ValueError):
            velocity_score(self.x, 0, self.z)

    def test_conditioning(self):
        net = Toy()
        a = net(self.x, .5, self.labels)
        b = net(self.x, .5, self.labels + 1)
        self.assertTrue(torch.allclose(b - a, torch.full_like(a, .1)))

    def test_nfe_and_gradient_boundaries(self):
        for nfe in [1, 2, 3, 4]:
            noises = [torch.randn_like(self.x) for _ in range(nfe)]
            net = Toy()
            out = student_rollout(net, noises, self.labels, nfe)
            self.assertEqual(len(net.calls), nfe)
            self.assertFalse(out.requires_grad)
            self.assertTrue(torch.allclose(torch.tensor([c[2] for c in net.calls]),
                                           torch.tensor(ladder(nfe))))
            for step in range(nfe):
                net = Toy()
                out = student_rollout(net, noises, self.labels, nfe, train_step=step)
                out.sum().backward()
                self.assertEqual(len(net.calls), step + 1)
                self.assertEqual([c[0] for c in net.calls], [False] * step + [True])
                self.assertFalse(net.calls[-1][1])
                self.assertIsNotNone(net.weight.grad)
                manual = torch.zeros_like(self.x)
                for i, t in enumerate(ladder(nfe)[:step + 1]):
                    xt = corrupt(manual.detach(), noises[i], t)
                    manual = xt - t * (net.weight * xt + self.labels[:, None, None, None] / 10)
                self.assertTrue(torch.equal(out, manual))

    def test_sid_full_input_jacobians(self):
        real, fake = Toy(.2).requires_grad_(False), Toy(.5).requires_grad_(False)
        t = torch.tensor([.2, .5, .8], dtype=torch.float64)
        def loss(x):
            xt = corrupt(x, self.z, t)
            r, f = denoise(real, xt, t, self.labels), denoise(fake, xt, t, self.labels)
            return sid_surrogate(x, r, f, t).mean()
        x = self.x.clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(loss, (x,), eps=1e-6, atol=1e-5))
        grad = torch.autograd.grad(loss(x), x)[0]
        xt = corrupt(x, self.z, t)
        r, f = denoise(real, xt, t, self.labels), denoise(fake, xt, t, self.labels)
        stopped = torch.autograd.grad(sid_surrogate(x, r.detach(), f.detach(), t).mean(), x)[0]
        self.assertFalse(torch.allclose(grad, stopped))
        self.assertTrue(all(p.grad is None for p in list(real.parameters()) + list(fake.parameters())))

    def test_detached_fake_target_and_bce(self):
        generator, fake = Toy(), Toy(.8)
        x = generator(self.x, .5, self.labels).detach()
        t = torch.full((3,), .4, dtype=torch.float64)
        prediction = fake(corrupt(x, self.z, t), t, self.labels)
        (prediction - (self.z - x)).square().mean().backward()
        self.assertIsNone(generator.weight.grad)
        self.assertIsNotNone(fake.weight.grad)
        r = torch.zeros(3, 1, 2, 2, requires_grad=True)
        f = torch.zeros_like(r, requires_grad=True)
        discriminator_loss(r, f).mean().backward()
        self.assertTrue((r.grad < 0).all() and (f.grad > 0).all())
        f.grad = None
        generator_adversarial(f, t).mean().backward()
        self.assertTrue((f.grad < 0).all())
        self.assertEqual(adversarial_scale(10, 20, 10, .01), 0)
        self.assertAlmostEqual(adversarial_scale(25, 20, 10, .01), .005)
        self.assertAlmostEqual(adversarial_scale(50, 20, 10, .01), .01)

    def test_accumulation_reductions(self):
        t = torch.tensor([.2, .5, .8], dtype=torch.float64)
        real, fake = Toy(.2).requires_grad_(False), Toy(.5).requires_grad_(False)
        gradients = []
        for accumulation in [False, True]:
            generator = Toy(.4)
            slices = [slice(i, i + 1) for i in range(3)] if accumulation else [slice(None)]
            for sl in slices:
                image = generator(self.x[sl], t[sl], self.labels[sl])
                noisy = corrupt(image, self.z[sl], t[sl])
                r = denoise(real, noisy, t[sl], self.labels[sl])
                f = denoise(fake, noisy, t[sl], self.labels[sl])
                loss = sid_surrogate(image, r, f, t[sl]).sum() / 3
                loss.backward()
            gradients.append(generator.weight.grad)
        self.assertTrue(torch.allclose(gradients[0], gradients[1], atol=1e-12, rtol=1e-12))


if __name__ == '__main__':
    unittest.main()
