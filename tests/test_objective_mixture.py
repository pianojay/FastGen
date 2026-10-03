import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from soflow_fastgen.common import seed_all
from soflow_fastgen.models import Soflow
from soflow_fastgen.training import Trainer, flow_matching_count


def mixture_config(microbatch=5):
    config = json.loads(Path('configs/pokemon.json').read_text())
    config.update(architecture={'model_channels': 32, 'num_blocks': 1, 'dropout': .1},
                  batch_size=8, microbatch_size=microbatch, fm_fraction=.25,
                  fm_time_mean=-.9, fm_time_std=1.6)
    return config


class AnalyticObjectives(torch.nn.Module):
    """Different per-example quadratics expose double weighting and dropped tails."""
    def __init__(self, **kwargs):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(.25))
        self.seen = []

    def objective(self, images, labels, velocity):
        self.seen.extend((int(label), velocity) for label in labels)
        residual = self.slope * images.flatten(1).mean(1) + (-1 if velocity else 2)
        mse = residual.square().mean()
        weight = torch.tensor(1. if velocity else 3., device=images.device)
        return weight * mse, mse, weight

    def get_velocity_loss(self, images, t, labels, *args):
        return self.objective(images, labels, True)

    def get_consistency_loss(self, images, t, l, s, labels, *args):
        return self.objective(images, labels, False)

    def get_middle_time(self, t, s, *args):
        return (t+s)/2


class MixtureContracts(unittest.TestCase):
    def test_partition_is_exact_and_independent_of_microbatch(self):
        self.assertEqual(flow_matching_count(128, 0), 0)
        self.assertEqual(flow_matching_count(128, .25), 32)
        self.assertEqual(flow_matching_count(8, .25), 2)
        for batch, fraction in ((128, -.1), (128, 1), (128, float('nan')),
                                (128, float('inf')), (128, True), (3, .25), (0, .25)):
            with self.subTest(batch=batch, fraction=fraction), self.assertRaises(ValueError):
                flow_matching_count(batch, fraction)

    def test_effective_batch_gradient_and_metrics_with_uneven_microbatches(self):
        x = torch.arange(1, 9, dtype=torch.float32)
        labels = torch.arange(8)
        fm_mse = (.25*x[:2]-1).square().mean()
        scm_mse = (.25*x[2:]+2).square().mean()
        expected_grad = (2*(.25*x[:2]-1)*x[:2]).sum()/8
        expected_grad += (6*(.25*x[2:]+2)*x[2:]).sum()/8
        with patch('soflow_fastgen.training.Soflow', AnalyticObjectives), \
             patch('torch.cuda.synchronize'), \
             patch('soflow_fastgen.training.resources', return_value={'allocated_bytes': 0}):
            for microbatch in (1, 3, 5, 8):
                with self.subTest(microbatch=microbatch):
                    config = mixture_config(microbatch)
                    config['augmentation'] = False
                    trainer = Trainer(config, 'cpu')
                    row = trainer.update(x.view(8, 1, 1, 1), labels)
                    torch.testing.assert_close(trainer.model.slope.grad, expected_grad)
                    self.assertEqual(trainer.model.seen,
                                     [(i, i < 2) for i in range(8)])
                    self.assertEqual((row['fm_examples'], row['consistency_examples']), (2, 6))
                    self.assertAlmostEqual(row['velocity_mse'], float(fm_mse), places=5)
                    self.assertAlmostEqual(row['consistency_mse'], float(scm_mse), places=5)
                    self.assertAlmostEqual(row['mixed_loss_surrogate'],
                                           float(.25*fm_mse+.75*3*scm_mse), places=5)
                    self.assertEqual((trainer.step, trainer.cursor), (1, 8))
                    self.assertEqual(int(trainer.optimizer.state[trainer.model.slope]['step']), 1)

    def test_resume_rejects_changed_or_removed_objective_fields(self):
        config = mixture_config()
        with patch('soflow_fastgen.training.Soflow', AnalyticObjectives), \
             tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'resume.pt'
            Trainer(config, 'cpu').save(path)
            for key, value in (('fm_fraction', .5), ('fm_time_mean', 0.),
                               ('fm_time_std', 1.), ('total_steps', 60000)):
                changed = dict(config, **{key: value})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'semantics'):
                    Trainer(changed, 'cpu').load(path)
            removed = {key: value for key, value in config.items() if not key.startswith('fm_')}
            with self.assertRaisesRegex(ValueError, 'semantics'):
                Trainer(removed, 'cpu').load(path)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MixtureGPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        seed_all(42)

    def test_velocity_gradient_matches_detached_adaptive_mse_and_conditioning(self):
        net = Soflow(model_channels=32, num_blocks=1, dropout=0).cuda()
        images = torch.randn(3, 3, 64, 64, device='cuda')
        noise = torch.randn_like(images)
        t = torch.tensor([.01, .5, .99], device='cuda')
        labels = torch.tensor([0, 30, 150], device='cuda')
        aug = torch.randn(3, 6, device='cuda')
        broadcast = t.view(3, 1, 1, 1)
        xt, target = (1-broadcast)*images+broadcast*noise, noise-images
        prediction = net.network(xt, t, t, labels, aug)
        mse = (prediction-target).square().flatten(1).mean(1)
        weights = (mse+.001).pow(-.75).detach()
        expected_loss = (weights*mse).mean()
        expected_grads = torch.autograd.grad(expected_loss, tuple(net.parameters()), allow_unused=True)
        seen = []
        handle = net.network.register_forward_pre_hook(
            lambda module, args: seen.append(args))
        with patch.object(net.scheduler, 'get_x_t_v_t', return_value=(xt, target)):
            loss, actual_mse, actual_weight = net.get_velocity_loss(
                images, t, labels, None, 1., 1., .75, .001, 1., aug)
        loss.backward()
        handle.remove()
        self.assertEqual(len(seen), 1)
        for actual, expected in zip(seen[0][1:], (t, t, labels, aug)):
            self.assertTrue(torch.equal(actual, expected))
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(actual_mse, mse.mean())
        torch.testing.assert_close(actual_weight, weights.mean())
        for parameter, expected in zip(net.parameters(), expected_grads):
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-6)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_default_and_zero_mixture_have_matching_updates_and_identical_rng(self):
        images = torch.randn(8, 3, 64, 64)
        labels = torch.arange(8)
        config = {key: value for key, value in mixture_config().items() if not key.startswith('fm_')}
        seed_all(91)
        first = Trainer(config)
        initial = {key: value.clone() for key, value in first.model.state_dict().items()}
        first_row = first.update(images, labels)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        seed_all(91)
        second = Trainer(dict(config, fm_fraction=0.))
        for key, value in second.model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[key]))
        second_row = second.update(images, labels)
        # Repeating the default path also has sub-ULP GPU reduction variation.
        # Keep exact initial states, RNG and diagnostics; use the existing
        # numerical-recovery tolerance for updated model/EMA tensors.
        for p, q in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
        for p, q in zip(first.ema.parameters(), second.ema.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for key in ('loss_surrogate', 'consistency_mse', 'weight', 'grad_norm'):
            self.assertEqual(first_row[key], second_row[key])

    def test_mixed_resume_reproduces_next_augmented_update(self):
        images, labels = torch.randn(8, 3, 64, 64), torch.arange(8)
        config = mixture_config()
        first = Trainer(config)
        first.update(images, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'resume.pt'
            first.save(path)
            expected_row = first.update(images, labels)
            expected_rng = torch.cuda.get_rng_state()
            second = Trainer(copy.deepcopy(config))
            second.load(path)
            actual_row = second.update(images, labels)
            self.assertEqual((second.step, second.cursor), (2, 16))
            self.assertTrue(torch.equal(expected_rng, torch.cuda.get_rng_state()))
            for left, right in ((first.model, second.model), (first.ema, second.ema)):
                for p, q in zip(left.parameters(), right.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
            for key in ('loss_surrogate', 'consistency_mse', 'velocity_loss',
                        'velocity_mse', 'mixed_loss_surrogate', 'grad_norm'):
                self.assertEqual(expected_row[key], actual_row[key])


if __name__ == '__main__':
    unittest.main()
