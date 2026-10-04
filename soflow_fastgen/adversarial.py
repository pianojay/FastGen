"""AYF-inspired SoFlow endpoint components, without a training campaign loop."""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from .augment import AugmentPipe
from .common import atomic_save, digest
from .models import Soflow
from .stylegan2 import StyleGAN2Critic


@contextmanager
def evaluation_mode(model):
    previous = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in previous:
            module.training = training


@contextmanager
def frozen_parameters(model):
    previous = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        yield
    finally:
        for p, required in zip(model.parameters(), previous):
            p.requires_grad_(required)


def endpoint(model, noise, categories):
    """Differentiable deployed endpoint, including its real output clamp."""
    with evaluation_mode(model):
        zero = noise.new_zeros(len(noise))
        raw = model.solution_operator(noise, zero + 1, zero, categories,
                                      noise.new_zeros(len(noise), 6))
    return raw, raw.clamp(-1, 1)


def paired_flip(real, fake, mask):
    if mask.shape != (len(real),) or mask.dtype != torch.bool:
        raise ValueError('Expected one fixed boolean flip per real/fake pair')
    mask = mask[:, None, None, None]
    return (torch.where(mask, real.flip(-1), real),
            torch.where(mask, fake.flip(-1), fake))


def input_penalty(logits, image):
    """Group-summed score gradient: includes minibatch-statistic interactions."""
    gradient, = torch.autograd.grad(logits.sum(), image, create_graph=True)
    return gradient.square().flatten(1).sum(1).mean()


def discriminator_objective(critic, real, fake, categories, mask, beta):
    # Fresh visible-image leaves: neither R2 nor D's adversarial loss reaches G.
    real = real.detach().requires_grad_(True)
    fake = fake.detach().requires_grad_(True)
    xr, xf = paired_flip(real, fake, mask)
    dr, df = critic(xr, categories), critic(xf, categories)
    adv = F.softplus(df - dr).mean()
    r1, r2 = input_penalty(dr, real), input_penalty(df, fake)
    return {'loss': adv + beta * (r1 + r2), 'adversarial': adv,
            'r1': r1, 'r2': r2, 'real_logit': dr.mean(), 'fake_logit': df.mean()}


def generator_objective(critic, real, visible, categories, mask):
    xr, xf = paired_flip(real, visible, mask)
    with frozen_parameters(critic):
        with torch.no_grad():
            dr = critic(xr, categories)
        df = critic(xf, categories)
        return F.softplus(dr - df).mean()


def balanced_gradients(consistency, adversarial, rho, *, cap=1e4, eps=1e-12):
    """Take norms after effective-batch accumulation, never per microbatch."""
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    if len(consistency) != len(adversarial) or not consistency:
        raise ValueError('Gradient vectors must have matching nonempty structure')
    with torch.no_grad():
        nc = torch.stack([g.square().sum() for g in consistency]).sum().sqrt()
        na = torch.stack([g.square().sum() for g in adversarial]).sum().sqrt()
        if not torch.isfinite(nc + na):
            raise FloatingPointError('Non-finite component gradients')
        scale = 0. if na.item() == 0 else rho * min(cap, (nc / (na + eps)).item())
        dot = sum((c * a).sum() for c, a in zip(consistency, adversarial))
        metrics = {'consistency_grad_norm': nc.item(), 'adversarial_grad_norm': na.item(),
                   'adversarial_coefficient': scale, 'adversarial_gradient_zero': na.item() == 0,
                   'gradient_cosine': (dot / (nc * na + eps)).item()}
        return [c.add(a, alpha=scale) for c, a in zip(consistency, adversarial)], metrics


@dataclass(frozen=True)
class ComponentConfig:
    """Explicit single-update fixtures; no horizons, warmup or automatic ramps."""
    consistency_microbatch: int = 2
    rho: float = .1
    penalty_beta: float = .1
    flip_probability: float = .25
    generator_lr: float = 2e-5
    discriminator_lr: float = 2e-5
    ema_decay: float = .99  # disposable smoke policy, not a training recommendation
    seed: int = 20261004
    memory_limit_bytes: int = 20_000_000_000

    def __post_init__(self):
        if type(self.consistency_microbatch) is not int or self.consistency_microbatch < 1:
            raise ValueError('Invalid consistency microbatch')
        for key in ('rho', 'penalty_beta', 'generator_lr', 'discriminator_lr'):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {key}')
        if not 0 <= self.flip_probability <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('Invalid augmentation/EMA coefficient')


class RandomStreams:
    """Keep SCM/dropout draws independent of all extra adversarial calls."""
    def __init__(self, device, seed):
        self.device = torch.device(device)
        self.states = {}
        for index, name in enumerate(('consistency', 'd_noise', 'd_flip', 'g_noise', 'g_flip')):
            generator = torch.Generator(device=self.device).manual_seed(seed + index * 100003)
            self.states[name] = generator.get_state()

    @contextmanager
    def use(self, name):
        cuda = self.device.type == 'cuda'
        index = (self.device.index if self.device.index is not None else torch.cuda.current_device()) if cuda else None
        with torch.random.fork_rng(devices=[index] if cuda else []):
            if cuda:
                torch.cuda.set_rng_state(self.states[name], index)
            else:
                torch.set_rng_state(self.states[name])
            try:
                yield
            finally:
                self.states[name] = (torch.cuda.get_rng_state(index) if cuda else torch.get_rng_state()).clone()


class AdversarialComponents:
    """Single-update interface. Callers provide batches and bound every use."""
    format_version = 1

    def __init__(self, source, config=None, *, device='cuda', critic=None):
        self.config = config or ComponentConfig()
        self.device = torch.device(device)
        self.source_config = copy.deepcopy(source['config'])
        if self.source_config.get('fm_fraction', 0) != 0:
            raise ValueError('Only completed consistency-only sources are supported')
        if self.source_config['l_end_ratio'] != .0003125:
            raise ValueError('Unexpected mature consistency gap')
        self.origin = copy.deepcopy(source['origin'])
        self.model = Soflow(**self.source_config['architecture']).to(self.device).train()
        self.model.load_state_dict(source['ema'], strict=True)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        # Initialization must not perturb the named training streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config.seed + 77)
            self.critic = (critic if critic is not None else StyleGAN2Critic()).to(self.device).train()
        self.g_optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.config.generator_lr,
                                             betas=(.9, .999), weight_decay=0)
        self.d_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.discriminator_lr,
                                            betas=(0., .99), weight_decay=0)
        self.augment = AugmentPipe(p=.12, xflip=1e8, yflip=0, scale=1,
                                   rotate_frac=0, aniso=1, translate_frac=1)
        self.streams = RandomStreams(self.device, self.config.seed)
        self.g_steps = self.d_steps = self.g_images = self.d_images = 0
        self.parameters = list(self.model.parameters())
        self.counts = {'generator': sum(p.numel() for p in self.parameters),
                       'discriminator': sum(p.numel() for p in self.critic.parameters())}
        if max(self.counts.values()) >= 100_000_000:
            raise ValueError('A learned model exceeds the separate parameter limit')

    @staticmethod
    def checkpoint_source(path, expected_hash=None):
        sha = digest(path)
        if expected_hash is not None and sha != expected_hash:
            raise ValueError('Source checkpoint hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        return {'ema': state['ema'], 'config': state['config'],
                'origin': {'checkpoint': str(path), 'sha256': sha, 'state_key': 'ema',
                           'source_step': state['step']}}

    def _batch(self, images, categories):
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError('These components require FP32 execution with autocast disabled')
        if len(images) == 0 or len(images) % 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError('Batch must contain complete ordered groups of four 64x64 RGB images')
        if images.dtype != torch.float32 or categories.dtype != torch.long or categories.shape != (len(images),):
            raise ValueError('Expected FP32 images and int64 class vector')
        if not torch.isfinite(images).all() or images.abs().max() > 1 or ((categories < 0) | (categories >= 151)).any():
            raise ValueError('Images/labels outside the declared domain')
        return images.to(self.device), categories.to(self.device)

    def _draw(self, name, images):
        with self.streams.use(name):
            return torch.randn_like(images)

    def _mask(self, name, count):
        with self.streams.use(name):
            return torch.rand(count, device=self.device) < self.config.flip_probability

    def _finite_step(self, optimizer, parameters):
        norm = torch.stack([p.grad.square().sum() for p in parameters if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite update gradient')
        self._memory()
        optimizer.step()
        self._memory()
        return norm.item()

    def _memory(self):
        if self.device.type == 'cuda' and torch.cuda.max_memory_allocated(self.device) >= self.config.memory_limit_bytes:
            raise RuntimeError('Combined workload exceeded 20 GB allocation')

    def discriminator_update(self, images, categories):
        images, categories = self._batch(images, categories)
        noise = self._draw('d_noise', images)
        mask = self._mask('d_flip', len(images))
        self.g_optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        totals = {}
        for start in range(0, len(images), 4):
            sl = slice(start, start + 4)
            with torch.no_grad():
                _, fake = endpoint(self.model, noise[sl], categories[sl])
            row = discriminator_objective(self.critic, images[sl], fake, categories[sl],
                                          mask[sl], self.config.penalty_beta)
            weight = 4 / len(images)
            (row['loss'] * weight).backward()
            for key, value in row.items():
                totals[key] = totals.get(key, 0.) + value.detach().item() * weight
            del row
        totals['grad_norm'] = self._finite_step(self.d_optimizer, list(self.critic.parameters()))
        self.d_optimizer.zero_grad(set_to_none=True)
        self.d_steps += 1
        self.d_images += len(images)
        return dict(totals, d_steps=self.d_steps, g_steps=self.g_steps)

    def consistency_backward(self, images, categories):
        """Exact existing SCM, fixed terminal gap, still paired training dropout."""
        self.model.train()
        stats = dict(loss_surrogate=0., consistency_mse=0., weight=0.)
        cfg = self.source_config
        with self.streams.use('consistency'):
            for start in range(0, len(images), self.config.consistency_microbatch):
                x = images[start:start + self.config.consistency_microbatch]
                c = categories[start:start + self.config.consistency_microbatch]
                if cfg['augmentation']:
                    x, augmentation = self.augment(x)
                else:
                    augmentation = x.new_zeros(len(x), 6)
                time1 = torch.sigmoid(torch.randn(len(x), device=self.device) * cfg['time_std'][0] + cfg['time_mean'][0])
                time2 = torch.sigmoid(torch.randn(len(x), device=self.device) * cfg['time_std'][1] + cfg['time_mean'][1])
                t = time2.clamp(min=1e-4)
                s = torch.minimum(time1, t - 1e-4)
                l = torch.minimum(t - .0003125 * (t - s), t - 1e-4)
                values = self.model.get_consistency_loss(x, t, l, s, c, None, 1., 1.,
                                                         cfg['p'], cfg['eps'], 1., augmentation)
                if not all(torch.isfinite(v) for v in values):
                    raise FloatingPointError('Non-finite consistency loss')
                fraction = len(x) / len(images)
                (values[0] * fraction).backward()
                for key, value in zip(stats, values):
                    stats[key] += fraction * value.detach().item()
        return stats

    def generator_update(self, images, categories, *, rho=None):
        images, categories = self._batch(images, categories)
        self.g_optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        rho = self.config.rho if rho is None else float(rho)
        if not math.isfinite(rho) or rho < 0:
            raise ValueError('rho must be finite and nonnegative')
        stats = self.consistency_backward(images, categories)
        gc = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in self.parameters]
        self.g_optimizer.zero_grad(set_to_none=True)
        stats.update(adversarial_loss=0., clipped_fraction=0., masked_gradient_energy_fraction=0.)
        if rho != 0:
            noise = self._draw('g_noise', images)
            mask = self._mask('g_flip', len(images))
            for start in range(0, len(images), 4):
                sl = slice(start, start + 4)
                raw, visible = endpoint(self.model, noise[sl], categories[sl])
                visible.retain_grad()
                loss = generator_objective(self.critic, images[sl], visible, categories[sl], mask[sl])
                fraction = 4 / len(images)
                (loss * fraction).backward()
                with torch.no_grad():
                    clipped = (raw < -1) | (raw > 1)
                    energy = visible.grad.square()
                    stats['adversarial_loss'] += loss.item() * fraction
                    stats['clipped_fraction'] += clipped.float().mean().item() * fraction
                    stats['masked_gradient_energy_fraction'] += ((energy * clipped).sum() / energy.sum().clamp_min(1e-30)).item() * fraction
                del raw, visible, loss
        ga = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in self.parameters]
        gradients, balance = balanced_gradients(gc, ga, rho)
        stats.update(balance)
        for p, gradient in zip(self.parameters, gradients):
            p.grad = gradient
        # Compute the actual optimizer displacement without another GPU model copy.
        before = [p.detach().cpu().clone() for p in self.parameters]
        stats['grad_norm'] = self._finite_step(self.g_optimizer, self.parameters)
        stats['parameter_update_norm'] = math.sqrt(sum((p.detach().cpu() - old).square().sum().item()
                                                      for p, old in zip(self.parameters, before)))
        with torch.no_grad():
            for target, source in zip(self.ema.parameters(), self.parameters):
                target.lerp_(source, 1 - self.config.ema_decay)
            for target, source in zip(self.ema.buffers(), self.model.buffers()):
                target.copy_(source)
        self.g_optimizer.zero_grad(set_to_none=True)
        self.g_steps += 1
        self.g_images += len(images)
        return dict(stats, g_steps=self.g_steps, d_steps=self.d_steps)

    def save(self, path, *, extra=None):
        atomic_save(path, {'format_version': self.format_version, 'config': asdict(self.config),
            'source_config': self.source_config, 'origin': self.origin,
            'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
            'critic': self.critic.state_dict(), 'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(), 'streams': self.streams.states,
            'counters': [self.g_steps, self.d_steps, self.g_images, self.d_images],
            'extra': extra or {}})

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in (('format_version', self.format_version), ('config', asdict(self.config)),
                           ('source_config', self.source_config), ('origin', self.origin)):
            if state[key] != value:
                raise ValueError(f'Component recovery changes {key}')
        for key in ('model', 'ema', 'critic', 'g_optimizer', 'd_optimizer'):
            getattr(self, key).load_state_dict(state[key])
        self.streams.states = {key: value.cpu() for key, value in state['streams'].items()}
        self.g_steps, self.d_steps, self.g_images, self.d_images = state['counters']
        return state['extra']
