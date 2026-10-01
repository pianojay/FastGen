"""Complete alternating updates, explicit RNG streams and lossless recovery."""
import contextlib
import copy
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .common import PARTITION_SHA, atomic_json, host_memory, object_hash
from .data import BalancedData
from .flow import (corrupt, denoise, student_rollout, sid_surrogate,
                   discriminator_loss, generator_adversarial, adversarial_scale)
from .networks import FlowNet, parameter_count


def setup_torch():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1
    from .gpu import EXPECTED_UUID
    prop = torch.cuda.get_device_properties(0)
    assert str(prop.uuid).removeprefix('GPU-') == EXPECTED_UUID.removeprefix('GPU-'), (str(prop.uuid), EXPECTED_UUID)
    assert torch.cuda.get_device_capability(0) == (7, 5)
    torch.cuda.set_per_process_memory_fraction(19_500_000_000 / prop.total_memory, 0)


def tensor_hash(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


class Streams:
    names = ['teacher_data', 'teacher_noise', 'teacher_dropout', 'fake', 'generator',
             'fake_dropout', 'adversarial_data', 'adversarial_noise', 'adversarial_dropout']

    def __init__(self, seed):
        self.generators = {name: torch.Generator().manual_seed(seed + 1009 * i)
                           for i, name in enumerate(self.names)}
        self.draws = {name: 0 for name in self.names}

    def get(self, name):
        self.draws[name] += 1
        return self.generators[name]

    def state_dict(self):
        return {'states': {k: v.get_state() for k, v in self.generators.items()},
                'draws': self.draws.copy()}

    def load_state_dict(self, state):
        for k, value in state['states'].items():
            self.generators[k].set_state(value.cpu())
        self.draws = state['draws'].copy()

    @contextlib.contextmanager
    def dropout(self, name):
        seed = int(torch.randint(2**31 - 1, (), generator=self.get(name)))
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            yield


def draw_batch(rng, batch, nfe):
    labels = torch.randint(151, (batch,), generator=rng)
    step = int(torch.randint(nfe, (), generator=rng))
    noises = [torch.randn(batch, 3, 64, 64, generator=rng) for _ in range(nfe)]
    noise = torch.randn(batch, 3, 64, 64, generator=rng)
    t = torch.sigmoid(math.log(2) + 1.6 * torch.randn(batch, generator=rng)).clamp(.01, .99)
    identity = tensor_hash([labels, torch.tensor(step), *noises, noise, t])
    return labels, step, noises, noise, t, identity


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class TrainingSystem:
    def __init__(self, config, run, teacher_state=None, teacher_hash=None):
        self.config, self.run = copy.deepcopy(config), run
        self.nfe = 0 if run == 'T' else int(run[1:])
        self.adversarial = run.startswith('A')
        self.step = 0
        self.teacher_hash = teacher_hash
        self.batch = config['effective_batch']
        self.micro = config['microbatch']
        assert self.batch % self.micro == 0
        self.seed = config['teacher_seed'] if run == 'T' else config['student_seed'] + self.nfe * 10000
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.streams = Streams(self.seed)
        self.data = BalancedData()
        self.g = FlowNet(config['checkpointing']).cuda()
        self.teacher = self.fake = None
        self.recipe = config['teacher'] if run == 'T' else config['student']
        if run != 'T':
            if teacher_state is None:
                raise ValueError('Students must be initialized from the selected eligible teacher')
            self.g.load_state_dict(teacher_state)
            self.teacher = FlowNet(config['checkpointing']).cuda().eval().requires_grad_(False)
            self.teacher.load_state_dict(teacher_state)
            self.fake = FlowNet(config['checkpointing']).cuda()
            self.fake.load_state_dict(teacher_state)
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=self.recipe['lr'],
            betas=tuple(self.recipe['betas']), eps=1e-8, foreach=False)
        self.fake_optimizer = (torch.optim.Adam(self.fake.parameters(),
            lr=self.recipe['fake_lr'], betas=tuple(self.recipe['betas']), eps=1e-8, foreach=False)
            if self.fake is not None else None)
        self.ema = {k: v.detach().clone() for k, v in self.g.state_dict().items()}
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16',
            init_scale=config['loss_scale'], growth_interval=1000000000)
        self.fake_scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16',
            init_scale=config['loss_scale'], growth_interval=1000000000)
        self.initial_hash = tensor_hash(list(self.g.state_dict().values()))
        self.exposures = {'teacher_images': 0, 'g_images': 0, 'fake_images': 0,
                          'd_real_images': 0, 'g_updates': 0, 'fake_updates': 0}
        self.counts = {'teacher': parameter_count(self.teacher) if self.teacher else parameter_count(self.g),
                       'student': parameter_count(self.g) if self.teacher else None,
                       'fake_score_discriminator': parameter_count(self.fake) if self.fake else None,
                       'discriminator_head': 0,
                       'ema_state_elements': sum(v.numel() for v in self.ema.values())}

    def amp(self):
        # A rollout mixes no-grad preceding calls with the trained final call.
        # Cached no-grad weight casts must never leak into that final call.
        return torch.autocast('cuda', dtype=torch.float16, enabled=self.config['precision'] == 'fp16',
                              cache_enabled=False)

    def schedule(self):
        # Warmup in complete update units, then constant LR. No hidden scheduler state.
        scale = min(1., (self.step + 1) / max(1, self.recipe['lr_warmup']))
        for group in self.optimizer.param_groups:
            group['lr'] = self.recipe['lr'] * scale
        if self.fake_optimizer:
            for group in self.fake_optimizer.param_groups:
                group['lr'] = self.recipe['fake_lr'] * scale

    def backward(self, loss, scaler):
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f'Nonfinite loss at {self.run} {self.step}; no samples discarded')
        scaler.scale(loss).backward()

    def finish_optimizer(self, model, optimizer, scaler):
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.recipe['grad_clip'],
                                            error_if_nonfinite=True, foreach=False)
        # Reject overflow instead of silently skipping a phase or breaking pair exposure.
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient {self.run} {self.step}')
        scaler.step(optimizer)
        scaler.update()
        return float(norm)

    @torch.no_grad()
    def update_ema(self):
        # Half-life in presentations; ramp-up avoids stale scratch initialization.
        half = min(self.recipe['ema_half_life_images'], (self.step + 1) * self.batch * .05)
        beta = 0.5 ** (self.batch / max(1., half))
        for k, v in self.g.state_dict().items():
            if v.is_floating_point():
                self.ema[k].lerp_(v.detach(), 1 - beta)
            else:
                self.ema[k].copy_(v)

    def teacher_update(self):
        self.g.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        x, labels, draw = self.data.take(self.batch, self.streams.get('teacher_data'))
        rng = self.streams.get('teacher_noise')
        noise = torch.randn(x.shape, generator=rng)
        t = torch.rand(self.batch, generator=rng)
        identity = tensor_hash([draw['indices'], draw['flip'], noise, t])
        loss_sum = 0.
        for start in range(0, self.batch, self.micro):
            sl = slice(start, start + self.micro)
            xi, yi, ni, ti = (v[sl].cuda() for v in [x, labels, noise, t])
            with self.streams.dropout('teacher_dropout'), self.amp():
                prediction = self.g(corrupt(xi, ni, ti), ti, yi)
                loss = (prediction - (ni - xi)).square().mean() * self.micro / self.batch
            self.backward(loss, self.scaler)
            loss_sum += float(loss.detach())
        norm = self.finish_optimizer(self.g, self.optimizer, self.scaler)
        self.exposures['teacher_images'] += self.batch
        self.exposures['g_updates'] += 1
        return {'fm_loss': loss_sum, 'g_grad_norm': norm, 'draw_teacher': identity}

    def student_update(self):
        self.g.eval().requires_grad_(False)
        self.fake.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        self.fake_optimizer.zero_grad(set_to_none=True)
        labels, step, noises, noise, t, fake_id = draw_batch(self.streams.get('fake'), self.batch, self.nfe)
        if self.config.get('probe_full_rollout'):
            step = self.nfe - 1
        d_weight = self.recipe['d_weight'] if self.adversarial else 0.
        if d_weight:
            real, real_labels, real_draw = self.data.take(self.batch, self.streams.get('adversarial_data'))
            d_rng = self.streams.get('adversarial_noise')
            real_noise = torch.randn(real.shape, generator=d_rng)
            real_t = torch.sigmoid(math.log(2) + 1.6 * torch.randn(self.batch, generator=d_rng)).clamp(.01, .99)
        fm_total = d_total = 0.
        for start in range(0, self.batch, self.micro):
            sl = slice(start, start + self.micro)
            yi, ni, ti = (v[sl].cuda() for v in [labels, noise, t])
            zi = [v[sl].cuda() for v in noises]
            with torch.no_grad(), self.amp():
                images = student_rollout(self.g, zi, yi, self.nfe, train_step=step)
            images = images.detach()
            with self.streams.dropout('fake_dropout'), self.amp():
                velocity, logits = self.fake(corrupt(images, ni, ti), ti, yi, return_flag='encoder_decoder')
                fm = (velocity - (ni - images)).square().mean()
                loss = fm * self.micro / self.batch
            if d_weight:
                ri, ryi, rni, rti = (v[sl].cuda() for v in [real, real_labels, real_noise, real_t])
                with self.streams.dropout('adversarial_dropout'), self.amp():
                    real_logits = self.fake(corrupt(ri, rni, rti), rti, ryi, return_flag='encoder')
                    d_loss = discriminator_loss(real_logits, logits).mean()
                    loss = loss + d_weight * d_loss * self.micro / self.batch
                d_total += float(d_loss.detach()) * self.micro / self.batch
            self.backward(loss, self.fake_scaler)
            fm_total += float(fm.detach()) * self.micro / self.batch
        fake_norm = self.finish_optimizer(self.fake, self.fake_optimizer, self.fake_scaler)
        self.fake_optimizer.zero_grad(set_to_none=True)
        self.fake.eval().requires_grad_(False)
        self.g.eval().requires_grad_(True)  # Dropout disabled in all generator calls.
        labels, step_g, noises, noise, t, g_id = draw_batch(self.streams.get('generator'), self.batch, self.nfe)
        if self.config.get('probe_full_rollout'):
            step_g = self.nfe - 1
        weight = (adversarial_scale(self.step, self.recipe['gan_warmup'], self.recipe['gan_ramp'],
                                   self.recipe['g_gan_weight']) if self.adversarial else 0.)
        sid_total = gan_total = 0.
        for start in range(0, self.batch, self.micro):
            sl = slice(start, start + self.micro)
            yi, ni, ti = (v[sl].cuda() for v in [labels, noise, t])
            zi = [v[sl].cuda() for v in noises]
            with self.amp():
                images = student_rollout(self.g, zi, yi, self.nfe, train_step=step_g)
                xt = corrupt(images, ni, ti)
                fake_clean, logits = denoise(self.fake, xt, ti, yi, with_logits=True)
                real_clean = denoise(self.teacher, xt, ti, yi)
                sid = sid_surrogate(images, real_clean, fake_clean, ti, self.recipe['alpha']).mean()
                loss = self.recipe['sid_weight'] * sid
                if weight:
                    gan = generator_adversarial(logits, ti).mean()
                    loss = loss + weight * gan
                    gan_total += float(gan.detach()) * self.micro / self.batch
                loss = loss * self.micro / self.batch
            self.backward(loss, self.scaler)
            sid_total += float(sid.detach()) * self.micro / self.batch
        g_norm = self.finish_optimizer(self.g, self.optimizer, self.scaler)
        assert self.g.backbone.dec['64x64_aux_conv'].weight.grad is not None
        assert all(p.grad is None for p in self.teacher.parameters())
        self.exposures['g_images'] += self.batch
        self.exposures['fake_images'] += self.batch
        self.exposures['d_real_images'] += self.batch if d_weight else 0
        self.exposures['g_updates'] += 1
        self.exposures['fake_updates'] += 1
        return {'sid_loss': sid_total, 'fake_fm_loss': fm_total, 'd_bce_loss': d_total,
                'g_gan_loss': gan_total, 'gan_coefficient': weight, 'g_grad_norm': g_norm,
                'fake_D_grad_norm': fake_norm, 'draw_fake': fake_id, 'draw_g': g_id,
                'step_fake': step, 'step_g': step_g}

    def update(self):
        torch.cuda.synchronize()
        start = time.monotonic()
        self.schedule()
        metrics = self.teacher_update() if self.run == 'T' else self.student_update()
        self.update_ema()
        self.step += 1
        torch.cuda.synchronize()
        metrics.update({'step': self.step, 'seconds': time.monotonic() - start,
                        'exposures': self.exposures.copy(), 'lr': self.optimizer.param_groups[0]['lr'],
                        'loss_scale': self.scaler.get_scale()})
        if torch.cuda.max_memory_allocated() >= 20_000_000_000:
            raise RuntimeError('Peak allocated memory violates strict 20GB limit')
        return metrics

    def snapshot(self, identity=None):
        return cpu_copy({'format': 1, 'run': self.run, 'config': self.config,
            'config_sha256': object_hash(self.config), 'partition_sha256': PARTITION_SHA,
            'identity': identity, 'teacher_sha256': self.teacher_hash, 'initial_sha256': self.initial_hash,
            'step': self.step, 'student': self.g.state_dict(), 'ema': self.ema,
            'fake': self.fake.state_dict() if self.fake else None,
            'optimizer': self.optimizer.state_dict(),
            'fake_optimizer': self.fake_optimizer.state_dict() if self.fake_optimizer else None,
            'scaler': self.scaler.state_dict(), 'fake_scaler': self.fake_scaler.state_dict(),
            'streams': self.streams.state_dict(), 'exposures': self.exposures,
            'data_presentations': self.data.presentations,
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(0),
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state()})

    def restore(self, state, identity=None):
        assert state['run'] == self.run and state['config_sha256'] == object_hash(self.config)
        assert state['partition_sha256'] == PARTITION_SHA
        assert state['teacher_sha256'] == self.teacher_hash
        if identity is not None:
            assert state['identity'] == identity
        self.g.load_state_dict(state['student'])
        self.ema = {k: v.cuda().clone() for k, v in state['ema'].items()}
        if self.fake:
            self.fake.load_state_dict(state['fake'])
            self.fake_optimizer.load_state_dict(state['fake_optimizer'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        self.fake_scaler.load_state_dict(state['fake_scaler'])
        self.streams.load_state_dict(state['streams'])
        self.step = state['step']
        self.initial_hash = state['initial_sha256']
        self.exposures = copy.deepcopy(state['exposures'])
        self.data.presentations = state['data_presentations']
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state(state['rng_cuda'], 0)
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])


def memory_metrics():
    from .gpu import device_status
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'device': device_status(), 'host': host_memory()}


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    with tmp.open('rb') as f:
        import os
        os.fsync(f.fileno())
    tmp.replace(path)
