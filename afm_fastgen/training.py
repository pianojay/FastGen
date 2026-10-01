"""Alternating D/G cycles, full recoverable state, FP32 engineering baseline."""
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .common import (ROOT, atomic_torch, digest, restore_rng, rng_state, seed_all,
                     source_identity, verify_protected)
from .data import DataStream, Images, prepare_cache
from .losses import (GradientNormalization, augmentation_parameters, cosine,
                     discriminator_losses, generator_losses, paired_augment)
from .models import Generator, Discriminator


class Trainer:
    def __init__(self, config, resume=None):
        self.config = config
        self.identity = {'source': source_identity(), 'protected': verify_protected(),
                         'config': copy.deepcopy(config)}
        seed_all(config['seed'])
        self.g = Generator(**config['generator']).cuda()
        self.d = Discriminator(**config['discriminator']).cuda()
        self.ema = copy.deepcopy(self.g).eval().requires_grad_(False)
        self.norm = GradientNormalization().cuda()
        self.parameters = {'generator': sum(p.numel() for p in self.g.parameters()),
                           'discriminator': sum(p.numel() for p in self.d.parameters())}
        assert max(self.parameters.values()) <= 100_000_000
        self.go = torch.optim.AdamW(self.g.parameters(), **config['optimizer'], foreach=False)
        self.do = torch.optim.AdamW(self.d.parameters(), **config['optimizer'], foreach=False)
        cache, record = prepare_cache(config)
        self.identity['data_cache_sha256'] = record['cache_sha256']
        self.identity['data_sources_sha256'] = digest(ROOT/'artifacts/data_cache/manifest.json')
        pixels = np.load(cache, mmap_mode='r')
        labels = Images(config, 'optimization').labels
        self.gs = DataStream(pixels, labels, config['seed']+11)
        self.ds = DataStream(pixels, labels, config['seed']+23)
        self.step = 0; self.g_updates = 0; self.d_updates = 0
        self.seconds = 0.; self.started = time.monotonic()
        if resume:
            self.load(resume)

    def cycle(self):
        t0 = time.monotonic()
        c = self.config; batch = c['batch_size']; micro = c['microbatch']
        ot = cosine(self.step, c['ot_decay_updates'], c['ot_start'], c['ot_end'])
        result = {'step': self.step+1, 'ot_coefficient': ot}
        self.g.train(); self.d.train()
        self.g.requires_grad_(False); self.d.requires_grad_(True)
        self.go.zero_grad(set_to_none=True); self.do.zero_grad(set_to_none=True)
        real, labels = self.ds.take(batch)
        noise = torch.randn_like(real)
        gp_count = max(1, round(batch*c['gp_fraction']))
        gp_mask = torch.zeros(batch, dtype=torch.bool, device='cuda')
        gp_mask[torch.randperm(batch, device='cuda')[:gp_count]] = True
        seen_gp = 0
        for lo in range(0, batch, micro):
            hi = min(batch, lo+micro); w = (hi-lo)/batch
            x, y, z = real[lo:hi], labels[lo:hi], noise[lo:hi]
            with torch.no_grad():
                fake = self.g(z, y)
            params = augmentation_parameters(len(x), x.device, c['augment_probability'],
                                             c['translate_pixels'], c['cutout_pixels'])
            ra, fa = paired_augment(x, fake, params)
            ix = torch.where(gp_mask[lo:hi])[0]; n = len(ix); seen_gp += n
            rg = ra[ix]+torch.randn_like(ra[ix])*c['gp_epsilon']
            fg = fa[ix]+torch.randn_like(fa[ix])*c['gp_epsilon']
            logits = self.d(torch.cat([ra, fa, rg, fg]), torch.cat([y, y, y[ix], y[ix]]))
            lr, lf, lrg, lfg = logits.split([len(x), len(x), n, n])
            if n:
                losses = discriminator_losses(lr, lf, lrg, lfg, ix, c['gp_epsilon'], c['gp_scale'], c['center_scale'])
                weights = {'d_adv': w, 'd_center': w, 'd_r1': n/gp_count, 'd_r2': n/gp_count}
            else:
                losses = {'d_adv': torch.nn.functional.softplus(lf-lr).mean(),
                          'd_center': (lr+lf).square().mean()*c['center_scale']}
                weights = {key: w for key in losses}
            total = sum(losses[k]*weights[k] for k in losses)
            total.backward()
            for key, value in losses.items():
                result[key] = result.get(key, 0.) + float(value.detach())*weights[key]
            result['d_real_logit'] = result.get('d_real_logit', 0.)+float(lr.detach().mean())*w
            result['d_fake_logit'] = result.get('d_fake_logit', 0.)+float(lf.detach().mean())*w
        assert seen_gp == gp_count
        result['gp_samples'] = seen_gp
        result['d_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(self.d.parameters(), c['gradient_clip'], error_if_nonfinite=True))
        self.do.step(); self.d_updates += 1; self.do.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); result['d_seconds'] = time.monotonic()-t0
        tg = time.monotonic()
        self.g.requires_grad_(True); self.d.requires_grad_(False)
        real, labels = self.gs.take(batch)
        noise = torch.randn_like(real)
        for lo in range(0, batch, micro):
            hi = min(batch, lo+micro); w = (hi-lo)/batch
            x, y, z = real[lo:hi], labels[lo:hi], noise[lo:hi]
            fake = self.g(z, y)
            params = augmentation_parameters(len(x), x.device, c['augment_probability'],
                                             c['translate_pixels'], c['cutout_pixels'])
            ra, fa = paired_augment(x, fake, params)
            with torch.no_grad():
                lr = self.d(ra, y)
            lf = self.d(self.norm(fa, loss_weight=w), y)
            losses = generator_losses(lr, lf, fake, z, ot)
            (sum(losses.values())*w).backward()
            for key, value in losses.items():
                result[key] = result.get(key, 0.)+float(value.detach())*w
            result['g_real_logit'] = result.get('g_real_logit', 0.)+float(lr.mean())*w
            result['g_fake_logit'] = result.get('g_fake_logit', 0.)+float(lf.detach().mean())*w
            result['fake_outside_unit_fraction'] = result.get('fake_outside_unit_fraction', 0.)+float((fake.detach().abs()>1).float().mean())*w
        result['g_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(self.g.parameters(), c['gradient_clip'], error_if_nonfinite=True))
        self.go.step(); self.g_updates += 1; self.go.zero_grad(set_to_none=True)
        with torch.no_grad():
            for ema, p in zip(self.ema.parameters(), self.g.parameters()):
                ema.lerp_(p, 1-c['ema_decay'])
        self.step += 1
        torch.cuda.synchronize()
        result['g_seconds'] = time.monotonic()-tg
        elapsed = time.monotonic()-t0; self.seconds += elapsed
        result.update({'cycle_seconds': elapsed, 'optimizer_seconds_total': self.seconds,
                       'g_updates': self.g_updates, 'd_updates': self.d_updates,
                       'g_real_presentations': self.gs.presentations, 'd_real_presentations': self.ds.presentations,
                       'generated_presentations': self.gs.presentations+self.ds.presentations,
                       'norm_square_avg': float(self.norm.square_avg), 'norm_backwards': int(self.norm.backwards)})
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError('Nonfinite metric')
        peak = torch.cuda.max_memory_allocated()
        if peak >= 20_000_000_000:
            raise MemoryError(f'Complete training workload peak {peak} exceeds strict 20GB limit')
        return result

    def state_dict(self):
        return {'identity': self.identity, 'g': self.g.state_dict(), 'd': self.d.state_dict(),
                'ema': self.ema.state_dict(), 'go': self.go.state_dict(), 'do': self.do.state_dict(),
                'norm': self.norm.state_dict(), 'rng': rng_state(),
                'gs': self.gs.state_dict(), 'ds': self.ds.state_dict(),
                'step': self.step, 'g_updates': self.g_updates, 'd_updates': self.d_updates,
                'optimizer_seconds': self.seconds}

    def save(self, path):
        atomic_torch(path, self.state_dict())

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['identity'] != self.identity:
            raise ValueError('Source, recipe, protected files or data identity changed on resume')
        for name in ['g', 'd', 'ema', 'go', 'do', 'norm', 'gs', 'ds']:
            getattr(self, name).load_state_dict(state[name])
        self.step, self.g_updates, self.d_updates = state['step'], state['g_updates'], state['d_updates']
        self.seconds = state['optimizer_seconds']; restore_rng(state['rng'])

    def export(self, path):
        state = {'backbone.'+key: value.detach().cpu() for key, value in self.ema.state_dict().items()}
        atomic_torch(path, {'state_dict': state})
