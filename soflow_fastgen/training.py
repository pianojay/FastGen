import copy
import math
import time

import torch
from .augment import AugmentPipe
from .models import Soflow
from .common import atomic_save, rng_state, restore_rng, resources


def flow_matching_count(batch_size, fraction):
    """Partition the effective batch before microbatching, without rounding its recipe."""
    if (type(batch_size) is not int or batch_size <= 0
            or type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction < 1):
        raise ValueError('Flow-matching fraction must be finite in [0, 1) with a positive batch')
    count = batch_size * fraction
    if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9):
        raise ValueError('Flow-matching fraction must allocate a whole number of examples')
    return round(count)


class Trainer:
    def __init__(self, config, device='cuda'):
        fm_count = flow_matching_count(config['batch_size'], config.get('fm_fraction', 0.))
        if fm_count and (not math.isfinite(config['fm_time_mean'])
                         or not math.isfinite(config['fm_time_std']) or config['fm_time_std'] <= 0):
            raise ValueError('Flow-matching time distribution must have finite mean and positive std')
        self.config, self.device = config, torch.device(device)
        self.model = Soflow(**config['architecture']).to(device).train()
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=config['learning_rate'],
            betas=config['betas'], weight_decay=config['weight_decay'])
        self.augment = AugmentPipe(p=.12, xflip=1e8, yflip=0, scale=1,
            rotate_frac=0, aniso=1, translate_frac=1)
        self.step, self.epoch, self.cursor = 0, 0, 0

    def backward(self, images, categories, fraction, objective='consistency'):
        if objective not in ('consistency', 'velocity'):
            raise ValueError('Unknown training objective')
        images = images.to(self.device, non_blocking=True)
        categories = categories.to(self.device, non_blocking=True)
        if self.config['augmentation']:
            images, augmentation = self.augment(images)
        else:
            augmentation = torch.zeros(len(images), 6, device=self.device)
        count = len(images)
        if objective == 'velocity':
            t = torch.sigmoid(torch.randn(count, device=self.device) * self.config['fm_time_std']
                              + self.config['fm_time_mean']).clamp(min=1e-4)
            loss, mse, weight = self.model.get_velocity_loss(images, t, categories,
                None, 1., 1., self.config['p'], self.config['eps'], 1., augmentation)
        else:
            # Preserve the original consistency-only path and its RNG draw order.
            time1 = torch.sigmoid(torch.randn(count, device=self.device) * self.config['time_std'][0] + self.config['time_mean'][0])
            time2 = torch.sigmoid(torch.randn(count, device=self.device) * self.config['time_std'][1] + self.config['time_mean'][1])
            t = time2.clamp(min=1e-4)
            s = torch.minimum(time1, t - 1e-4)
            l = self.model.get_middle_time(t, s, self.step, self.config['total_steps'],
                self.config['l_init_ratio'], self.config['l_end_ratio'], 'Exponent')
            l = torch.minimum(l, t - 1e-4)
            loss, mse, weight = self.model.get_consistency_loss(images, t, l, s, categories,
                None, 1., 1., self.config['p'], self.config['eps'], 1., augmentation)
        if not all(torch.isfinite(v).all() for v in (loss, mse, weight)):
            raise FloatingPointError(f'Non-finite loss at update {self.step}')
        (fraction * loss).backward()
        return torch.stack([loss.detach(), mse.detach(), weight.detach()]) * fraction

    def update(self, images, categories):
        fm_count = flow_matching_count(len(images), self.config.get('fm_fraction', 0.))
        if len(categories) != len(images):
            raise ValueError('Image and category batch sizes differ')
        torch.cuda.synchronize(self.device)
        start = time.monotonic()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(3, device=self.device)
        microbatch = self.config['microbatch_size']
        # The loader shuffles the effective batch. Assign its first fm_count
        # examples to FM and the rest to consistency, each exactly once.
        if fm_count:
            fm_stats = torch.zeros(3, device=self.device)
            for offset in range(0, fm_count, microbatch):
                stop = min(offset + microbatch, fm_count)
                fm_stats += self.backward(images[offset:stop], categories[offset:stop],
                                          (stop-offset)/len(images), 'velocity')
        for offset in range(fm_count, len(images), microbatch):
            stop = min(offset + microbatch, len(images))
            stats += self.backward(images[offset:stop], categories[offset:stop], (stop-offset)/len(images))
        norm = torch.stack([p.grad.detach().float().square().sum() for p in self.model.parameters() if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Non-finite gradient at update {self.step}')
        self.optimizer.step()
        decay = min(self.config['ema_decay'], (1 + self.step)/(10 + self.step))
        with torch.no_grad():
            for target, source in zip(self.ema.parameters(), self.model.parameters()):
                target.lerp_(source, 1-decay)
            for target, source in zip(self.ema.buffers(), self.model.buffers()):
                target.copy_(source)
        self.step += 1
        self.cursor += len(images)
        torch.cuda.synchronize(self.device)
        result = dict(zip(('loss_surrogate','consistency_mse','weight'), stats.tolist()))
        if fm_count:
            fm_fraction = fm_count / len(images)
            # Report each objective's conditional means; never label a blended
            # velocity/consistency MSE as a consistency diagnostic.
            result = {key: value / (1-fm_fraction) for key, value in result.items()}
            result.update(dict(zip(('velocity_loss', 'velocity_mse', 'velocity_weight'),
                                   (fm_stats / fm_fraction).tolist())))
            result.update(fm_examples=fm_count, consistency_examples=len(images)-fm_count,
                          fm_fraction=fm_fraction,
                          mixed_loss_surrogate=float(stats[0] + fm_stats[0]))
        result.update(step=self.step, grad_norm=float(norm), seconds=time.monotonic()-start,
            ema_decay=decay, **resources())
        if result['allocated_bytes'] >= self.config['memory_limit_bytes']:
            raise RuntimeError('Training exceeded the 20 GB allocated-memory limit')
        return result

    def save(self, path, extra=None):
        atomic_save(path, {'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'step': self.step, 'epoch': self.epoch,
            'cursor': self.cursor, 'rng': rng_state(), 'config': self.config, 'extra': extra or {}})

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        runtime_fields = {'run_dir', 'data_root', 'partition', 'num_workers', 'torch_threads',
                          'microbatch_size', 'logging_steps', 'checkpoint_steps', 'sample_steps',
                          'eval_batch_size'}
        differences = [key for key in sorted(self.config.keys() | state['config'].keys())
                       if key not in runtime_fields and self.config.get(key) != state['config'].get(key)]
        if differences:
            raise ValueError(f'Resume changes training semantics: {differences}')
        self.model.load_state_dict(state['model'])
        self.ema.load_state_dict(state['ema'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step, self.epoch, self.cursor = (state[k] for k in ('step','epoch','cursor'))
        restore_rng(state['rng'])
        return state['extra']
