import copy
import time
from pathlib import Path

import torch
from .augment import AugmentPipe
from .models import Soflow
from .common import atomic_save, rng_state, restore_rng, resources


class Trainer:
    def __init__(self, config, device='cuda'):
        self.config, self.device = config, torch.device(device)
        self.model = Soflow(**config['architecture']).to(device).train()
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=config['learning_rate'],
            betas=config['betas'], weight_decay=config['weight_decay'])
        self.augment = AugmentPipe(p=.12, xflip=1e8, yflip=0, scale=1,
            rotate_frac=0, aniso=1, translate_frac=1)
        self.step, self.epoch, self.cursor = 0, 0, 0

    def backward(self, images, categories, fraction):
        images = images.to(self.device, non_blocking=True)
        categories = categories.to(self.device, non_blocking=True)
        if self.config['augmentation']:
            images, augmentation = self.augment(images)
        else:
            augmentation = torch.zeros(len(images), 6, device=self.device)
        count = len(images)
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
        torch.cuda.synchronize(self.device)
        start = time.monotonic()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(3, device=self.device)
        microbatch = self.config['microbatch_size']
        for offset in range(0, len(images), microbatch):
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
        differences = [key for key in self.config if key not in runtime_fields and self.config[key] != state['config'].get(key)]
        if differences:
            raise ValueError(f'Resume changes training semantics: {differences}')
        self.model.load_state_dict(state['model'])
        self.ema.load_state_dict(state['ema'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step, self.epoch, self.cursor = (state[k] for k in ('step','epoch','cursor'))
        restore_rng(state['rng'])
        return state['extra']
