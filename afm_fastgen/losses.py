"""AFM ab60013 formulas, specialized to s=1,t=0; see docs/IMPLEMENTATION.md.

Gradient normalization adapted from ByteDance's MIT-licensed grad_norm.py.
Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
The license is preserved at vendor/adversarial-flow-models/LICENSE.txt.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def cosine(step, steps, start, end):
    if steps <= 1 or step >= steps - 1:
        return end
    if step < 0:
        raise ValueError('Negative schedule index')
    return end + (start - end) * (1 + math.cos(math.pi * step / (steps-1))) / 2


def discriminator_losses(real, fake, real_gp, fake_gp, gp_indices, eps, gp, cp):
    return {
        'd_adv': F.softplus(fake - real).mean(),
        'd_r1': (real_gp-real[gp_indices]).square().mean() * (gp/eps**2),
        'd_r2': (fake_gp-fake[gp_indices]).square().mean() * (gp/eps**2),
        'd_center': (real+fake).square().mean() * cp,
    }


def generator_losses(real, fake, prediction, source, ot):
    return {'g_adv': F.softplus(real-fake).mean(),
            'g_ot': (prediction-source).square().mean() * ot}


class GradientNormalization(nn.Module):
    def __init__(self, decay=0.9, eps=1e-8, target_scale=1.0):
        super().__init__()
        self.decay, self.eps = decay, eps
        self.target_scale = target_scale
        self.register_buffer('square_avg', torch.tensor(0.0))
        self.register_buffer('backwards', torch.tensor(0, dtype=torch.long))

    def forward(self, x, loss_weight=1.0):
        return _Normalize.apply(x, self.square_avg, self.backwards,
                                self.decay, self.eps, loss_weight, self.target_scale)


class _Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, square_avg, backwards, decay, eps, weight, target_scale):
        ctx.state = square_avg, backwards, decay, eps, weight, target_scale
        return x.clone()

    @staticmethod
    def backward(ctx, gradient):
        avg, backwards, decay, eps, weight, target_scale = ctx.state
        # Track the LOCAL MEAN adversarial gradient, then retain the microbatch
        # weight in the outgoing gradient. Upstream cancels accumulation weights.
        local = gradient.float() / weight
        square = local.square().sum() * local.numel()
        avg.lerp_(square, 1-decay)
        backwards.add_(1)
        return gradient * (target_scale / (avg.sqrt()+eps)), None, None, None, None, None, None


def augmentation_parameters(batch, device, probability=0.5, translation=8, cutout=16):
    active = torch.rand(batch, device=device) < probability
    shift = torch.randint(-translation, translation+1, (batch, 2), device=device)
    shift = shift * active[:, None]
    centers = torch.randint(0, 64, (batch, 2), device=device)
    erase = torch.rand(batch, device=device) < probability
    return {'shift': shift, 'centers': centers, 'erase': erase, 'cutout': cutout}


def apply_augmentation(image, params):
    # Integer translation with white fill; same differentiable map on real/fake.
    b, c, h, w = image.shape
    yy = torch.arange(h, device=image.device)[None, :, None] - params['shift'][:, 0, None, None]
    xx = torch.arange(w, device=image.device)[None, None, :] - params['shift'][:, 1, None, None]
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    shifted = image[torch.arange(b, device=image.device)[:, None, None],
                    :, yy.clamp(0, h-1), xx.clamp(0, w-1)].permute(0, 3, 1, 2)
    shifted = torch.where(valid[:, None], shifted, 1.0)
    cy, cx = params['centers'].unbind(1)
    cut = params['cutout']
    yy0 = torch.arange(h, device=image.device)[None, :, None]
    xx0 = torch.arange(w, device=image.device)[None, None, :]
    mask = ((yy0 >= cy[:, None, None]-cut//2) & (yy0 < cy[:, None, None]+cut//2)
            & (xx0 >= cx[:, None, None]-cut//2) & (xx0 < cx[:, None, None]+cut//2)
            & params['erase'][:, None, None])
    return torch.where(mask[:, None], 1.0, shifted)


def paired_augment(real, fake, params):
    return apply_augmentation(real, params), apply_augmentation(fake, params)
