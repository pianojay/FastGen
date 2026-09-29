"""Endpoint prediction and re-noising, matching upstream SoFlow by default."""
import math

import torch


def query_times(nfe, intermediate_times=None):
    """Validate re-noising levels; the first network query is always at t=1."""
    if type(nfe) is not int or nfe not in (1, 2, 3, 4):
        raise ValueError('NFE must be 1, 2, 3, or 4')
    levels = (tuple(1 - i / nfe for i in range(1, nfe))
              if intermediate_times is None else tuple(intermediate_times))
    if len(levels) != nfe - 1:
        raise ValueError('Expected NFE-1 intermediate times')
    times = (1., *levels, 0.)
    if (not all(isinstance(t, (int, float)) and math.isfinite(t) for t in times)
            or not all(a > b for a, b in zip(times, times[1:]))):
        raise ValueError('Times must decrease strictly from 1 to 0')
    # Distinct Python floats must also remain ordered at the model's precision.
    rounded = torch.tensor(times, dtype=torch.float32)
    if not bool(torch.all(rounded[:-1] > rounded[1:])):
        raise ValueError('Times are not distinct at float32 precision')
    return times[:-1]


@torch.no_grad()
def sample(backbone, shape, *, device, category=None, nfe=1, generator=None,
           intermediate_times=None, noise=None):
    if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64) or shape[0] < 1:
        raise ValueError('Expected shape (B, 3, 64, 64), B > 0')
    times = query_times(nfe, intermediate_times)
    device = torch.device(device)
    if noise is not None:
        if generator is not None or category is None:
            raise ValueError('Explicit noise requires categories and no generator')
        if len(noise) != nfe:
            raise ValueError('Expected one initial and NFE-1 re-noising tensors')
        for draw in noise:
            if (draw.shape != torch.Size(shape) or draw.device.type != device.type
                    or (device.index is not None and draw.device.index != device.index)
                    or draw.dtype != torch.float32 or not torch.isfinite(draw).all()):
                raise ValueError('Noise must be finite float32 tensors matching shape/device')
    if category is None:
        category = torch.randint(151, (shape[0],), device=device, generator=generator)
    else:
        category = category.to(device)
    if category.dtype != torch.long or category.shape != (shape[0],):
        raise ValueError('category must be an int64 vector of length B')
    if ((category < 0) | (category >= 151)).any():
        raise ValueError('category IDs must be in [0, 150]')
    x = (torch.randn(shape, device=device, generator=generator)
         if noise is None else noise[0])
    zero = torch.zeros(shape[0], device=device)
    augmentation = torch.zeros(shape[0], 6, device=device)
    for index in range(nfe):
        t = torch.full_like(zero, times[index])
        x = backbone.solution_operator(x, t, zero, category, augmentation)
        if index + 1 < nfe:
            next_t = times[index + 1]
            draw = (torch.randn(shape, device=device, generator=generator)
                    if noise is None else noise[index + 1])
            x = (1 - next_t) * x + next_t * draw
    if not torch.isfinite(x).all():
        raise FloatingPointError('Non-finite generated samples')
    return x.clamp(-1, 1)
