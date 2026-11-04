"""Endpoint prediction and re-noising, matching upstream SoFlow inference."""
import torch


@torch.no_grad()
def sample(backbone, shape, *, device, category=None, nfe=1, generator=None):
    if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64) or shape[0] < 1:
        raise ValueError('Expected shape (B, 3, 64, 64), B > 0')
    if nfe not in (1, 2, 3, 4):
        raise ValueError('NFE must be 1, 2, 3, or 4')
    device = torch.device(device)
    if category is None:
        category = torch.randint(151, (shape[0],), device=device, generator=generator)
    else:
        category = category.to(device)
    if category.dtype != torch.long or category.shape != (shape[0],):
        raise ValueError('category must be an int64 vector of length B')
    if ((category < 0) | (category >= 151)).any():
        raise ValueError('category IDs must be in [0, 150]')
    x = torch.randn(shape, device=device, generator=generator)
    zero = torch.zeros(shape[0], device=device)
    augmentation = torch.zeros(shape[0], 6, device=device)
    for index in range(nfe):
        t = torch.full_like(zero, 1 - index / nfe)
        x = backbone.solution_operator(x, t, zero, category, augmentation)
        if index + 1 < nfe:
            next_t = 1 - (index + 1) / nfe
            noise = torch.randn(shape, device=device, generator=generator)
            x = (1 - next_t) * x + next_t * noise
    if not torch.isfinite(x).all():
        raise FloatingPointError('Non-finite generated samples')
    return x.clamp(-1, 1)
