"""Flow and SiD mathematics adapted from Apple SiD-DiT, pinned 626b79f.

See docs/IMPLEMENTATION.md and the unmodified upstream LICENSE/ACKNOWLEDGEMENTS.
Full R/F input Jacobians are intentional; this is not a stopped DMD target.
"""
import contextlib
import torch
import torch.nn.functional as F


def broadcast(t, x):
    return torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)


def corrupt(clean, noise, t):
    t = broadcast(t, clean)
    return (1 - t) * clean + t * noise


def denoise(net, x, t, labels, with_logits=False):
    if with_logits:
        v, logits = net(x, t, labels, return_flag='encoder_decoder')
        return x - broadcast(t, x) * v, logits
    return x - broadcast(t, x) * net(x, t, labels)


def velocity_score(x, t, velocity):
    t = broadcast(t, x)
    if (t <= 0).any():
        raise ValueError('Score conversion is defined only at positive t')
    return -(x + (1 - t) * velocity) / t


def ladder(nfe):
    if nfe not in [1, 2, 3, 4]:
        raise ValueError('Student NFE must be 1, 2, 3 or 4')
    return tuple(1 - k / nfe for k in range(nfe))


def student_rollout(net, noises, labels, nfe, train_step=None):
    """Fresh noise; chosen step is zero-based; prior calls have no gradient."""
    if len(noises) != nfe:
        raise ValueError('Supply exactly N noise tensors')
    if train_step is not None and not 0 <= train_step < nfe:
        raise ValueError('Invalid chosen step')
    clean = torch.zeros_like(noises[0])
    for i, t in enumerate(ladder(nfe)):
        active = train_step is not None and i == train_step
        with torch.set_grad_enabled(active and torch.is_grad_enabled()):
            x = corrupt(clean.detach(), noises[i], t)
            clean = denoise(net, x, t, labels)
        if active:
            return clean
    return clean


@torch.no_grad()
def teacher_sample(net, noise, labels, steps=32):
    """32-call uniform Euler by default; t=1 -> 0 with no score division."""
    x = noise
    for i in range(steps):
        t = 1 - i / steps
        x = x - net(x, t, labels) / steps
    return x


def sid_surrogate(images, real_clean, fake_clean, t, alpha=1.0):
    difference = real_clean - fake_clean
    value = difference * ((real_clean - images) - alpha * difference)
    return (value * (1 - broadcast(t, images))).flatten(1).mean(1)


def discriminator_loss(real_logits, fake_logits):
    r = F.softplus(-real_logits.clamp(-10, 10)).flatten(1).mean(1)
    f = F.softplus(fake_logits.clamp(-10, 10)).flatten(1).mean(1)
    return (r + f) * 0.5


def generator_adversarial(logits, t):
    return F.softplus(-logits.clamp(-10, 10)).flatten(1).mean(1) * (1 - t.reshape(-1))


def adversarial_scale(step, warmup, ramp, maximum):
    return maximum * min(1., max(0., (step - warmup) / max(1, ramp)))

