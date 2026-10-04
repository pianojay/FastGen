"""Small, deterministic helpers for the finite pre-program audit."""
from dataclasses import asdict, dataclass
import copy
import math
import numpy as np
import torch


def range_penalty(raw):
    return raw.abs().sub(1).relu().square().mean()


@dataclass(frozen=True)
class PhaseFixture:
    d_warmup: int = 2
    gan_delay: int = 1
    ramp_updates: int = 2
    target_rho: float = .1

    def rho(self, d_steps, g_steps):
        if d_steps < self.d_warmup:
            raise ValueError('Generator step before discriminator preparation')
        return self.target_rho * min(1., max(0., (g_steps - self.gan_delay + 1) / self.ramp_updates))


class ClassSampler:
    """Uniform class, then uniform fit image; no workers/prefetch to hide cursor."""
    def __init__(self, groups, seed):
        self.groups = groups
        self.rng = np.random.default_rng(seed)
        self.images = 0

    def draw(self, size):
        labels = self.rng.integers(0, len(self.groups), size=size)
        positions = [int(self.rng.choice(self.groups[int(c)])) for c in labels]
        self.images += size
        return positions, labels.tolist()

    def state_dict(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state), images=self.images)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.images = state['images']


def add_vector(total, gradients, scale=1.):
    if total is None:
        total = [torch.zeros_like(g) if g is not None else None for g in gradients]
    for a, g in zip(total, gradients):
        if g is not None:
            if not torch.isfinite(g).all():
                raise FloatingPointError('Nonfinite diagnostic gradient')
            a.add_(g, alpha=scale)
    return total


def vector_norm(vector):
    return math.sqrt(sum(float(x.double().square().sum()) for x in vector if x is not None))


def vector_cosine(a, b):
    na, nb = vector_norm(a), vector_norm(b)
    dot = sum(float(x.double().mul(y).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return dot / (na * nb) if na * nb else None


def job_seconds(jobs, now):
    """Sum each exclusive leased job once, including overlapping jobs."""
    return sum(max(0., row.get('ended', now) - row['started']) for row in jobs)
