"""Pinned NVIDIA StyleGAN2 discriminator; no pretrained weights or ADA trainer.

The upstream files remain byte-identical. Their top-level import names are
guarded against sibling-project collisions. Only reference operator dispatch
is selected at runtime; layer definitions and state-dict names stay upstream.
"""
import functools
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parents[1] / 'vendor/stylegan2_ada'
DEFAULTS = dict(c_dim=151, img_resolution=64, img_channels=3,
                architecture='resnet', channel_base=32768, channel_max=512,
                cmap_dim=512, num_fp16_res=0, conv_clamp=None,
                mapping_kwargs={'num_layers': 8},
                epilogue_kwargs={'mbstd_group_size': 4, 'mbstd_num_channels': 1})
_reference = None


def verify_vendor():
    record = json.loads((VENDOR / 'PROVENANCE.json').read_text())
    for name, expected in record['files'].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Modified StyleGAN2 source: {name}')
    return record


def reference_module():
    global _reference
    if _reference is not None:
        return _reference
    verify_vendor()
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('training', 'torch_utils', 'dnnlib'):
            origin = getattr(module, '__file__', None)
            if origin is None or not Path(origin).resolve().is_relative_to(VENDOR):
                raise RuntimeError(f'Conflicting upstream import {name}: {origin}')
    sys.path.insert(0, str(VENDOR))
    try:
        reference = importlib.import_module('training.networks')
    finally:
        sys.path.remove(str(VENDOR))
    # No JIT/CUDA extension compilation or unsupported old conv2d gradfix.
    # CPU upstream parity still executes the identical native reference path.
    for name, function in (('bias_act', 'bias_act'), ('upfirdn2d', 'upfirdn2d')):
        module = importlib.import_module('torch_utils.ops.' + name)
        setattr(module, function, functools.partial(getattr(module, function), impl='ref'))
    importlib.import_module('torch_utils.ops.conv2d_gradfix').enabled = False
    _reference = reference
    return _reference


class StyleGAN2Critic(nn.Module):
    """One immutable logical group of four; higher score means more real.

Small channel overrides exist for numerical test fixtures. Production smoke
uses DEFAULTS. Accumulation must call this on the same ordered groups rather
than repacking a larger batch through the upstream minibatch-std layer.
"""
    group_size = 4

    def __init__(self, *, channel_base=32768, channel_max=512, cmap_dim=512):
        super().__init__()
        kwargs = dict(DEFAULTS, channel_base=channel_base,
                      channel_max=channel_max, cmap_dim=cmap_dim)
        self.network = reference_module().Discriminator(**kwargs)

    def forward(self, image, category):
        if image.shape != (4, 3, 64, 64) or image.dtype != torch.float32:
            raise ValueError('StyleGAN2 requires an FP32 logical group (4,3,64,64)')
        if category.shape != (4,) or category.dtype != torch.long:
            raise ValueError('Expected four int64 class IDs')
        if category.device != image.device or ((category < 0) | (category >= 151)).any():
            raise ValueError('Class IDs must be on the image device and in [0,150]')
        return self.network(image, F.one_hot(category, 151).float(), force_fp32=True).flatten()
