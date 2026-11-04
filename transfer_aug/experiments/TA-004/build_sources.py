"""Extract the pinned, dependency-free CUT network subset; never fetch code."""
import ast
from pathlib import Path
root = Path(__file__).resolve().parent
source = root.parents[1] / 'audit_sources/cut'
s = (source / 'models/networks.py').read_text()
wanted = {'get_filter','Downsample','Upsample','get_pad_layer','Normalize','PatchSampleF','ResnetGenerator','ResnetBlock','NLayerDiscriminator','init_weights','init_net'}
parts = [ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
header = '''# Adapted/extracted from CUT b3ac297708dfb6f7589d04662277e53c0d579c27.
# Copyright (c) 2020, Taesung Park and Jun-Yan Zhu. All rights reserved.
# See LICENSE.upstream for retained notices and redistribution conditions.
# Unrelated StyleGAN/custom-extension dependencies intentionally excluded.
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import numpy as np
'''
(root/'networks.py').write_text(header+'\n\n'.join(parts)+'\n')
(root/'LICENSE.upstream').write_text((source/'LICENSE').read_text())
