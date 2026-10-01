"""Run from a copied, independent export directory with no checkout on sys.path."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from model import Model

torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model.load_checkpoint('one_nfe.pt', evaluate_mode='one_nfe', device=device)
count = [0]
def hook(*args):
    count[0] += 1
handle = model.backbone.register_forward_hook(hook)
torch.manual_seed(1187)
image = model.sample((3, 3, 64, 64), device=device, category=torch.tensor([0, 25, 150], device=device))
handle.remove()
assert count[0] == 1 and image.shape == (3, 3, 64, 64) and torch.isfinite(image).all()
assert image.min() >= -1 and image.max() <= 1 and model.num_parameters <= 100_000_000
assert all(k.startswith('backbone.network.') for k in model.state_dict())
for name, module in sys.modules.copy().items():
    if name.startswith(('afm_fastgen', 'model', 'src')) and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd())
report = {'one_nfe_calls': count[0], 'parameters': model.num_parameters,
          'cwd': str(Path.cwd()), 'device': device,
          'sample_sha256': hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest(),
          'output_shape': list(image.shape), 'external_project_imports': False}
Path('loader_check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
