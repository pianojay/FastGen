# Selected SiD/SiDA students

Run Python from this directory. Each adjacent checkpoint contains one selected student and its NFE/ladder buffers. No teacher or fake-score network is required. Tested runtime: Python 3.10, PyTorch 2.6.0 and NumPy 2.2.6.

```python
import torch
from model import Model
model = Model.load_checkpoint("../S2.ckpt", evaluate_mode="few_nfe", device="cpu")
images = model.sample((1, 3, 64, 64), category=torch.tensor([25]), device="cpu")
```

Use `one_nfe` for S1/A1; use `few_nfe` for S2/A2/S3/A3/S4/A4. The checkpoint fixes the exact call count. Output is RGB in the model’s normalized pixel coordinates; the official image writer applies `(images + 1) / 2` and clips to [0,1]. Category IDs are listed in `category_to_id.json`. These are local development results, with one training realization per arm. NVIDIA notices remain in the copied U-Net source.
