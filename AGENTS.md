# Python environment notes

On 2026-09-24, `soflow_eval` was found to contain Python but no pip after
`conda create -n soflow_eval python=3.10 -y`. Although the environment was
activated, bare `pip` resolved to `/usr/bin/pip`, which used system packages
and `/home/user/.local/lib/python3.10/site-packages`.

Pip was installed with `conda install -n soflow_eval pip -y`. Its verified
location is `/home/user/miniconda3/envs/soflow_eval/lib/python3.10/site-packages/pip`.
The evaluation dependencies were not installed as part of that initial repair.

- Include `pip` explicitly when creating a new Conda environment.
- Use `python -m pip` to target the active Python interpreter. Check
  `python -m pip --version` when diagnosing installation issues.
- Use `PYTHONNOUSERSITE=1` for dependency installation and execution in both environments
  to prevent packages in `~/.local` from interfering.
- After fixing pip, run `hash -r` in an existing shell to clear cached paths.

Install evaluation dependencies with:

```bash
conda activate soflow_eval
hash -r
PYTHONNOUSERSITE=1 python -m pip install -r requirements_soflow_eval.txt
```

`evaluator.py` uses TensorFlow; neither PyGObject nor Torch is required by
the evaluator. Torch is used elsewhere for training and inference. The
reported `meson executable "meson" not found` error occurred while system
pip tried to build PyCairo during `pip install pygobject torch`; do not treat
that error as evidence that PyGObject is an evaluation dependency.

## Follow-up repair on 2026-09-24

`soflow` had the same missing-pip issue. All nine project dependencies loaded
from `~/.local`, including NumPy 2.2.6 instead of the required 1.26.4.
Pip and `requirements_soflow.txt` have now been installed inside `soflow`.
All nine pinned versions and package locations were checked; `pip check`,
training/inference imports, and a SoFlow UNet CUDA forward/backward pass passed.

`requirements_soflow_eval.txt` has also been installed inside `soflow_eval`.
Verified versions include TensorFlow 2.21.0, NumPy 2.2.6, and SciPy 1.15.3.
The evaluator requirements do not pin NumPy to the training version.
`pip check`, environment-local package locations, a TensorFlow v1 session
CUDA convolution, evaluator FID statistics, and actual Inception feature
extraction on a synthetic image all passed on an NVIDIA TITAN RTX.

For this repair, "fixed" means environment-local pip and required packages,
disabled user-site imports after activation, consistent dependencies, and
successful application imports and GPU smoke tests. It does not establish
full training convergence or reproduce a published FID score.

Both environments now have the Conda environment variable `PYTHONNOUSERSITE=1`.
Reactivate an already-open environment to apply it, then run `hash -r`.
When invoking an environment's Python by absolute path without activation,
set `PYTHONNOUSERSITE=1` explicitly; the training launcher does this too.

Environment repair does not migrate already-running processes. A training
process started before this repair may still use packages from `~/.local`.

## Training scale and documentation

Read the [few-NFE training-cost audit](../FastGen-SoFlow/docs/soflow-cifar10/few-nfe-training-cost-audit.md)
for comparisons with Shortcut, MeanFlow, iCT, distillation, pretrained
consistency tuning, and LD3/BézierFlow sampler optimization. Keep sampler
calibration and fine-tuning cost distinct from backbone/teacher pretraining,
and do not translate published A100/TPU runtimes directly into TITAN runtimes.

Read [CIFAR-10 training scale and runtime](../FastGen-SoFlow/docs/soflow-cifar10/cifar10-training-cost.md)
before planning a full CIFAR-10 reproduction or interpreting its runtime.
The paper's 800,000 updates at effective batch 1,024 correspond to 819.2
million image presentations (16,384 equivalent dataset passes). Local
single-TITAN-RTX estimates are approximately four months; these are measured
throughput extrapolations, not elapsed times reported by the paper.

Distinguish GPU microbatch size from effective optimizer batch size. Consult
[the documentation index](../FastGen-SoFlow/docs/soflow-cifar10/README.md) and [the run record](../FastGen-SoFlow/docs/soflow-cifar10/TRAINING_RUN.md)
for measurements and current configuration.

All documentation produced by this CIFAR-10 work belongs in
`../FastGen-SoFlow/docs/soflow-cifar10/`, the shared canonical location in
Astra's checkout. Keep that index and both repositories' AGENTS.md references
current when adding or moving documents. Do not maintain duplicate local
copies. Training code and runtime artifacts remain in this SoFlow checkout;
CIFAR measurements must not be presented as FastGen/Pokémon measurements.


## CIFAR experiment disposition

Read the shared [formal experiment report](../FastGen-SoFlow/docs/soflow-cifar10/EXPERIMENT_REPORT.md)
for the 2026-09-24 findings and archived evidence. The full run was stopped
by an authorized feasibility decision: trainability was observed, while
full training projected to about 126 days and generation quality was not
evaluated. Last logged step: 130; latest recoverable main checkpoint: 5.
Do not confuse logged progress with saved weights or call this a completed
reproduction. Earlier launch instructions are historical; do not restart
the closed experiment automatically.
