# TA-005 reproducible commands

Run from `/home/user/jw_soflow/exploration`. No dependency installation is required.

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/verify_setup.py
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/verify_data.py
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/protected_integrity.py before
TRANSFER_AUG_GPU=1 bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/runner.py --stage all
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/report.py
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/protected_integrity.py after
# After direct review records and SUMMARY.md are written:
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/finalize.py
```

GPU execution requires the existing physical GPU1 lease and UUID match. `--resume` is only for interrupted runs of this contract and preserves cumulative resource accounting and checkpoint RNG/optimizer/sampler state. Do not rerun completed experiments automatically.

Final outputs use nine entries per source in `outputs/final_float.npy`: four reference translations, four Gaussian-style translations, one self-reconstruction, in frozen evaluation manifest order. Values are raw network output in nominal [-1,1]; display PNGs apply affine mapping, rounding and clipping. Milestone arrays retain reference variant0 and latent variant0 only. No checkpoint is selected using diagnostics.

Clipped RMS and clipped diversity use floating RGB after `np.clip((raw.astype(float64)+1)/2,0,1)`; no uint8 rounding enters these metrics. Only descriptors/PCA and exact displayed-image comparisons use quantized PNG RGB. See `report_corrections/clipped_float_v1/verification.json` for the targeted correction proof.
