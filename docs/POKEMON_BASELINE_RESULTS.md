# Pokémon baseline: completed experiment EXP-1

Reviewed 2026-09-28, Asia/Seoul. Run: `pokemon_unet128_seed42`, local TITAN RTX GPU 1. [Notion EXP-1](https://app.notion.com/p/3e5f3544609181f5a1d7e29d547ea7fc).

The 20,000-update baseline completed its schedule, all scheduled development evaluations, inference export and artifact packaging. It establishes a practical few-call starting point: four-call images have recognizable category-dependent structure, while one-call quality remains substantially weaker. It does not establish convergence, class accuracy, freedom from memorization or an official FastGen score.

## Quantitative results

FID is lower-is-better. Each evaluation uses 3,020 generated images (20/category) and the same 3,020-image holdout drawn exclusively from official **training** data. Final results use two sampling seeds for one trained checkpoint, not two independently trained models. Their spread is not a confidence interval.

| Final NFE | Seed 42 | Seed 43 | Mean development FID |
|---|---:|---:|---:|
| 1 | 114.6036 | 113.3066 | **113.9551** |
| 2 | 46.4976 | 46.6981 | **46.5978** |
| 3 | 40.2009 | 40.3537 | **40.2773** |
| 4 | 38.3041 | 38.2675 | **38.2858** |

Four calls were selected for the few-NFE export by mean development FID. The largest measured gain comes from adding the second call; the fourth improves mean FID by 1.9915 over three calls. This is a distributional metric, not a percentage measure of visual quality or category correctness.

| Update | One-call FID, seed 42 | Four-call FID, seed 42 |
|---|---:|---:|
| 5,000 | 326.9586 | 103.8197 |
| 10,000 | 174.6829 | 57.4169 |
| 20,000 | 114.6036 | 38.3041 |

Quality improved at every evaluated checkpoint for both modes. The final point is the best **evaluated** checkpoint; most saved checkpoints were not scored. These three measurements do not establish a plateau or guarantee that additional training will help.

## Apparent sample quality

Inspection of the fixed final panels (eight categories, four samples each) shows recognizable Gengar, Kabuto, Metapod and other category-dependent forms at four calls, with cleaner contours and flatter backgrounds than at one call. There are still malformed details, unusual poses, clipping and dark/background artifacts. One-call samples retain category cues but have much noisier textures and less coherent shapes.

The panels reuse the same four initial random draws across categories; similarities in pose, layout and background across rows are therefore partly designed into the comparison. They are not, by themselves, evidence of mode collapse. A broader category/diversity audit and nearest-neighbor review are still needed. No semantic classifier or memorization metric was measured.

Local sample panels remain in `runs/pokemon_unet128_seed42/samples/020000/nfe1.png` through `nfe4.png`, with the complete local report at `runs/pokemon_unet128_seed42/report.html`. Generated image collections and weights remain outside Git.

## Training and resource findings

| Measurement | Observed value |
|---|---:|
| Logged optimizer updates | 20,000, contiguous; all logged numeric values finite |
| Image presentations | 2,560,000; approximately 182 optimization-partition equivalents |
| Effective batch / microbatch | 128 / 32 |
| Mean optimizer update duration | 1.7201 seconds |
| Sum of measured optimizer update durations | 9.5562 hours |
| Recorded cumulative GPU-job budget use | **10.1978 hours** of 24 hours |
| Peak training torch allocation | **8.613 GB** |
| Peak training torch reservation | 9.727 GB |
| Peak parent host RSS | 2.094 GB |
| Peak sampled process-tree PSS, including loader workers | 2.120 GB |
| Parameters per loaded inference model | **39,667,843** |
| Exported backbone call counts | Exactly 1 and 4 |

GB means decimal bytes/1e9. Allocator values exclude CUDA context/library overhead; PSS is sampled and can miss transient peaks. Final generation-plus-FID passes recorded about 1.394 GB peak allocation at batch 32, but this includes the metric pipeline and is not a dedicated inference memory benchmark. Their 40–73-second durations likewise include image writing and FID, and must not be reported as pure sampling latency.

The workflow began 2026-09-24 at 21:18 KST. Filesystem timestamps place schedule completion at 2026-09-25 07:12 and workflow finalization at 07:21 KST. The recorded budget includes qualification, training and evaluation; CPU export/verification/archive finalization lies outside that GPU-job counter. Completion was schedule-driven, not budget exhaustion.

The raw consistency diagnostic becomes spikier later (maximum 46.3461). It measures the squared difference quotient `(f_t - f_l)/(t - l)` while the curriculum shrinks the gap. It is not a fixed validation objective comparable across the whole run. Adaptive weighting also changes the gradient contribution. Mean gradient norm fell from 2.7334 in the first 1,000 updates to 0.5735 in the final 1,000, while FID improved. No logged nonfinite event occurred; the loss trend alone neither establishes divergence nor proves the absence of optimization issues.

The EMA configuration has a 0.9999 cap, but its warmup formula yields an actual final decay of **0.9995502**; the cap is never reached within this 20,000-update run. Longer training experiments must specify both the consistency-gap curriculum and EMA behavior. Simply increasing the configured horizon changes the learning recipe, and the existing exact-resume guard rejects that change.

## Provenance and handoff

Generative weights started from scratch. Training source revision: `38137db`; the terminal source bundle records `ff321c6`. Source hashes inside the checkpoint retain the actual launch provenance. Recipe: class-conditioned 64×64 U-Net128, 151 classes, upstream SoFlow augmentation, FP32 RAdam at 1e-4, consistency-only objective, no classifier-free guidance. See [status and recipe](STATUS.md).

Final full checkpoint SHA-256: `068ae45cd3fee3300cfc0d9089723bf44dfae6678ac9deb4933038db4d736518`.

Development partition SHA-256: `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`.

Exports are under `runs/pokemon_unet128_seed42/export/checkpoints/`. Each contains one EMA backbone without optimizer, duplicate training weights or teacher. The saved terminal verification confirms official-loader compatibility and actual 1/4 call counts. Review on September 28 rechecked all export-manifest hashes and all three entries in the SFTP bundle's checksum file; no GPU evaluation was repeated.

The completed transfer bundle is `dist/pokemon_unet128_seed42/` (source archive about 3 MiB, artifact archive about 881 MiB). It has not been transferred to KCloud. Its embedded documentation is the historical completion-time snapshot; this later report lives in the source repository. KCloud still requires fresh environment/memory qualification and a fresh training run.

Small immutable result records are preserved in [pokemon-baseline-evidence](pokemon-baseline-evidence/): checkpoint FIDs, configuration, budget, selection, export verification, metric summary and training curves. Full per-update logs, checkpoints and generated images remain in the local run and artifact bundle.

## Limitations and next discussion

This is one training seed. Sampling seeds 42/43 only test generation variability. Global FID does not validate requested-class fidelity or within-class diversity. The split has zero exact-byte overlap but near-duplicates were not excluded, and holding out globally byte-unique images affects its distribution. Development FID selected the few-call mode, so it is a selection metric, not an untouched final estimate. Official validation was not used.

The observations support the following order for discussion, without launching another experiment:

1. **Audit category fidelity, diversity and nearest training neighbors** across a wider sample. This clarifies whether the remaining problem is semantics, artifacts or memorization.
2. **Try a small fixed-backbone sampling comparison at 2–4 calls.** Predeclare a small family of intermediate re-noising levels, retain uniform sampling as control, use matched random draws, and distinguish calibration from verification. Reusing subsets of the already consulted development data cannot create an untouched holdout; disclose that limitation. Preserve the original 3,020-image baseline metric alongside any subset metrics.
3. **Treat one-call improvement as a separate training question.** The present one-call sampler has no intermediate time to tune. Compare a budgeted training/curriculum change or locally trained teacher approach only after defining its objective and total cost. Four-call sample quality alone does not validate the diagonal velocity field as a higher-NFE ODE teacher.
4. **Test augmentation or capacity one factor at a time.** There is room under the parameter/memory limits, but this run does not show that insufficient width caused the errors. A matched augmentation-off comparison or explicit longer-training recipe would resolve different questions from widening the U-Net. Keep training presentations, EMA, curriculum and evaluation protocol explicit.

No new training, calibration or GPU evaluation was started for this review. See the earlier [strategy and cost distinctions](FASTGEN_STRATEGY.md) for context.
