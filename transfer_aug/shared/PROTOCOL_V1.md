# First-wave common protocol — frozen revision 1

## Scope and inputs

CPU-only direct transforms; no training, model loading, pretrained feature
extraction, classifier, FID, SiDA integration, or A2/B analysis. Use only the
frozen `shared/cohort_v1` bundle. `cohort.json` row order aligns with the three
uint8 NPY arrays (content, intra donor, inter donor); shape is 604×64×64×3.
Images are official-export RGB. Convert to float64 sRGB in [0,1] for operations.
The bundle manifest pins every file and protected input. Do not modify them.

Use `bash transfer_aug/run_python.sh <worker script>`. No package installations.
Workers may read pinned audit source files and original input hashes but write
only their assigned experiment directory. Finish all declared conditions;
do not select an augmentation recipe. Compute normalized metrics before uint8
display export; display export is round-to-nearest after RGB clipping, with the
same convention used by both workers. Explicit alpha=0 returns the input exactly.

## Verification and provenance

Check frozen contract/common-protocol/cohort/bundle hashes before and after.
Verify every `protected_inputs` hash against the workspace and compare source
file sizes/mtime_ns with `source_stat_snapshot.json` at both endpoints. Verify
the selected original A1 source-file hashes from the cohort. This is integrity
checking, not permission to analyze excluded images. Record stage timings and
all commands, environment version identifiers, implementation hashes, outputs
and SHA-256 hashes. Recompute all declared outputs and compare exact arrays or
hashes; do not rewrite successful results for this check. Cohort regeneration
is checked by the master separately. No model or GPU allocation is expected.

Numerical acceptance: finite outputs; byte-exact alpha=0 bypass; self-donor
RGB maximum absolute reconstruction error <=1e-6 in [0,1] and exported pixels
within 1 LSB. Include constant and near-constant synthetic fixtures. Check
algorithm-specific properties from the experiment contract. Implementation
faults can be fixed within the contract, preserving failed-attempt evidence.

## Shared descriptive measurements

Per output, record sample ID, class, donor arm, condition, then:

- `rms_change`: sqrt(mean((clip(raw,0,1)-content)^2)).
- `clipped_fraction`: fraction of raw scalar RGB channel values outside [0,1].
- `max_excursion`: max(max(-raw), max(raw-1), 0).
- `rgb_mean_shift_l2`: L2 distance between displayed and content channel means.
- `edge_energy_delta`: displayed minus content mean absolute neighbor difference,
  averaging the horizontal and vertical means equally over RGB channels.
- `boundary_jump_delta`: displayed minus content mean absolute opposite-edge
  difference, averaging first/last-row and first/last-column means equally.
- `hf_fraction_delta`: displayed minus content Fourier energy fraction at
  max(|fx|,|fy|)>=16, divided by total non-DC energy (zero if total <=1e-20).

Here 'displayed' in measurements means clipped floating RGB, before quantization.
Report median and p95 by condition; do not treat boundary/high-frequency
statistics as ground-truth artifact labels. Add method-specific mechanism
measurements. Save unclipped RGB arrays for every output, either grouped NPY
arrays or an indexed memmap. Record layout and dtype. Preserve clipped RGB PNGs.

## Visual evidence

Create labeled contact sheets for each of the 16 frozen review sample IDs,
showing content, donor and every declared condition. Use readable nearest-
neighbor enlargements of the actual 64-pixel outputs, not upscaled synthesis.
FDA arms must be distinct. Include eight separately labeled tail cases as
specified by each contract. A simple static gallery linking sheets is useful.

The worker must actually inspect all required sheets with image-viewing tools.
Save per-output review rows for representative outputs and the eight selected
tails. Fields: sample/condition/arm, review_group, recognizable identity
(retained/disrupted/uncertain), anatomy/markings (retained/disrupted/uncertain),
background contamination (none/visible/uncertain), artifacts
(none/visible/uncertain), and a short observation. These are unblinded AI visual
judgments relative to the input, not measured category accuracy. Preserve
uncertainty; do not substitute a numeric heuristic for visual inspection.

## Execution, stopping and handoff

Finite grid; two CPU threads; no GPU; up to two hours elapsed per initial
worker assignment including implementation and reporting. Report a blocker or
budget-limited result if that cannot suffice; no automatic extension. Update
`status.json` at stage transitions and at least once per minute of sustained
execution; send master brief milestones. Never kill another process.

Final `SUMMARY.md` <=500 words and `result.json` must include:
`experiment_id`, `contract_revision`, `status`, `contract_sha256`,
`common_protocol_sha256`, `cohort_sha256`, `bundle_manifest_sha256`,
`training_applicable`, expected/completed output and review counts,
`resources` (device, parameters, elapsed seconds, GPU seconds, peak host memory),
`checks` (each pass/fail and proof path), `key_findings`, `limitations`, and
`evidence` (relative paths to summary, condition metrics, per-output metrics,
review rows, gallery/sheets, numeric arrays, verification, commands and hashes).
Worker owns all detailed verification. Master consumes this compact handoff.
Status `completed` means the contracted evidence exists and checks pass,
regardless of whether any transformed images appear useful.
