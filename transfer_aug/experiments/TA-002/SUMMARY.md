# TA-002 FDA — completed (contract revision 1)

Completed **14,496 transforms + 604 unchanged controls**, 25 condition/control summaries, **384 representative + 8 tail judgments**, and **34 actually inspected sheets**. The frozen A1 cohort, contract, protocol, protected hashes, 20,099 source metadata records, and 1,687 selected original A1 hashes passed before/after integrity checks.

The pinned NumPy centered-square convention passed all 7,248 self-donor cases (maximum RGB error 8.89e-16; 0 exported LSB), 4,832 exact alpha-zero checks, constant/near-constant fixtures, and reference/linear/DC checks. Raw outside-mask complex error was at most **4.55e-13**; imaginary residual was 4.99e-16. Deterministic replay exactly matched every raw output, PNG, metric row, tail selection and contact sheet without rewriting those outputs.

Clipping broke spectral locality: at **r=4, alpha=1**, median outside-mask relative L2 changed from numerical noise to **0.203 intra / 0.244 inter** (p95 0.542 / 0.538). Full-strength median RGB RMS rose from r=0 to r=4: **0.190→0.238 intra**, **0.231→0.274 inter**. Inter had larger median RMS in every matched condition for these fixed donors.

Visual review found mostly retained layouts alongside stronger spatial color patches/halos and altered surface markings as radius/strength increased. Representative identity was retained/uncertain in **179/13 intra** and **176/16 inter** outputs; anatomy/markings was disrupted in **15/192 intra** and **33/192 inter**. Dense card details and blown-out highlights sometimes prevented confident assessment. These subjective, repeated-condition judgments are not accuracy rates.

High clipping frequency did not imply severe visible damage. All four intra tails appeared mild against white backgrounds despite **61.8–66.0%** clipped channels (RMS **0.00586–0.0382**, maximum excursions **0.0131–0.0827**, not roundoff). Inter tails included mild recoloring, substantial darkening, and cyan contamination.

CPU only; **0 learned parameters, 0 GPU seconds**, no training. Measured peak process host memory: **1.706 GB**. Conservative elapsed wall time including implementation/review: **1074.4 seconds**, below 7,200 seconds.

Evidence: `result.json`, `condition_metrics.csv`, `per_output_metrics.csv`, `visual_reviews.csv`, `gallery.html`, `raw/layout.json`, `verification/`, `hashes.json`, and `REPRODUCE.md`. All outputs remain in this experiment directory.

Limits: A1-only fixed donors, small unblinded visual subset, and global subject/background amplitude mixing. No discriminator behavior, class correctness, training utility, FID, preferred recipe, or SiDA-integration claim is supported.
