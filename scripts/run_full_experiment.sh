#!/usr/bin/env bash
# Full reproduction experiment (n=800, 1000 instances, 3600 s timeout).
# Hardware note: requires ~450,000 CPU-hours.
# Equivalent to the complete manuscript experiment.
set -euo pipefail
OUTPUT_DIR="${1:-results}"
N_JOBS="${2:-$(nproc)}"

echo "Starting full experiment  →  output: $OUTPUT_DIR  jobs: $N_JOBS"

python experiments/alpha_sweep.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_min 3.0 --alpha_max 5.0 --alpha_step 0.05 \
  --seed 20240223 --output_dir "$OUTPUT_DIR" --n_jobs "$N_JOBS"

python experiments/finite_size_scaling.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.05 \
  --seed 20240223 --output_dir "$OUTPUT_DIR" --n_jobs "$N_JOBS"

python experiments/hardness_peak.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_center 4.20 --alpha_width 0.40 --n_alpha_points 40 \
  --seed 20240223 --output_dir "$OUTPUT_DIR"

python experiments/scaling_law_verification.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.10 \
  --seed 20240223 --output_dir "$OUTPUT_DIR"

python src/validation.py --results_dir "$OUTPUT_DIR"
echo "Full experiment complete.  Results in $OUTPUT_DIR/"
