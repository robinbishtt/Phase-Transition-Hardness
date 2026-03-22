#!/usr/bin/env bash
# Quick 5-minute smoke test at small scale.
# Verifies the pipeline end-to-end before committing to a full run.
set -euo pipefail
OUTPUT_DIR="${1:-results_quick}"

echo "Quick test  →  $OUTPUT_DIR"

python experiments/alpha_sweep.py \
  --n 30 50 \
  --n_instances 50 \
  --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.25 \
  --seed 20240223 --output_dir "$OUTPUT_DIR" --n_jobs 1

python experiments/finite_size_scaling.py \
  --n 30 50 \
  --n_instances 50 \
  --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.25 \
  --seed 20240223 --output_dir "$OUTPUT_DIR" --n_jobs 1

python src/validation.py --results_dir "$OUTPUT_DIR"
echo "Quick test complete."
