#!/usr/bin/env bash
# Generate all manuscript figures.
# Pass --results_dir if you have pre-computed data; otherwise uses synthetic data.
set -euo pipefail
RESULTS_DIR="${1:-results}"
OUTPUT_DIR="${2:-results/figures}"
FORMAT="${3:-png}"
DPI="${4:-300}"

mkdir -p "$OUTPUT_DIR"

python figures/generate_all_figures.py \
  --results_dir "$RESULTS_DIR" \
  --output_dir  "$OUTPUT_DIR" \
  --format "$FORMAT" \
  --dpi "$DPI"

echo "Generated figures in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.${FORMAT} 2>/dev/null | awk '{print $5, $9}'
