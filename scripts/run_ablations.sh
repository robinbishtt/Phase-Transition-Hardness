#!/usr/bin/env bash
# Run all ablation studies.
set -euo pipefail
echo "Running all ablation studies..."
for script in ablation/0*.py; do
  echo "  → $script"
  python "$script"
done
echo "All ablations complete.  Figures in results/figures/ablation_*.png"
