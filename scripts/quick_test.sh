#!/bin/bash
# quick_test.sh
# Quick test script for CI/CD and development
# Usage: ./scripts/quick_test.sh

set -e

OUTPUT_DIR="results/quick_test"
MASTER_SEED=42

echo "========================================"
echo "Phase-Transition-Hardness Quick Test"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install package if needed
if ! python -c "import src" 2>/dev/null; then
    echo "Installing package..."
    pip install -e . -q
fi

# Run quick alpha sweep
echo ""
echo "Running quick alpha sweep..."
python experiments/alpha_sweep.py \
    --n 50 100 \
    --n_instances 20 \
    --alpha_min 3.5 \
    --alpha_max 4.5 \
    --alpha_step 0.25 \
    --seed $MASTER_SEED \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs 1

# Run validation
echo ""
echo "Running validation..."
python src/validation.py --results_dir "$OUTPUT_DIR" || true

echo ""
echo "========================================"
echo "Quick test complete!"
echo "Results: $OUTPUT_DIR"
echo "========================================"
