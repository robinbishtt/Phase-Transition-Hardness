#!/bin/bash
# reproduce_all_figures.sh
# Complete reproduction script for all manuscript figures
# Usage: ./scripts/reproduce_all_figures.sh [output_dir]

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${1:-results/manuscript}"
MASTER_SEED=20240223
NS="100 200 400 800"
N_INSTANCES=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Phase-Transition-Hardness Reproduction"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Master seed: $MASTER_SEED"
echo "System sizes: $NS"
echo "Instances per point: $N_INSTANCES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check Python installation
log "Checking Python installation..."
if ! command -v python &> /dev/null; then
    error "Python not found. Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log "Python version: $PYTHON_VERSION"

# Check if package is installed
log "Checking package installation..."
if ! python -c "import src" 2>/dev/null; then
    warn "Package not installed. Installing..."
    pip install -e .
fi

# Estimate runtime
log "Estimating computational requirements..."
TOTAL_INSTANCES=$(echo "$NS" | wc -w)
log "System sizes: $NS ($TOTAL_INSTANCES configurations)"
log "Estimated time: ~4.5 hours for n=800 with $N_INSTANCES instances"
log "Estimated CPU-hours: ~450,000 for complete reproduction"
log ""

read -p "Continue with full reproduction? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Reproduction cancelled."
    exit 0
fi

# Figure 3: Hardness Density Curve
log "========================================"
log "Figure 3: Hardness Density Curve"
log "========================================"
python experiments/alpha_sweep.py \
    --n $NS \
    --n_instances $N_INSTANCES \
    --alpha_min 3.0 \
    --alpha_max 5.0 \
    --alpha_step 0.05 \
    --seed $MASTER_SEED \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs -1

if [ $? -eq 0 ]; then
    log "Figure 3 data generation complete."
else
    error "Figure 3 data generation failed."
    exit 1
fi

# Figure 4: Finite-Size Scaling Collapse
log "========================================"
log "Figure 4: Finite-Size Scaling Collapse"
log "========================================"
python experiments/finite_size_scaling.py \
    --n $NS \
    --n_instances $N_INSTANCES \
    --alpha_min 3.5 \
    --alpha_max 5.0 \
    --alpha_step 0.05 \
    --seed $MASTER_SEED \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs -1

if [ $? -eq 0 ]; then
    log "Figure 4 data generation complete."
else
    error "Figure 4 data generation failed."
    exit 1
fi

# Figure 5: Hardness Peak Localization
log "========================================"
log "Figure 5: Hardness Peak Localization"
log "========================================"
python experiments/hardness_peak.py \
    --n $NS \
    --n_instances $N_INSTANCES \
    --alpha_center 4.20 \
    --alpha_width 0.40 \
    --n_alpha_points 40 \
    --seed $MASTER_SEED \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    log "Figure 5 data generation complete."
else
    error "Figure 5 data generation failed."
    exit 1
fi

# Scaling Law Verification
log "========================================"
log "Scaling Law Verification"
log "========================================"
python experiments/scaling_law_verification.py \
    --n $NS \
    --n_instances $N_INSTANCES \
    --alpha_min 3.5 \
    --alpha_max 5.0 \
    --alpha_step 0.1 \
    --seed $MASTER_SEED \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs -1

if [ $? -eq 0 ]; then
    log "Scaling law verification complete."
else
    error "Scaling law verification failed."
    exit 1
fi

# Run validation
log "========================================"
log "Running Validation Suite"
log "========================================"
python src/validation.py --results_dir "$OUTPUT_DIR"

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -eq 0 ]; then
    log "All validation checks passed!"
else
    warn "Some validation checks failed. Review output above."
fi

# Generate summary
log "========================================"
log "Generating Summary"
log "========================================"

cat > "$OUTPUT_DIR/reproduction_summary.txt" << EOF
Phase-Transition-Hardness Reproduction Summary
==============================================

Date: $(date)
Master Seed: $MASTER_SEED
System Sizes: $NS
Instances per point: $N_INSTANCES
Output Directory: $OUTPUT_DIR

Generated Files:
EOF

ls -la "$OUTPUT_DIR" >> "$OUTPUT_DIR/reproduction_summary.txt"

echo "" >> "$OUTPUT_DIR/reproduction_summary.txt"
echo "Validation Results:" >> "$OUTPUT_DIR/reproduction_summary.txt"
echo "===================" >> "$OUTPUT_DIR/reproduction_summary.txt"

python src/validation.py --results_dir "$OUTPUT_DIR" >> "$OUTPUT_DIR/reproduction_summary.txt" 2>&1 || true

log "========================================"
log "Reproduction Complete!"
log "========================================"
log "Results saved to: $OUTPUT_DIR"
log "Summary: $OUTPUT_DIR/reproduction_summary.txt"
log ""

if [ $VALIDATION_STATUS -eq 0 ]; then
    log "All validation checks passed. Results match manuscript predictions."
else
    warn "Some validation checks failed. Review $OUTPUT_DIR/reproduction_summary.txt"
fi

exit $VALIDATION_STATUS
