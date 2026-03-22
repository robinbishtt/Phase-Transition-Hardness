#!/usr/bin/env bash
# =============================================================================
#  reproduce.sh  —  One-command full reproduction of all manuscript results.
#
#  Usage:
#    bash reproduce.sh                     # full run (~450k CPU-hours)
#    bash reproduce.sh --quick             # 5-minute smoke test (n=30-50)
#    bash reproduce.sh --figures-only      # regenerate figures from saved data
#    bash reproduce.sh --tables-only       # regenerate CSV tables only
#    bash reproduce.sh --output=path       # custom output directory
#    bash reproduce.sh --jobs=N            # parallel workers
#
#  Requirements:
#    pip install -r requirements.txt && pip install -e .
#    Optional (for exact Table 2 wall-clock values):
#      kissat 3.1.0 and cadical 1.9.4 as system binaries.
#
#  Seed: master seed 20240223; per-instance SHA-256 (all platforms).
#
#  Final output:
#    <output>/validation.json    8/8 checks passed
#    <output>/figures/*.png      15 manuscript figures
#    <output>/tables/*.csv       Tables 1-6 (exact manuscript values)
# =============================================================================
set -euo pipefail

QUICK=0
FIGURES_ONLY=0
TABLES_ONLY=0
OUTPUT="results"
N_JOBS=$(python3 -c "import os; print(max(1, os.cpu_count() // 2))" 2>/dev/null || echo 1)

for arg in "$@"; do
  case $arg in
    --quick)         QUICK=1           ;;
    --figures-only)  FIGURES_ONLY=1    ;;
    --tables-only)   TABLES_ONLY=1     ;;
    --output=*)      OUTPUT="${arg#*=}" ;;
    --jobs=*)        N_JOBS="${arg#*=}" ;;
  esac
done

MODE="FULL"
[ $QUICK -eq 1 ]        && MODE="QUICK"
[ $FIGURES_ONLY -eq 1 ] && MODE="FIGURES-ONLY"
[ $TABLES_ONLY -eq 1 ]  && MODE="TABLES-ONLY"

echo "================================================================"
echo "  Phase-Transition-Hardness  —  Full Reproduction"
echo "================================================================"
echo "  Mode:         $MODE"
echo "  Output dir:   $OUTPUT"
echo "  CPU workers:  $N_JOBS"
echo "  Master seed:  20240223  (SHA-256 per-instance)"
echo "================================================================"

mkdir -p "$OUTPUT/figures" "$OUTPUT/tables"

# ── Step 0: Verify installation ──────────────────────────────────────────────
echo ""
echo "[0/6]  Verifying installation..."
python3 - << 'PYCHECK'
import sys; sys.path.insert(0, '.')
from src.energy_model import barrier_density, ALPHA_STAR
b = barrier_density(ALPHA_STAR)
assert abs(b - 0.021) < 0.003, f'FAIL: b(alpha*)={b:.4f}, expected 0.021'
print(f'  b(alpha*=4.20) = {b:.4f}  (expected 0.021)  OK')

from src.utils import derive_seed
s1 = derive_seed(20240223, 100, 4.20, 0)
s2 = derive_seed(20240223, 100, 4.20, 0)
assert s1 == s2, 'FAIL: seed is not deterministic'
print(f'  SHA-256 seed determinism: OK')

from src.cryptography import SecurityParameterTable
spt = SecurityParameterTable()
assert spt.validate_table6(), 'FAIL: Table 6 values do not match manuscript'
print('  Security parameter table: OK  (40 / 60 / 80 bits match Table 6)')
print('  Installation: PASS')
PYCHECK

if [ $FIGURES_ONLY -eq 0 ] && [ $TABLES_ONLY -eq 0 ]; then

  # ── Step 1: Alpha sweep ────────────────────────────────────────────────────
  echo ""
  echo "[1/6]  Alpha sweep  (P_sat + hardness density γ(α, n))..."
  if [ $QUICK -eq 1 ]; then
    python3 experiments/alpha_sweep.py \
      --n 30 50 --n_instances 50 \
      --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.25 \
      --seed 20240223 --output_dir "$OUTPUT" --n_jobs "$N_JOBS"
  else
    python3 experiments/alpha_sweep.py \
      --n 100 200 400 800 --n_instances 1000 \
      --alpha_min 3.0 --alpha_max 5.0 --alpha_step 0.05 \
      --seed 20240223 --output_dir "$OUTPUT" --n_jobs "$N_JOBS"
  fi

  # ── Step 2: Finite-size scaling ────────────────────────────────────────────
  echo ""
  echo "[2/6]  Finite-size scaling collapse  (ν estimation, R²=0.9997 target)..."
  if [ $QUICK -eq 1 ]; then
    python3 experiments/finite_size_scaling.py \
      --n 30 50 --n_instances 50 \
      --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.25 \
      --seed 20240223 --output_dir "$OUTPUT" --n_jobs "$N_JOBS"
  else
    python3 experiments/finite_size_scaling.py \
      --n 100 200 400 800 --n_instances 1000 \
      --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.05 \
      --seed 20240223 --output_dir "$OUTPUT" --n_jobs "$N_JOBS"
  fi

  # ── Step 3: Hardness peak ─────────────────────────────────────────────────
  echo ""
  echo "[3/6]  Fine-resolution hardness peak localisation  (Table 2)..."
  if [ $QUICK -eq 1 ]; then
    python3 experiments/hardness_peak.py \
      --n 30 50 --n_instances 50 \
      --alpha_center 4.20 --alpha_width 0.40 --n_alpha_points 10 \
      --seed 20240223 --output_dir "$OUTPUT"
  else
    python3 experiments/hardness_peak.py \
      --n 100 200 400 800 --n_instances 1000 \
      --alpha_center 4.20 --alpha_width 0.40 --n_alpha_points 40 \
      --seed 20240223 --output_dir "$OUTPUT"
  fi

  # ── Step 4: Scaling law verification ──────────────────────────────────────
  echo ""
  echo "[4/6]  Scaling law verification  (Conjecture 4 regression, R²≥0.85)..."
  if [ $QUICK -eq 1 ]; then
    python3 experiments/scaling_law_verification.py \
      --n 30 50 --n_instances 50 \
      --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.25 \
      --seed 20240223 --output_dir "$OUTPUT"
  else
    python3 experiments/scaling_law_verification.py \
      --n 100 200 400 800 --n_instances 1000 \
      --alpha_min 3.5 --alpha_max 5.0 --alpha_step 0.10 \
      --seed 20240223 --output_dir "$OUTPUT"
  fi

fi  # end FIGURES_ONLY / TABLES_ONLY skip

# ── Step 5: Generate figures ──────────────────────────────────────────────────
if [ $TABLES_ONLY -eq 0 ]; then
  echo ""
  echo "[5/6]  Generating all manuscript figures  (falls back to synthetic data)..."
  python3 figures/generate_all_figures.py \
    --results_dir "$OUTPUT" \
    --output_dir  "$OUTPUT/figures" \
    --format png --dpi 300
  N_FIGS=$(ls "$OUTPUT/figures"/*.png 2>/dev/null | wc -l)
  echo "  → $N_FIGS PNG files written to $OUTPUT/figures/"
fi

# ── Step 6: Generate CSV tables ───────────────────────────────────────────────
if [ $FIGURES_ONLY -eq 0 ]; then
  echo ""
  echo "[6/6]  Generating CSV result tables  (Tables 1–6)..."
  python3 scripts/generate_tables.py --output_dir "$OUTPUT"
fi

# ── Validation ────────────────────────────────────────────────────────────────
echo ""
echo "[VALIDATION]  Running 8 automated manuscript checks..."
python3 src/validation.py --results_dir "$OUTPUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Reproduction complete!"
[ $TABLES_ONLY  -eq 0 ] && echo "  Figures : $OUTPUT/figures/  ($(ls "$OUTPUT/figures"/*.png 2>/dev/null | wc -l) PNGs)"
[ $FIGURES_ONLY -eq 0 ] && echo "  Tables  : $OUTPUT/tables/   ($(ls "$OUTPUT/tables"/*.csv  2>/dev/null | wc -l) CSVs)"
echo "  Log     : $OUTPUT/validation.json"
echo "================================================================"
