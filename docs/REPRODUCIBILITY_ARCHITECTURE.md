# Reproducibility Guide

Complete guide for reproducing all results from the manuscript.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Full Reproduction](#full-reproduction)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Hardware Requirements](#hardware-requirements)
- [Computational Budget](#computational-budget)

---

## Quick Start

### 5-Minute Validation

Run a minimal experiment to verify the installation:

```bash
# Clone repository
git clone https://github.com/[ANONYMOUS_USER]/Phase-Transition-Hardness.git
cd Phase-Transition-Hardness

# Install package
pip install -e .

# Run quick validation (n=50, 20 instances)
python experiments/alpha_sweep.py \
    --n 50 \
    --n_instances 20 \
    --alpha_min 3.5 \
    --alpha_max 5.0 \
    --alpha_step 0.5 \
    --seed 42

# Check results
python src/validation.py --results_dir results
```

Expected output: 6-8 validation checks passing (some may fail due to small sample size).

---

## Environment Setup

### Option 1: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n phase-transition python=3.11

# Activate
conda activate phase-transition

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 3: Docker (Most Reproducible)

```bash
# Build Docker image
docker build -t phase-transition .

# Run container
docker run -v $(pwd)/results:/app/results phase-transition \
    python experiments/alpha_sweep.py --n 100 200 --n_instances 100
```

### Verify Installation

```bash
# Run unit tests
pytest tests/unit/ -v --tb=short

# Check specific module
python -c "from src.energy_model import ALPHA_S; print(f'α_s = {ALPHA_S}')"
```

---

## Full Reproduction

### Manuscript Figure 3: Hardness Density Curve

Reproduces the main hardness density γ(α) curve:

```bash
# Full-scale experiment (matches manuscript)
python experiments/alpha_sweep.py \
    --n 100 200 400 800 \
    --n_instances 1000 \
    --alpha_min 3.0 \
    --alpha_max 5.0 \
    --alpha_step 0.05 \
    --seed 20240223 \
    --output_dir results/manuscript

# Expected runtime: ~4.5 hours (n=800 with 1000 instances)
# CPU-hours: ~450,000 for complete experiment
```

**Quick version** (for testing):
```bash
python experiments/alpha_sweep.py \
    --n 100 200 \
    --n_instances 200 \
    --alpha_min 3.0 \
    --alpha_max 5.0 \
    --alpha_step 0.1 \
    --seed 42
```

### Manuscript Figure 4: Finite-Size Scaling Collapse

Reproduces the FSS collapse:

```bash
python experiments/finite_size_scaling.py \
    --n 100 200 400 800 \
    --n_instances 1000 \
    --alpha_min 3.5 \
    --alpha_max 5.0 \
    --alpha_step 0.05 \
    --seed 20240223 \
    --output_dir results/manuscript

# Expected output: α_s ≈ 4.27, ν ≈ 2.30
```

### Manuscript Figure 5: Hardness Peak Localization

High-resolution peak localization:

```bash
python experiments/hardness_peak.py \
    --n 100 200 400 800 \
    --n_instances 1000 \
    --alpha_center 4.20 \
    --alpha_width 0.40 \
    --n_alpha_points 40 \
    --seed 20240223 \
    --output_dir results/manuscript

# Expected output: α*(∞) ≈ 4.20
```

### Scaling Law Verification

Verifies Conjecture 1 (Barrier-Hardness Correspondence):

```bash
python experiments/scaling_law_verification.py \
    --n 100 200 400 800 \
    --n_instances 1000 \
    --alpha_min 3.5 \
    --alpha_max 5.0 \
    --alpha_step 0.1 \
    --seed 20240223 \
    --output_dir results/manuscript

# Expected output: Mean R² ≥ 0.85, positive correlation
```

---

## Validation

### Run All Validation Checks

```bash
python src/validation.py --results_dir results/manuscript
```

### Expected Output

```
================================================================================
  Phase-Transition-Hardness  Validation Suite
================================================================================
  Results directory: results/manuscript

  [1] ✓ PASS  Satisfiability threshold
        α_s = 4.271  (expected [4.20, 4.35])
        Manuscript value: ≈ 4.267

  [2] ✓ PASS  Hardness peak location
        α* = 4.198  (expected [4.10, 4.40])
        Manuscript value: ≈ 4.20

  [3] ✓ PASS  Peak hardness density
        γ_max = 0.0147  (expected [0.005, 0.05])
        Manuscript value: ≈ 0.015

  [4] ✓ PASS  Exponential scaling fit
        R² = 0.923  (R² ≥ 0.85, γ > 0)
        Manuscript value: Conjecture 1

  [5] ✓ PASS  FSS collapse quality
        Residual = 0.067  (expected < 0.10)
        Manuscript value: Figure 4

  [6] ✓ PASS  FSS critical exponent
        ν = 2.31  (expected [1.5, 3.5])
        Manuscript value: ≈ 2.3

  [7] ✓ PASS  Barrier density positivity
        min b(α) in hard phase = 0.000123  (expected > 0)
        Manuscript value: Conjecture 1

  [8] ✓ PASS  P_sat monotonicity
        P_sat is monotone non-increasing (violations = 0)
        Manuscript value: Threshold structure

================================================================================
  Total: 8/8 checks passed
================================================================================
```

### Interpreting Results

| Check | If Failed | Action |
|-------|-----------|--------|
| 1 (α_s) | Outside [4.20, 4.35] | Increase n_instances or extend alpha range |
| 2 (α*) | Outside [4.10, 4.40] | Increase resolution near α=4.2 |
| 3 (γ_max) | Outside [0.005, 0.05] | Check solver configuration |
| 4 (R²) | < 0.85 | Increase system sizes (n ≥ 400) |
| 5 (Residual) | ≥ 0.10 | Increase n_instances or check data quality |
| 6 (ν) | Outside [1.5, 3.5] | Verify FSS convergence |
| 7 (Barrier) | ≤ 0 | Check energy model implementation |
| 8 (Monotonicity) | Violations > 0 | Statistical fluctuations expected at small n |

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'src'

**Cause**: Running script from wrong directory

**Solution**:
```bash
# Always run from repository root
cd /path/to/Phase-Transition-Hardness
python experiments/alpha_sweep.py ...

# Or set PYTHONPATH
export PYTHONPATH=/path/to/Phase-Transition-Hardness:$PYTHONPATH
```

### Issue: MemoryError at n=800

**Cause**: Insufficient RAM for large instances

**Solutions**:
```bash
# Option 1: Reduce parallel jobs
python experiments/alpha_sweep.py ... --n_jobs 1

# Option 2: Reduce n_instances
python experiments/alpha_sweep.py ... --n_instances 500

# Option 3: Use smaller n values
python experiments/alpha_sweep.py --n 100 200 400 ...
```

### Issue: Validation fails with 'file not found'

**Cause**: Experiment not run yet

**Solution**:
```bash
# Run experiment before validation
python experiments/alpha_sweep.py ...
python src/validation.py --results_dir results
```

### Issue: alpha_s estimate is NaN

**Cause**: P_sat curve doesn't cross 0.5 in alpha range

**Solution**:
```bash
# Extend alpha range to include threshold
python experiments/alpha_sweep.py \
    --alpha_min 3.0 \
    --alpha_max 5.5 \
    ...
```

### Issue: FSS residual exceeds 0.10

**Cause**: Insufficient data for quality collapse

**Solutions**:
```bash
# Option 1: Increase system sizes
python experiments/finite_size_scaling.py --n 100 200 400 800 ...

# Option 2: Increase instances per point
python experiments/finite_size_scaling.py --n_instances 1000 ...

# Option 3: Finer alpha resolution
python experiments/finite_size_scaling.py --alpha_step 0.02 ...
```

### Issue: Figures display empty data

**Cause**: Experiments didn't complete successfully

**Solution**:
```bash
# Check experiment output
ls -la results/

# Re-run experiments with verbose logging
python experiments/alpha_sweep.py ... --verbose
```

### Issue: Different results on different machines

**Cause**: Hardware/compiler differences

**Expected**: Results should agree within:
- α_s: ±0.05
- ν: ±0.2
- γ_max: ±0.005

**Verification**: Cross-platform validation documented in manuscript

---

## Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 4 cores |
| RAM | 8 GB |
| Disk | 1 GB |
| OS | Linux/Mac/Windows |
| Python | 3.10+ |

### Recommended for Full Reproduction

| Component | Specification |
|-----------|---------------|
| CPU | 8+ cores |
| RAM | 16+ GB |
| Disk | 5+ GB |
| OS | Linux (tested) |
| Python | 3.11 |

### Cloud Infrastructure (Manuscript)

Experiments in the manuscript were executed on:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Gold 6248R @ 3.0 GHz |
| Cores | 24 physical per node |
| RAM | 256 GB DDR4-2933 per node |
| Nodes | Multiple (cloud-provisioned) |
| Total CPU-hours | ~450,000 |

---

## Computational Budget

### Time Estimates (per configuration)

| n | Instances | Time | Output | RAM |
|---|-----------|------|--------|-----|
| 100 | 200 | ~2 min | ~2 MB | ~2 GB |
| 200 | 200 | ~8 min | ~4 MB | ~4 GB |
| 400 | 500 | ~45 min | ~15 MB | ~8 GB |
| 800 | 1000 | ~4.5 hours | ~50 MB | ~16 GB |

### Scaling Behavior

- **Time**: Scales approximately as O(n²) in critical regime
- **Memory**: Scales linearly with n
- **Disk**: Scales linearly with n_instances

### Cost Estimation (Cloud)

Using AWS c6i.8xlarge instances (32 vCPU, 64 GB RAM):

| Experiment | Instances | Hours | Cost (approx) |
|------------|-----------|-------|---------------|
| Quick test | 1 | 0.5 | $0.50 |
| Medium | 4 | 2 | $4.00 |
| Full manuscript | 16 | 300 | $2,400 |

*Prices approximate, based on on-demand pricing*

---

## Determinism Verification

### Bit-Identical Reproduction

```bash
# First execution
python experiments/alpha_sweep.py --n 100 --n_instances 10 --seed 42
sha256sum results/alpha_sweep.npz > checksum1.txt

# Second execution (after removing output)
rm results/alpha_sweep.npz
python experiments/alpha_sweep.py --n 100 --n_instances 10 --seed 42
sha256sum results/alpha_sweep.npz > checksum2.txt

# Compare
diff checksum1.txt checksum2.txt
# Should be identical
```

### Cross-Platform Validation

Results may vary slightly across platforms due to:
- Floating-point arithmetic differences
- NumPy/SciPy version differences
- Compiler optimizations

**Acceptable tolerance**: 1% relative difference in median values

---

## Data Preservation

### Archiving Results

```bash
# Create archive
tar czvf results_$(date +%Y%m%d).tar.gz results/

# Include metadata
python -c "import json; json.dump({'date': '$(date)', 'commit': '$(git rev-parse HEAD)'}, open('results/metadata.json', 'w'))"
```

### Long-term Storage

Recommended for publication:
- Zenodo DOI: [10.5281/zenodo.18764848](https://doi.org/10.5281/zenodo.18764848)
- GitHub Releases: Tag with version number
- Institutional repository: Follow institutional guidelines

---

## Citation

If you use this code, please cite:

```bibtex
@software{phase_transition_hardness_2026,
  author = {Anonymous},
  title = {Phase-Transition Structure as Foundation for Cryptographic Hardness},
  year = {2026},
  url = {https://github.com/[ANONYMOUS_USER]/Phase-Transition-Hardness},
  doi = {10.5281/zenodo.18764848}
}
```

---

## Support

For reproducibility issues:
1. Check this guide first
2. Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Open an issue on GitHub with:
   - Command used
   - Error message
   - System information (OS, Python version)
   - Expected vs actual behavior
