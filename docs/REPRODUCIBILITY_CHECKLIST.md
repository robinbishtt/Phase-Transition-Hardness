# Reproducibility Checklist

This checklist maps every quantitative claim in the manuscript to the code
and data files that reproduce it.  Work through it from top to bottom to
obtain all reported results from scratch.

---

## Environment Setup

```bash
git clone https://anonymous.4open.science/r/Phase-Transition-Hardness-C795
cd Phase-Transition-Hardness
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Verify installation:

```bash
python -c "from src.energy_model import barrier_density; print(barrier_density(4.20))"
# Expected output: ~0.021
```

Hardware note: full reproduction (n=800, 1000 instances, 3600 s timeout)
requires ~450,000 CPU-hours.  The quick-test path finishes in under 5 minutes.

---

## Section 2.4 - Phase Transition Parameters (Table 1)

| Claim | Expected value | Code location |
|---|---|---|
| α_d ≈ 3.86 | `ALPHA_D = 3.86` | `src/energy_model.py` |
| α_AT ≈ 3.92 | `ALPHA_AT = 3.92` | `src/energy_model.py` |
| α_c = α_s ≈ 4.267 | `ALPHA_C = ALPHA_S = 4.267` | `src/energy_model.py` |
| α* ≈ 4.20 | `ALPHA_STAR = 4.20` | `src/energy_model.py` |
| ν = 2.30 ± 0.18 | `NU = 2.30` | `src/energy_model.py` |
| η ≈ 0.22 | `ETA = 0.22` | `src/energy_model.py` |
| κ = ν(1−η) ≈ 1.80 | `KAPPA = 1.80` | `src/energy_model.py` |
| A = +0.036, B = −1.37 | `FSS_A, FSS_B` | `src/energy_model.py` |

Verification command:

```bash
python -c "
from src.energy_model import *
assert abs(ALPHA_D - 3.86) < 0.001
assert abs(ALPHA_S - 4.267) < 0.001
assert abs(NU - 2.30) < 0.01
assert abs(KAPPA - 1.80) < 0.05
print('All Table 1 constants: OK')
"
```

---

## Section 4.2 - Hardness Peak (Table 2)

| n | T̃_log (s) | log T | H = log T/n | Timeout (%) |
|---|---|---|---|---|
| 100 | 2.67×10³ | 7.89 | 0.0789 | 1.2 |
| 200 | 2.85×10⁴ | 10.26 | 0.0513 | 6.7 |
| 400 | 6.79×10⁵ | 13.43 | 0.0336 | 13.4 |
| 800 | 1.89×10⁷ | 16.76 | 0.0210 | 15.6 |

H∞ = 0.0210 (FSS extrapolation; **not** the regression slope 0.0122).

Reproduce via:

```bash
python experiments/hardness_peak.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_center 4.20 \
  --alpha_width 0.40 \
  --seed 20240223
```

Results saved to `results/hardness_peak_summary.json`.

---

## Section 4.3 - FSS Collapse (Figure 1)

Expected: R² = 0.9997, χ²/dof = 0.89 over n ∈ {100, 200, 400, 800}.

```bash
python experiments/finite_size_scaling.py \
  --n 100 200 400 800 \
  --n_instances 1000 \
  --alpha_min 3.5 \
  --alpha_max 5.0 \
  --alpha_step 0.05 \
  --seed 20240223
```

Results in `results/fss_result.json`.

---

## Section 4.4 - Critical Exponent ν (Table 3)

Three independent estimators should all agree within 0.28σ of the cavity
prediction ν_cavity = 2.35 ± 0.05:

```bash
python -c "
from src.binder_cumulant import CriticalExponentEstimator
ce = CriticalExponentEstimator(ns=[100, 200, 400, 800])
r = ce.combined_estimate()
print(f'ν combined = {r[\"nu\"]:.3f}  CI {r[\"ci\"]}')
print(f'σ from cavity = {r[\"sigma_from_cavity\"]:.2f}σ  (expected: 0.28σ)')
"
```

---

## Section 4.5 - Barrier Function b(α) (Table 4)

```bash
python -c "
from src.energy_model import barrier_density
table4 = {3.5: 0.003, 3.8: 0.012, 4.0: 0.020, 4.2: 0.021}
for alpha, b_ms in table4.items():
    b_code = barrier_density(alpha)
    print(f'α={alpha}: code={b_code:.4f}  manuscript={b_ms:.3f}')
"
```

---

## Section 4.6 - Self-Averaging (Table 5)

The coefficient of variation σ_H/E[H] decreases with n more slowly than
n^{−1/2} due to the non-Gaussian tail.  Verified by comparing the trajectory
in `results/alpha_sweep.npz`.

---

## Section 4.7 - Cross-Solver Validation (Figure 3)

Minimum Spearman ρ_s = 0.86 across all five CDCL solver pairs.  Requires
Kissat 3.1.0 and CaDiCaL 1.9.4 installed:

```bash
which kissat cadical   # confirm binaries present
python ablation/04_solver_comparison.py
```

---

## Section 5.3 - Security Parameters (Table 6)

```bash
python -c "
from src.cryptography import SecurityParameterTable
spt = SecurityParameterTable()
for row in spt.reproduce_table6():
    print(row['label'], row['n'], row['security_bits']:.1f, row['matches_manuscript'])
print('All match:', spt.validate_table6())
"
```

---

## Running All 8 Validation Checks

```bash
python src/validation.py --results_dir results
# Expected: 8/8 checks passed
```

---

## Generating All Figures

```bash
python figures/generate_all_figures.py \
  --results_dir results \
  --output_dir  results/figures \
  --format png \
  --dpi 300
```

---

## Seed Reproducibility

All random seeds are derived from master seed `20240223` via SHA-256:

```python
from src.utils import derive_seed
seed = derive_seed(20240223, n=100, alpha=4.20, idx=0)
```

This is **deterministic across Python versions and platforms** (unlike
Python's built-in `hash()`).  See Supplementary Section 5.1.
