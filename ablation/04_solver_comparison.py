"""Ablation: DPLL vs WalkSAT hardness metric comparison.

The paper uses Kissat/CaDiCaL (CDCL) as the primary solver.  This ablation
compares DPLL and WalkSAT proxies to verify that the qualitative hardness
profile — peak near α*≈4.20, zero outside (α_d, α_s) — is solver-independent.
Quantitatively, DPLL decision counts and WalkSAT flip counts are different
units from wall-clock seconds, but the shape of H(α) should be consistent.
This corroborates the cross-solver Spearman ρ_s≥0.86 claim from Section 4.7.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.hardness_metrics import measure_hardness, hardness_curve
from src.energy_model import barrier_density, ALPHA_D, ALPHA_S, ALPHA_STAR

N = 30
N_INSTANCES = 50
MASTER_SEED = 20240223
ALPHAS = np.linspace(3.2, 5.0, 18)


def run():
    print(f"Solver comparison ablation  (n={N}, {N_INSTANCES} instances, {len(ALPHAS)} α values)")

    mean_dpll,   lo_d, hi_d = hardness_curve(
        n=N, alphas=ALPHAS, n_instances=N_INSTANCES, k=3,
        solver='dpll', master_seed=MASTER_SEED, max_decisions=50000)

    mean_wsat, lo_w, hi_w = hardness_curve(
        n=N, alphas=ALPHAS, n_instances=N_INSTANCES, k=3,
        solver='walksat', master_seed=MASTER_SEED)

    rho, p_val = spearmanr(mean_dpll, mean_wsat)
    print(f"Spearman ρ (DPLL vs WalkSAT): {rho:.3f}  (p={p_val:.4f})")
    print(f"DPLL peak α:   {ALPHAS[np.argmax(mean_dpll)]:.3f}   (expected ≈4.20)")
    print(f"WalkSAT peak α:{ALPHAS[np.argmax(mean_wsat)]:.3f}   (expected ≈4.20)")

    b_theory = np.array([barrier_density(a) for a in ALPHAS])
    scale_d = mean_dpll.max() / max(b_theory.max(), 1e-10)
    scale_w = mean_wsat.max() / max(b_theory.max(), 1e-10)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ALPHAS, mean_dpll, 'o-', color='#1f77b4', lw=1.8, ms=4, label='DPLL (decisions)')
    ax.fill_between(ALPHAS, lo_d, hi_d, color='#1f77b4', alpha=0.15)
    ax.plot(ALPHAS, mean_wsat, 's--', color='#ff7f0e', lw=1.5, ms=4, label='WalkSAT (flips)')
    ax.fill_between(ALPHAS, lo_w, hi_w, color='#ff7f0e', alpha=0.15)
    ax.plot(ALPHAS, b_theory * scale_d, 'k:', lw=1.2, label=r'$b(\alpha)$ (rescaled)')
    ax.axvline(ALPHA_D, color='gray', ls=':', lw=0.9)
    ax.axvline(ALPHA_S, color='gray', ls=':', lw=0.9)
    ax.set_xlabel(r'Clause density $\alpha$')
    ax.set_ylabel(r'Hardness proxy $\gamma$')
    ax.set_title(f'DPLL vs WalkSAT hardness profile  (Spearman ρ = {rho:.3f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_solver_comparison.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_solver_comparison.png")


if __name__ == '__main__':
    run()
