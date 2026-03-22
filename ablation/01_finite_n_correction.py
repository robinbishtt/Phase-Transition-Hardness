"""Ablation: Effect of the 1/N correction in the FSS threshold shift.

Tests whether removing the sub-leading B·n^{-2/ν} term from Eq. 15
degrades the accuracy of α*(n) predictions.  The manuscript uses two-term
FSS:  α*(n) = α*_∞ + A·n^{-1/ν} + B·n^{-2/ν}  with A=+0.036, B=−1.37.

This ablation quantifies how much the sub-leading term improves accuracy,
justifying its inclusion in the manuscript's finite-size analysis.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.proofs.fss_derivation import FSSAnsatz, MANUSCRIPT_FSS, fss_threshold_shift
from src.energy_model import ALPHA_STAR, NU, FSS_A, FSS_B

NS = [50, 100, 200, 400, 800, 1600]


def two_term(ns):
    return fss_threshold_shift(ns, nu=NU, A=FSS_A, B=FSS_B)


def one_term(ns):
    return fss_threshold_shift(ns, nu=NU, A=FSS_A, B=0.0)


def run():
    ns_arr = np.array(NS, dtype=float)
    two = two_term(NS)
    one = one_term(NS)

    print("1/N correction ablation — α*(n) predictions:")
    print(f"{'n':>6}  {'two-term':>10}  {'one-term':>10}  {'diff':>8}")
    for n, t, o in zip(NS, two, one):
        print(f"{n:6d}  {t:10.5f}  {o:10.5f}  {abs(t-o):8.5f}")

    rmse_two  = float(np.sqrt(np.mean((two - ALPHA_STAR) ** 2)))
    rmse_one  = float(np.sqrt(np.mean((one - ALPHA_STAR) ** 2)))
    print(f"\nRMSE from α*_∞ — two-term: {rmse_two:.5f},  one-term: {rmse_one:.5f}")
    print(f"Sub-leading term reduces RMSE by {100*(rmse_one-rmse_two)/rmse_one:.1f}%")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogx(ns_arr, two, 'bo-', lw=1.8, ms=5, label=r'Two-term (A·n$^{-1/\nu}$ + B·n$^{-2/\nu}$)')
    ax.semilogx(ns_arr, one, 's--', color='#ff7f0e', lw=1.5, ms=5,
                label=r'One-term (A·n$^{-1/\nu}$ only)')
    ax.axhline(ALPHA_STAR, color='gray', ls=':', lw=1.0, label=r'$\alpha^*_\infty = 4.20$')
    ax.set_xlabel('System size n')
    ax.set_ylabel(r'$\alpha^*(n)$')
    ax.set_title(r'Impact of sub-leading $1/N$ correction on $\alpha^*(n)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_finite_n_correction.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_finite_n_correction.png")


if __name__ == '__main__':
    run()
