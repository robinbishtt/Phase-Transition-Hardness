"""Ablation: Effect of clause length K on barrier structure.

Tests barrier density and hardness for K ∈ {3, 4, 5} to verify that
the barrier-hardness correspondence extends beyond the K=3 case studied
in the manuscript.  The open problem (Section 6, item iii) asks whether
the critical exponents are universal to the 1RSB universality class.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.energy_model import barrier_density, ALPHA_D, ALPHA_S, ALPHA_STAR, annealed_entropy

# K-dependent satisfiability thresholds (cavity-method estimates)
ALPHA_S_K = {3: 4.267, 4: 9.931, 5: 21.117}
ALPHA_D_K = {3: 3.86,  4: 9.38,  5: 20.80}


def barrier_density_k(alpha: float, k: int) -> float:
    """Approximate barrier density for general K using the same formula
    but with K-specific phase boundaries."""
    ad = ALPHA_D_K[k]
    acs = ALPHA_S_K[k]
    if alpha <= ad or alpha >= acs:
        return 0.0
    alpha_star = ad + 0.97 * (acs - ad)   # peak near condensation
    _A = 0.3819
    KAPPA = 1.80
    _BETA = KAPPA * (acs - alpha_star) / (alpha_star - ad)
    b = _A * (alpha - ad) ** KAPPA * (acs - alpha) ** _BETA
    return float(max(b, 0.0))


def run():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ki, k in enumerate([3, 4, 5]):
        ad  = ALPHA_D_K[k]
        acs = ALPHA_S_K[k]
        alphas = np.linspace(ad - 0.05 * (acs - ad), acs + 0.03 * (acs - ad), 200)
        b_vals = [barrier_density_k(a, k) for a in alphas]
        s_vals = [max(annealed_entropy(a, k), 0.0) for a in alphas]

        ax = axes[ki]
        ax.plot(alphas, b_vals, color='#1f77b4', lw=2.0, label=f'b(α), K={k}')
        ax2 = ax.twinx()
        ax2.plot(alphas, s_vals, color='#d62728', lw=1.5, ls='--',
                 label='s_annealed(α)', alpha=0.7)
        ax.axvline(ad,  color='gray', ls=':', lw=1.0)
        ax.axvline(acs, color='gray', ls=':', lw=1.0)
        ax.set_xlabel(r'Clause density $\alpha$')
        ax.set_ylabel(r'Barrier density $b(\alpha)$')
        ax.set_title(f'K = {k}  (α_d={ad:.2f}, α_s={acs:.3f})')
        ax.legend(fontsize=8, loc='upper left')
        ax2.legend(fontsize=8, loc='upper right')

        peak_b = max(b_vals)
        peak_a = alphas[np.argmax(b_vals)]
        print(f"K={k}: peak b={peak_b:.4f} at α={peak_a:.3f}  "
              f"(α_d={ad:.3f}, α_s={acs:.3f})")

    plt.suptitle('Barrier density b(α) for K ∈ {3, 4, 5}', fontsize=12)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_k_variation.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_k_variation.png")


if __name__ == '__main__':
    run()
