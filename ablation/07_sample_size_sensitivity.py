"""Ablation: Sensitivity of the hardness peak estimate to sample size.

Tests how n_instances per (n, α) point affects the precision of the peak
location α*(n) and hardness maximum γ_max.  The manuscript uses 1000
instances per point.  This ablation checks whether fewer samples give
consistent estimates, validating the sample-size choice.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.hardness_metrics import hardness_curve
from src.scaling_analysis import locate_hardness_peak
from src.energy_model import ALPHA_STAR

N = 30
ALPHAS = np.linspace(3.6, 5.0, 14)
N_INST_LIST = [20, 50, 100, 200]
MASTER_SEED = 20240223


def run():
    results = {}
    for n_inst in N_INST_LIST:
        print(f"  n_instances={n_inst} ...", flush=True)
        mean, lo, hi = hardness_curve(
            n=N, alphas=ALPHAS, n_instances=n_inst,
            solver='dpll', master_seed=MASTER_SEED, max_decisions=30000)
        alpha_peak, gamma_max = locate_hardness_peak(ALPHAS, mean)
        results[n_inst] = {
            'mean': mean, 'lo': lo, 'hi': hi,
            'alpha_peak': alpha_peak,
            'gamma_max':  gamma_max,
        }

    print("\nSample-size sensitivity (n={N}):")
    print(f"{'n_inst':>8}  {'α_peak':>8}  {'γ_max':>8}  {'CI_width':>10}")
    for n_inst, r in results.items():
        ci_width = float(np.mean(r['hi'] - r['lo']))
        print(f"{n_inst:8d}  {r['alpha_peak']:8.4f}  {r['gamma_max']:8.5f}  {ci_width:10.5f}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for (n_inst, r), col in zip(results.items(), colors):
        ax1.plot(ALPHAS, r['mean'], '-o', color=col, ms=4, lw=1.5,
                 label=f'n_inst={n_inst}')
        ax1.fill_between(ALPHAS, r['lo'], r['hi'], color=col, alpha=0.12)
    ax1.axvline(ALPHA_STAR, color='gray', ls='--', lw=1.0, label=r'$\alpha^*=4.20$')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\gamma$')
    ax1.set_title(f'Hardness curve vs sample size (n={N})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(N_INST_LIST, [results[n]['alpha_peak'] for n in N_INST_LIST],
                 'bs-', lw=1.8, ms=5, label=r'$\alpha^*(n)$')
    ax2.axhline(ALPHA_STAR, color='gray', ls='--', lw=1.0, label=r'$\alpha^*_\infty=4.20$')
    ax2.set_xlabel('Instances per (n, α)')
    ax2.set_ylabel(r'Estimated $\alpha^*$')
    ax2.set_title('Peak location vs sample size')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_sample_size.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_sample_size.png")


if __name__ == '__main__':
    run()
