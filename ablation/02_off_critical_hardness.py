"""Ablation: Hardness behaviour far from the clustering transition.

Tests whether the barrier-hardness correspondence holds in the Easy-SAT
regime (α < α_d) and the UNSAT regime (α > α_s), where b(α) = 0.
The key prediction is that hardness should be sub-exponential (H ≈ 0)
for α outside (α_d, α_s), confirming the barrier as the mechanistic cause.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.energy_model import barrier_density, ALPHA_D, ALPHA_S, ALPHA_STAR
from src.hardness_metrics import measure_hardness, dpll_solve
from src.instance_generator import generate_instance_batch

ALPHA_TEST = [2.5, 3.0, 3.5, 3.86, 4.0, 4.20, 4.267, 4.5, 5.0]
N_TEST = 50
N_INST = 50
MASTER_SEED = 20240223


def run():
    results = {}
    for alpha in ALPHA_TEST:
        instances = generate_instance_batch(N_TEST, alpha, N_INST, k=3,
                                            master_seed=MASTER_SEED)
        gammas = [measure_hardness(inst, solver='dpll', max_decisions=50000)
                  for inst in instances]
        b = barrier_density(alpha)
        results[alpha] = {
            'gamma_mean': float(np.mean(gammas)),
            'gamma_std':  float(np.std(gammas)),
            'b_alpha':    b,
            'regime':     ('Easy-SAT' if alpha < ALPHA_D
                           else ('UNSAT' if alpha > ALPHA_S
                                 else 'Hard-SAT')),
        }

    print(f"Off-critical hardness ablation  (n={N_TEST}, {N_INST} instances each):")
    print(f"{'α':>6} | {'γ_mean':>8} | {'b(α)':>8} | {'regime'}")
    for alpha, r in results.items():
        print(f"{alpha:6.3f} | {r['gamma_mean']:8.5f} | {r['b_alpha']:8.5f} | {r['regime']}")

    alphas_theory = np.linspace(2.0, 5.5, 200)
    b_theory = [barrier_density(a) for a in alphas_theory]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas_theory, b_theory, 'k-', lw=2.0, label='b(α) theory')
    alphas_obs = list(results.keys())
    gammas_obs = [results[a]['gamma_mean'] for a in alphas_obs]
    ax.scatter(alphas_obs, gammas_obs, color='red', zorder=5,
               label=f'Empirical γ (n={N_TEST})')
    ax.axvline(ALPHA_D, color='gray', ls='--', lw=0.9)
    ax.axvline(ALPHA_S, color='gray', ls='--', lw=0.9)
    ax.text(ALPHA_D + 0.03, max(b_theory) * 0.95, r'$\alpha_d$', fontsize=10, color='gray')
    ax.text(ALPHA_S + 0.03, max(b_theory) * 0.95, r'$\alpha_s$', fontsize=10, color='gray')
    ax.set_xlabel(r'Clause density $\alpha$')
    ax.set_ylabel(r'Hardness density $\gamma$  /  $b(\alpha)$')
    ax.set_title('Off-critical hardness: b(α)=0 outside hard phase')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_off_critical.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_off_critical.png")


if __name__ == '__main__':
    run()
