"""Ablation: BP convergence behaviour across the AT instability threshold.

Tests whether the Belief Propagation equations (Theorem 1) converge in the
RS phase (α < α_AT ≈ 3.92) and diverge above it — confirming the de
Almeida-Thouless instability that necessitates the 1RSB (SP) treatment.
This validates the manuscript's claim that the RS solution is valid for
α < α_AT and that 1RSB is required in the hard phase.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.survey_propagation import BeliefPropagation
from src.instance_generator import generate_ksat_instance
from src.energy_model import ALPHA_D, ALPHA_S

ALPHAS     = [3.0, 3.5, 3.86, 3.92, 4.0, 4.1, 4.2]
N          = 20
N_INSTANCES = 10
SEED       = 20240223


def run():
    results = {}
    rng = np.random.RandomState(SEED)
    for alpha in ALPHAS:
        conv_count = 0
        iter_list  = []
        for i in range(N_INSTANCES):
            inst = generate_ksat_instance(N, alpha, k=3, seed=rng.randint(2**30))
            bp   = BeliefPropagation(inst, beta=3.0, damping=0.5, max_iter=300)
            res  = bp.run()
            if res.converged:
                conv_count += 1
            iter_list.append(res.n_iterations)
        results[alpha] = {
            'conv_frac':  conv_count / N_INSTANCES,
            'mean_iters': float(np.mean(iter_list)),
        }

    print("BP convergence across the AT instability threshold:")
    print(f"{'α':>5}  {'conv_frac':>10}  {'mean_iters':>12}  {'phase'}")
    AT = 3.92
    for alpha, r in results.items():
        phase = 'RS' if alpha < AT else '1RSB'
        print(f"{alpha:5.2f}  {r['conv_frac']:10.3f}  {r['mean_iters']:12.1f}  {phase}")

    alphas_arr  = np.array(ALPHAS)
    conv_fracs  = np.array([results[a]['conv_frac']  for a in ALPHAS])
    mean_iters  = np.array([results[a]['mean_iters'] for a in ALPHAS])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(alphas_arr, conv_fracs, 'bo-', lw=1.8, ms=5)
    ax1.axvline(3.92, color='red', ls='--', lw=1.0, label=r'$\alpha_{AT} \approx 3.92$')
    ax1.set_xlabel(r'Clause density $\alpha$')
    ax1.set_ylabel('BP convergence fraction')
    ax1.set_title('BP convergence vs AT instability')
    ax1.set_ylim(-0.05, 1.10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas_arr, mean_iters, 'rs--', lw=1.5, ms=5)
    ax2.axvline(3.92, color='red', ls='--', lw=1.0, label=r'$\alpha_{AT} \approx 3.92$')
    ax2.set_xlabel(r'Clause density $\alpha$')
    ax2.set_ylabel('Mean iterations to convergence (or max)')
    ax2.set_title('BP iteration count vs AT instability')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_bp_convergence.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_bp_convergence.png")


if __name__ == '__main__':
    run()
