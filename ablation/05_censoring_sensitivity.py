"""Ablation: Sensitivity of E[log T] to the censoring-correction method.

The manuscript applies Tobit regression + Kaplan-Meier estimation to handle
censored runtimes (15.6% at n=800, α=4.20).  This ablation compares three
approaches to understand how much the choice of censoring correction matters:

  (A) Naive: treat censored instances as T = T_timeout (lower bound).
  (B) Conservative Tobit lower bound (implemented in src/statistics.py).
  (C) Imputation: replace censored values with T_timeout × exp(σ) (upper bound).

This tests whether the manuscript's key quantitative claims
(H_∞ = 0.021 at α=4.20, ν = 2.30) are robust to censoring correction.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.statistics import censored_log_mean
from src.utils import make_rng

# Reproduce the censoring scenario at n=800, α=4.20 from manuscript
# Censoring fraction 15.6%, mean log T ≈ 16.76 (Table 2)
N_INSTANCES  = 1000
CENS_FRAC    = 0.156
LOG_T_MEAN   = 16.76
LOG_T_STD    = 2.5
LOG_T_CUTOFF = np.log(3600)   # 3600 s timeout


def run():
    rng = make_rng(20240223)

    log_T_true = rng.normal(LOG_T_MEAN, LOG_T_STD, N_INSTANCES)
    censored   = log_T_true >= LOG_T_CUTOFF
    log_T_obs  = np.where(censored, LOG_T_CUTOFF, log_T_true)

    cens_frac_actual = float(np.mean(censored))

    # Method A: naive
    mean_A = float(np.mean(log_T_obs))

    # Method B: conservative Tobit lower bound (src/statistics.py)
    mean_B = censored_log_mean(log_T_obs, censored, LOG_T_CUTOFF)

    # Method C: upper bound imputation
    log_T_imputed = np.where(censored, LOG_T_CUTOFF + LOG_T_STD, log_T_obs)
    mean_C = float(np.mean(log_T_imputed))

    # True (unobservable) mean
    mean_true = float(np.mean(log_T_true))

    n = 800
    print(f"Censoring sensitivity at n={n}, α=4.20  ({cens_frac_actual*100:.1f}% censored)")
    print(f"  True E[log T]:       {mean_true:.3f}   (unobservable)")
    print(f"  Method A (naive):    {mean_A:.3f}   bias = {mean_A - mean_true:+.3f}")
    print(f"  Method B (Tobit lb): {mean_B:.3f}   bias = {mean_B - mean_true:+.3f}")
    print(f"  Method C (imputed):  {mean_C:.3f}   bias = {mean_C - mean_true:+.3f}")
    print(f"\n  H(n,α) = E[log T] / n:")
    for method, mean in [('A', mean_A), ('B', mean_B), ('C', mean_C), ('True', mean_true)]:
        print(f"    Method {method}: H = {mean/n:.5f}  (manuscript H_∞ = 0.0210)")

    methods = ['True (unknown)', 'A (naive)', 'B (Tobit lb)', 'C (imputed)']
    vals    = [mean_true, mean_A, mean_B, mean_C]
    colors  = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(methods, [v / n for v in vals], color=colors)
    ax1.axhline(0.021, color='gray', ls='--', lw=1.0, label='H_∞=0.021 (manuscript)')
    ax1.set_ylabel(r'H(n, α) = E[log T] / n')
    ax1.set_title('Censoring correction methods (n=800, α=4.20)')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=20)

    sorted_obs = np.sort(log_T_obs)
    ax2.hist(log_T_true, bins=40, alpha=0.5, color='#2ca02c', label='True log T (unknown)', density=True)
    ax2.hist(log_T_obs,  bins=40, alpha=0.5, color='#1f77b4', label='Observed log T',       density=True)
    ax2.axvline(LOG_T_CUTOFF, color='red', ls='--', lw=1.2, label=f'Timeout log({3600}s)')
    ax2.set_xlabel(r'log T')
    ax2.set_ylabel('Density')
    ax2.set_title('Runtime distribution with censoring')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ablation_censoring_sensitivity.png', dpi=150)
    plt.close()
    print("Figure saved: results/figures/ablation_censoring_sensitivity.png")


if __name__ == '__main__':
    run()
