from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict

from .instance_generator import generate_instance_batch
from .hardness_metrics import measure_hardness, dpll_solve, walksat_solve
from .scaling_analysis import locate_hardness_peak, finite_size_peak_extrapolation
from .statistics import bootstrap_ci, lognormal_mean_ci
from .utils import (
    make_rng, derive_seed, get_logger, save_json, save_npz,
    ensure_dir, progress,
)

logger = get_logger(__name__)






def measure_runtime_distribution(
    n: int,
    alpha: float,
    n_instances: int = 200,
    k: int = 3,
    solver: str = "dpll",
    master_seed: int = 42,
    max_decisions: int = 100_000,
) -> Dict:
    
    instances = generate_instance_batch(n, alpha, n_instances, k, master_seed)
    decisions_list = []
    gamma_list     = []

    for j, inst in enumerate(instances):
        ws = derive_seed(master_seed, n, alpha, j)
        if solver == "dpll":
            res = dpll_solve(inst, max_decisions=max_decisions)
            t   = res["decisions"]
        else:
            res = walksat_solve(inst, seed=ws)
            t   = res["flips"]

        decisions_list.append(t)
        gamma_list.append(float(np.log(t + 1) / n))

    decisions = np.array(decisions_list, dtype=float)
    gammas    = np.array(gamma_list)

    lo, hi   = bootstrap_ci(gammas, n_boot=500, ci=0.95, seed=master_seed)
    log_mean = float(np.mean(np.log(decisions + 1)))

    return {
        "decisions":  decisions,
        "gamma":      gammas,
        "gamma_mean": float(np.mean(gammas)),
        "gamma_lo":   lo,
        "gamma_hi":   hi,
        "log_mean":   log_mean,
        "n":          n,
        "alpha":      alpha,
        "n_instances": n_instances,
    }






def alpha_sweep(
    ns: List[int],
    alphas: np.ndarray,
    n_instances: int = 200,
    k: int = 3,
    solver: str = "dpll",
    master_seed: int = 42,
    max_decisions: int = 100_000,
    output_dir: str = "results",
) -> Dict:
    
    ensure_dir(output_dir)

    gamma_mean_matrix = np.zeros((len(ns), len(alphas)))
    gamma_lo_matrix   = np.zeros((len(ns), len(alphas)))
    gamma_hi_matrix   = np.zeros((len(ns), len(alphas)))
    alpha_stars       = np.zeros(len(ns))
    gamma_maxima      = np.zeros(len(ns))

    for i, n in enumerate(ns):
        logger.info(f"Alpha sweep: n={n}")
        for j, alpha in enumerate(progress(alphas, desc=f"  n={n}")):
            dist = measure_runtime_distribution(
                n=n, alpha=alpha, n_instances=n_instances, k=k,
                solver=solver, master_seed=master_seed,
                max_decisions=max_decisions,
            )
            gamma_mean_matrix[i, j] = dist["gamma_mean"]
            gamma_lo_matrix[i, j]   = dist["gamma_lo"]
            gamma_hi_matrix[i, j]   = dist["gamma_hi"]


        a_star, g_max = locate_hardness_peak(alphas, gamma_mean_matrix[i])
        alpha_stars[i] = a_star
        gamma_maxima[i] = g_max
        logger.info(f"  α*(n={n}) = {a_star:.4f},  γ_max = {g_max:.5f}")


    extrap = finite_size_peak_extrapolation(ns, alpha_stars)
    logger.info(f"α*(∞) ≈ {extrap['alpha_star_inf']:.4f}  (R²={extrap['r2']:.4f})")

    result = {
        "alphas":            alphas,
        "ns":                np.array(ns),
        "gamma_mean_matrix": gamma_mean_matrix,
        "gamma_lo_matrix":   gamma_lo_matrix,
        "gamma_hi_matrix":   gamma_hi_matrix,
        "alpha_stars":       alpha_stars,
        "gamma_maxima":      gamma_maxima,
        "alpha_star_inf":    extrap["alpha_star_inf"],
        "extrap_r2":         extrap["r2"],
    }

    save_npz(f"{output_dir}/alpha_sweep.npz", **result)
    save_json(
        {
            "ns":             ns,
            "alpha_stars":    alpha_stars.tolist(),
            "gamma_maxima":   gamma_maxima.tolist(),
            "alpha_star_inf": extrap["alpha_star_inf"],
            "extrap_c":       extrap["c"],
            "extrap_r2":      extrap["r2"],
        },
        f"{output_dir}/alpha_sweep_summary.json",
    )

    return result






def localise_hardness_peak(
    ns: List[int],
    alpha_center: float = 4.20,
    width: float = 0.30,
    n_points: int = 30,
    n_instances: int = 500,
    k: int = 3,
    solver: str = "dpll",
    master_seed: int = 42,
    max_decisions: int = 100_000,
    output_dir: str = "results",
) -> Dict:
    
    alphas_fine = np.linspace(
        max(alpha_center - width, 3.0),
        min(alpha_center + width, 5.5),
        n_points,
    )

    result = alpha_sweep(
        ns=ns,
        alphas=alphas_fine,
        n_instances=n_instances,
        k=k,
        solver=solver,
        master_seed=master_seed,
        max_decisions=max_decisions,
        output_dir=output_dir,
    )


    save_npz(f"{output_dir}/hardness_peak.npz", **result)
    save_json(
        {
            "alpha_stars":    result["alpha_stars"].tolist(),
            "gamma_maxima":   result["gamma_maxima"].tolist(),
            "alpha_star_inf": result["alpha_star_inf"],
        },
        f"{output_dir}/hardness_peak_summary.json",
    )

    return result