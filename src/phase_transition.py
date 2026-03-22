from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict
from joblib import Parallel, delayed

from .instance_generator import generate_instance_batch
from .energy_model import (
    rs_entropy_density, cluster_complexity, frozen_fraction,
    ALPHA_D, ALPHA_R, ALPHA_C, ALPHA_S,
)
from .utils import (
    make_rng, interpolate_threshold, save_json, save_npz,
    get_logger, progress,
)

logger = get_logger(__name__)






def estimate_psat_single(
    n: int,
    alpha: float,
    n_instances: int = 1000,
    k: int = 3,
    master_seed: int = 20240223,
    solver: str = "dpll",
) -> float:
    
    from .hardness_metrics import dpll_solve, walksat_solve

    instances = generate_instance_batch(n, alpha, n_instances, k, master_seed)
    n_sat = 0
    for inst in instances:
        if solver == "dpll":
            result = dpll_solve(inst)
            if result["satisfiable"]:
                n_sat += 1
        else:
            result = walksat_solve(inst)
            if result["satisfiable"]:
                n_sat += 1
    return n_sat / n_instances


def psat_curve(
    n: int,
    alphas: np.ndarray,
    n_instances: int = 1000,
    k: int = 3,
    master_seed: int = 20240223,
    solver: str = "dpll",
    n_jobs: int = 1,
) -> np.ndarray:
    
    def _one(alpha):
        return estimate_psat_single(
            n=n, alpha=alpha, n_instances=n_instances, k=k,
            master_seed=master_seed, solver=solver,
        )

    if n_jobs == 1:
        psats = [_one(a) for a in progress(alphas, desc=f"P_sat n={n}")]
    else:
        psats = Parallel(n_jobs=n_jobs)(delayed(_one)(a) for a in alphas)

    return np.array(psats)






def locate_threshold(
    alphas: np.ndarray,
    psats: np.ndarray,
    target: float = 0.5,
) -> float:
    
    return interpolate_threshold(alphas, psats, target)






def theoretical_order_parameters(
    alphas: np.ndarray,
) -> Dict[str, np.ndarray]:
    
    entropy     = np.array([rs_entropy_density(a) for a in alphas])
    complexity  = np.array([cluster_complexity(a)  for a in alphas])
    frozen      = np.array([frozen_fraction(a)      for a in alphas])
    return {
        "entropy": entropy,
        "cluster_complexity": complexity,
        "frozen_fraction": frozen,
    }






def run_psat_sweep(
    ns: List[int],
    alphas: np.ndarray,
    n_instances: int = 1000,
    k: int = 3,
    master_seed: int = 20240223,
    solver: str = "dpll",
    output_dir: str = "results",
    n_jobs: int = 1,
) -> Dict:
    
    from .utils import ensure_dir
    ensure_dir(output_dir)

    psat_matrix = np.zeros((len(ns), len(alphas)))
    thresholds = {}

    for i, n in enumerate(ns):
        logger.info(f"P_sat sweep: n={n}")
        psats = psat_curve(
            n=n, alphas=alphas, n_instances=n_instances, k=k,
            master_seed=master_seed, solver=solver, n_jobs=n_jobs,
        )
        psat_matrix[i] = psats
        alpha_s_est = locate_threshold(alphas, psats, target=0.5)
        thresholds[n] = alpha_s_est
        logger.info(f"  α_s estimate (n={n}): {alpha_s_est:.4f}")

    theory = theoretical_order_parameters(alphas)

    result = {
        "alphas": alphas,
        "ns": np.array(ns),
        "psat_matrix": psat_matrix,
        "thresholds": {str(k_): float(v) for k_, v in thresholds.items()},
        **{f"theory_{k_}": v for k_, v in theory.items()},
    }

    # The thresholds field is a Python dict; it cannot be stored as a typed
    # numeric ndarray and would produce an object array in the npz archive,
    # preventing load with allow_pickle=False.  Only numeric arrays go into
    # the npz; thresholds are persisted exclusively in the JSON summary.
    npz_arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    save_npz(f"{output_dir}/phase_transition.npz", **npz_arrays)
    save_json(
        {
            "alphas": alphas.tolist(),
            "ns": ns,
            "thresholds": {str(k_): float(v) for k_, v in thresholds.items()},
            "literature_alpha_s": ALPHA_S,
            "literature_alpha_d": ALPHA_D,
        },
        f"{output_dir}/phase_transition_summary.json",
    )

    return result