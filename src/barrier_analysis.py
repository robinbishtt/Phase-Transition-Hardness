from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict

from .energy_model import barrier_density, barrier_height, ALPHA_D, ALPHA_S
from .instance_generator import count_violated_clauses
from .utils import make_rng, get_logger, save_json, save_npz, ensure_dir

logger = get_logger(__name__)






def path_barrier(
    instance: dict,
    assign1: dict,
    assign2: dict,
    n_samples: int = 500,
    rng: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
) -> float:
    
    if rng is None:
        rng = make_rng(seed)

    n = instance["n"]

    diff_vars = [v for v in range(1, n + 1)
                 if assign1.get(v, False) != assign2.get(v, False)]

    if not diff_vars:
        return 0.0

    barriers = []
    for _ in range(n_samples):
        rng.shuffle(diff_vars)
        current = dict(assign1)
        path_energies = [count_violated_clauses(instance, current)]
        for v in diff_vars:
            current[v] = not current[v]
            path_energies.append(count_violated_clauses(instance, current))
        barriers.append(max(path_energies) - path_energies[0])

    return float(np.mean(barriers))






def theoretical_barrier_curve(alphas: np.ndarray, k: int = 3) -> np.ndarray:
    
    return np.array([barrier_density(a, k) for a in alphas])






def barrier_scaling_data(
    ns: List[int],
    alpha: float,
    k: int = 3,
) -> Dict:
    
    b = barrier_density(alpha, k)
    barriers = [n * b for n in ns]
    return {
        "ns":       np.array(ns),
        "barriers": np.array(barriers),
        "b_alpha":  b,
        "alpha":    alpha,
    }


def run_barrier_scaling_sweep(
    ns: List[int],
    alphas: np.ndarray,
    k: int = 3,
    output_dir: str = "results",
) -> Dict:
    
    ensure_dir(output_dir)

    b_curve   = theoretical_barrier_curve(alphas, k)
    peak_idx  = int(np.argmax(b_curve))
    alpha_peak = float(alphas[peak_idx])
    b_peak    = float(b_curve[peak_idx])


    selected_alphas = [ALPHA_D + 0.1, 4.10, 4.20, ALPHA_S - 0.1]
    scaling_data = {}
    for a in selected_alphas:
        sd = barrier_scaling_data(ns, a, k)
        scaling_data[f"alpha_{a:.2f}"] = {k_: v.tolist() if isinstance(v, np.ndarray) else v
                                            for k_, v in sd.items()}

    result = {
        "alphas":      alphas,
        "b_curve":     b_curve,
        "alpha_peak":  alpha_peak,
        "b_peak":      b_peak,
        "ns":          np.array(ns),
        "alpha_d":     ALPHA_D,
        "alpha_s":     ALPHA_S,
    }

    save_npz(f"{output_dir}/barrier_analysis.npz", **result)
    save_json(
        {
            "alpha_peak": alpha_peak,
            "b_peak":     b_peak,
            "alpha_d":    ALPHA_D,
            "alpha_s":    ALPHA_S,
            "scaling_data": scaling_data,
        },
        f"{output_dir}/barrier_analysis_summary.json",
    )

    logger.info(f"Barrier peak: b={b_peak:.5f} at α={alpha_peak:.4f}")
    return result






def barrier_hardness_correlation(
    alphas: np.ndarray,
    gamma_mean: np.ndarray,
    k: int = 3,
) -> Dict:
    
    from scipy.stats import pearsonr

    b_curve = theoretical_barrier_curve(alphas, k)

    mask = (alphas > ALPHA_D) & (alphas < ALPHA_S) & (b_curve > 0) & (gamma_mean > 0)

    if mask.sum() < 3:
        return {"correlation": float("nan"), "p_value": float("nan"),
                "b_curve": b_curve, "gamma_mean": gamma_mean}

    r, p = pearsonr(b_curve[mask], gamma_mean[mask])
    return {
        "correlation": float(r),
        "p_value":     float(p),
        "b_curve":     b_curve,
        "gamma_mean":  gamma_mean,
    }