from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict

from .statistics import fss_collapse, exponential_scaling_fit
from .utils import get_logger, save_json, save_npz, ensure_dir

logger = get_logger(__name__)






def run_fss_analysis(
    alphas: np.ndarray,
    ns: List[int],
    psat_matrix: np.ndarray,
    output_dir: str = "results",
) -> Dict:
    
    ensure_dir(output_dir)
    ns_arr = np.array(ns)

    logger.info("Running FSS collapse optimisation …")
    result = fss_collapse(alphas, ns_arr, psat_matrix)

    logger.info(f"FSS: α_s={result['alpha_s']:.4f}, ν={result['nu']:.3f}, "
                f"residual={result['residual']:.6f}")

    save_json(
        {
            "alpha_s":    result["alpha_s"],
            "nu":         result["nu"],
            "residual":   result["residual"],
            "converged":  result["converged"],
        },
        f"{output_dir}/fss_result.json",
    )
    save_npz(
        f"{output_dir}/fss_collapse.npz",
        x_data=result["x_data"],
        psat_data=result["psat_data"],
        alphas=alphas,
        ns=ns_arr,
        psat_matrix=psat_matrix,
    )

    return result






def run_exponential_scaling(
    ns: List[int],
    alphas: np.ndarray,
    gamma_matrix: np.ndarray,
    output_dir: str = "results",
) -> Dict:
    
    ensure_dir(output_dir)
    ns_arr = np.array(ns, dtype=float)

    gamma_slope  = np.zeros(len(alphas))
    r2_values    = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):


        log_T_mean = gamma_matrix[:, i] * ns_arr
        fit = exponential_scaling_fit(ns_arr, log_T_mean)
        gamma_slope[i] = fit["gamma"]
        r2_values[i]   = fit["r2"]

    result = {
        "alphas":      alphas,
        "gamma_slope": gamma_slope,
        "r2_values":   r2_values,
        "mean_r2":     float(np.nanmean(r2_values)),
    }

    save_npz(f"{output_dir}/exponential_scaling.npz", **result)
    save_json(
        {
            "mean_r2":    result["mean_r2"],
            "min_r2":     float(np.nanmin(r2_values)),
            "max_gamma":  float(np.nanmax(gamma_slope)),
        },
        f"{output_dir}/exponential_scaling_summary.json",
    )

    logger.info(f"Exponential scaling: mean R²={result['mean_r2']:.4f}")
    return result






def locate_hardness_peak(
    alphas: np.ndarray,
    gamma_mean: np.ndarray,
) -> Tuple[float, float]:
    
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import minimize_scalar

    if len(alphas) < 4:
        idx = int(np.argmax(gamma_mean))
        return float(alphas[idx]), float(gamma_mean[idx])

    try:
        spl = UnivariateSpline(alphas, gamma_mean, s=0, k=4)
        res = minimize_scalar(lambda a: -spl(a),
                              bounds=(float(alphas[0]), float(alphas[-1])),
                              method="bounded")
        alpha_star = float(res.x)
        gamma_max  = float(spl(alpha_star))
    except Exception:
        idx        = int(np.argmax(gamma_mean))
        alpha_star = float(alphas[idx])
        gamma_max  = float(gamma_mean[idx])

    return alpha_star, gamma_max


def finite_size_peak_extrapolation(
    ns: List[int],
    alpha_stars: np.ndarray,
) -> dict:
    
    from scipy.stats import linregress

    inv_ns = 1.0 / np.array(ns, dtype=float)
    slope, intercept, r, _, _ = linregress(inv_ns, alpha_stars)

    return {
        "alpha_star_inf": float(intercept),
        "c":              float(slope),
        "r2":             float(r ** 2),
    }