from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

from .utils import make_rng






def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    
    rng = make_rng(seed)
    n   = len(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha_tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_stats, 100 * alpha_tail))
    hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha_tail)))
    return lo, hi






def lognormal_mean_ci(
    data: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    
    log_data  = np.log(np.maximum(data, 1e-300))
    mu, sigma = float(np.mean(log_data)), float(np.std(log_data, ddof=1))
    n         = len(data)
    se        = sigma / np.sqrt(n)
    t_crit    = float(stats.t.ppf(0.5 + ci / 2.0, df=n - 1))

    mean_orig = float(np.exp(mu + 0.5 * sigma ** 2))
    lo        = float(np.exp(mu - t_crit * se))
    hi        = float(np.exp(mu + t_crit * se))
    return mean_orig, lo, hi






def exponential_scaling_fit(
    ns: np.ndarray,
    log_mean_runtimes: np.ndarray,
) -> dict:
    
    slope, intercept, r, p, se = stats.linregress(ns, log_mean_runtimes)
    residuals = log_mean_runtimes - (slope * ns + intercept)
    return {
        "gamma":     float(slope),
        "intercept": float(intercept),
        "r2":        float(r ** 2),
        "p_value":   float(p),
        "stderr":    float(se),
        "residuals": residuals,
    }






def fss_scaling_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    
    return a + b * x + c * x ** 2


def fss_collapse(
    alphas: np.ndarray,
    ns: np.ndarray,
    psat_matrix: np.ndarray,
    alpha_s_init: float = 4.267,
    nu_init: float = 2.3,
) -> dict:
    
    from scipy.optimize import minimize
    from scipy.interpolate import interp1d

    def collapse_residual(params):
        alpha_s, nu = params
        if nu <= 0.1 or alpha_s <= 3.0 or alpha_s >= 5.5:
            return 1e10
        xs, ys = [], []
        for i, n in enumerate(ns):
            scaled = (alphas - alpha_s) * n ** (1.0 / nu)
            xs.extend(scaled.tolist())
            ys.extend(psat_matrix[i].tolist())
        xs = np.array(xs)
        ys = np.array(ys)

        try:
            coeffs = np.polyfit(xs, ys, deg=3)
            y_pred = np.polyval(coeffs, xs)
            residual = float(np.mean((ys - y_pred) ** 2))
        except Exception:
            residual = 1e10
        return residual

    result = minimize(
        collapse_residual,
        x0=[alpha_s_init, nu_init],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )

    alpha_s_opt, nu_opt = result.x
    residual = result.fun


    x_data, psat_data = [], []
    for i, n in enumerate(ns):
        scaled = (alphas - alpha_s_opt) * n ** (1.0 / nu_opt)
        x_data.extend(scaled.tolist())
        psat_data.extend(psat_matrix[i].tolist())

    return {
        "alpha_s":  float(alpha_s_opt),
        "nu":       float(nu_opt),
        "residual": float(residual),
        "x_data":   np.array(x_data),
        "psat_data": np.array(psat_data),
        "converged": bool(result.success),
    }






def fit_lognormal(data: np.ndarray) -> dict:
    
    log_data = np.log(np.maximum(data, 1.0))
    mu  = float(np.mean(log_data))
    sig = float(np.std(log_data, ddof=1))


    standardised = (log_data - mu) / sig
    ks, p = stats.kstest(standardised, "norm")
    return {"mu": mu, "sigma": sig, "ks_stat": float(ks), "ks_pvalue": float(p)}


def fit_exponential_tail(
    data: np.ndarray,
    tail_quantile: float = 0.90,
) -> dict:
    
    threshold = float(np.quantile(data, tail_quantile))
    tail_data = data[data >= threshold]
    if len(tail_data) < 5:
        return {"lambda": float("nan"), "tail_threshold": threshold, "n_tail": 0}

    excess = tail_data - threshold
    lam    = 1.0 / float(np.mean(excess)) if np.mean(excess) > 0 else float("nan")
    return {"lambda": lam, "tail_threshold": threshold, "n_tail": int(len(tail_data))}