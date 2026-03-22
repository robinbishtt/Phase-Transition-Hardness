"""Statistical functions for Phase-Transition-Hardness.

All confidence intervals and fitting procedures match the manuscript.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy import stats
from scipy.optimize import minimize

from .utils import make_rng


# =============================================================================
# Bootstrap confidence intervals
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Parameters
    ----------
    data : array
        Sample data.
    statistic : callable
        Summary statistic (default: mean).
    n_boot : int
        Number of bootstrap resamples (manuscript uses 1000).
    ci : float
        Coverage level (manuscript uses 0.95).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    (lo, hi) : Tuple[float, float]
        Lower and upper CI bounds.
    """
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


# =============================================================================
# Log-normal mean + CI
# =============================================================================

def lognormal_mean_ci(
    data: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Estimate log-normal mean and Student-t CI on the log scale.

    Returns (mean_orig, lo, hi) where mean_orig = exp(mu + sigma²/2).
    """
    log_data  = np.log(np.maximum(data, 1e-300))
    mu, sigma = float(np.mean(log_data)), float(np.std(log_data, ddof=1))
    n         = len(data)
    se        = sigma / np.sqrt(n)
    t_crit    = float(stats.t.ppf(0.5 + ci / 2.0, df=n - 1))

    mean_orig = float(np.exp(np.clip(mu + 0.5 * sigma ** 2, -700.0, 700.0)))
    lo        = float(np.exp(mu - t_crit * se))
    hi        = float(np.exp(mu + t_crit * se))
    return mean_orig, lo, hi


# =============================================================================
# Exponential-scaling linear regression
# =============================================================================

def exponential_scaling_fit(
    ns: np.ndarray,
    log_mean_runtimes: np.ndarray,
) -> dict:
    """OLS fit of log T̄ = γ · n + c.

    Tests Conjecture 1: log T(n,α) = Θ(n · b(α)).

    Returns
    -------
    dict with keys: gamma, intercept, r2, p_value, stderr, residuals.
    """
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


# =============================================================================
# FSS collapse
# =============================================================================

def fss_scaling_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Universal FSS master curve (quadratic ansatz)."""
    return a + b * x + c * x ** 2


def fss_collapse(
    alphas: np.ndarray,
    ns: np.ndarray,
    psat_matrix: np.ndarray,
    alpha_s_init: float = 4.267,
    nu_init: float = 2.3,
) -> dict:
    """Optimise FSS collapse: find (α_s, ν) minimising cross-curve scatter.

    Uses Nelder-Mead on the mean-squared scatter of P_sat data after
    rescaling the abscissa to the FSS variable x = (α − α_s)·n^{1/ν}.

    Initial values are chosen from the paper's known values.

    Parameters
    ----------
    alphas : array, shape (M,)
        Constraint densities.
    ns : array, shape (K,)
        System sizes.
    psat_matrix : array, shape (K, M)
        P_sat(n, α) estimates.
    alpha_s_init : float
        Starting point for α_s optimisation (literature value).
    nu_init : float
        Starting point for ν optimisation (literature value).

    Returns
    -------
    dict with keys: alpha_s, nu, residual, x_data, psat_data, converged.
    """
    from scipy.interpolate import interp1d

    def collapse_residual(params):
        alpha_s, nu = params
        if nu <= 0.1 or nu > 6.0 or alpha_s <= 3.0 or alpha_s >= 5.5:
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

    x_data, psat_data = [], []
    for i, n in enumerate(ns):
        scaled = (alphas - alpha_s_opt) * n ** (1.0 / nu_opt)
        x_data.extend(scaled.tolist())
        psat_data.extend(psat_matrix[i].tolist())

    return {
        "alpha_s":   float(alpha_s_opt),
        "nu":        float(nu_opt),
        "residual":  float(result.fun),
        "x_data":    np.array(x_data),
        "psat_data": np.array(psat_data),
        "converged": bool(result.success),
    }


# =============================================================================
# Log-normal and tail fits
# =============================================================================

def fit_lognormal(data: np.ndarray) -> dict:
    """Fit a log-normal distribution and run a KS normality test."""
    log_data = np.log(np.maximum(data, 1e-300))
    mu  = float(np.mean(log_data))
    sig = float(np.std(log_data, ddof=1))
    standardised = (log_data - mu) / sig
    ks, p = stats.kstest(standardised, "norm")
    return {"mu": mu, "sigma": sig, "ks_stat": float(ks), "ks_pvalue": float(p)}


def fit_exponential_tail(
    data: np.ndarray,
    tail_quantile: float = 0.90,
) -> dict:
    """Fit an exponential distribution to the upper tail of *data*."""
    threshold = float(np.quantile(data, tail_quantile))
    tail_data = data[data >= threshold]
    if len(tail_data) < 5:
        return {"lambda": float("nan"), "tail_threshold": threshold, "n_tail": int(len(tail_data))}
    excess = tail_data - threshold
    lam = 1.0 / float(np.mean(excess)) if np.mean(excess) > 0 else float("nan")
    return {"lambda": lam, "tail_threshold": threshold, "n_tail": int(len(tail_data))}


# =============================================================================
# Simple censoring approximation (Tobit-style lower bound)
# =============================================================================

def censored_log_mean(
    log_runtimes: np.ndarray,
    censored: np.ndarray,
    cutoff_log: float,
) -> float:
    """Censoring-adjusted E[log T] using a simple Tobit-style correction.

    For censored observations (runtime hit the wall-clock limit), we observe
    only T ≥ T_cutoff.  This function implements a first-order correction:

        E[log T] ≈ (1/n) * [Σ_uncens log(t_i) + n_cens * cutoff_log]

    This is a conservative lower bound (the true mean is at least this large).
    For the manuscript's full Kaplan-Meier + Tobit regression pipeline see
    Supplementary Section 5.3; that analysis is deferred to future code
    releases.  The relative bias is bounded by the censoring fraction
    (≤ 15.6% at peak hardness for n=800, per Table S6).

    Parameters
    ----------
    log_runtimes : array
        log(T) for all instances (observed or censored).
    censored : boolean array
        True where the instance hit the timeout.
    cutoff_log : float
        log(T_cutoff) - the wall-clock timeout.

    Returns
    -------
    float
        Censoring-adjusted estimate of E[log T].
    """
    n_total   = len(log_runtimes)
    n_cens    = int(np.sum(censored))
    uncens    = log_runtimes[~censored]
    adj_mean  = (np.sum(uncens) + n_cens * cutoff_log) / n_total
    return float(adj_mean)
