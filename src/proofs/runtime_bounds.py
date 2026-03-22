from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..energy_model import barrier_density, ALPHA_D, ALPHA_S, ALPHA_STAR, NU, KAPPA
from .barrier_bounds import ArrheniusLowerBound, ConflictGraphUpperBound


@dataclass
class BoundResult:
    """Runtime bound evaluation at a single (n, α) point."""
    n: int
    alpha: float
    b_alpha: float
    log_T_lower: float
    log_T_upper: float
    theoretical: float
    sandwich_ratio: float
    conjecture_consistent: bool


@dataclass
class RuntimeBoundsReport:
    """Full evaluation of Conjecture 4 across a grid of (n, α) values."""
    n_values: List[int]
    alpha_values: np.ndarray
    lower_bounds: np.ndarray     # shape (len(ns), len(alphas))
    upper_bounds: np.ndarray     # shape (len(ns), len(alphas))
    theoretical: np.ndarray      # n * b(α)
    mean_sandwich_ratio: float
    conjecture_support: float    # fraction of (n,α) points where lb ≤ theory ≤ ub
    critical_exponent_nu: float
    barrier_growth_exponent_kappa: float


class RuntimeBounds:
    """Evaluate both sides of Conjecture 4: log T(n,α) = Θ(n·b(α)).

    This class computes the Arrhenius lower bound (Proposition 5 of the
    manuscript) and the conflict-graph upper bound simultaneously, and checks
    whether the observed or theoretical hardness density lies within the
    sandwich for each (n, α) point.

    The 'theoretical' value used for the sandwich check is n · b(α) directly,
    which is the leading-order prediction from Conjecture 4.  In the manuscript
    this is validated empirically via the regression log T̄ = γ(α) · n +
    intercept with R² = 0.9876 at α = 4.20.
    """

    def __init__(
        self,
        c1: float = ArrheniusLowerBound.C1_DEFAULT,
        c2: float = ArrheniusLowerBound.C2_DEFAULT,
        c3: float = ConflictGraphUpperBound.C3_DEFAULT,
        c4: float = ConflictGraphUpperBound.C4_DEFAULT,
    ):
        self.lower = ArrheniusLowerBound(c1=c1, c2=c2)
        self.upper = ConflictGraphUpperBound(c3=c3, c4=c4)

    def evaluate_point(self, n: int, alpha: float) -> BoundResult:
        """Evaluate bounds at a single (n, α) point."""
        b = barrier_density(alpha)
        lb = self.lower.log_T_lower(n, alpha)
        ub = self.upper.log_T_upper(n, alpha)
        theory = float(n * b)
        sandwich = float(ub / max(lb, 1e-10))
        consistent = (lb <= theory <= ub) or b <= 0.0
        return BoundResult(
            n=n, alpha=alpha, b_alpha=b,
            log_T_lower=lb, log_T_upper=ub,
            theoretical=theory,
            sandwich_ratio=sandwich,
            conjecture_consistent=consistent,
        )

    def evaluate_grid(
        self, ns: List[int], alphas: np.ndarray
    ) -> RuntimeBoundsReport:
        """Evaluate bounds on a full (n, α) grid."""
        lower_mat   = np.zeros((len(ns), len(alphas)))
        upper_mat   = np.zeros((len(ns), len(alphas)))
        theory_mat  = np.zeros((len(ns), len(alphas)))

        for i, n in enumerate(ns):
            for j, alpha in enumerate(alphas):
                b = barrier_density(alpha)
                lower_mat[i, j]  = max(0.0, self.lower.log_T_lower(n, alpha))
                upper_mat[i, j]  = self.upper.log_T_upper(n, alpha)
                theory_mat[i, j] = float(n * b)

        mask = theory_mat > 0
        consistent = (lower_mat <= theory_mat) & (theory_mat <= upper_mat)
        support = float(np.sum(consistent & mask) / max(np.sum(mask), 1))

        sandwiches = upper_mat[mask] / np.maximum(lower_mat[mask], 1e-10)
        mean_sandwich = float(np.mean(sandwiches)) if len(sandwiches) > 0 else float("nan")

        return RuntimeBoundsReport(
            n_values=ns,
            alpha_values=alphas,
            lower_bounds=lower_mat,
            upper_bounds=upper_mat,
            theoretical=theory_mat,
            mean_sandwich_ratio=mean_sandwich,
            conjecture_support=support,
            critical_exponent_nu=NU,
            barrier_growth_exponent_kappa=KAPPA,
        )


def conjecture4_bounds(
    n: int,
    alpha: float,
    c1: float = 0.80,
    c2: float = 2.50,
    c3: float = 1.20,
    c4: float = 3.00,
) -> Tuple[float, float, float]:
    """Convenience function: return (lower, theoretical, upper) for (n, α).

    Returns
    -------
    Tuple[float, float, float]
        (log_T_lower, n*b(α), log_T_upper)
    """
    rb = RuntimeBounds(c1=c1, c2=c2, c3=c3, c4=c4)
    r  = rb.evaluate_point(n, alpha)
    return r.log_T_lower, r.theoretical, r.log_T_upper


def verify_theta_scaling(
    ns: List[int],
    alpha: float = ALPHA_STAR,
    regression_slope: float = 0.0122,
) -> Dict[str, float]:
    """Verify that the empirical regression slope is consistent with Θ(n·b(α)).

    The manuscript reports a regression slope of 0.0122 ± 0.0004 for log T̄
    vs n at α = 4.20 (Table 2, R² = 0.9876).  The thermodynamic-limit
    barrier b(4.20) = 0.021 (H∞ from the FSS extrapolation).  The slope
    0.0122 is the finite-n effective quantity; b(α) = 0.021 is the n → ∞
    limit.  Both are Θ(b(α)) confirming the Conjecture.

    Parameters
    ----------
    regression_slope : float
        Empirical slope from linear regression of log T̄ on n.
    alpha : float
        Constraint density at which slope was measured.

    Returns
    -------
    dict with keys:
        b_alpha, regression_slope, ratio_slope_to_barrier,
        is_theta_consistent, n_inf_extrapolation
    """
    b = barrier_density(alpha)
    ratio = float(regression_slope / b) if b > 0 else float("nan")
    is_theta = 0.1 <= ratio <= 10.0 if not np.isnan(ratio) else False
    return {
        "b_alpha":                b,
        "regression_slope":       regression_slope,
        "ratio_slope_to_barrier": ratio,
        "is_theta_consistent":    is_theta,
        "n_inf_extrapolation":    b,
        "comment": (
            "Slope is the finite-n effective quantity; b(α) = H∞ = n→∞ limit. "
            "Both are Θ(b(α)) as required by Conjecture 4."
        ),
    }
