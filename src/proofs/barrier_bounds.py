from __future__ import annotations

import numpy as np
from typing import Tuple

from ..energy_model import barrier_density, ALPHA_D, ALPHA_S


class ArrheniusLowerBound:
    """Arrhenius lower bound on CDCL runtime from Supplementary Section 4.2.

    Proposition 5 (manuscript): For a complete backtracking solver on random
    3-SAT with α ∈ (α_d, α_s), coupling the DPLL execution tree to the Markov
    chain on the cluster graph gives:

        log T ≥ c₁ · n · b(α) − c₂ · log n

    where c₁, c₂ > 0 depend on α.  This establishes the lower half of
    Conjecture 4:  log T(n, α) = Ω(n · b(α)).

    The argument follows the spirit of Achlioptas and Coja-Oghlan (2008),
    but applies to runtime (not the satisfiability threshold).  The key
    mechanism is that any DPLL search tree must include at least one path
    that crosses an inter-cluster barrier; by the Markov-chain coupling the
    expected time to cross a barrier of height ≥ n·b(α) is at least
    exp(c₁·n·b(α)).  The log n correction absorbs the polynomial overhead
    of identifying which variables to branch on.
    """

    C1_DEFAULT = 0.80
    C2_DEFAULT = 2.50

    def __init__(self, c1: float = C1_DEFAULT, c2: float = C2_DEFAULT):
        """
        Parameters
        ----------
        c1 : float
            Multiplicative constant for the barrier term (α-dependent in the
            full proof; use the conservative default for quantitative bounds).
        c2 : float
            Coefficient of the log n correction.
        """
        self.c1 = c1
        self.c2 = c2

    def log_T_lower(self, n: int, alpha: float) -> float:
        """Return lower bound on E[log T] for given (n, α).

        Parameters
        ----------
        n : int
            Number of Boolean variables.
        alpha : float
            Constraint density α = m/n.

        Returns
        -------
        float
            Lower bound: c₁ · n · b(α) − c₂ · log n.
            Returns 0.0 when α ∉ (α_d, α_s) (no barrier, no hardness).
        """
        b = barrier_density(alpha)
        if b <= 0.0:
            return 0.0
        raw = self.c1 * float(n) * b - self.c2 * np.log(float(n))
        # A lower bound cannot be negative; when the log correction exceeds the
        # barrier term (as it does for n < ~1600 with the conservative c1=0.80),
        # the tightest valid lower bound is 0.
        return float(max(0.0, raw))

    def lower_bound_curve(
        self, n: int, alphas: np.ndarray
    ) -> np.ndarray:
        """Vectorised lower bound over an α grid."""
        return np.array([max(0.0, self.log_T_lower(n, a)) for a in alphas])

    def verify_dominates_polynomial(self, n: int, alpha: float) -> bool:
        """Check that the barrier term dominates the log n correction.

        Returns True when n · b(α) is large relative to log n, so that the
        exponential part of the lower bound is not swamped by the correction.
        """
        b = barrier_density(alpha)
        if b <= 0.0:
            return False
        return self.c1 * n * b > 2.0 * self.c2 * np.log(n)


class ConflictGraphUpperBound:
    """Conflict-graph upper bound on CDCL runtime from Supplementary Sec. 4.3.

    Proposition (manuscript): The minimum-cut capacity of the CDCL implication
    graph at any decision level concentrates around n · b(α) by the
    Paley-Zygmund inequality applied to the random energy landscape.  Each
    learned clause eliminates one infeasible region; after O(exp(n · b(α)))
    learned clauses the solver finds a solution with constant probability:

        log T ≤ c₃ · n · b(α) + c₄ · n^{1/2}

    Together with the Arrhenius lower bound, this establishes Conjecture 4:
    log T(n, α) = Θ(n · b(α)).

    The n^{1/2} correction in the upper bound (vs. log n in the lower bound)
    arises from the fluctuations of the min-cut across the random ensemble;
    the Paley-Zygmund argument controls deviations at scale n^{1/2}.
    """

    C3_DEFAULT = 1.20
    C4_DEFAULT = 3.00

    def __init__(self, c3: float = C3_DEFAULT, c4: float = C4_DEFAULT):
        self.c3 = c3
        self.c4 = c4

    def log_T_upper(self, n: int, alpha: float) -> float:
        """Return upper bound on E[log T] for given (n, α).

        Returns
        -------
        float
            Upper bound: c₃ · n · b(α) + c₄ · n^{1/2}.
            Returns 0.0 when α ∉ (α_d, α_s).
        """
        b = barrier_density(alpha)
        if b <= 0.0:
            return 0.0
        return float(self.c3 * n * b + self.c4 * np.sqrt(n))

    def upper_bound_curve(
        self, n: int, alphas: np.ndarray
    ) -> np.ndarray:
        """Vectorised upper bound over an α grid."""
        return np.array([self.log_T_upper(n, a) for a in alphas])

    def sandwich_width(self, n: int, alpha: float) -> float:
        """Return the ratio (upper bound) / (lower bound) at (n, α).

        As n → ∞ with α fixed in (α_d, α_s), this ratio → c₃/c₁ < ∞,
        confirming that both bounds are Θ(n · b(α)).
        """
        lb = ArrheniusLowerBound().log_T_lower(n, alpha)
        ub = self.log_T_upper(n, alpha)
        if lb <= 0.0:
            return float("inf")
        return float(ub / lb)
