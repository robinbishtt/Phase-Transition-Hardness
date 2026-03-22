from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SPResult:
    """Result of running Survey Propagation on a K-SAT instance."""
    converged:          bool
    n_iterations:       int
    eta_surveys:        Dict[Tuple[int, int], float]  # fraction of non-trivial surveys
    complexity:         float
    frozen_fraction:    float
    biases:             Dict[int, float]              # bias toward True for each variable


class SurveyPropagation:
    """Survey Propagation (Theorem 2, manuscript Eq. 8–9) for random K-SAT.

    Survey Propagation extends Belief Propagation to the 1RSB phase by
    propagating distributions P_{i→a}(u) over cavity fields rather than
    individual field values.  In the zero-temperature limit each survey
    η_{a→i} ∈ [0, 1] represents the fraction of clusters in which variable i
    is forced to satisfy clause a.

    Theorem 2 (SP Fixed-Point, manuscript Eq. 8):
        P_{i→a}(u) = (1/Z_{i→a}) Π_{b ∈ ∂i\\a} ∫ dP_{b→i}(u_{b→i}) δ(u − û({u_{b→i}}))

    where û implements the BP update (Eq. 7).  The 1RSB complexity functional
    (manuscript Eq. 9, CORRECTED three-term form):
        Σ[{P}] = Σ_{(i,a)∈E} log Z_{i→a}
                 − Σ_a (|∂a|−1) log Z_a
                 − Σ_i (|∂i|−1) log Z_i

    At zero temperature the distributions collapse to three-mass distributions:
        η_{a→i} = P(u_{a→i} = 1)   (clause satisfied by others)
        The update rule for η_{a→i} is:
            η_{a→i} = Π_{j ∈ ∂a\\i} (1 − Π_{b ∈ ∂j\\a} (1 − η_{b→j}))

    This implementation runs the simplified zero-temperature SP
    (one scalar per directed edge) as in Mézard, Parisi, Zecchina (2002).
    """

    MAX_ITER    = 500
    TOL         = 1e-5

    def __init__(
        self,
        instance: dict,
        max_iter: int  = MAX_ITER,
        tol:      float = TOL,
        seed:     int  = 42,
    ):
        self.instance = instance
        self.max_iter = max_iter
        self.tol      = tol
        self.rng      = np.random.RandomState(seed)

        n       = instance["n"]
        clauses = instance["clauses"]

        # Build adjacency
        self._var_adj: Dict[int, List[Tuple[int, int]]] = {
            v: [] for v in range(1, n + 1)
        }
        for ci, clause in enumerate(clauses):
            for lit in clause:
                vi   = abs(lit)
                sign = 1 if lit > 0 else -1
                self._var_adj[vi].append((ci, sign))

        # Initialise η_{a→i} surveys uniformly in [0, 1]
        self._eta: Dict[Tuple[int, int], float] = {}
        for ci, clause in enumerate(clauses):
            for lit in clause:
                vi = abs(lit)
                self._eta[(ci, vi)] = float(self.rng.uniform(0.05, 0.95))

    def run(self) -> SPResult:
        """Run zero-temperature SP until convergence.

        Convergence is declared when the max change in any η is < tol.
        If SP does not converge, the instance may be in the RS phase (too
        easy) or at/above condensation (too close to threshold).
        """
        n       = self.instance["n"]
        clauses = self.instance["clauses"]

        for iteration in range(self.max_iter):
            new_eta: Dict[Tuple[int, int], float] = {}
            max_delta = 0.0

            for ci, clause in enumerate(clauses):
                for lit_i in clause:
                    vi     = abs(lit_i)
                    sign_i = 1 if lit_i > 0 else -1
                    product = 1.0

                    for lit_j in clause:
                        vj = abs(lit_j)
                        if vj == vi:
                            continue
                        inner_product = 1.0
                        for (ci2, _s2) in self._var_adj[vj]:
                            if ci2 == ci:
                                continue
                            inner_product *= (1.0 - self._eta.get((ci2, vj), 0.5))
                        product *= (1.0 - inner_product)

                    new_eta_val = float(np.clip(product, 0.0, 1.0))
                    old = self._eta.get((ci, vi), 0.5)
                    new_eta[(ci, vi)] = new_eta_val
                    max_delta = max(max_delta, abs(new_eta_val - old))

            self._eta = new_eta
            if max_delta < self.tol:
                return self._build_result(iteration + 1, converged=True)

        return self._build_result(self.max_iter, converged=False)

    def _build_result(self, n_iter: int, converged: bool) -> SPResult:
        n       = self.instance["n"]
        clauses = self.instance["clauses"]

        # ── Compute biases ───────────────────────────────────────────────────
        biases: Dict[int, float] = {}
        for vi in range(1, n + 1):
            adj = self._var_adj[vi]
            if not adj:
                biases[vi] = 0.5
                continue
            pi_plus, pi_minus, pi_zero = 1.0, 1.0, 1.0
            for (ci, sign) in adj:
                eta_here = self._eta.get((ci, vi), 0.5)
                if sign == +1:
                    pi_plus  *= (1.0 - eta_here)
                    pi_zero  *= (1.0 - eta_here)
                else:
                    pi_minus *= (1.0 - eta_here)
                    pi_zero  *= (1.0 - eta_here)
            W_plus  = (1.0 - pi_plus)  * pi_minus
            W_minus = (1.0 - pi_minus) * pi_plus
            W_zero  = pi_plus * pi_minus
            Z = W_plus + W_minus + W_zero + 1e-12
            biases[vi] = float((W_plus - W_minus + W_zero * 0.5) / Z)

        # ── Complexity from fraction of non-trivial surveys ──────────────────
        all_eta = list(self._eta.values())
        frac_nontrivial = float(np.mean([e > 0.01 and e < 0.99 for e in all_eta]))
        complexity = float(-frac_nontrivial * np.log(max(frac_nontrivial, 1e-300)))

        # ── Frozen fraction ──────────────────────────────────────────────────
        frozen_count = sum(
            1 for vi in range(1, n + 1)
            if abs(biases.get(vi, 0.0)) > 0.7
        )
        frozen_frac = float(frozen_count / max(n, 1))

        return SPResult(
            converged=converged,
            n_iterations=n_iter,
            eta_surveys=dict(self._eta),
            complexity=complexity,
            frozen_fraction=frozen_frac,
            biases=biases,
        )

    def decimation_assignment(self, result: SPResult, threshold: float = 0.7) -> Dict[int, bool]:
        """SP-guided decimation: assign frozen variables first.

        Returns a partial assignment for variables with |bias| > threshold.
        """
        assignment: Dict[int, bool] = {}
        for vi, bias in result.biases.items():
            if abs(bias) > threshold:
                assignment[vi] = (bias > 0.0)
        return assignment
