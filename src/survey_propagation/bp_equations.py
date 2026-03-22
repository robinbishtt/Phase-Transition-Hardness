from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BPResult:
    """Result of running Belief Propagation on a K-SAT instance."""
    converged:          bool
    n_iterations:       int
    magnetisations:     Dict[int, float]
    cavity_fields:      Dict[Tuple[int, int], float]
    free_energy:        float
    entropy_density:    float


class BeliefPropagation:
    """Belief Propagation (Theorem 1 of the manuscript) for random K-SAT.

    All-positive gauge (J_{a,j} = +1) as adopted throughout the manuscript.

    Theorem 1 (BP Fixed-Point Equations, manuscript Eq. 6–7):

        m_{i→a} = tanh(Σ_{b ∈ ∂i\\a} u_{b→i})          [variable-to-clause]

        u_{a→i} = tanh(β) · Π_{j ∈ ∂a\\i} (1 − m_{j→a}·tanh(β))  /
                                              (1 + m_{j→a}·tanh(β))  [clause-to-variable]

    At zero temperature β → ∞ these reduce to Warning Propagation (WP) with
    messages in {1, 0, *} per the Mézard-Montanari convention [7]:
        u_{a→i} = 1  (clause satisfied by some j ≠ i)
        u_{a→i} = 0  (clause forces s_i)
        u_{a→i} = *  (indeterminate)

    The RS solution is valid for α < α_AT ≈ 3.92; above α_AT the 1RSB
    ansatz (Survey Propagation) is required.

    This implementation runs finite-β BP using the all-positive gauge;
    variable signs are handled by flipping messages according to the literal
    signs of each clause-variable incidence.
    """

    MAX_ITER     = 1000
    CONVERGENCE  = 1e-5
    BETA_DEFAULT = 2.0

    def __init__(
        self,
        instance:    dict,
        beta:        float = BETA_DEFAULT,
        damping:     float = 0.5,
        max_iter:    int   = MAX_ITER,
        tol:         float = CONVERGENCE,
        seed:        int   = 42,
    ):
        self.instance = instance
        self.beta     = beta
        self.damping  = damping
        self.max_iter = max_iter
        self.tol      = tol
        self.rng      = np.random.RandomState(seed)

        n = instance["n"]
        self._var_nodes: List[int]              = list(range(1, n + 1))
        self._clauses:   List[List[int]]         = instance["clauses"]

        # Adjacency: var_i → list of (clause_idx, sign)
        self._var_adj: Dict[int, List[Tuple[int, int]]] = {v: [] for v in self._var_nodes}
        for ci, clause in enumerate(self._clauses):
            for lit in clause:
                vi = abs(lit)
                sign = 1 if lit > 0 else -1
                self._var_adj[vi].append((ci, sign))

        # Initialise messages m_{i→a} = 0 (neutral), u_{a→i} = 0 (neutral)
        self._m: Dict[Tuple[int, int], float] = {}
        self._u: Dict[Tuple[int, int], float] = {}
        for ci, clause in enumerate(self._clauses):
            for lit in clause:
                vi = abs(lit)
                self._m[(vi, ci)] = float(self.rng.uniform(-0.1, 0.1))
                self._u[(ci, vi)] = float(self.rng.uniform(-0.1, 0.1))

    def run(self) -> BPResult:
        """Run BP until convergence or max_iter.

        Returns BPResult with convergence flag, final messages, and
        approximate free energy and entropy density.
        """
        tb = float(np.tanh(self.beta))

        for iteration in range(self.max_iter):
            max_delta = 0.0

            # ── Update u_{a→i} (clause → variable) ──────────────────────────
            new_u: Dict[Tuple[int, int], float] = {}
            for ci, clause in enumerate(self._clauses):
                for lit_i in clause:
                    vi   = abs(lit_i)
                    sign_i = 1 if lit_i > 0 else -1
                    product = 1.0
                    for lit_j in clause:
                        vj = abs(lit_j)
                        if vj == vi:
                            continue
                        sign_j = 1 if lit_j > 0 else -1
                        m_val  = float(sign_j) * self._m.get((vj, ci), 0.0)
                        denom  = 1.0 + m_val * tb
                        numer  = 1.0 - m_val * tb
                        product *= (numer / denom) if abs(denom) > 1e-12 else 0.0
                    new_u_val = float(sign_i) * tb * product
                    new_u_val = np.clip(new_u_val, -1 + 1e-7, 1 - 1e-7)
                    old       = self._u.get((ci, vi), 0.0)
                    new_u[(ci, vi)] = (1 - self.damping) * new_u_val + self.damping * old
                    max_delta = max(max_delta, abs(new_u[(ci, vi)] - old))

            self._u = new_u

            # ── Update m_{i→a} (variable → clause) ──────────────────────────
            new_m: Dict[Tuple[int, int], float] = {}
            for vi in self._var_nodes:
                for (ci_target, _sign_target) in self._var_adj[vi]:
                    total_field = sum(
                        self._u.get((ci, vi), 0.0)
                        for (ci, _s) in self._var_adj[vi]
                        if ci != ci_target
                    )
                    m_new = float(np.tanh(total_field))
                    m_new = np.clip(m_new, -1 + 1e-7, 1 - 1e-7)
                    old   = self._m.get((vi, ci_target), 0.0)
                    new_m[(vi, ci_target)] = (1 - self.damping) * m_new + self.damping * old
                    max_delta = max(max_delta, abs(new_m[(vi, ci_target)] - old))

            self._m = new_m

            if max_delta < self.tol:
                return self._build_result(iteration + 1, converged=True)

        return self._build_result(self.max_iter, converged=False)

    def _build_result(self, n_iter: int, converged: bool) -> BPResult:
        """Assemble BPResult from final messages."""
        n = self.instance["n"]
        mags: Dict[int, float] = {}
        for vi in self._var_nodes:
            total = sum(self._u.get((ci, vi), 0.0) for (ci, _) in self._var_adj[vi])
            mags[vi] = float(np.tanh(total))

        f  = self._free_energy_estimate()
        s  = max(0.0, float(-f * n * self.beta) / n) if n > 0 else 0.0

        return BPResult(
            converged=converged,
            n_iterations=n_iter,
            magnetisations=mags,
            cavity_fields=dict(self._u),
            free_energy=f,
            entropy_density=s,
        )

    def _free_energy_estimate(self) -> float:
        """Bethe free energy approximation from BP messages."""
        f = 0.0
        tb = float(np.tanh(self.beta))
        for ci, clause in enumerate(self._clauses):
            prod = 1.0
            for lit in clause:
                vi = abs(lit)
                sign = 1 if lit > 0 else -1
                m = float(sign) * self._m.get((vi, ci), 0.0)
                denom = 1.0 + m * tb
                if abs(denom) > 1e-12:
                    prod *= (1.0 - m * tb) / denom
            f -= np.log(max(1.0 - 0.5 ** len(clause) * prod, 1e-300))
        return float(f / max(len(self._clauses), 1))
