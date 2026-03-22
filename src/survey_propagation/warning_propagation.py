from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WPResult:
    """Result of Warning Propagation (zero-temperature BP)."""
    converged:       bool
    n_iterations:    int
    warnings:        Dict[Tuple[int, int], int]   # 0, 1, or -1 (*)
    forced_vars:     Dict[int, bool]              # variables forced by unit propagation
    n_contradictions: int


class WarningPropagation:
    """Warning Propagation — the β→∞ limit of Belief Propagation.

    At zero temperature BP reduces to Warning Propagation (WP) with
    messages in {1, 0, *} per the Mézard-Montanari convention [7]:

        u_{a→i} = 1  (clause a satisfied by some literal j ≠ i → no warning)
        u_{a→i} = 0  (clause a unsatisfied by all j ≠ i → forces s_i)
        u_{a→i} = *  (indeterminate: clause has no information about i)

    Update rule:
        u_{a→i} = 1  if ∃ j ∈ ∂a\\i s.t. m_{j→a} = s_{j,a}
                   0  if ∀ j ∈ ∂a\\i  m_{j→a} ≠ s_{j,a}
                   *  otherwise

    where s_{j,a} is the literal sign of variable j in clause a.

    WP is equivalent to unit propagation in SAT solvers:  a clause forces
    its unique remaining free literal when all other literals are falsified.
    WP is valid in the RS phase (α < α_AT ≈ 3.92) but does not capture
    cluster structure in the 1RSB phase.
    """

    MAX_ITER = 200

    def __init__(self, instance: dict, seed: int = 42):
        self.instance = instance
        self.rng      = np.random.RandomState(seed)

        n       = instance["n"]
        clauses = instance["clauses"]

        self._var_adj: Dict[int, List[Tuple[int, int]]] = {
            v: [] for v in range(1, n + 1)
        }
        for ci, clause in enumerate(clauses):
            for lit in clause:
                vi = abs(lit)
                self._var_adj[vi].append((ci, 1 if lit > 0 else -1))

        # WP messages: 1 = satisfied, 0 = forcing, -1 = indeterminate (*)
        self._u: Dict[Tuple[int, int], int] = {}
        for ci, clause in enumerate(clauses):
            for lit in clause:
                vi = abs(lit)
                self._u[(ci, vi)] = -1  # all indeterminate initially

        # Variable spin estimates: None = unforced
        self._spin: Dict[int, Optional[int]] = {v: None for v in range(1, n + 1)}

    def run(self) -> WPResult:
        """Run WP until convergence or max_iter."""
        n       = self.instance["n"]
        clauses = self.instance["clauses"]
        contradictions = 0

        for iteration in range(self.MAX_ITER):
            new_u: Dict[Tuple[int, int], int] = {}
            changed = False

            for ci, clause in enumerate(clauses):
                for lit_i in clause:
                    vi     = abs(lit_i)
                    sign_i = 1 if lit_i > 0 else -1

                    satisfied_by_other = False
                    all_forced_false   = True

                    for lit_j in clause:
                        vj     = abs(lit_j)
                        sign_j = 1 if lit_j > 0 else -1
                        if vj == vi:
                            continue
                        u_val = self._u.get((ci, vj), -1)
                        spin  = self._spin.get(vj)
                        if u_val == 0 and spin is not None:
                            if spin == sign_j:
                                satisfied_by_other = True
                        if u_val != 0:
                            all_forced_false = False

                    if satisfied_by_other:
                        new_u_val = 1
                    elif all_forced_false:
                        new_u_val = 0
                        # This message forces sign_i on vi
                        old_spin = self._spin.get(vi)
                        if old_spin is None:
                            self._spin[vi] = sign_i
                        elif old_spin != sign_i:
                            contradictions += 1
                    else:
                        new_u_val = -1

                    if new_u.get((ci, vi), -2) != new_u_val:
                        changed = True
                    new_u[(ci, vi)] = new_u_val

            self._u = new_u
            if not changed:
                return self._build_result(iteration + 1, True, contradictions)

        return self._build_result(self.MAX_ITER, False, contradictions)

    def _build_result(self, n_iter: int, converged: bool, contradictions: int) -> WPResult:
        forced = {
            vi: (spin == 1)
            for vi, spin in self._spin.items()
            if spin is not None
        }
        return WPResult(
            converged=converged,
            n_iterations=n_iter,
            warnings=dict(self._u),
            forced_vars=forced,
            n_contradictions=contradictions,
        )
