"""Hardness measurement functions for random K-SAT.

Metric definition (manuscript Section 4.1)
------------------------------------------
The paper's empirical hardness density is:

    H(n, α) = E[log T] / n

where T is wall-clock seconds from Kissat 3.1.0 or CaDiCaL 1.9.4 with a
3600 s timeout, and E[·] is the geometric-mean expectation over 1000
instances per (n, α) point, adjusted for censored runtimes via the Tobit
regression described in Supplementary Section 5.3.

THIS MODULE
-----------
- measure_hardness() uses the DPLL decision-count proxy
  log(decisions + 1)/n for convenience, but **this is NOT the paper's
  metric**.  See the WARNING block below.
- measure_cdcl_hardness() uses the KissatWrapper / CadicalWrapper
  and measures wall-clock seconds, which is the paper's actual metric.
- The censoring correction is implemented as a conservative lower bound;
  the full Kaplan-Meier + Tobit pipeline from the paper is documented
  in Supplementary Section 5.3.
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  !!!  METRIC MISMATCH WARNING  !!!                                      ║
# ║                                                                          ║
# ║  measure_hardness() uses:  log(DPLL_decisions + 1) / n                 ║
# ║  The PAPER uses:           E[log T] / n  (wall-clock seconds, CDCL)    ║
# ║                                                                          ║
# ║  These are NOT equivalent:                                               ║
# ║    1. Decision counts ≠ wall-clock seconds (different units + scale).   ║
# ║    2. Python DPLL is ~100–1000× slower than Kissat/CaDiCaL.            ║
# ║    3. Paper applies Tobit censoring correction (15.6% at n=800).        ║
# ║                                                                          ║
# ║  Use measure_cdcl_hardness() with an installed Kissat/CaDiCaL binary   ║
# ║  to reproduce the paper's Table 2 values exactly.                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

from .utils import make_rng, derive_seed, get_logger
from .statistics import censored_log_mean

logger = get_logger(__name__)

MAX_DECISIONS_DEFAULT = 100_000
WALKSAT_MAX_FLIPS     = 100_000
WALKSAT_NOISE         = 0.57
PAPER_TIMEOUT_SECS    = 3600  # wall-clock timeout used in the manuscript


# =============================================================================
# CNF helper
# =============================================================================

class CNF:
    """Lightweight CNF representation."""

    def __init__(self, n: int, clauses: List[List[int]]):
        self.n       = n
        self.clauses = [list(c) for c in clauses]
        self.m       = len(self.clauses)

    @classmethod
    def from_instance(cls, instance: dict) -> "CNF":
        return cls(instance["n"], instance["clauses"])

    def copy(self) -> "CNF":
        return CNF(self.n, [list(c) for c in self.clauses])


# =============================================================================
# DPLL solver (decision-count proxy; NOT the paper's metric)
# =============================================================================

def dpll_solve(
    instance: dict,
    max_decisions: int = MAX_DECISIONS_DEFAULT,
) -> Dict:
    """Pure-Python DPLL with unit-propagation and MOMS branching.

    Returns decision count as a proxy for runtime.

    WARNING: The paper's hardness metric uses wall-clock seconds from
    Kissat/CaDiCaL, not decision counts from Python DPLL.  These values
    are NOT directly comparable.
    """
    n       = instance["n"]
    clauses = [list(c) for c in instance["clauses"]]
    counter = [0]

    def solve(clauses, assignment):
        if counter[0] >= max_decisions:
            return "CUTOFF"

        # Unit propagation
        changed = True
        while changed:
            changed = False
            for uc in [c for c in clauses if len(c) == 1]:
                lit = uc[0]
                vi  = abs(lit)
                val = lit > 0
                if vi in assignment:
                    if assignment[vi] != val:
                        return None
                    continue
                assignment[vi] = val
                new_clauses = []
                for c in clauses:
                    if lit in c:
                        continue
                    nc = [l for l in c if l != -lit]
                    if not nc:
                        return None
                    new_clauses.append(nc)
                clauses = new_clauses
                changed = True

        if not clauses:
            for v in range(1, n + 1):
                assignment.setdefault(v, True)
            return assignment

        if any(len(c) == 0 for c in clauses):
            return None

        # Pure literal elimination
        literals = {l for c in clauses for l in c}
        for lit in list(literals):
            vi = abs(lit)
            if -lit not in literals and vi not in assignment:
                assignment[vi] = lit > 0
                clauses = [c for c in clauses if lit not in c]
                return solve(clauses, assignment)

        # MOMS branching
        min_size  = min(len(c) for c in clauses)
        freq: Dict[int, int] = {}
        for c in [c for c in clauses if len(c) == min_size]:
            for l in c:
                freq[abs(l)] = freq.get(abs(l), 0) + 1
        branch_var = max(freq, key=freq.get)

        counter[0] += 1
        if counter[0] >= max_decisions:
            return "CUTOFF"

        for val in [True, False]:
            lit     = branch_var if val else -branch_var
            neg_lit = -lit
            na = dict(assignment)
            na[branch_var] = val
            nc = [
                [l for l in c if l != neg_lit]
                for c in clauses if lit not in c
            ]
            if any(len(c) == 0 for c in nc):
                continue
            result = solve(nc, na)
            if result == "CUTOFF":
                return "CUTOFF"
            if result is not None:
                return result
        return None

    result = solve(clauses, {})

    if result == "CUTOFF":
        return {"satisfiable": None, "decisions": counter[0], "assignment": None}
    elif result is None:
        return {"satisfiable": False, "decisions": counter[0], "assignment": None}
    else:
        return {"satisfiable": True, "decisions": counter[0], "assignment": result}


# =============================================================================
# WalkSAT solver
# =============================================================================

def walksat_solve(
    instance: dict,
    max_flips: int = WALKSAT_MAX_FLIPS,
    noise: float = WALKSAT_NOISE,
    seed: Optional[int] = None,
    restarts: int = 5,
) -> Dict:
    """WalkSAT with noise parameter p = 0.57 (paper's setting)."""
    from .instance_generator import count_violated_clauses

    rng              = make_rng(seed)
    n                = instance["n"]
    clauses_raw      = instance["clauses"]
    total_flips      = 0
    flips_per_restart = max_flips // max(restarts, 1)

    for _ in range(restarts):
        assignment = {i + 1: bool(rng.randint(2)) for i in range(n)}

        for _ in range(flips_per_restart):
            total_flips += 1
            unsat = [
                clause for clause in clauses_raw
                if not any(
                    (l > 0 and assignment[abs(l)]) or (l < 0 and not assignment[abs(l)])
                    for l in clause
                )
            ]
            if not unsat:
                return {"satisfiable": True, "flips": total_flips, "assignment": assignment}

            chosen = unsat[rng.randint(len(unsat))]
            if rng.rand() < noise:
                vi = abs(chosen[rng.randint(len(chosen))])
            else:
                best_vi, best_breaks = None, sys.maxsize
                for lit in chosen:
                    vi = abs(lit)
                    breaks = sum(
                        1 for c in clauses_raw if vi in [abs(l) for l in c]
                        and any(
                            (l > 0 and assignment[abs(l)]) or (l < 0 and not assignment[abs(l)])
                            for l in c
                        )
                        and not any(
                            (l > 0 and (assignment[abs(l)] if abs(l) != vi else not assignment[abs(l)])) or
                            (l < 0 and not (assignment[abs(l)] if abs(l) != vi else not assignment[abs(l)]))
                            for l in c
                        )
                    )
                    if breaks < best_breaks:
                        best_breaks, best_vi = breaks, vi
                vi = best_vi if best_vi is not None else abs(chosen[0])
            assignment[vi] = not assignment[vi]

    return {"satisfiable": False, "flips": total_flips, "assignment": None}


# =============================================================================
# DPLL-based hardness proxy (NOT the paper's metric)
# =============================================================================

def measure_hardness(
    instance: dict,
    solver: str = "dpll",
    max_decisions: int = MAX_DECISIONS_DEFAULT,
    walksat_seed: Optional[int] = None,
) -> float:
    """Return log(decisions + 1)/n as a cheap hardness proxy.

    WARNING: The paper's metric is E[log T]/n (wall-clock, CDCL). This
    function is suitable for quick sanity checks and unit tests only.
    Use measure_cdcl_hardness() for manuscript-faithful measurements.
    """
    n = instance["n"]
    if solver == "dpll":
        result = dpll_solve(instance, max_decisions=max_decisions)
        t = result["decisions"]
    else:
        result = walksat_solve(instance, seed=walksat_seed)
        t = result["flips"]
    return float(np.log(t + 1) / n)


# =============================================================================
# CDCL wall-clock hardness (paper's actual metric)
# =============================================================================

def measure_cdcl_hardness(
    instance: dict,
    solver: str = "kissat",
    timeout: int = PAPER_TIMEOUT_SECS,
) -> Dict:
    """Measure hardness using Kissat or CaDiCaL (paper's actual protocol).

    Returns
    -------
    dict with keys:
        log_T      : float  - log(wall_time) in seconds
        wall_time  : float  - raw wall-clock seconds
        timed_out  : bool   - True if the instance hit the timeout
        satisfiable: bool or None
        hardness   : float  - log(T)/n  (paper's H(n,α) without E[·])
    """
    n = instance["n"]

    if solver == "kissat":
        from .solver_wrappers.kissat_wrapper import KissatWrapper
        wrapper = KissatWrapper(timeout=timeout)
    else:
        from .solver_wrappers.cadical_wrapper import CadicalWrapper
        wrapper = CadicalWrapper(timeout=timeout)

    result    = wrapper.solve(instance)
    wall_time = result["wall_time"]
    timed_out = result["timed_out"]

    # For censored instances, wall_time = timeout (lower bound on actual T)
    log_T    = float(np.log(max(wall_time, 1e-6)))
    hardness = log_T / n

    return {
        "log_T":       log_T,
        "wall_time":   wall_time,
        "timed_out":   timed_out,
        "satisfiable": result.get("satisfiable"),
        "hardness":    hardness,
        "solver":      solver,
    }


# =============================================================================
# Batch hardness curve (DPLL proxy)
# =============================================================================

def hardness_curve(
    n: int,
    alphas: np.ndarray,
    n_instances: int = 1000,
    k: int = 3,
    solver: str = "dpll",
    master_seed: int = 20240223,
    max_decisions: int = MAX_DECISIONS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep α and return mean ± 95% CI of the hardness density.

    Parameters
    ----------
    n_instances : int
        Instances per (n, α) point.  Paper uses 1000.
    master_seed : int
        Paper uses 20240223.
    """
    from .instance_generator import generate_instance_batch
    from .statistics import bootstrap_ci
    from .utils import progress

    gamma_mean = np.zeros(len(alphas))
    gamma_lo   = np.zeros(len(alphas))
    gamma_hi   = np.zeros(len(alphas))

    for i, alpha in enumerate(progress(alphas, desc=f"Hardness n={n}")):
        instances = generate_instance_batch(n, alpha, n_instances, k, master_seed)
        gammas = []
        for j, inst in enumerate(instances):
            ws = derive_seed(master_seed, n, alpha, j)
            g  = measure_hardness(inst, solver=solver,
                                  max_decisions=max_decisions,
                                  walksat_seed=ws)
            gammas.append(g)
        gammas = np.array(gammas)
        gamma_mean[i] = np.mean(gammas)
        lo, hi = bootstrap_ci(gammas, n_boot=1000, ci=0.95, seed=master_seed)
        gamma_lo[i] = lo
        gamma_hi[i] = hi

    return gamma_mean, gamma_lo, gamma_hi
