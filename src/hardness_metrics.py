from __future__ import annotations

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

from .utils import make_rng, derive_seed, get_logger

logger = get_logger(__name__)


MAX_DECISIONS_DEFAULT = 100_000
WALKSAT_MAX_FLIPS     = 100_000
WALKSAT_NOISE         = 0.57






class CNF:
    

    def __init__(self, n: int, clauses: List[List[int]]):
        self.n = n
        self.clauses: List[List[int]] = [list(c) for c in clauses]
        self.m = len(self.clauses)

    @classmethod
    def from_instance(cls, instance: dict) -> "CNF":
        return cls(instance["n"], instance["clauses"])

    def copy(self) -> "CNF":
        return CNF(self.n, [list(c) for c in self.clauses])






def dpll_solve(
    instance: dict,
    max_decisions: int = MAX_DECISIONS_DEFAULT,
) -> Dict:
    
    n       = instance["n"]
    clauses = [list(c) for c in instance["clauses"]]
    counter = [0]

    def solve(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        if counter[0] >= max_decisions:
            return "CUTOFF"


        changed = True
        while changed:
            changed = False
            unit_clauses = [c for c in clauses if len(c) == 1]
            for uc in unit_clauses:
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
                    new_c = [l for l in c if l != -lit]
                    if len(new_c) == 0:
                        return None
                    new_clauses.append(new_c)
                clauses = new_clauses
                changed = True

        if not clauses:

            for v in range(1, n + 1):
                assignment.setdefault(v, True)
            return assignment


        if any(len(c) == 0 for c in clauses):
            return None


        literals = {l for c in clauses for l in c}
        for lit in list(literals):
            vi = abs(lit)
            if -lit not in literals and vi not in assignment:
                assignment[vi] = lit > 0
                clauses = [c for c in clauses if lit not in c]

                return solve(clauses, assignment)


        min_size = min(len(c) for c in clauses)
        min_clauses = [c for c in clauses if len(c) == min_size]
        freq: Dict[int, int] = {}
        for c in min_clauses:
            for l in c:
                vi = abs(l)
                freq[vi] = freq.get(vi, 0) + 1
        branch_var = max(freq, key=freq.get)

        counter[0] += 1
        if counter[0] >= max_decisions:
            return "CUTOFF"


        for val in [True, False]:
            lit     = branch_var if val else -branch_var
            neg_lit = -lit
            new_assignment = dict(assignment)
            new_assignment[branch_var] = val
            new_clauses = [
                [l for l in c if l != neg_lit]
                for c in clauses
                if lit not in c
            ]

            if any(len(c) == 0 for c in new_clauses):
                continue
            result = solve(new_clauses, new_assignment)
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






def walksat_solve(
    instance: dict,
    max_flips: int = WALKSAT_MAX_FLIPS,
    noise: float = WALKSAT_NOISE,
    seed: Optional[int] = None,
    restarts: int = 5,
) -> Dict:
    
    from .instance_generator import count_violated_clauses

    rng         = make_rng(seed)
    n           = instance["n"]
    clauses_raw = instance["clauses"]
    total_flips = 0
    flips_per_restart = max_flips // max(restarts, 1)

    for _ in range(restarts):

        assignment = {i + 1: bool(rng.randint(2)) for i in range(n)}

        for flip in range(flips_per_restart):
            total_flips += 1


            unsat = []
            for clause in clauses_raw:
                sat = any(
                    (l > 0 and assignment[abs(l)]) or (l < 0 and not assignment[abs(l)])
                    for l in clause
                )
                if not sat:
                    unsat.append(clause)

            if not unsat:
                return {"satisfiable": True, "flips": total_flips, "assignment": assignment}


            chosen = unsat[rng.randint(len(unsat))]

            if rng.rand() < noise:

                lit = chosen[rng.randint(len(chosen))]
                vi  = abs(lit)
            else:

                best_vi   = None
                best_breaks = sys.maxsize
                for lit in chosen:
                    vi = abs(lit)

                    breaks = 0
                    for clause in clauses_raw:
                        if vi not in [abs(l) for l in clause]:
                            continue
                        was_sat = any(
                            (l > 0 and assignment[abs(l)]) or (l < 0 and not assignment[abs(l)])
                            for l in clause
                        )
                        if was_sat:

                            new_assignment = dict(assignment)
                            new_assignment[vi] = not new_assignment[vi]
                            still_sat = any(
                                (l > 0 and new_assignment[abs(l)]) or
                                (l < 0 and not new_assignment[abs(l)])
                                for l in clause
                            )
                            if not still_sat:
                                breaks += 1
                    if breaks < best_breaks:
                        best_breaks = breaks
                        best_vi     = vi
                vi = best_vi if best_vi is not None else abs(chosen[0])

            assignment[vi] = not assignment[vi]

    return {"satisfiable": False, "flips": total_flips, "assignment": None}






def measure_hardness(
    instance: dict,
    solver: str = "dpll",
    max_decisions: int = MAX_DECISIONS_DEFAULT,
    walksat_seed: Optional[int] = None,
) -> float:
    
    n = instance["n"]
    if solver == "dpll":
        result = dpll_solve(instance, max_decisions=max_decisions)
        t = result["decisions"]
    else:
        result = walksat_solve(instance, seed=walksat_seed)
        t = result["flips"]

    return float(np.log(t + 1) / n)


def hardness_curve(
    n: int,
    alphas: np.ndarray,
    n_instances: int = 200,
    k: int = 3,
    solver: str = "dpll",
    master_seed: int = 42,
    max_decisions: int = MAX_DECISIONS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    from .instance_generator import generate_instance_batch
    from .statistics import bootstrap_ci

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
        lo, hi = bootstrap_ci(gammas, n_boot=200, ci=0.95, seed=master_seed)
        gamma_lo[i] = lo
        gamma_hi[i] = hi

    return gamma_mean, gamma_lo, gamma_hi



from .utils import progress