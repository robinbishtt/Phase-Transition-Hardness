from __future__ import annotations

import numpy as np
from typing import List, Optional

from .utils import make_rng, derive_seed






def generate_ksat_instance(
    n: int,
    alpha: float,
    k: int = 3,
    seed: Optional[int] = None,
) -> dict:
    
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if alpha <= 0:
        raise ValueError(f"alpha={alpha} must be positive")

    rng = make_rng(seed)
    m = int(round(alpha * n))

    clauses: List[List[int]] = []
    for _ in range(m):

        vars_ = rng.choice(n, size=k, replace=False) + 1

        signs = rng.choice([-1, 1], size=k)
        clause = (signs * vars_).tolist()
        clauses.append(clause)

    return {
        "n": n,
        "k": k,
        "alpha": alpha,
        "m": m,
        "clauses": clauses,
        "seed": seed,
    }


def generate_instance_batch(
    n: int,
    alpha: float,
    n_instances: int,
    k: int = 3,
    master_seed: int = 42,
) -> List[dict]:
    
    instances = []
    for idx in range(n_instances):
        child_seed = derive_seed(master_seed, n, alpha, idx)
        inst = generate_ksat_instance(n=n, alpha=alpha, k=k, seed=child_seed)
        instances.append(inst)
    return instances






def instance_to_adjacency(instance: dict):
    
    n = instance["n"]
    clauses = instance["clauses"]

    var_to_clauses: List[List[int]] = [[] for _ in range(n + 1)]
    clause_to_vars: List[List[int]] = []

    for ci, clause in enumerate(clauses):
        cvars = []
        for lit in clause:
            vi = abs(lit)
            var_to_clauses[vi].append(ci)
            cvars.append(vi)
        clause_to_vars.append(cvars)

    return var_to_clauses, clause_to_vars


def count_violated_clauses(instance: dict, assignment: dict) -> int:
    
    violated = 0
    for clause in instance["clauses"]:
        satisfied = False
        for lit in clause:
            vi = abs(lit)
            val = assignment.get(vi, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if not satisfied:
            violated += 1
    return violated


def is_satisfying(instance: dict, assignment: dict) -> bool:
    
    return count_violated_clauses(instance, assignment) == 0