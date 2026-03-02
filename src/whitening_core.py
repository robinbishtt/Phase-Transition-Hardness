"""Whitening core computation for k-SAT instances."""

import numpy as np
from typing import Dict, List, Set, Tuple
from collections import deque


def compute_whitening_core(instance: Dict) -> Set[int]:
    """Compute the whitening core of a k-SAT instance.

    The whitening core is the set of variables that can be removed
    by iteratively deleting clauses that have a unique satisfying literal.
    """
    n = instance["n"]
    clauses = [set(c) for c in instance["clauses"]]

    removed_clauses = set()
    removed_vars = set()

    changed = True
    while changed:
        changed = False

        for ci, clause in enumerate(clauses):
            if ci in removed_clauses:
                continue

            active_lits = [
                lit for lit in clause
                if abs(lit) not in removed_vars
            ]

            if len(active_lits) == 1:
                removed_clauses.add(ci)
                removed_vars.add(abs(active_lits[0]))
                changed = True

    return removed_vars


def compute_core_size(instance: Dict) -> int:
    """Compute the size of the whitening core."""
    core = compute_whitening_core(instance)
    return len(core)


def compute_core_fraction(instance: Dict) -> float:
    """Compute the fraction of variables in the whitening core."""
    n = instance["n"]
    if n == 0:
        return 0.0
    core_size = compute_core_size(instance)
    return core_size / n


def is_in_whitening_core(instance: Dict, var: int) -> bool:
    """Check if a specific variable is in the whitening core."""
    core = compute_whitening_core(instance)
    return var in core


def compute_clause_whitening_levels(instance: Dict) -> Dict[int, int]:
    """Compute whitening level for each clause.

    Level 0: Clause in the core
    Level k: Clause becomes unit after k whitening rounds
    """
    n = instance["n"]
    clauses = [set(c) for c in instance["clauses"]]
    m = len(clauses)

    levels = {}
    removed_vars = set()
    current_level = 0

    remaining_clauses = set(range(m))

    while remaining_clauses:
        clauses_to_remove = set()

        for ci in remaining_clauses:
            clause = clauses[ci]
            active_lits = [
                lit for lit in clause
                if abs(lit) not in removed_vars
            ]

            if len(active_lits) == 1:
                clauses_to_remove.add(ci)
                levels[ci] = current_level
                removed_vars.add(abs(active_lits[0]))

        if not clauses_to_remove:
            for ci in remaining_clauses:
                levels[ci] = -1
            break

        remaining_clauses -= clauses_to_remove
        current_level += 1

    return levels


def compute_whitening_distribution(instance: Dict) -> Dict[int, int]:
    """Compute distribution of clause whitening levels."""
    levels = compute_clause_whitening_levels(instance)
    distribution = {}

    for level in levels.values():
        distribution[level] = distribution.get(level, 0) + 1

    return distribution


def estimate_core_size_distribution(
    n: int,
    alpha: float,
    n_instances: int = 100,
    k: int = 3,
    seed: int = 42
) -> Dict:
    """Estimate distribution of core sizes across instances."""
    from src.instance_generator import generate_instance_batch

    instances = generate_instance_batch(n, alpha, n_instances, k, seed)

    core_sizes = [compute_core_size(inst) for inst in instances]
    core_fractions = [compute_core_fraction(inst) for inst in instances]

    return {
        "core_sizes": core_sizes,
        "core_fractions": core_fractions,
        "mean_size": np.mean(core_sizes),
        "std_size": np.std(core_sizes),
        "mean_fraction": np.mean(core_fractions),
        "std_fraction": np.std(core_fractions),
        "n": n,
        "alpha": alpha,
        "k": k
    }


def analyze_whitening_transition(
    n: int,
    alphas: np.ndarray,
    n_instances: int = 100,
    k: int = 3,
    seed: int = 42
) -> Dict:
    """Analyze whitening core transition across alpha values."""
    mean_fractions = []
    std_fractions = []

    for alpha in alphas:
        result = estimate_core_size_distribution(
            n, alpha, n_instances, k, seed
        )
        mean_fractions.append(result["mean_fraction"])
        std_fractions.append(result["std_fraction"])

    return {
        "alphas": alphas,
        "mean_fractions": np.array(mean_fractions),
        "std_fractions": np.array(std_fractions),
        "n": n,
        "k": k
    }


def compute_residual_formula(instance: Dict) -> Dict:
    """Compute the residual formula after whitening."""
    core_vars = compute_whitening_core(instance)

    residual_clauses = []
    for clause in instance["clauses"]:
        residual_lits = [
            lit for lit in clause
            if abs(lit) not in core_vars
        ]
        if residual_lits:
            residual_clauses.append(residual_lits)

    residual_vars = set()
    for clause in residual_clauses:
        for lit in clause:
            residual_vars.add(abs(lit))

    return {
        "n": len(residual_vars),
        "m": len(residual_clauses),
        "clauses": residual_clauses,
        "vars": sorted(residual_vars)
    }