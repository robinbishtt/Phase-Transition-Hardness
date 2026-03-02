"""Rigidity analysis for frozen variables in k-SAT clusters."""

import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def compute_variable_dependencies(instance: Dict) -> Dict[int, Set[int]]:
    """Compute clause dependencies for each variable."""
    deps = defaultdict(set)
    for ci, clause in enumerate(instance["clauses"]):
        for lit in clause:
            deps[abs(lit)].add(ci)
    return dict(deps)


def find_unit_clauses(instance: Dict) -> List[Tuple[int, int]]:
    """Find all unit clauses and their implications."""
    units = []
    for clause in instance["clauses"]:
        if len(clause) == 1:
            units.append((abs(clause[0]), 1 if clause[0] > 0 else 0))
    return units


def propagate_units(instance: Dict, assignment: Dict[int, bool]) -> Dict[int, bool]:
    """Unit propagation to extend partial assignment."""
    result = dict(assignment)
    changed = True

    while changed:
        changed = False
        for clause in instance["clauses"]:
            unassigned = []
            satisfied = False

            for lit in clause:
                var = abs(lit)
                if var in result:
                    if (lit > 0 and result[var]) or (lit < 0 and not result[var]):
                        satisfied = True
                        break
                else:
                    unassigned.append(lit)

            if not satisfied and len(unassigned) == 1:
                lit = unassigned[0]
                var = abs(lit)
                result[var] = lit > 0
                changed = True

    return result


def compute_rigidity_profile(
    instance: Dict,
    assignment: Dict[int, bool]
) -> Dict[int, bool]:
    """Compute rigidity profile: which variables are frozen."""
    n = instance["n"]
    frozen = {}

    for var in range(1, n + 1):
        if var not in assignment:
            frozen[var] = False
            continue

        test_assign = dict(assignment)
        test_assign[var] = not test_assign[var]

        propagated = propagate_units(instance, test_assign)

        contradiction = False
        for clause in instance["clauses"]:
            all_assigned = True
            satisfied = False
            for lit in clause:
                v = abs(lit)
                if v not in propagated:
                    all_assigned = False
                    break
                if (lit > 0 and propagated[v]) or (lit < 0 and not propagated[v]):
                    satisfied = True
                    break
            if all_assigned and not satisfied:
                contradiction = True
                break

        frozen[var] = contradiction

    return frozen


def compute_frozen_fraction(
    instance: Dict,
    assignment: Dict[int, bool]
) -> float:
    """Compute fraction of frozen variables."""
    frozen = compute_rigidity_profile(instance, assignment)
    if not frozen:
        return 0.0
    return sum(frozen.values()) / len(frozen)


def estimate_cluster_rigidity(
    instance: Dict,
    n_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Estimate rigidity by sampling from cluster."""
    from src.hardness_metrics import walksat_solve

    rng = np.random.RandomState(seed)
    frozen_fractions = []

    for _ in range(n_samples):
        result = walksat_solve(
            instance,
            max_flips=10000,
            noise=0.57,
            seed=rng.randint(2**31)
        )

        if result["satisfiable"]:
            ff = compute_frozen_fraction(instance, result["assignment"])
            frozen_fractions.append(ff)

    if not frozen_fractions:
        return {"mean": 0.0, "std": 0.0, "n_samples": 0}

    return {
        "mean": np.mean(frozen_fractions),
        "std": np.std(frozen_fractions),
        "n_samples": len(frozen_fractions)
    }


def compute_rigidity_threshold_indicator(
    instance: Dict,
    threshold: float = 0.5
) -> bool:
    """Check if instance is above rigidity threshold."""
    result = estimate_cluster_rigidity(instance, n_samples=50)
    return result["mean"] > threshold


def analyze_rigidity_transition(
    n: int,
    alphas: np.ndarray,
    n_instances: int = 100,
    k: int = 3,
    seed: int = 42
) -> Dict:
    """Analyze rigidity transition across alpha values."""
    from src.instance_generator import generate_instance_batch

    frozen_fractions = []

    for alpha in alphas:
        instances = generate_instance_batch(
            n, alpha, n_instances, k, master_seed=seed
        )

        fractions = []
        for inst in instances:
            result = estimate_cluster_rigidity(inst, n_samples=20)
            fractions.append(result["mean"])

        frozen_fractions.append({
            "alpha": alpha,
            "mean": np.mean(fractions),
            "std": np.std(fractions),
            "sem": np.std(fractions) / np.sqrt(len(fractions))
        })

    return {
        "alphas": alphas,
        "frozen_fractions": frozen_fractions,
        "n": n,
        "k": k
    }