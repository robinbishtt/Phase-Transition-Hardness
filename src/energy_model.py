from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .utils import safe_log, binary_entropy






ALPHA_D   = 3.86
ALPHA_R   = 4.00
ALPHA_C   = 4.10
ALPHA_S   = 4.267
K_DEFAULT = 3






def annealed_entropy(alpha: float, k: int = K_DEFAULT) -> float:
    
    return float(np.log(2.0) + alpha * np.log(1.0 - 2.0 ** (1 - k)))






def rs_entropy_density(alpha: float, k: int = K_DEFAULT) -> float:
    
    if k == 3:

        s = np.log(2.0) * (1.0 - (alpha / ALPHA_S) ** 2.5)
    else:

        s = annealed_entropy(alpha, k)

    return float(np.clip(s, 0.0, np.log(2.0)))






def cluster_complexity(alpha: float) -> float:
    
    if alpha <= ALPHA_D or alpha >= ALPHA_C:
        return 0.0

    t = (ALPHA_C - alpha) / (ALPHA_C - ALPHA_D)
    sigma = 0.5 * t
    return float(max(sigma, 0.0))






def free_energy_density(
    alpha: float,
    beta: float = 1.0,
    k: int = K_DEFAULT,
) -> float:
    
    s = rs_entropy_density(alpha, k)

    annealed = annealed_entropy(alpha, k)
    f = -(1.0 / beta) * max(s, 0.0) + (1.0 / beta) * max(annealed - s, 0.0) * np.exp(-beta)
    return float(f)






def frozen_fraction(alpha: float) -> float:
    
    if alpha < ALPHA_D:
        return 0.0
    if alpha < ALPHA_R:

        t = (alpha - ALPHA_D) / (ALPHA_R - ALPHA_D)
        return float(0.3 * t ** 1.5)
    if alpha < ALPHA_S:

        t = (alpha - ALPHA_R) / (ALPHA_S - ALPHA_R)
        return float(0.3 + 0.7 * t)
    return 1.0






def barrier_density(alpha: float, k: int = K_DEFAULT) -> float:
    
    if alpha <= ALPHA_D or alpha >= ALPHA_S:
        return 0.0


    alpha_peak = 4.20
    width      = 0.18
    b = 0.015 * np.exp(-((alpha - alpha_peak) ** 2) / (2 * width ** 2))
    return float(max(b, 0.0))


def barrier_height(n: int, alpha: float, k: int = K_DEFAULT) -> float:
    
    return float(n) * barrier_density(alpha, k)






def compute_partition_function_log(
    instance: dict,
    beta: float = 1.0,
) -> float:
    
    from .instance_generator import count_violated_clauses

    n = instance["n"]
    if n > 20:
        raise ValueError("Brute-force partition function requires n ≤ 20.")

    log_z = -np.inf
    for mask in range(2 ** n):
        assignment = {i + 1: bool((mask >> i) & 1) for i in range(n)}
        e = count_violated_clauses(instance, assignment)
        log_z = np.logaddexp(log_z, -beta * e)

    return float(log_z)


def gibbs_sample(
    instance: dict,
    beta: float = 5.0,
    n_steps: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[dict, float]:
    
    from .instance_generator import count_violated_clauses
    from .utils import make_rng

    rng = make_rng(seed)
    n = instance["n"]


    assignment = {i + 1: bool(rng.randint(2)) for i in range(n)}
    energy = float(count_violated_clauses(instance, assignment))

    for _ in range(n_steps):
        vi = int(rng.randint(1, n + 1))

        assignment[vi] = not assignment[vi]
        new_energy = float(count_violated_clauses(instance, assignment))
        delta_e = new_energy - energy

        if delta_e <= 0 or rng.rand() < np.exp(-beta * delta_e):
            energy = new_energy
        else:
            assignment[vi] = not assignment[vi]

    return assignment, energy