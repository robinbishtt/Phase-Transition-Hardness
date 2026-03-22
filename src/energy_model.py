"""Energy model and statistical mechanics for random K-SAT.

All formulas and constants match the submitted FOCS manuscript exactly.

Physical constants for K=3:
  α_d  ≈ 3.86   clustering / dynamical threshold
  α_AT ≈ 3.92   de Almeida-Thouless instability
  α_c  ≈ 4.267  condensation = satisfiability (for K=3, α_c = α_s)
  α_s  ≈ 4.267  satisfiability threshold (Ding-Sly-Sun 2015)
  α*   ≈ 4.20   peak hardness density
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .utils import safe_log, binary_entropy


# =============================================================================
# Phase transition constants - K=3 (from cavity-method literature)
# =============================================================================
ALPHA_D   = 3.86    # clustering / dynamical threshold
ALPHA_AT  = 3.92    # de Almeida-Thouless instability onset
ALPHA_C   = 4.267   # condensation threshold; coincides with α_s for K=3 (Ding-Sly-Sun 2015)
ALPHA_S   = 4.267   # SAT/UNSAT threshold (Ding-Sly-Sun 2015)
ALPHA_R   = 3.86    # rigidity threshold ≈ α_d for K=3

# Barrier-hardness parameters (manuscript Section 2.4)
ALPHA_STAR = 4.20   # peak-hardness density
KAPPA      = 1.80   # barrier-growth exponent κ = ν(1 − η) ≈ 2.30 × 0.78
NU         = 2.30   # FSS correlation-length exponent
ETA        = 0.22   # Fisher anomalous dimension

# FSS shift coefficients (Eq. 15 in manuscript)
FSS_A      = +0.036  # leading coefficient
FSS_B      = -1.37   # sub-leading coefficient

K_DEFAULT  = 3


# =============================================================================
# Annealed entropy (exact formula for K-SAT)
# =============================================================================
def annealed_entropy(alpha: float, k: int = K_DEFAULT) -> float:
    """Compute the annealed entropy density for random K-SAT.

    s_annealed(α) = log 2 + α · log(1 − 2^{1-k})

    This is exact for the random ensemble.
    """
    return float(np.log(2.0) + alpha * np.log(1.0 - 2.0 ** (1 - k)))


# =============================================================================
# RS entropy density - cavity-method approximation for K=3
# =============================================================================
def rs_entropy_density(alpha: float, k: int = K_DEFAULT) -> float:
    """RS entropy density s(α) from the replica-symmetric cavity solution.

    For K=3: uses the power-law form calibrated to K=3 RS cavity numerics
    (Mézard-Montanari 2009, Chapter 19):

        s(α) = log(2) · max(1 − (α/α_s)^γ, 0),   γ ≈ 1.387

    Calibration: γ chosen so that s(α_d) ≈ 0.09, matching the known RS
    entropy near the clustering threshold; s(α_s) = 0 by construction;
    s(0) = log 2 exactly.

    For K ≠ 3: falls back to the annealed entropy (which is exact for
    the random ensemble and serves as the RS approximation for other K).

    Note: we do NOT clamp against the annealed entropy for K=3, which is
    negative for K=3 when α > log(2)/log(4/3) ≈ 2.41, and thus provides
    no useful upper bound in the regime of interest.
    """
    if k != K_DEFAULT:
        return float(max(annealed_entropy(alpha, k), 0.0))

    if alpha <= 0.0:
        return float(np.log(2.0))
    if alpha >= ALPHA_S:
        return 0.0

    _GAMMA_S = 1.387   # calibrated: s(α_d=3.86) ≈ 0.09
    t = alpha / ALPHA_S
    s = float(np.log(2.0) * max(1.0 - t ** _GAMMA_S, 0.0))
    return s


# =============================================================================
# Cluster complexity Σ(α) - 1RSB phase (α_d, α_c)
# =============================================================================
def cluster_complexity(alpha: float) -> float:
    """1RSB cluster complexity Σ(α) = lim_{n→∞} n^{-1} log N_clusters.

    Uses the two-sided power-law form derived from the SP functional
    (Mézard-Parisi 2003; manuscript Section 2.3):

        Σ(α) = A · (α − α_d)^a · (α_s − α)^b

    Calibration:
      - Peak at α_peak ≈ 4.05 with Σ_max ≈ 0.047 (manuscript Figure S9)
      - Σ(4.20) ≈ 0.027 (PRG stretch bound, manuscript Section 5.2)
      - Σ(α_d) = Σ(α_s) = 0 by construction

    Parameters: A=0.566, a=0.731, b=0.835 (derived analytically from the
    peak location and calibration constraints above).
    """
    if alpha <= ALPHA_D or alpha >= ALPHA_S:
        return 0.0

    _A = 0.566
    _a = 0.731
    _b = 0.835
    sigma = _A * (alpha - ALPHA_D) ** _a * (ALPHA_S - alpha) ** _b
    return float(max(sigma, 0.0))


# =============================================================================
# Barrier density b(α) - 1RSB power-law formula (manuscript Eq. 1)
# =============================================================================
def barrier_density(alpha: float, k: int = K_DEFAULT) -> float:
    """Intensive inter-cluster energy barrier b(α) from 1RSB analysis.

    Two-sided power-law formula (manuscript Definition 2.3):

        b(α) = A · (α − α_d)^κ · (α_s − α)^β

    where the peak constraint b'(α*) = 0 at α* = 4.20 gives:

        β = κ · (α_s − α*) / (α* − α_d)  ≈  1.80 × 0.067/0.34  ≈  0.3547

    Calibration:
      A = 0.3819 chosen so that b(α*=4.20) = H∞ = 0.0210, matching
      the thermodynamic-limit barrier obtained by FSS extrapolation of
      H(n, α) = E[log T]/n (manuscript Table 2, H∞ column).

      Note on the two calibration targets:
        H∞ = 0.0210  — the n→∞ FSS extrapolation of the intensive
                        hardness density (used for b(α) calibration).
        slope = 0.0122 ± 0.0004  — the finite-n linear-regression slope
                        of log T̄ vs n at n ∈ {100,...,800}, α = 4.20.
      These are numerically distinct: H∞ is the thermodynamic limit while
      the regression slope reflects finite-size corrections at the
      experimental system sizes.  A = 0.3819 calibrates b(α) to H∞.

    This form:
      - Rises from zero at α_d as (α−α_d)^1.80  [power-law onset]
      - Peaks exactly at α* = 4.20              [barrier-hardness max]
      - Vanishes at α_s = 4.267                 [condensation cutoff]
      - Gives zero for α < α_d and α > α_s
    """
    if alpha <= ALPHA_D or alpha >= ALPHA_S:
        return 0.0

    # A = 0.3819 calibrated so that b(α*=4.20) = 0.021, matching the
    # thermodynamic-limit barrier H∞ = 0.0210 from Table 2 of the manuscript
    # (FSS extrapolation, not the finite-n regression slope 0.0122).
    _A    = 0.3819
    _BETA = KAPPA * (ALPHA_S - ALPHA_STAR) / (ALPHA_STAR - ALPHA_D)  # ≈ 0.3547
    b = _A * (alpha - ALPHA_D) ** KAPPA * (ALPHA_S - alpha) ** _BETA
    return float(max(b, 0.0))


def barrier_height(n: int, alpha: float, k: int = K_DEFAULT) -> float:
    """Extensive barrier height B(n,α) = n · b(α) (manuscript Definition 2.3)."""
    return float(n) * barrier_density(alpha, k)


# =============================================================================
# Free energy density
# =============================================================================
def free_energy_density(
    alpha: float,
    beta: float = 1.0,
    k: int = K_DEFAULT,
) -> float:
    """Replica-symmetric free energy density f(β, α).

    f = -β^{-1} · s_RS(α) + corrections from cluster structure.
    """
    s = rs_entropy_density(alpha, k)
    annealed = annealed_entropy(alpha, k)
    f = -(1.0 / beta) * max(s, 0.0) + (1.0 / beta) * max(annealed - s, 0.0) * np.exp(-beta)
    return float(f)


# =============================================================================
# Frozen variable fraction
# =============================================================================
def frozen_fraction(alpha: float) -> float:
    """Fraction of variables frozen within a 1RSB solution cluster.

    Piecewise model from the whitening / rigidity analysis:
      - Below α_d: no frozen variables
      - α_d to α_s: fraction rises from 0 to ~1
    Consistent with Figure S3 (entropy density curve) of the manuscript.
    """
    if alpha < ALPHA_D:
        return 0.0
    if alpha >= ALPHA_S:
        return 1.0

    t = (alpha - ALPHA_D) / (ALPHA_S - ALPHA_D)
    # Sigmoid-like profile calibrated to cavity numerics
    return float(t ** 1.5 * (1.5 - 0.5 * t))


# =============================================================================
# Partition function (brute-force, n ≤ 20 only)
# =============================================================================
def compute_partition_function_log(
    instance: dict,
    beta: float = 1.0,
) -> float:
    """Exact log-partition function via brute force.  Requires n ≤ 20."""
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


# =============================================================================
# Gibbs sampler
# =============================================================================
def gibbs_sample(
    instance: dict,
    beta: float = 5.0,
    n_steps: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[dict, float]:
    """Metropolis–Hastings Gibbs sampler on the clause Hamiltonian."""
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
