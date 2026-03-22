"""Phase-Transition-Hardness: computational artifact for FOCS 2026.

Implements the theoretical framework linking computational hardness in
random 3-SAT to phase-transition phenomena in statistical mechanics,
including the barrier-hardness correspondence, finite-size scaling
analysis, and cryptographic hardness implications.

Public API (most-used objects exported at the package root):

    Phase constants
    ---------------
    ALPHA_D, ALPHA_R      Clustering / rigidity threshold ≈ 3.86
    ALPHA_AT              de Almeida-Thouless instability ≈ 3.92
    ALPHA_C, ALPHA_S      Condensation = SAT/UNSAT threshold ≈ 4.267
    ALPHA_STAR            Peak hardness density ≈ 4.20
    NU                    FSS correlation-length exponent = 2.30
    KAPPA                 Barrier growth exponent = 1.80
    ETA                   Fisher anomalous dimension = 0.22
    FSS_A, FSS_B          FSS shift coefficients (+0.036, -1.37)

    Core functions
    --------------
    barrier_density(alpha)       Intensive inter-cluster barrier b(α)
    cluster_complexity(alpha)    1RSB complexity Σ(α)
    rs_entropy_density(alpha)    RS entropy density s(α)

    Instance generation
    -------------------
    generate_ksat_instance(n, alpha, k, seed)
    generate_instance_batch(n, alpha, n_instances, k, master_seed)

    Measurement
    -----------
    dpll_solve(instance, max_decisions)
    walksat_solve(instance, max_flips, noise, seed)
    measure_hardness(instance, solver, max_decisions)
"""

# Phase transition constants
from .energy_model import (
    ALPHA_D,
    ALPHA_R,
    ALPHA_AT,
    ALPHA_C,
    ALPHA_S,
    ALPHA_STAR,
    NU,
    KAPPA,
    ETA,
    FSS_A,
    FSS_B,
)

# Core energy functions
from .energy_model import (
    barrier_density,
    barrier_height,
    cluster_complexity,
    rs_entropy_density,
    frozen_fraction,
    annealed_entropy,
    free_energy_density,
)

# Instance generation
from .instance_generator import (
    generate_ksat_instance,
    generate_instance_batch,
    count_violated_clauses,
    is_satisfying,
)

# Solvers and measurement
from .hardness_metrics import (
    dpll_solve,
    walksat_solve,
    measure_hardness,
    measure_cdcl_hardness,
    hardness_curve,
)

# Utilities
from .utils import derive_seed, get_logger
from .statistics import bootstrap_ci

__version__ = "1.0.0"
__all__ = [
    # Constants
    "ALPHA_D", "ALPHA_R", "ALPHA_AT", "ALPHA_C", "ALPHA_S", "ALPHA_STAR",
    "NU", "KAPPA", "ETA", "FSS_A", "FSS_B",
    # Energy functions
    "barrier_density", "barrier_height", "cluster_complexity",
    "rs_entropy_density", "frozen_fraction", "annealed_entropy",
    "free_energy_density",
    # Instance generation
    "generate_ksat_instance", "generate_instance_batch",
    "count_violated_clauses", "is_satisfying",
    # Solvers
    "dpll_solve", "walksat_solve", "measure_hardness", "measure_cdcl_hardness",
    "hardness_curve",
    # Utilities
    "derive_seed", "get_logger", "bootstrap_ci",
]
