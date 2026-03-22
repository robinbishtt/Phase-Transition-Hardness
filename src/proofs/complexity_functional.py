from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from ..energy_model import ALPHA_D, ALPHA_S, ALPHA_STAR, cluster_complexity


class ComplexityFunctional:
    """1RSB complexity functional Σ[{P}] from Theorem 2 (manuscript Eq. 9).

    The CORRECT three-term form is (manuscript Section 2.3, Eq. 9):

        Σ[{P}] = Σ_{(i,a)∈E} log Z_{i→a}
                 − Σ_a (|∂a| − 1) log Z_a
                 − Σ_i (|∂i| − 1) log Z_i

    where:
        Z_{i→a}  = normalisation on edge (i, a) (variable-to-clause direction)
        Z_a      = clause-node normalisation
        Z_i      = variable-node normalisation
        |∂a| = K  (clause degree, always K for K-SAT)
        |∂i|     = variable degree (Poisson(Kα) distributed)

    IMPORTANT CORRECTION: Several references use the incorrect form
    −n log Z_var in place of the three-term sum above.  The manuscript
    explicitly notes this error and validates the correct form through the
    barrier-scaling data b(α) ∼ (α − α_d)^{1.80±0.12} in Section 4.5.

    This class computes the complexity from the analytical calibrated formula
    (see energy_model.cluster_complexity) rather than solving the SP equations
    numerically, which requires population dynamics and is deferred to
    src/survey_propagation/sp_solver.py.
    """

    def __init__(self, k: int = 3):
        self.k = k

    def sigma(self, alpha: float) -> float:
        """Return Σ(α) = lim_{n→∞} n⁻¹ log N_clusters(α).

        Uses the calibrated two-sided power-law from energy_model:
            Σ(α) = A · (α − α_d)^a · (α_s − α)^b
        with A=0.566, a=0.731, b=0.835, yielding:
            Σ(4.20) ≈ 0.027 (the PRG stretch bound from Section 5.2)
            Σ_max   ≈ 0.047 near α ≈ 4.05 (manuscript Figure S9)
        """
        return cluster_complexity(alpha)

    def sigma_curve(self, alphas: np.ndarray) -> np.ndarray:
        """Return Σ(α) for an array of constraint densities."""
        return np.array([cluster_complexity(a) for a in alphas])

    def validate_key_values(self) -> Dict[str, float]:
        """Check Σ at key values reported in the manuscript.

        Manuscript claims:
            Σ(α_d) = 0  (onset of shattered phase)
            Σ(α_s) = 0  (condensation)
            Σ(4.20) ≈ 0.027  (PRG stretch Section 5.2)
        """
        return {
            "sigma_at_alpha_d":   self.sigma(ALPHA_D),
            "sigma_at_alpha_s":   self.sigma(ALPHA_S),
            "sigma_at_alpha_star": self.sigma(ALPHA_STAR),
            "sigma_max":          max(self.sigma(a) for a in np.linspace(ALPHA_D, ALPHA_S, 200)),
            "sigma_peak_alpha":   float(
                np.linspace(ALPHA_D, ALPHA_S, 200)[
                    np.argmax([self.sigma(a) for a in np.linspace(ALPHA_D, ALPHA_S, 200)])
                ]
            ),
        }

    @staticmethod
    def edge_normalisation(
        z_edge: float,
        z_clause: float,
        z_var: float,
        k: int = 3,
        degree_var: float = 3.0 * 4.2,
    ) -> float:
        """Compute Σ from the three normalisation factors at a single edge.

        Parameters
        ----------
        z_edge   : Z_{i→a}, the edge normalisation.
        z_clause : Z_a, the clause normalisation (|∂a|=K for K-SAT).
        z_var    : Z_i, the variable normalisation.
        k        : clause length.
        degree_var : mean variable degree = Kα (Poisson).

        Returns the per-edge contribution to Σ:
            log(z_edge) − (k−1)·log(z_clause) − (degree_var−1)·log(z_var)
        """
        return (
            np.log(max(z_edge,   1e-300))
            - (k - 1.0)          * np.log(max(z_clause, 1e-300))
            - (degree_var - 1.0) * np.log(max(z_var,    1e-300))
        )


def compute_sp_complexity(
    alpha: float,
    k: int = 3,
    use_analytical: bool = True,
) -> float:
    """Compute the SP complexity Σ(α) at a given constraint density.

    Parameters
    ----------
    alpha : float
        Constraint density α = m/n.
    k : int
        Clause length.
    use_analytical : bool
        If True (default), return the analytically calibrated value from
        energy_model.cluster_complexity.  If False, the full population-
        dynamics SP solver (src/survey_propagation/sp_solver.py) would be
        invoked; that path is not yet implemented numerically.

    Returns
    -------
    float
        Σ(α) ≥ 0.  Returns 0.0 for α ∉ (α_d, α_s).
    """
    if use_analytical:
        return cluster_complexity(alpha)
    raise NotImplementedError(
        "Numerical SP population dynamics not yet implemented. "
        "Use use_analytical=True for the calibrated analytical value."
    )
