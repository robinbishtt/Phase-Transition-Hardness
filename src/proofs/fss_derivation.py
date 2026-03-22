from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..energy_model import ALPHA_S, ALPHA_STAR, NU, FSS_A, FSS_B, ETA, KAPPA


@dataclass
class FSSParameters:
    """All parameters governing the finite-size scaling ansatz (Eq. 15)."""
    alpha_star_inf: float   # thermodynamic-limit hardness peak = 4.20
    nu:             float   # correlation-length exponent = 2.30 ± 0.18
    eta:            float   # Fisher anomalous dimension ≈ 0.22
    kappa:          float   # barrier-growth exponent κ = ν(1−η) ≈ 1.80
    A:              float   # leading FSS shift coefficient = +0.036
    B:              float   # sub-leading FSS shift coefficient = −1.37
    r2_collapse:    float   # quality of data collapse (R² = 0.9997 in manuscript)
    chi2_dof:       float   # χ²/dof of the collapse fit (= 0.89 in manuscript)


MANUSCRIPT_FSS = FSSParameters(
    alpha_star_inf=ALPHA_STAR,
    nu=NU,
    eta=ETA,
    kappa=KAPPA,
    A=FSS_A,
    B=FSS_B,
    r2_collapse=0.9997,
    chi2_dof=0.89,
)


class FSSAnsatz:
    """Finite-size scaling ansatz for the hardness density H(n, α).

    The manuscript (Section 3.3, Eq. 14–15) proposes:

        H(n, α) = n⁻¹ · F(n^{1/ν} · (α − α*(n)))

    with finite-size pseudo-critical density (Eq. 15):

        α*(n) = α*_∞ + A · n^{−1/ν} + B · n^{−2/ν} + O(n^{−3/ν})

    where:
        α*_∞ = 4.20      (thermodynamic-limit hardness peak)
        ν    = 2.30      (correlation-length exponent)
        A    = +0.036    (leading coefficient;  positive  →  finite-n systems
                          peak at slightly higher α than thermodynamic limit)
        B    = −1.37     (sub-leading correction; negative → corrects extrapolation
                          downward at small n, preventing over-shoot)

    Predicted peak locations verified against manuscript Table 2:
        n=100:  α*(100) = 4.18  ✓
        n=200:  α*(200) = 4.19  ✓
        n=400:  α*(400) ≈ 4.20  ✓
        n=800:  α*(800) ≈ 4.20  ✓
    """

    def __init__(self, params: FSSParameters = MANUSCRIPT_FSS):
        self.p = params

    def alpha_star_n(self, n: int) -> float:
        """Finite-size pseudo-critical density α*(n) from Eq. 15.

        Parameters
        ----------
        n : int
            System size (number of Boolean variables).

        Returns
        -------
        float
            α*(n) = 4.20 + 0.036·n^{−1/ν} − 1.37·n^{−2/ν}
        """
        x1 = float(n) ** (-1.0 / self.p.nu)
        x2 = float(n) ** (-2.0 / self.p.nu)
        return float(self.p.alpha_star_inf + self.p.A * x1 + self.p.B * x2)

    def fss_variable(self, n: int, alpha: float) -> float:
        """Return the FSS scaling variable x = n^{1/ν} · (α − α*(n)).

        Under the FSS ansatz H(n,α) = n⁻¹ · F(x), all system sizes collapse
        onto the single master function F when plotted against x.
        """
        a_star = self.alpha_star_n(n)
        return float(n ** (1.0 / self.p.nu) * (alpha - a_star))

    def predict_alpha_stars(self, ns: List[int]) -> np.ndarray:
        """Vectorised α*(n) for a list of system sizes."""
        return np.array([self.alpha_star_n(n) for n in ns])

    def validate_against_manuscript(self) -> dict:
        """Check predicted α*(n) against the four values in Table 2.

        The manuscript states:
            α*(100) = 4.18,  α*(200) = 4.19,
            α*(400) ≈ α*(800) ≈ 4.20.
        """
        predictions = {n: self.alpha_star_n(n) for n in [100, 200, 400, 800]}
        manuscript  = {100: 4.18, 200: 4.19, 400: 4.20, 800: 4.20}
        agreement   = {
            n: abs(predictions[n] - manuscript[n]) < 0.01
            for n in [100, 200, 400, 800]
        }
        return {
            "predictions":  predictions,
            "manuscript":   manuscript,
            "agreement":    agreement,
            "all_agree":    all(agreement.values()),
        }

    def correlation_length(self, alpha: float, alpha_d: float = 3.86) -> float:
        """Return the correlation length ξ ∼ |α − α_d|^{−ν}.

        Near the clustering transition ξ diverges as (α − α_d)^{−ν}.
        This divergence drives finite-size effects and governs how quickly
        the transition sharpens with increasing n.
        """
        diff = abs(alpha - alpha_d)
        if diff < 1e-10:
            return float("inf")
        return float(diff ** (-self.p.nu))

    def barrier_critical_scaling(self, alpha: float, alpha_d: float = 3.86) -> float:
        """Return b(α) ∼ (α − α_d)^κ  near threshold.

        κ = ν(1 − η) ≈ 1.80.  The η ≈ 0.22 correction arises from loop
        corrections on the sparse factor graph; mean-field (η = 0) gives
        κ = ν = 2.30, which the manuscript excludes at > 5σ (Section 4.5).
        """
        diff = alpha - alpha_d
        if diff <= 0.0:
            return 0.0
        return float(diff ** self.p.kappa)


def fss_threshold_shift(
    ns: List[int],
    nu: float = NU,
    A: float = FSS_A,
    B: float = FSS_B,
    alpha_star_inf: float = ALPHA_STAR,
) -> np.ndarray:
    """Convenience function: return α*(n) for a list of system sizes.

    Uses the two-term expansion from manuscript Eq. 15.
    """
    ns_arr = np.array(ns, dtype=float)
    x1 = ns_arr ** (-1.0 / nu)
    x2 = ns_arr ** (-2.0 / nu)
    return alpha_star_inf + A * x1 + B * x2
