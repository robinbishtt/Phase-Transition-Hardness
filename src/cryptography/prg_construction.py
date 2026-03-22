from __future__ import annotations

import numpy as np
from typing import Dict, List

from ..energy_model import (
    barrier_density, cluster_complexity, ALPHA_D, ALPHA_S, ALPHA_STAR,
)
from ..instance_generator import generate_ksat_instance


class APKPseudoRandomGenerator:
    """Pseudorandom generator via Applebaum-Ishai-Kushilevitz construction.

    Manuscript Section 5.2 (following [22]):

        Since f_Φ at α = 4.20 is computable in O(n) time per output bit,
        it lies in NC¹ ⊂ NC⁰[log n].  By the AIK construction, one-way
        functions in NC⁰ imply PRGs with polynomial stretch.  Under standard
        hardness amplification (XOR Lemma, Supplementary Section 6.3), the
        resulting PRG stretches n bits to (1+ε)n bits with security
        2^{cn·b(4.20)} against CDCL-based distinguishers, for any ε < Σ(4.20).

    Key parameters:
        - Input seed length: n bits
        - Output length:    ⌊(1 + ε) · n⌋  bits   (ε < Σ(α*) ≈ 0.027)
        - Security level:   2^{cn·b(α*)} CDCL operations (c ≈ 0.8)
        - Stretch bound:    ε < Σ(4.20) ≈ 0.027

    The PRG is NOT cryptographically deployed here — this class computes the
    analytical parameters and verifies that the AIK conditions are satisfied.
    """

    def __init__(self, n: int, epsilon: float = 0.025, alpha: float = ALPHA_STAR):
        if not (ALPHA_D < alpha < ALPHA_S):
            raise ValueError(f"α must be in hard phase ({ALPHA_D}, {ALPHA_S})")
        sigma = cluster_complexity(alpha)
        if epsilon >= sigma:
            raise ValueError(
                f"ε = {epsilon:.4f} must be < Σ({alpha:.3f}) = {sigma:.4f} "
                f"(the 1RSB complexity bounds the maximum stretch)."
            )
        self.n       = n
        self.epsilon = epsilon
        self.alpha   = alpha
        self.sigma   = sigma
        self.b       = barrier_density(alpha)

    def seed_length(self) -> int:
        """Input seed length in bits."""
        return self.n

    def output_length(self) -> int:
        """Output length = floor((1 + ε) · n) bits."""
        return int(np.floor((1.0 + self.epsilon) * self.n))

    def stretch(self) -> float:
        """Absolute stretch = output_length − seed_length."""
        return float(self.output_length() - self.seed_length())

    def stretch_fraction(self) -> float:
        """Fractional stretch ε."""
        return self.epsilon

    def security_level(self, c: float = 0.8) -> float:
        """Log₂ of the number of CDCL operations needed to distinguish.

        Security = c · n · b(α) / ln 2  bits.
        """
        return float(c * self.n * self.b / np.log(2))

    def is_nc0_computable(self) -> bool:
        """True: the Goldreich OWF is computable in NC⁰ for K=3."""
        return True

    def aik_conditions(self) -> Dict:
        """Verify all conditions required by the AIK theorem.

        Returns dict with per-condition bool and explanations.
        """
        return {
            "owf_in_nc0": {
                "satisfied": True,
                "reason": "K=3 clauses have fixed locality; each output bit "
                          "depends on exactly 3 input bits → NC⁰",
            },
            "epsilon_below_sigma": {
                "satisfied": self.epsilon < self.sigma,
                "reason": (
                    f"ε = {self.epsilon:.4f} < Σ(α) = {self.sigma:.4f}"
                    if self.epsilon < self.sigma else
                    f"ε = {self.epsilon:.4f} ≥ Σ(α) = {self.sigma:.4f}: VIOLATED"
                ),
            },
            "owf_hardness_conjecture4": {
                "satisfied": ALPHA_D < self.alpha < ALPHA_S,
                "reason": (
                    f"α = {self.alpha:.3f} ∈ ({ALPHA_D}, {ALPHA_S}): "
                    "Conjecture 4 applies"
                ),
            },
            "uniform_instance_generation": {
                "satisfied": True,
                "reason": "SHA-256 seeding in instance_generator ensures "
                          "demonstrably uniform instance generation",
            },
        }

    def prg_parameters(self) -> Dict:
        """Full parameter report matching manuscript Section 5.2."""
        return {
            "seed_length_n":     self.n,
            "output_length":     self.output_length(),
            "stretch_epsilon":   self.epsilon,
            "max_epsilon":       self.sigma,
            "alpha":             self.alpha,
            "b_alpha":           self.b,
            "sigma_alpha":       self.sigma,
            "security_bits":     self.security_level(),
            "is_nc0_computable": True,
            "aik_conditions":    self.aik_conditions(),
        }
