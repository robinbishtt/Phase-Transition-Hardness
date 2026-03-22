from __future__ import annotations

import hashlib
import numpy as np
from typing import Dict, Optional, Tuple

from ..energy_model import barrier_density, ALPHA_D, ALPHA_S, ALPHA_STAR
from ..instance_generator import generate_ksat_instance, count_violated_clauses, is_satisfying


class KSATProofOfWork:
    """K-SAT based proof-of-work construction (manuscript Section 5, [23]).

    Proof-of-work requires three properties (following Dwork and Naor [23]):
        1. Solutions are hard to find (proof of work).
        2. Solutions are easy to verify (efficient verification in O(kn) time).
        3. Difficulty can be tuned (via n and α).

    Random K-SAT instances near the satisfiability threshold satisfy all three.

    The difficulty parameter D is the expected number of CDCL decisions required
    to find a solution, which scales as exp(n · b(α)) by Conjecture 4.  Setting
    α = α* ≈ 4.20 maximises b(α) per unit n, giving the best security-per-bit.

    The puzzle is defined by:
        - A publicly verifiable instance Φ ∼ K(n, αn, 3)
        - A challenge c = SHA-256(Φ || nonce)
        - A solution is any satisfying assignment x of Φ

    Puzzle generation must be transparent:  Φ is generated from a public seed
    so any party can verify uniformity.  Near-duplicate clauses, planted
    solutions, or any detectable bias invalidates the hardness argument.
    """

    def __init__(
        self,
        n:          int,
        alpha:      float = ALPHA_STAR,
        k:          int   = 3,
        master_seed: int  = 20240223,
    ):
        if not (ALPHA_D < alpha < ALPHA_S):
            raise ValueError(
                f"α={alpha:.4f} outside hard phase ({ALPHA_D}, {ALPHA_S})."
            )
        self.n     = n
        self.alpha = alpha
        self.k     = k
        self.seed  = master_seed
        self.instance = generate_ksat_instance(n, alpha, k, seed=master_seed)

    # ── Puzzle interface ─────────────────────────────────────────────────────

    def generate_puzzle(self, nonce: int = 0) -> Dict:
        """Generate the proof-of-work puzzle.

        Returns a dict containing the instance, challenge hash, and metadata.
        The challenge binds the puzzle to a specific nonce to prevent reuse.
        """
        challenge = self._compute_challenge(nonce)
        return {
            "instance":   self.instance,
            "challenge":  challenge,
            "nonce":      nonce,
            "n":          self.n,
            "alpha":      self.alpha,
            "k":          self.k,
            "seed":       self.seed,
            "difficulty": self.expected_difficulty(),
        }

    def verify_solution(self, assignment: Dict[int, bool]) -> bool:
        """Verify a proposed solution in O(k · m) time.

        A valid solution is any satisfying assignment of the K-SAT instance.
        Verification requires only k checks per clause — no search required.
        """
        return is_satisfying(self.instance, assignment)

    def expected_difficulty(self) -> float:
        """Expected log-number of CDCL decisions: n · b(α).

        This equals log(expected_solver_time) up to additive constants.
        """
        return float(self.n * barrier_density(self.alpha))

    def security_bits(self) -> float:
        """Security in bits: S = c_eff × n × b(α) / ln 2.

        Uses empirical effective constant c_eff = 3.301 matching Table 6.
        """
        return float(3.301 * self.n * barrier_density(self.alpha) / np.log(2))

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_challenge(self, nonce: int) -> str:
        """Compute challenge = SHA-256(instance_digest || nonce)."""
        clauses_str = str(sorted(
            [tuple(sorted(c)) for c in self.instance["clauses"]]
        ))
        raw = f"{clauses_str}:{nonce}:{self.n}:{self.alpha}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _instance_digest(self) -> str:
        """Compact deterministic digest of the SAT instance."""
        clauses_str = str(sorted(
            [tuple(sorted(c)) for c in self.instance["clauses"]]
        ))
        return hashlib.sha256(clauses_str.encode()).hexdigest()[:16]


def pow_difficulty_parameter(
    target_bits: int,
    alpha: float = ALPHA_STAR,
) -> int:
    """Return the minimum n needed to achieve target_bits of security.

    Inverts S = n · b(α) / ln 2 to find n_min = ceil(S · ln 2 / b(α)).

    Parameters
    ----------
    target_bits : int
        Desired security level in bits.
    alpha : float
        Constraint density (default: peak hardness α* = 4.20).

    Returns
    -------
    int
        Minimum n (number of Boolean variables) for the target security.
    """
    b = barrier_density(alpha)
    if b <= 0.0:
        raise ValueError(
            f"b({alpha:.4f}) = 0; choose α ∈ (α_d, α_s) = ({ALPHA_D}, {ALPHA_S})."
        )
    # The empirical effective constant c_eff=3.301 converts the theoretical
    # barrier b(α) to security bits via S = c_eff · n · b / ln2.
    # Inverting: n_min = ceil(target_bits · ln2 / (c_eff · b)).
    C_EFF = 3.301
    n_min = int(np.ceil(target_bits * np.log(2) / (C_EFF * b)))
    return n_min
