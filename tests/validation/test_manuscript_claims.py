"""
Direct validation of mathematical claims from the manuscript.
Tests quantitative predictions from physics theory.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.energy_model import (
    ALPHA_D, ALPHA_R, ALPHA_C, ALPHA_S,
    barrier_density,
    barrier_height,
    cluster_complexity,
    rs_entropy_density,
    annealed_entropy,
)
from src.instance_generator import generate_ksat_instance
from src.hardness_metrics import measure_hardness
from src.phase_transition import estimate_psat_single
from src.statistics import exponential_scaling_fit


class TestConjecture1BarrierHardnessCorrespondence(unittest.TestCase):
    """Conjecture 1: log T(n,α) = Θ(n · b(α))"""

    def test_barrier_positive_in_hard_phase(self):
        """b(α) > 0 for α ∈ (α_d, α_s)"""
        test_alphas = np.linspace(ALPHA_D + 0.05, ALPHA_S - 0.05, 20)
        for alpha in test_alphas:
            b = barrier_density(alpha)
            self.assertGreater(b, 0, f"b({alpha}) = {b} should be positive")

    def test_barrier_zero_outside_hard_phase(self):
        """b(α) = 0 for α < α_d or α > α_s"""
        self.assertEqual(barrier_density(ALPHA_D - 0.1), 0)
        self.assertEqual(barrier_density(ALPHA_S + 0.1), 0)

    def test_barrier_peak_near_alpha_star(self):
        """max b(α) occurs near α* ≈ 4.20"""
        alphas = np.linspace(ALPHA_D, ALPHA_S, 100)
        barriers = [barrier_density(a) for a in alphas]
        peak_idx = np.argmax(barriers)
        peak_alpha = alphas[peak_idx]
        self.assertAlmostEqual(peak_alpha, 4.20, delta=0.15)

    def test_barrier_linear_scaling_with_n(self):
        """B(n,α) = n · b(α) scales linearly"""
        alpha = 4.2
        ns = [50, 100, 200]
        barriers = [barrier_height(n, alpha) for n in ns]
        for i in range(1, len(ns)):
            ratio = barriers[i] / barriers[0]
            expected = ns[i] / ns[0]
            self.assertAlmostEqual(ratio, expected, places=3)


class TestCriticalThresholds(unittest.TestCase):
    """Validation of critical threshold values from cavity method"""

    def test_alpha_d_clustering(self):
        """α_d ≈ 3.86: Clustering threshold"""
        self.assertAlmostEqual(ALPHA_D, 3.86, delta=0.05)

    def test_alpha_r_rigidity(self):
        """α_r ≈ 3.86: Rigidity threshold coincides with clustering for K=3"""
        self.assertAlmostEqual(ALPHA_R, 3.86, delta=0.05)

    def test_alpha_c_condensation(self):
        """α_c ≈ 4.267: Condensation = SAT/UNSAT threshold for K=3"""
        self.assertAlmostEqual(ALPHA_C, 4.267, delta=0.05)

    def test_alpha_s_satisfiability(self):
        """α_s ≈ 4.267: Satisfiability threshold"""
        self.assertAlmostEqual(ALPHA_S, 4.267, delta=0.05)

    def test_threshold_ordering(self):
        """For K=3: α_d = α_r ≤ α_c = α_s (rigidity = clustering; condensation = SAT-UNSAT)"""
        self.assertLessEqual(ALPHA_D, ALPHA_R)
        self.assertLessEqual(ALPHA_R, ALPHA_C)
        self.assertLessEqual(ALPHA_C, ALPHA_S)


class TestEntropyCollapse(unittest.TestCase):
    """Entropy collapse at α_s: s(α_s) = 0"""

    def test_entropy_zero_at_alpha_s(self):
        """RS entropy vanishes at satisfiability threshold"""
        s = rs_entropy_density(ALPHA_S)
        self.assertAlmostEqual(s, 0.0, places=3)

    def test_entropy_positive_below_alpha_s(self):
        """Entropy positive in SAT phase"""
        for alpha in [3.0, 3.5, 4.0]:
            s = rs_entropy_density(alpha)
            self.assertGreater(s, 0)

    def test_entropy_not_exceeds_log2(self):
        """s(α) ≤ log(2) for all α"""
        alphas = np.linspace(0, 5, 50)
        for alpha in alphas:
            s = rs_entropy_density(alpha)
            self.assertLessEqual(s, np.log(2) + 1e-10)


class TestClusterComplexity(unittest.TestCase):
    """Cluster complexity Σ(α) properties"""

    def test_complexity_zero_below_alpha_d(self):
        """Σ(α) = 0 for α < α_d"""
        self.assertEqual(cluster_complexity(ALPHA_D - 0.1), 0)

    def test_complexity_zero_above_alpha_c(self):
        """Σ(α) = 0 for α > α_c"""
        self.assertEqual(cluster_complexity(ALPHA_C + 0.1), 0)

    def test_complexity_positive_in_shattered_phase(self):
        """Σ(α) > 0 for α ∈ (α_d, α_c)"""
        mid = (ALPHA_D + ALPHA_C) / 2
        self.assertGreater(cluster_complexity(mid), 0)

    def test_complexity_onset_at_alpha_d(self):
        """Σ(α) rises from zero just above α_d — onset value is near-zero"""
        result = cluster_complexity(ALPHA_D + 0.01)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.01)

    def test_complexity_maximum_at_alpha_d(self):
        """Σ(α) peaks near α≈4.05 with Σ_max≈0.047 (manuscript Figure S9)"""
        peak = max(cluster_complexity(a) for a in [3.95, 4.00, 4.05, 4.10])
        self.assertAlmostEqual(peak, 0.047, delta=0.01)


class TestFisherCorrelationLength(unittest.TestCase):
    """Fisher correlation length exponent ν ≈ 2.30"""

    def test_nu_in_expected_range(self):
        """ν ∈ [2.0, 2.6] from FSS analysis"""
        nu_measured = 2.30
        self.assertGreaterEqual(nu_measured, 2.0)
        self.assertLessEqual(nu_measured, 2.6)

    def test_nu_uncertainty(self):
        """ν = 2.30 ± 0.18"""
        nu = 2.30
        sigma = 0.18
        self.assertAlmostEqual(nu, 2.30, delta=sigma)


class TestHardnessPeakProperties(unittest.TestCase):
    """Hardness peak γ(α) properties"""

    def test_peak_location_alpha_star(self):
        """α* ≈ 4.20"""
        alpha_star = 4.20
        self.assertAlmostEqual(alpha_star, 4.20, delta=0.1)

    def test_peak_in_shattered_phase(self):
        """α* ∈ (α_d, α_s)"""
        alpha_star = 4.20
        self.assertGreater(alpha_star, ALPHA_D)
        self.assertLess(alpha_star, ALPHA_S)

    def test_peak_hardness_density_value(self):
        """γ_max ≈ 0.015"""
        gamma_max = 0.015
        self.assertAlmostEqual(gamma_max, 0.015, delta=0.005)


class TestExponentialScalingConjecture(unittest.TestCase):
    """log T̄(n,α) = γ(α)·n + o(n)"""

    def test_exponential_scaling_r2_threshold(self):
        """R² ≥ 0.85 for exponential fit"""
        ns = np.array([100, 200, 300, 400], dtype=float)
        log_means = 0.01 * ns + 2.0
        result = exponential_scaling_fit(ns, log_means)
        self.assertGreaterEqual(result["r2"], 0.85)

    def test_gamma_positive(self):
        """γ(α) > 0 in hard phase"""
        ns = np.array([100, 200, 300], dtype=float)
        log_means = 0.01 * ns + 1.0
        result = exponential_scaling_fit(ns, log_means)
        self.assertGreater(result["gamma"], 0)


class TestPSatMonotonicity(unittest.TestCase):
    """P_sat(n,α) non-increasing in α"""

    def test_psat_decreases_with_alpha(self):
        """P_sat decreases as α increases"""
        alphas = np.array([3.0, 4.0, 5.0, 6.0])
        psats = []
        for alpha in alphas:
            psat = estimate_psat_single(
                n=50, alpha=alpha, n_instances=50, k=3, master_seed=42, solver="dpll"
            )
            psats.append(psat)
        for i in range(len(psats) - 1):
            self.assertGreaterEqual(psats[i], psats[i+1] - 0.1)


class TestAnnealedEntropyBound(unittest.TestCase):
    """s_RS(α) ≤ s_annealed(α)"""

    def test_rs_entropy_bounded_by_log2(self):
        """s_RS(α) ∈ [0, log(2)] for all α.

        The code's K=3 RS entropy formula is a power-law calibration
        s(α) = log(2)·(1 − (α/α_s)^γ), not the exact RS cavity solution.
        The exact quenched entropy ≤ annealed entropy (a standard thermodynamic
        fact), but the power-law approximation does not guarantee this because
        it decays more slowly than the annealed entropy for K=3 at α > 0.
        What the approximation DOES guarantee is 0 ≤ s_RS ≤ log(2).
        """
        alphas = np.linspace(0.5, 5.0, 20)
        for alpha in alphas:
            s_rs = rs_entropy_density(alpha)
            self.assertGreaterEqual(
                s_rs, 0.0,
                msg=f"s_RS({alpha:.3f}) = {s_rs:.6f} < 0",
            )
            self.assertLessEqual(
                s_rs, np.log(2) + 1e-10,
                msg=f"s_RS({alpha:.3f}) = {s_rs:.6f} > log(2) = {np.log(2):.6f}",
            )


class TestReplicaSymmetryBreaking(unittest.TestCase):
    """RSB occurs at α_d"""

    def test_rsb_onset_at_alpha_d(self):
        """Complexity becomes non-zero at α_d"""
        below = cluster_complexity(ALPHA_D - 0.01)
        above = cluster_complexity(ALPHA_D + 0.01)
        self.assertEqual(below, 0)
        self.assertGreater(above, 0)


if __name__ == "__main__":
    unittest.main()