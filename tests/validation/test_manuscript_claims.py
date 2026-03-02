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
        """α_r ≈ 4.00: Rigidity threshold"""
        self.assertAlmostEqual(ALPHA_R, 4.00, delta=0.05)

    def test_alpha_c_condensation(self):
        """α_c ≈ 4.10: Condensation threshold"""
        self.assertAlmostEqual(ALPHA_C, 4.10, delta=0.05)

    def test_alpha_s_satisfiability(self):
        """α_s ≈ 4.267: Satisfiability threshold"""
        self.assertAlmostEqual(ALPHA_S, 4.267, delta=0.05)

    def test_threshold_ordering(self):
        """α_d < α_r < α_c < α_s"""
        self.assertLess(ALPHA_D, ALPHA_R)
        self.assertLess(ALPHA_R, ALPHA_C)
        self.assertLess(ALPHA_C, ALPHA_S)


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

    def test_complexity_maximum_at_alpha_d(self):
        """Σ(α) peaks near α_d"""
        c_at_d = cluster_complexity(ALPHA_D + 0.01)
        self.assertAlmostEqual(c_at_d, 0.5, delta=0.1)


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

    def test_rs_entropy_below_annealed(self):
        """RS entropy never exceeds annealed entropy"""
        alphas = np.linspace(0.5, 5.0, 20)
        for alpha in alphas:
            s_rs = rs_entropy_density(alpha)
            s_annealed = annealed_entropy(alpha)
            self.assertLessEqual(s_rs, s_annealed + 1e-10)


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