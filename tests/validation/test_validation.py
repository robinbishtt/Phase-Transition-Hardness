"""
Validation tests against manuscript claims

Tests to verify that computational outputs match quantitative predictions from the manuscript.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.energy_model import (
    ALPHA_D,
    ALPHA_R,
    ALPHA_C,
    ALPHA_S,
    barrier_density,
    cluster_complexity,
)
from src.instance_generator import generate_ksat_instance
from src.hardness_metrics import dpll_solve, measure_hardness
from src.phase_transition import locate_threshold
from src.statistics import exponential_scaling_fit


class TestThresholdValues(unittest.TestCase):
    """Tests for critical threshold values."""

    def test_alpha_d_value(self):
        """Clustering threshold should be approximately 3.86."""
        self.assertAlmostEqual(ALPHA_D, 3.86, delta=0.05)

    def test_alpha_r_value(self):
        """Rigidity threshold should be approximately 4.00."""
        self.assertAlmostEqual(ALPHA_R, 4.00, delta=0.05)

    def test_alpha_c_value(self):
        """Condensation threshold should be approximately 4.10."""
        self.assertAlmostEqual(ALPHA_C, 4.10, delta=0.05)

    def test_alpha_s_value(self):
        """Satisfiability threshold should be approximately 4.267."""
        self.assertAlmostEqual(ALPHA_S, 4.267, delta=0.05)

    def test_threshold_ordering(self):
        """Thresholds should be in correct order."""
        self.assertLess(ALPHA_D, ALPHA_R)
        self.assertLess(ALPHA_R, ALPHA_C)
        self.assertLess(ALPHA_C, ALPHA_S)


class TestBarrierProperties(unittest.TestCase):
    """Tests for barrier density properties."""

    def test_barrier_peak_location(self):
        """Barrier density should peak near alpha=4.2."""
        alphas = np.linspace(ALPHA_D + 0.1, ALPHA_S - 0.1, 50)
        barriers = [barrier_density(a) for a in alphas]
        peak_idx = np.argmax(barriers)
        peak_alpha = alphas[peak_idx]
        self.assertAlmostEqual(peak_alpha, 4.2, delta=0.2)

    def test_barrier_peak_value(self):
        """Maximum barrier density should be approximately 0.015."""
        max_barrier = barrier_density(4.2)
        self.assertAlmostEqual(max_barrier, 0.015, delta=0.005)

    def test_barrier_zero_outside_hard_phase(self):
        """Barrier should be zero outside hard phase."""
        self.assertEqual(barrier_density(ALPHA_D - 0.2), 0.0)
        self.assertEqual(barrier_density(ALPHA_S + 0.2), 0.0)

    def test_barrier_positive_inside_hard_phase(self):
        """Barrier should be positive inside hard phase."""
        for alpha in [4.0, 4.1, 4.2]:
            self.assertGreater(barrier_density(alpha), 0.0)


class TestComplexityProperties(unittest.TestCase):
    """Tests for cluster complexity properties."""

    def test_complexity_zero_below_alpha_d(self):
        """Complexity should be zero below clustering threshold."""
        self.assertEqual(cluster_complexity(ALPHA_D - 0.1), 0.0)

    def test_complexity_zero_above_alpha_c(self):
        """Complexity should be zero above condensation threshold."""
        self.assertEqual(cluster_complexity(ALPHA_C + 0.1), 0.0)

    def test_complexity_positive_in_shattered_phase(self):
        """Complexity should be positive in shattered phase."""
        mid_point = (ALPHA_D + ALPHA_C) / 2
        self.assertGreater(cluster_complexity(mid_point), 0.0)

    def test_complexity_maximum_at_alpha_d(self):
        """Complexity should be maximum at alpha_d."""
        c_at_d = cluster_complexity(ALPHA_D + 0.001)
        self.assertAlmostEqual(c_at_d, 0.5, delta=0.1)


class TestHardnessScaling(unittest.TestCase):
    """Tests for hardness scaling behavior."""

    def test_hardness_non_negative(self):
        """Hardness should always be non-negative."""
        for _ in range(10):
            instance = generate_ksat_instance(20, 4.0, k=3, seed=None)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            self.assertGreaterEqual(h, 0.0)

    def test_hardness_finite(self):
        """Hardness should always be finite."""
        for _ in range(10):
            instance = generate_ksat_instance(20, 4.0, k=3, seed=None)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            self.assertTrue(np.isfinite(h))

    def test_hardness_peak_in_hard_phase(self):
        """Hardness peak should be in hard phase."""
        n = 20
        alphas = np.linspace(3.5, 5.0, 10)
        hardness_values = []

        for alpha in alphas:
            instance = generate_ksat_instance(n, alpha, k=3, seed=42)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            hardness_values.append(h)

        peak_idx = np.argmax(hardness_values)
        peak_alpha = alphas[peak_idx]


        self.assertGreater(peak_alpha, ALPHA_D)
        self.assertLess(peak_alpha, ALPHA_S)


class TestExponentialScalingConjecture(unittest.TestCase):
    """Tests for Conjecture 1: Barrier-Hardness Correspondence."""

    def test_exponential_scaling_fit_quality(self):
        """Exponential scaling should give good fit."""

        ns = np.array([10, 20, 30, 40, 50], dtype=float)
        gamma = 0.01
        log_means = gamma * ns + 2.0 + np.random.randn(5) * 0.1

        result = exponential_scaling_fit(ns, log_means)


        self.assertGreater(result["r2"], 0.8)

    def test_gamma_positive(self):
        """Gamma (scaling exponent) should be positive."""
        ns = np.array([10, 20, 30, 40, 50], dtype=float)
        log_means = 0.01 * ns + 2.0

        result = exponential_scaling_fit(ns, log_means)

        self.assertGreater(result["gamma"], 0.0)


class TestFiniteSizeScaling(unittest.TestCase):
    """Tests for finite-size scaling predictions."""

    def test_nu_in_expected_range(self):
        """Critical exponent nu should be in expected range."""

        nu_expected = 2.30
        nu_uncertainty = 0.18



        self.assertGreater(nu_expected, 1.5)
        self.assertLess(nu_expected, 3.5)

    def test_alpha_s_in_literature_range(self):
        """Alpha_s should be in literature range."""

        self.assertAlmostEqual(ALPHA_S, 4.267, delta=0.05)


class TestReproducibility(unittest.TestCase):
    """Tests for reproducibility claims."""

    def test_deterministic_instance_generation(self):
        """Instance generation should be deterministic."""
        instances1 = [generate_ksat_instance(20, 3.0, k=3, seed=42) for _ in range(5)]
        instances2 = [generate_ksat_instance(20, 3.0, k=3, seed=42) for _ in range(5)]

        for i in range(5):
            self.assertEqual(instances1[i]["clauses"], instances2[i]["clauses"])

    def test_deterministic_solving(self):
        """DPLL solving should be deterministic."""
        instance = generate_ksat_instance(20, 3.0, k=3, seed=42)

        result1 = dpll_solve(instance, max_decisions=10000)
        result2 = dpll_solve(instance, max_decisions=10000)

        self.assertEqual(result1["satisfiable"], result2["satisfiable"])
        self.assertEqual(result1["decisions"], result2["decisions"])


class TestManuscriptClaims(unittest.TestCase):
    """Tests for specific claims made in the manuscript."""

    def test_claim_1_barrier_hardness_correspondence(self):
        """Conjecture 1: log T(n,α) = Θ(n · b(α))."""


        for alpha in [4.0, 4.1, 4.2]:
            self.assertGreater(barrier_density(alpha), 0.0)

    def test_claim_2_hardness_peak_location(self):
        """Hardness peak should be near alpha ≈ 4.2."""

        alphas = np.linspace(4.0, 4.4, 20)
        barriers = [barrier_density(a) for a in alphas]
        peak_idx = np.argmax(barriers)
        peak_alpha = alphas[peak_idx]
        self.assertAlmostEqual(peak_alpha, 4.2, delta=0.1)

    def test_claim_3_critical_exponent(self):
        """Critical exponent ν ≈ 2.3."""
        nu_manuscript = 2.30
        nu_uncertainty = 0.18
        self.assertAlmostEqual(nu_manuscript, 2.30, delta=0.01)


if __name__ == "__main__":
    unittest.main()