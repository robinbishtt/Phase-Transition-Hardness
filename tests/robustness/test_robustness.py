"""
Robustness tests for stability under perturbations

Tests to ensure the system remains stable under various perturbations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance
from src.hardness_metrics import dpll_solve, walksat_solve, measure_hardness
from src.energy_model import (
    rs_entropy_density,
    cluster_complexity,
    barrier_density,
    ALPHA_D,
    ALPHA_S,
)


class TestSolverRobustness(unittest.TestCase):
    """Robustness tests for solvers."""

    def test_dpll_with_tight_decision_limit(self):
        """DPLL should handle tight decision limits gracefully."""
        instance = generate_ksat_instance(30, 4.5, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=100)

        self.assertIn(result["satisfiable"], [True, False, None])
        self.assertLessEqual(result["decisions"], 100 + 10)

    def test_dpll_with_generous_decision_limit(self):
        """DPLL should solve with generous decision limits."""
        instance = generate_ksat_instance(20, 3.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=100000)

        self.assertIn(result["satisfiable"], [True, False])

    def test_walksat_with_tight_flip_limit(self):
        """WalkSAT should handle tight flip limits."""
        instance = generate_ksat_instance(30, 4.5, k=3, seed=42)
        result = walksat_solve(instance, max_flips=100, noise=0.57, seed=42)

        self.assertIn(result["satisfiable"], [True, False])

    def test_walksat_extreme_noise(self):
        """WalkSAT should handle extreme noise values."""
        instance = generate_ksat_instance(20, 3.0, k=3, seed=42)

        for noise in [0.0, 0.99]:
            result = walksat_solve(instance, max_flips=5000, noise=noise, seed=42)
            self.assertIn(result["satisfiable"], [True, False])


class TestInstanceGenerationRobustness(unittest.TestCase):
    """Robustness tests for instance generation."""

    def test_very_low_alpha(self):
        """Should handle very low alpha values."""
        instance = generate_ksat_instance(100, 0.1, k=3, seed=42)
        self.assertEqual(instance["m"], 10)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertIn(result["satisfiable"], [True, False, None])

    def test_very_high_alpha(self):
        """Should handle very high alpha values."""
        instance = generate_ksat_instance(50, 10.0, k=3, seed=42)
        self.assertEqual(instance["m"], 500)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertIn(result["satisfiable"], [True, False, None])

    def test_large_n_small_alpha(self):
        """Should handle large n with small alpha."""
        instance = generate_ksat_instance(200, 1.0, k=3, seed=42)
        self.assertEqual(instance["m"], 200)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertIn(result["satisfiable"], [True, False, None])

    def test_boundary_k_equals_n(self):
        """Should handle k close to n."""
        instance = generate_ksat_instance(10, 2.0, k=9, seed=42)
        self.assertEqual(instance["k"], 9)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertIn(result["satisfiable"], [True, False, None])


class TestEnergyModelRobustness(unittest.TestCase):
    """Robustness tests for energy model."""

    def test_entropy_extreme_alpha(self):
        """Entropy functions should handle extreme alpha values."""
        for alpha in [0.0, 0.1, 1.0, 5.0, 10.0, 100.0]:
            result = rs_entropy_density(alpha, k=3)

            self.assertTrue(np.isfinite(result))

            self.assertGreaterEqual(result, 0.0)

    def test_complexity_boundary_values(self):
        """Complexity should handle boundary values."""

        self.assertEqual(cluster_complexity(ALPHA_D), 0.0)
        self.assertEqual(cluster_complexity(ALPHA_S), 0.0)


        self.assertGreaterEqual(cluster_complexity(ALPHA_D + 0.01), 0.0)
        self.assertGreaterEqual(cluster_complexity(ALPHA_S - 0.01), 0.0)

    def test_barrier_extreme_alpha(self):
        """Barrier density should handle extreme alpha values."""
        for alpha in [0.0, 1.0, 3.0, 5.0, 10.0]:
            result = barrier_density(alpha, k=3)

            self.assertTrue(np.isfinite(result))

            self.assertGreaterEqual(result, 0.0)


class TestHardnessMeasurementRobustness(unittest.TestCase):
    """Robustness tests for hardness measurement."""

    def test_hardness_various_instances(self):
        """Hardness measurement should work for various instance types."""
        test_cases = [
            (20, 2.0),
            (20, 3.0),
            (20, 4.0),
        ]

        for n, alpha in test_cases:
            instance = generate_ksat_instance(n, alpha, k=3, seed=42)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            self.assertGreaterEqual(h, 0.0)
            self.assertTrue(np.isfinite(h))

    def test_hardness_consistency(self):
        """Hardness should be consistent for same instance."""
        instance = generate_ksat_instance(20, 3.0, k=3, seed=42)
        h1 = measure_hardness(instance, solver="dpll", max_decisions=10000)
        h2 = measure_hardness(instance, solver="dpll", max_decisions=10000)
        self.assertAlmostEqual(h1, h2, places=10)


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def test_no_nan_in_entropy(self):
        """Entropy functions should not produce NaN."""
        alphas = np.linspace(0, 10, 100)
        for alpha in alphas:
            result = rs_entropy_density(alpha, k=3)
            self.assertFalse(np.isnan(result))

    def test_no_inf_in_barrier(self):
        """Barrier functions should not produce inf."""
        alphas = np.linspace(0, 10, 100)
        for alpha in alphas:
            result = barrier_density(alpha, k=3)
            self.assertFalse(np.isinf(result))

    def test_finite_hardness_values(self):
        """Hardness values should always be finite."""
        for _ in range(10):
            instance = generate_ksat_instance(20, 4.0, k=3, seed=None)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            self.assertTrue(np.isfinite(h))


if __name__ == "__main__":
    unittest.main()