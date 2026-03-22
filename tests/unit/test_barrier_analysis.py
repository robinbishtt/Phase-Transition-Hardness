"""
Unit tests for src/barrier_analysis.py

Tests barrier analysis functions including:
- Path barrier computation
- Theoretical barrier curves
- Barrier scaling data
- Barrier-hardness correlation
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.barrier_analysis import (
    path_barrier,
    theoretical_barrier_curve,
    barrier_scaling_data,
    run_barrier_scaling_sweep,
    barrier_hardness_correlation,
)
from src.instance_generator import generate_ksat_instance
from src.energy_model import ALPHA_D, ALPHA_S


class TestPathBarrier(unittest.TestCase):
    """Tests for path_barrier function."""

    def test_same_assignment_zero_barrier(self):
        """Barrier between same assignment should be zero."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        assignment = {i + 1: True for i in range(10)}
        result = path_barrier(instance, assignment, assignment, n_samples=10)
        self.assertEqual(result, 0.0)

    def test_returns_non_negative(self):
        """Barrier should be non-negative."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        assign1 = {i + 1: True for i in range(10)}
        assign2 = {i + 1: False for i in range(10)}
        result = path_barrier(instance, assign1, assign2, n_samples=10)
        self.assertGreaterEqual(result, 0.0)

    def test_returns_float(self):
        """Should return a float."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        assign1 = {i + 1: True for i in range(10)}
        assign2 = {i + 1: False for i in range(10)}
        result = path_barrier(instance, assign1, assign2, n_samples=10)
        self.assertIsInstance(result, float)

    def test_more_samples_more_stable(self):
        """More samples should give more stable estimate."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        assign1 = {i + 1: True for i in range(10)}
        assign2 = {i + 1: False for i in range(10)}
        result1 = path_barrier(instance, assign1, assign2, n_samples=10, seed=42)
        result2 = path_barrier(instance, assign1, assign2, n_samples=100, seed=42)

        self.assertGreaterEqual(result1, 0.0)
        self.assertGreaterEqual(result2, 0.0)


class TestTheoreticalBarrierCurve(unittest.TestCase):
    """Tests for theoretical_barrier_curve function."""

    def test_returns_array(self):
        """Should return a numpy array."""
        alphas = np.linspace(3.0, 5.0, 20)
        result = theoretical_barrier_curve(alphas, k=3)
        self.assertIsInstance(result, np.ndarray)

    def test_correct_length(self):
        """Output should have same length as input."""
        alphas = np.linspace(3.0, 5.0, 20)
        result = theoretical_barrier_curve(alphas, k=3)
        self.assertEqual(len(result), len(alphas))

    def test_zero_outside_hard_phase(self):
        """Barrier should be zero outside hard phase."""
        alphas = np.array([ALPHA_D - 0.2, ALPHA_S + 0.2])
        result = theoretical_barrier_curve(alphas, k=3)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)

    def test_positive_in_hard_phase(self):
        """Barrier should be positive in hard phase."""
        alphas = np.array([4.0, 4.1, 4.2])
        result = theoretical_barrier_curve(alphas, k=3)
        self.assertTrue(np.all(result > 0.0))

    def test_peak_near_4_2(self):
        """Barrier should peak near alpha=4.2."""
        alphas = np.linspace(ALPHA_D + 0.1, ALPHA_S - 0.1, 50)
        result = theoretical_barrier_curve(alphas, k=3)
        peak_idx = np.argmax(result)
        peak_alpha = alphas[peak_idx]
        self.assertAlmostEqual(peak_alpha, 4.2, delta=0.2)


class TestBarrierScalingData(unittest.TestCase):
    """Tests for barrier_scaling_data function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        ns = [10, 20, 30]
        result = barrier_scaling_data(ns, alpha=4.2, k=3)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        ns = [10, 20, 30]
        result = barrier_scaling_data(ns, alpha=4.2, k=3)
        required_keys = ["ns", "barriers", "b_alpha", "alpha"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_linear_scaling(self):
        """Barriers should scale linearly with n."""
        ns = [10, 20, 30, 40]
        result = barrier_scaling_data(ns, alpha=4.2, k=3)
        barriers = result["barriers"]

        for i in range(1, len(ns)):
            ratio = barriers[i] / barriers[0]
            expected_ratio = ns[i] / ns[0]
            self.assertAlmostEqual(ratio, expected_ratio, places=5)

    def test_b_alpha_matches(self):
        """b_alpha should match barrier density."""
        from src.energy_model import barrier_density
        ns = [10, 20, 30]
        alpha = 4.2
        result = barrier_scaling_data(ns, alpha=alpha, k=3)
        expected_b = barrier_density(alpha, k=3)
        self.assertAlmostEqual(result["b_alpha"], expected_b, places=10)


class TestRunBarrierScalingSweep(unittest.TestCase):
    """Tests for run_barrier_scaling_sweep function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        ns = [10, 20]
        alphas = np.linspace(3.5, 5.0, 10)
        result = run_barrier_scaling_sweep(ns, alphas, k=3, output_dir="/tmp/test_barrier")
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        ns = [10, 20]
        alphas = np.linspace(3.5, 5.0, 10)
        result = run_barrier_scaling_sweep(ns, alphas, k=3, output_dir="/tmp/test_barrier2")
        required_keys = ["alphas", "b_curve", "alpha_peak", "b_peak", "ns", "alpha_d", "alpha_s"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_peak_in_hard_phase(self):
        """Peak should be in hard phase."""
        ns = [10, 20]
        alphas = np.linspace(3.5, 5.0, 20)
        result = run_barrier_scaling_sweep(ns, alphas, k=3, output_dir="/tmp/test_barrier3")
        self.assertGreater(result["alpha_peak"], ALPHA_D)
        self.assertLess(result["alpha_peak"], ALPHA_S)

    def test_b_peak_positive(self):
        """Peak barrier density should be positive."""
        ns = [10, 20]
        alphas = np.linspace(3.5, 5.0, 20)
        result = run_barrier_scaling_sweep(ns, alphas, k=3, output_dir="/tmp/test_barrier4")
        self.assertGreater(result["b_peak"], 0.0)


class TestBarrierHardnessCorrelation(unittest.TestCase):
    """Tests for barrier_hardness_correlation function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        alphas = np.linspace(3.5, 5.0, 20)
        gamma_mean = np.random.rand(20) * 0.01
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        alphas = np.linspace(3.5, 5.0, 20)
        gamma_mean = np.random.rand(20) * 0.01
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)
        self.assertIn("correlation", result)
        self.assertIn("p_value", result)
        self.assertIn("b_curve", result)
        self.assertIn("gamma_mean", result)

    def test_correlation_in_range(self):
        """Correlation should be in [-1, 1]."""
        alphas = np.linspace(3.5, 5.0, 20)
        gamma_mean = np.random.rand(20) * 0.01
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)
        self.assertGreaterEqual(result["correlation"], -1.0)
        self.assertLessEqual(result["correlation"], 1.0)

    def test_p_value_in_range(self):
        """P-value should be in [0, 1]."""
        alphas = np.linspace(3.5, 5.0, 20)
        gamma_mean = np.random.rand(20) * 0.01
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_perfect_correlation(self):
        """Should detect perfect correlation."""
        alphas = np.linspace(ALPHA_D + 0.1, ALPHA_S - 0.1, 20)
        from src.energy_model import barrier_density
        b_curve = np.array([barrier_density(a) for a in alphas])

        gamma_mean = b_curve * 2.0
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)
        self.assertAlmostEqual(result["correlation"], 1.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_path_barrier_empty_diff(self):
        """Path barrier with no differing variables should be zero."""
        instance = generate_ksat_instance(5, 3.0, k=3, seed=42)
        assignment = {i + 1: True for i in range(5)}
        result = path_barrier(instance, assignment, assignment, n_samples=10)
        self.assertEqual(result, 0.0)

    def test_barrier_scaling_zero_ns(self):
        """Should handle empty ns list."""
        result = barrier_scaling_data([], alpha=4.2, k=3)
        self.assertEqual(len(result["ns"]), 0)
        self.assertEqual(len(result["barriers"]), 0)

    def test_correlation_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        alphas = np.array([4.0, 4.1])
        gamma_mean = np.array([0.01, 0.02])
        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)

        self.assertTrue(np.isnan(result["correlation"]))


if __name__ == "__main__":
    unittest.main()