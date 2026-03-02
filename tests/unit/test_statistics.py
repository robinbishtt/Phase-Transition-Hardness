"""
Unit tests for src/statistics.py

Tests statistical functions including:
- Bootstrap confidence intervals
- Lognormal fitting
- Exponential scaling fit
- Finite-size scaling collapse
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.statistics import (
    bootstrap_ci,
    lognormal_mean_ci,
    exponential_scaling_fit,
    fss_collapse,
    fss_scaling_function,
    fit_lognormal,
    fit_exponential_tail,
)


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap_ci function."""

    def test_returns_tuple(self):
        """Should return a tuple of two floats."""
        data = np.array([1, 2, 3, 4, 5])
        lo, hi = bootstrap_ci(data, n_boot=100, ci=0.95, seed=42)
        self.assertIsInstance(lo, float)
        self.assertIsInstance(hi, float)

    def test_lo_less_than_hi(self):
        """Lower bound should be less than upper bound."""
        data = np.array([1, 2, 3, 4, 5])
        lo, hi = bootstrap_ci(data, n_boot=100, ci=0.95, seed=42)
        self.assertLess(lo, hi)

    def test_mean_in_interval(self):
        """Sample mean should be within confidence interval."""
        data = np.array([1, 2, 3, 4, 5])
        lo, hi = bootstrap_ci(data, statistic=np.mean, n_boot=100, ci=0.95, seed=42)
        mean = np.mean(data)
        self.assertGreaterEqual(mean, lo)
        self.assertLessEqual(mean, hi)

    def test_median_in_interval(self):
        """Sample median should be within confidence interval."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lo, hi = bootstrap_ci(data, statistic=np.median, n_boot=100, ci=0.95, seed=42)
        median = np.median(data)
        self.assertGreaterEqual(median, lo)
        self.assertLessEqual(median, hi)

    def test_narrower_ci_with_more_data(self):
        """CI should be narrower with more data."""
        data_small = np.random.randn(20)
        data_large = np.random.randn(200)
        lo_small, hi_small = bootstrap_ci(data_small, n_boot=100, ci=0.95, seed=42)
        lo_large, hi_large = bootstrap_ci(data_large, n_boot=100, ci=0.95, seed=42)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large


        self.assertGreater(width_small, 0)
        self.assertGreater(width_large, 0)

    def test_deterministic_with_same_seed(self):
        """Same seed should give same result."""
        data = np.array([1, 2, 3, 4, 5])
        lo1, hi1 = bootstrap_ci(data, n_boot=100, ci=0.95, seed=42)
        lo2, hi2 = bootstrap_ci(data, n_boot=100, ci=0.95, seed=42)
        self.assertAlmostEqual(lo1, lo2, places=10)
        self.assertAlmostEqual(hi1, hi2, places=10)


class TestLognormalMeanCI(unittest.TestCase):
    """Tests for lognormal_mean_ci function."""

    def test_returns_three_values(self):
        """Should return three floats."""
        data = np.array([1, 2, 3, 4, 5])
        mean, lo, hi = lognormal_mean_ci(data, ci=0.95)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(lo, float)
        self.assertIsInstance(hi, float)

    def test_mean_positive(self):
        """Mean should be positive."""
        data = np.array([1, 2, 3, 4, 5])
        mean, _, _ = lognormal_mean_ci(data, ci=0.95)
        self.assertGreater(mean, 0.0)

    def test_lo_less_than_mean(self):
        """Lower bound should be less than mean."""
        data = np.array([1, 2, 3, 4, 5])
        mean, lo, _ = lognormal_mean_ci(data, ci=0.95)
        self.assertLess(lo, mean)

    def test_mean_less_than_hi(self):
        """Mean should be less than upper bound."""
        data = np.array([1, 2, 3, 4, 5])
        mean, _, hi = lognormal_mean_ci(data, ci=0.95)
        self.assertLess(mean, hi)

    def test_handles_zeros(self):
        """Should handle zeros in data."""
        data = np.array([0, 1, 2, 3, 4])
        mean, lo, hi = lognormal_mean_ci(data, ci=0.95)
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))


class TestExponentialScalingFit(unittest.TestCase):
    """Tests for exponential_scaling_fit function."""

    def test_perfect_fit(self):
        """Should give perfect fit for linear data."""
        ns = np.array([10, 20, 30, 40, 50])
        log_runtimes = 0.1 * ns + 2.0
        result = exponential_scaling_fit(ns, log_runtimes)
        self.assertAlmostEqual(result["gamma"], 0.1, places=5)
        self.assertAlmostEqual(result["intercept"], 2.0, places=5)
        self.assertAlmostEqual(result["r2"], 1.0, places=5)

    def test_returns_dict(self):
        """Should return a dictionary."""
        ns = np.array([10, 20, 30])
        log_runtimes = np.array([1.0, 2.0, 3.0])
        result = exponential_scaling_fit(ns, log_runtimes)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        ns = np.array([10, 20, 30])
        log_runtimes = np.array([1.0, 2.0, 3.0])
        result = exponential_scaling_fit(ns, log_runtimes)
        required_keys = ["gamma", "intercept", "r2", "p_value", "stderr", "residuals"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_gamma_positive(self):
        """Gamma should be positive for increasing data."""
        ns = np.array([10, 20, 30, 40, 50])
        log_runtimes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = exponential_scaling_fit(ns, log_runtimes)
        self.assertGreater(result["gamma"], 0.0)

    def test_r2_in_range(self):
        """R^2 should be in [0, 1]."""
        ns = np.array([10, 20, 30, 40, 50])
        log_runtimes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = exponential_scaling_fit(ns, log_runtimes)
        self.assertGreaterEqual(result["r2"], 0.0)
        self.assertLessEqual(result["r2"], 1.0)


class TestFSSScalingFunction(unittest.TestCase):
    """Tests for fss_scaling_function."""

    def test_returns_array(self):
        """Should return a numpy array."""
        x = np.array([1, 2, 3])
        result = fss_scaling_function(x, a=1.0, b=0.5, c=0.1)
        self.assertIsInstance(result, np.ndarray)

    def test_correct_length(self):
        """Output should have same length as input."""
        x = np.array([1, 2, 3, 4, 5])
        result = fss_scaling_function(x, a=1.0, b=0.5, c=0.1)
        self.assertEqual(len(result), len(x))

    def test_quadratic_behavior(self):
        """Should follow quadratic form."""
        x = np.array([0.0, 1.0, 2.0])
        result = fss_scaling_function(x, a=0.0, b=0.0, c=1.0)
        expected = np.array([0.0, 1.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_linear_behavior(self):
        """Should follow linear form when c=0."""
        x = np.array([0.0, 1.0, 2.0])
        result = fss_scaling_function(x, a=0.0, b=1.0, c=0.0)
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestFSSCollapse(unittest.TestCase):
    """Tests for fss_collapse function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        alphas = np.linspace(3.0, 5.0, 20)
        ns = [20, 30]

        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))
        result = fss_collapse(alphas, ns, psat_matrix, alpha_s_init=4.2, nu_init=2.5)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        alphas = np.linspace(3.0, 5.0, 20)
        ns = [20, 30]
        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))
        result = fss_collapse(alphas, ns, psat_matrix)
        required_keys = ["alpha_s", "nu", "residual", "x_data", "psat_data", "converged"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_nu_positive(self):
        """Estimated nu should be positive."""
        alphas = np.linspace(3.0, 5.0, 20)
        ns = [20, 30]
        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))
        result = fss_collapse(alphas, ns, psat_matrix)
        self.assertGreater(result["nu"], 0.0)

    def test_alpha_s_in_range(self):
        """Estimated alpha_s should be in reasonable range."""
        alphas = np.linspace(3.0, 5.0, 20)
        ns = [20, 30]
        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))
        result = fss_collapse(alphas, ns, psat_matrix)
        self.assertGreater(result["alpha_s"], 3.5)
        self.assertLess(result["alpha_s"], 5.0)

    def test_residual_non_negative(self):
        """Residual should be non-negative."""
        alphas = np.linspace(3.0, 5.0, 20)
        ns = [20, 30]
        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))
        result = fss_collapse(alphas, ns, psat_matrix)
        self.assertGreaterEqual(result["residual"], 0.0)


class TestFitLognormal(unittest.TestCase):
    """Tests for fit_lognormal function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        data = np.array([1, 2, 3, 4, 5])
        result = fit_lognormal(data)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        data = np.array([1, 2, 3, 4, 5])
        result = fit_lognormal(data)
        self.assertIn("mu", result)
        self.assertIn("sigma", result)
        self.assertIn("ks_stat", result)
        self.assertIn("ks_pvalue", result)

    def test_perfect_lognormal_data(self):
        """Should fit perfectly to lognormal data."""
        mu_true, sigma_true = 0.0, 1.0
        data = np.random.lognormal(mu_true, sigma_true, size=1000)
        result = fit_lognormal(data)
        self.assertAlmostEqual(result["mu"], mu_true, delta=0.2)
        self.assertAlmostEqual(result["sigma"], sigma_true, delta=0.2)


class TestFitExponentialTail(unittest.TestCase):
    """Tests for fit_exponential_tail function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        data = np.array([1, 2, 3, 4, 5])
        result = fit_exponential_tail(data, tail_quantile=0.5)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        data = np.array([1, 2, 3, 4, 5])
        result = fit_exponential_tail(data, tail_quantile=0.5)
        self.assertIn("lambda", result)
        self.assertIn("tail_threshold", result)
        self.assertIn("n_tail", result)

    def test_lambda_positive(self):
        """Lambda should be positive for exponential data."""
        data = np.random.exponential(scale=2.0, size=1000)
        result = fit_exponential_tail(data, tail_quantile=0.5)
        self.assertGreater(result["lambda"], 0.0)

    def test_n_tail_correct(self):
        """n_tail should be correct count."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = fit_exponential_tail(data, tail_quantile=0.7)

        self.assertEqual(result["n_tail"], 3)


if __name__ == "__main__":
    unittest.main()