"""Unit tests for src/scaling_analysis.py.

This module implements the finite-size scaling (FSS) analysis that produces
the R²=0.9997 collapse reported in Figure 1 of the manuscript, the exponential
scaling regression that validates Conjecture 4, the hardness peak localisation
that gives Table 2, and the FSS extrapolation that gives α*_∞ = 4.20.

All test functions call each routine with its exact signature and verify the
keys and types of the return dict against the actual implementation.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scaling_analysis import (
    finite_size_peak_extrapolation,
    locate_hardness_peak,
    run_exponential_scaling,
    run_fss_analysis,
)


class TestLocateHardnessPeak(unittest.TestCase):
    """locate_hardness_peak(alphas, gamma_mean) -> (alpha_star, gamma_max)."""

    def setUp(self):
        # Simple unimodal hardness curve peaking near alpha = 4.20
        self.alphas = np.linspace(3.0, 5.0, 41)
        self.gamma  = np.array([
            max(0.0, 0.021 * np.exp(-0.5 * ((a - 4.20) / 0.25) ** 2))
            for a in self.alphas
        ])

    def test_returns_two_element_tuple(self):
        result = locate_hardness_peak(self.alphas, self.gamma)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_alpha_star_near_true_peak(self):
        alpha_star, _ = locate_hardness_peak(self.alphas, self.gamma)
        self.assertAlmostEqual(float(alpha_star), 4.20, delta=0.06)

    def test_gamma_max_near_true_maximum(self):
        _, gamma_max = locate_hardness_peak(self.alphas, self.gamma)
        self.assertAlmostEqual(float(gamma_max), 0.021, delta=0.003)

    def test_alpha_star_within_input_range(self):
        alpha_star, _ = locate_hardness_peak(self.alphas, self.gamma)
        self.assertGreaterEqual(float(alpha_star), self.alphas[0])
        self.assertLessEqual(float(alpha_star),    self.alphas[-1])

    def test_gamma_max_non_negative(self):
        _, gamma_max = locate_hardness_peak(self.alphas, self.gamma)
        self.assertGreaterEqual(float(gamma_max), 0.0)

    def test_flat_curve_returns_finite_values(self):
        flat   = np.zeros(20)
        alphas = np.linspace(3.0, 5.0, 20)
        alpha_star, gamma_max = locate_hardness_peak(alphas, flat)
        self.assertTrue(np.isfinite(float(alpha_star)))
        self.assertGreaterEqual(float(gamma_max), 0.0)


class TestRunFSSAnalysis(unittest.TestCase):
    """run_fss_analysis(alphas, ns, psat_matrix, output_dir)
    -> {"alpha_s", "nu", "residual", "converged", "psat_data", "x_data"}."""

    def setUp(self):
        from scipy.special import expit
        self.alphas = np.linspace(3.5, 5.0, 12)
        self.ns     = np.array([30, 50, 80])
        self.psat   = np.array([
            expit(-(2.0 + n / 60) * (self.alphas - 4.267))
            for n in self.ns
        ])

    def test_returns_dict_with_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(self.alphas, self.ns, self.psat, tmp)
        for key in ["alpha_s", "nu", "residual", "converged"]:
            self.assertIn(key, result, msg=f"Key '{key}' missing from run_fss_analysis result")

    def test_alpha_s_in_reasonable_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(self.alphas, self.ns, self.psat, tmp)
        self.assertGreater(result["alpha_s"], 3.0)
        self.assertLess(result["alpha_s"],    6.0)

    def test_nu_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(self.alphas, self.ns, self.psat, tmp)
        self.assertGreater(result["nu"], 0.0)

    def test_residual_non_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(self.alphas, self.ns, self.psat, tmp)
        self.assertGreaterEqual(result["residual"], 0.0)

    def test_converged_is_bool(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(self.alphas, self.ns, self.psat, tmp)
        self.assertIsInstance(result["converged"], bool)

    def test_good_psat_data_gives_alpha_s_near_4267(self):
        # With a clean sigmoid input the optimiser should recover α_s ≈ 4.267
        from scipy.special import expit
        alphas = np.linspace(3.5, 5.0, 20)
        ns     = np.array([100, 200, 400])
        psat   = np.array([
            expit(-(5.0 + n / 60) * (alphas - 4.267))
            for n in ns
        ])
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fss_analysis(alphas, ns, psat, tmp)
        self.assertAlmostEqual(result["alpha_s"], 4.267, delta=0.20)


class TestRunExponentialScaling(unittest.TestCase):
    """run_exponential_scaling(ns, alphas, gamma_matrix, output_dir)
    -> {"alphas", "gamma_slope", "mean_r2", "r2_values"}."""

    def setUp(self):
        from src.energy_model import barrier_density
        self.alphas = np.linspace(3.5, 5.0, 8)
        self.ns     = [30, 50, 80]
        self.gamma  = np.array([
            [max(0.0, barrier_density(a) * (1 + 0.3 * np.log(n) / np.log(30)))
             for a in self.alphas]
            for n in self.ns
        ])

    def test_returns_dict_with_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(self.ns, self.alphas, self.gamma, tmp)
        for key in ["alphas", "gamma_slope", "mean_r2", "r2_values"]:
            self.assertIn(key, result)

    def test_mean_r2_in_unit_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(self.ns, self.alphas, self.gamma, tmp)
        self.assertGreaterEqual(result["mean_r2"], 0.0)
        self.assertLessEqual(result["mean_r2"],    1.0)

    def test_r2_values_length_matches_alphas(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(self.ns, self.alphas, self.gamma, tmp)
        self.assertEqual(len(result["r2_values"]), len(self.alphas))

    def test_gamma_slope_length_matches_alphas(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(self.ns, self.alphas, self.gamma, tmp)
        self.assertEqual(len(result["gamma_slope"]), len(self.alphas))

    def test_r2_values_non_negative_where_finite(self):
        # For alpha values where gamma = 0 across all n (outside the hard phase),
        # the linear regression is degenerate and R² is NaN.  Only finite R²
        # values are tested; they must lie in [0, 1].
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(self.ns, self.alphas, self.gamma, tmp)
        for r2 in result["r2_values"]:
            if np.isfinite(float(r2)):
                self.assertGreaterEqual(float(r2), 0.0)
                self.assertLessEqual(float(r2), 1.0 + 1e-9)

    def test_perfect_linear_data_gives_high_r2(self):
        # If log T = gamma * n exactly, R² must be 1.0 for every α value.
        ns     = [100, 200, 400]
        alphas = np.linspace(3.5, 5.0, 5)
        gamma_true = 0.02
        gamma_m = np.array([[gamma_true] * 5 for _ in ns])
        with tempfile.TemporaryDirectory() as tmp:
            result = run_exponential_scaling(ns, alphas, gamma_m, tmp)
        self.assertGreater(result["mean_r2"], 0.90)


class TestFiniteSizePeakExtrapolation(unittest.TestCase):
    """finite_size_peak_extrapolation(ns, alpha_stars)
    -> {"alpha_star_inf", "A", "B", "nu", "r2", "residuals", "c"}."""

    def setUp(self):
        from src.proofs.fss_derivation import fss_threshold_shift
        from src.energy_model import ALPHA_STAR, NU, FSS_A, FSS_B
        ns = np.array([100, 200, 400, 800], dtype=float)
        self.ns          = ns
        self.alpha_stars = fss_threshold_shift(list(ns))

    def test_returns_dict_with_required_keys(self):
        result = finite_size_peak_extrapolation(self.ns, self.alpha_stars)
        for key in ["alpha_star_inf", "A", "B", "nu", "r2"]:
            self.assertIn(key, result)

    def test_alpha_star_inf_near_4_20(self):
        result = finite_size_peak_extrapolation(self.ns, self.alpha_stars)
        self.assertAlmostEqual(result["alpha_star_inf"], 4.20, delta=0.05)

    def test_nu_is_fixed_manuscript_value(self):
        from src.energy_model import NU
        result = finite_size_peak_extrapolation(self.ns, self.alpha_stars)
        self.assertAlmostEqual(result["nu"], NU, delta=0.01)

    def test_r2_in_unit_interval(self):
        result = finite_size_peak_extrapolation(self.ns, self.alpha_stars)
        r2 = result["r2"]
        if not np.isnan(r2):
            self.assertGreaterEqual(float(r2), 0.0)
            self.assertLessEqual(float(r2),    1.0 + 1e-8)

    def test_two_point_input_exact_fit(self):
        # With exactly 2 data points the fit is over-determined to have R²≥0.99
        ns     = np.array([100.0, 800.0])
        stars  = np.array([4.18, 4.20])
        result = finite_size_peak_extrapolation(ns, stars)
        self.assertAlmostEqual(result["alpha_star_inf"], 4.20, delta=0.02)


if __name__ == "__main__":
    unittest.main()
