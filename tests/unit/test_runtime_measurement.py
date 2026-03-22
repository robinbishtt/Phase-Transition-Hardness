"""Unit tests for src/runtime_measurement.py.

This module implements the three primary measurement functions used by the
manuscript experiments:

measure_runtime_distribution — per-(n, α) distribution of DPLL decisions,
    producing the hardness density γ = log(decisions+1)/n (Eq. 16).

alpha_sweep — sweeps across the entire α grid for multiple n values,
    producing the gamma_mean_matrix that feeds Conjecture 4 regression
    and the FSS peak extrapolation.

localise_hardness_peak — fine-resolution sweep around α* ≈ 4.20,
    producing the per-n peak locations for Table 2 extrapolation.

All tests use small instances (n ≤ 25, n_instances ≤ 30) to keep wall-clock
time under 5 seconds.  The exact return dict keys are verified against the
source implementation.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.runtime_measurement import (
    alpha_sweep,
    localise_hardness_peak,
    measure_runtime_distribution,
)


class TestMeasureRuntimeDistribution(unittest.TestCase):
    """measure_runtime_distribution(n, alpha, ...) ->
    {"decisions", "gamma", "gamma_mean", "gamma_lo", "gamma_hi",
     "log_mean", "n", "alpha", "n_instances"}."""

    def setUp(self):
        self.result = measure_runtime_distribution(
            n=20, alpha=4.0, n_instances=20, k=3,
            solver="dpll", master_seed=20240223, max_decisions=10000,
        )

    def test_required_keys_present(self):
        for key in ["decisions", "gamma", "gamma_mean", "gamma_lo",
                    "gamma_hi", "log_mean", "n", "alpha", "n_instances"]:
            self.assertIn(key, self.result,
                          msg=f"Key '{key}' missing from measure_runtime_distribution result")

    def test_decisions_length_matches_n_instances(self):
        self.assertEqual(len(self.result["decisions"]), 20)

    def test_gamma_length_matches_n_instances(self):
        self.assertEqual(len(self.result["gamma"]), 20)

    def test_gamma_values_non_negative(self):
        self.assertTrue(np.all(self.result["gamma"] >= 0.0))

    def test_gamma_mean_is_mean_of_gamma(self):
        self.assertAlmostEqual(
            self.result["gamma_mean"],
            float(np.mean(self.result["gamma"])),
            places=10,
        )

    def test_gamma_lo_leq_mean_leq_gamma_hi(self):
        self.assertLessEqual(self.result["gamma_lo"], self.result["gamma_mean"] + 1e-9)
        self.assertGreaterEqual(self.result["gamma_hi"], self.result["gamma_mean"] - 1e-9)

    def test_log_mean_finite(self):
        self.assertTrue(np.isfinite(self.result["log_mean"]))

    def test_metadata_preserved(self):
        self.assertEqual(self.result["n"],          20)
        self.assertAlmostEqual(self.result["alpha"], 4.0, places=10)
        self.assertEqual(self.result["n_instances"], 20)

    def test_decisions_are_non_negative(self):
        self.assertTrue(np.all(self.result["decisions"] >= 0))

    def test_deterministic_with_same_seed(self):
        r2 = measure_runtime_distribution(
            n=20, alpha=4.0, n_instances=20, k=3,
            solver="dpll", master_seed=20240223, max_decisions=10000,
        )
        np.testing.assert_array_equal(self.result["decisions"], r2["decisions"])

    def test_walksat_solver_runs(self):
        result = measure_runtime_distribution(
            n=15, alpha=3.5, n_instances=10, k=3,
            solver="walksat", master_seed=42, max_decisions=5000,
        )
        self.assertIn("decisions", result)
        self.assertEqual(len(result["decisions"]), 10)

    def test_gamma_consistent_with_decisions(self):
        n = self.result["n"]
        for dec, g in zip(self.result["decisions"], self.result["gamma"]):
            expected_g = float(np.log(dec + 1) / n)
            self.assertAlmostEqual(g, expected_g, places=10)


class TestAlphaSweep(unittest.TestCase):
    """alpha_sweep(ns, alphas, ...) ->
    {"alphas", "ns", "gamma_mean_matrix", "gamma_lo_matrix", "gamma_hi_matrix",
     "alpha_stars", "gamma_maxima", "alpha_star_inf", "extrap_r2"}."""

    def setUp(self):
        self.ns     = [20, 25]
        self.alphas = np.array([3.5, 4.0, 4.2, 4.5, 5.0])
        with tempfile.TemporaryDirectory() as tmp:
            self.result = alpha_sweep(
                ns=self.ns, alphas=self.alphas,
                n_instances=10, k=3, solver="dpll",
                master_seed=20240223, max_decisions=5000,
                output_dir=tmp,
            )

    def test_required_keys_present(self):
        for key in ["alphas", "ns", "gamma_mean_matrix", "gamma_lo_matrix",
                    "gamma_hi_matrix", "alpha_stars", "gamma_maxima",
                    "alpha_star_inf", "extrap_r2"]:
            self.assertIn(key, self.result)

    def test_gamma_mean_matrix_shape(self):
        self.assertEqual(
            self.result["gamma_mean_matrix"].shape,
            (len(self.ns), len(self.alphas)),
        )

    def test_gamma_lo_hi_matrix_shapes_match(self):
        shape = self.result["gamma_mean_matrix"].shape
        self.assertEqual(self.result["gamma_lo_matrix"].shape, shape)
        self.assertEqual(self.result["gamma_hi_matrix"].shape, shape)

    def test_alpha_stars_length_matches_ns(self):
        self.assertEqual(len(self.result["alpha_stars"]), len(self.ns))

    def test_gamma_maxima_length_matches_ns(self):
        self.assertEqual(len(self.result["gamma_maxima"]), len(self.ns))

    def test_gamma_mean_non_negative(self):
        self.assertTrue(np.all(self.result["gamma_mean_matrix"] >= 0.0))

    def test_alpha_stars_within_input_range(self):
        for a_star in self.result["alpha_stars"]:
            self.assertGreaterEqual(float(a_star), self.alphas[0])
            self.assertLessEqual(float(a_star),    self.alphas[-1])

    def test_gamma_maxima_non_negative(self):
        self.assertTrue(np.all(self.result["gamma_maxima"] >= 0.0))

    def test_alpha_star_inf_finite(self):
        self.assertTrue(np.isfinite(self.result["alpha_star_inf"]))

    def test_files_written_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            alpha_sweep(
                ns=[20], alphas=np.array([3.5, 4.0, 4.5]),
                n_instances=5, k=3, solver="dpll",
                master_seed=20240223, max_decisions=3000,
                output_dir=tmp,
            )
            import os
            self.assertTrue(os.path.exists(f"{tmp}/alpha_sweep.npz"))
            self.assertTrue(os.path.exists(f"{tmp}/alpha_sweep_summary.json"))


class TestLocaliseHardnessPeak(unittest.TestCase):
    """localise_hardness_peak(ns, alpha_center, width, n_points, ...) ->
    {"alpha_stars", "gamma_maxima", "alpha_star_inf", "alphas", "gamma_matrices"}."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.result = localise_hardness_peak(
                ns=[20, 25],
                alpha_center=4.20,
                width=0.30,
                n_points=5,
                n_instances=10,
                k=3,
                solver="dpll",
                master_seed=20240223,
                max_decisions=5000,
                output_dir=tmp,
            )

    def test_result_is_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_alpha_stars_length_matches_ns(self):
        self.assertIn("alpha_stars", self.result)
        self.assertEqual(len(self.result["alpha_stars"]), 2)

    def test_gamma_maxima_length_matches_ns(self):
        self.assertIn("gamma_maxima", self.result)
        self.assertEqual(len(self.result["gamma_maxima"]), 2)

    def test_alpha_star_inf_within_search_range(self):
        self.assertIn("alpha_star_inf", self.result)
        a_inf = float(self.result["alpha_star_inf"])
        self.assertGreaterEqual(a_inf, 4.20 - 0.30 - 0.10)
        self.assertLessEqual(a_inf,    4.20 + 0.30 + 0.10)

    def test_gamma_maxima_non_negative(self):
        for g in self.result["gamma_maxima"]:
            self.assertGreaterEqual(float(g), 0.0)

    def test_alpha_stars_within_search_range(self):
        lo = 4.20 - 0.30
        hi = 4.20 + 0.30
        for a_star in self.result["alpha_stars"]:
            self.assertGreaterEqual(float(a_star), lo - 0.05)
            self.assertLessEqual(float(a_star),    hi + 0.05)


if __name__ == "__main__":
    unittest.main()
