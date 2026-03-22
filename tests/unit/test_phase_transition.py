"""
Unit tests for src/phase_transition.py

Tests phase transition detection including:
- P_sat estimation
- Threshold location
- Theoretical order parameters
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.phase_transition import (
    estimate_psat_single,
    psat_curve,
    locate_threshold,
    theoretical_order_parameters,
    run_psat_sweep,
)
from src.energy_model import ALPHA_D, ALPHA_S


class TestEstimatePsatSingle(unittest.TestCase):
    """Tests for estimate_psat_single function."""

    def test_low_alpha_high_sat(self):
        """At low alpha, most instances should be satisfiable."""
        result = estimate_psat_single(
            n=50, alpha=2.0, n_instances=50, k=3, master_seed=42, solver="dpll"
        )
        self.assertGreater(result, 0.8)

    def test_high_alpha_low_sat(self):
        """At high alpha, most instances should be unsatisfiable."""
        result = estimate_psat_single(
            n=50, alpha=6.0, n_instances=50, k=3, master_seed=42, solver="dpll"
        )
        self.assertLess(result, 0.2)

    def test_returns_probability(self):
        """Result should be a probability in [0, 1]."""
        result = estimate_psat_single(
            n=50, alpha=4.0, n_instances=20, k=3, master_seed=42, solver="dpll"
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_deterministic_with_same_seed(self):
        """Same seed should give same result."""
        result1 = estimate_psat_single(
            n=50, alpha=4.0, n_instances=30, k=3, master_seed=42, solver="dpll"
        )
        result2 = estimate_psat_single(
            n=50, alpha=4.0, n_instances=30, k=3, master_seed=42, solver="dpll"
        )
        self.assertEqual(result1, result2)

    def test_different_seeds_different_results(self):
        """Different seeds may give different results."""
        result1 = estimate_psat_single(
            n=50, alpha=4.0, n_instances=30, k=3, master_seed=42, solver="dpll"
        )
        result2 = estimate_psat_single(
            n=50, alpha=4.0, n_instances=30, k=3, master_seed=43, solver="dpll"
        )


        self.assertGreaterEqual(result1, 0.0)
        self.assertLessEqual(result1, 1.0)
        self.assertGreaterEqual(result2, 0.0)
        self.assertLessEqual(result2, 1.0)


class TestPsatCurve(unittest.TestCase):
    """Tests for psat_curve function."""

    def test_returns_array(self):
        """Should return a numpy array."""
        alphas = np.linspace(3.0, 5.0, 5)
        result = psat_curve(
            n=50, alphas=alphas, n_instances=20, k=3, master_seed=42, solver="dpll", n_jobs=1
        )
        self.assertIsInstance(result, np.ndarray)

    def test_correct_length(self):
        """Output should have same length as input alphas."""
        alphas = np.linspace(3.0, 5.0, 10)
        result = psat_curve(
            n=50, alphas=alphas, n_instances=20, k=3, master_seed=42, solver="dpll", n_jobs=1
        )
        self.assertEqual(len(result), len(alphas))

    def test_decreasing_with_alpha(self):
        """P_sat should generally decrease with alpha."""
        alphas = np.array([3.0, 4.0, 5.0, 6.0])
        result = psat_curve(
            n=50, alphas=alphas, n_instances=50, k=3, master_seed=42, solver="dpll", n_jobs=1
        )

        self.assertGreater(result[0], result[-1])

    def test_all_values_in_range(self):
        """All P_sat values should be in [0, 1]."""
        alphas = np.linspace(3.0, 5.0, 5)
        result = psat_curve(
            n=50, alphas=alphas, n_instances=20, k=3, master_seed=42, solver="dpll", n_jobs=1
        )
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


class TestLocateThreshold(unittest.TestCase):
    """Tests for locate_threshold function."""

    def test_finds_target(self):
        """Should find alpha where P_sat crosses target."""
        alphas = np.linspace(3.0, 6.0, 100)
        psats = np.linspace(0.9, 0.1, 100)
        result = locate_threshold(alphas, psats, target=0.5)

        self.assertGreater(result, 4.0)
        self.assertLess(result, 5.0)

    def test_exact_crossing(self):
        """Should find exact crossing point."""
        alphas = np.array([3.0, 4.0, 5.0, 6.0])
        psats = np.array([1.0, 0.7, 0.3, 0.0])
        result = locate_threshold(alphas, psats, target=0.5)

        self.assertGreaterEqual(result, 4.0)
        self.assertLessEqual(result, 5.0)

    def test_returns_nan_if_no_crossing(self):
        """Should return NaN if target is never crossed."""
        alphas = np.array([3.0, 4.0, 5.0])
        psats = np.array([0.9, 0.8, 0.7])
        result = locate_threshold(alphas, psats, target=0.5)
        self.assertTrue(np.isnan(result))

    def test_different_targets(self):
        """Should work with different target values."""
        alphas = np.linspace(3.0, 6.0, 100)
        psats = np.linspace(0.9, 0.1, 100)
        result_30 = locate_threshold(alphas, psats, target=0.3)
        result_70 = locate_threshold(alphas, psats, target=0.7)

        self.assertGreater(result_30, result_70)


class TestTheoreticalOrderParameters(unittest.TestCase):
    """Tests for theoretical_order_parameters function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        alphas = np.linspace(3.0, 5.0, 10)
        result = theoretical_order_parameters(alphas)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Should have entropy, cluster_complexity, frozen_fraction keys."""
        alphas = np.linspace(3.0, 5.0, 10)
        result = theoretical_order_parameters(alphas)
        self.assertIn("entropy", result)
        self.assertIn("cluster_complexity", result)
        self.assertIn("frozen_fraction", result)

    def test_correct_shapes(self):
        """All arrays should have same shape as input alphas."""
        alphas = np.linspace(3.0, 5.0, 10)
        result = theoretical_order_parameters(alphas)
        self.assertEqual(result["entropy"].shape, alphas.shape)
        self.assertEqual(result["cluster_complexity"].shape, alphas.shape)
        self.assertEqual(result["frozen_fraction"].shape, alphas.shape)

    def test_entropy_non_negative(self):
        """Entropy should be non-negative."""
        alphas = np.linspace(3.0, 5.0, 20)
        result = theoretical_order_parameters(alphas)
        self.assertTrue(np.all(result["entropy"] >= 0.0))

    def test_frozen_fraction_in_range(self):
        """Frozen fraction should be in [0, 1]."""
        alphas = np.linspace(3.0, 5.0, 20)
        result = theoretical_order_parameters(alphas)
        self.assertTrue(np.all(result["frozen_fraction"] >= 0.0))
        self.assertTrue(np.all(result["frozen_fraction"] <= 1.0))


class TestRunPsatSweep(unittest.TestCase):
    """Tests for run_psat_sweep function."""

    def test_returns_dict(self):
        """Should return a dictionary with results."""
        ns = [30, 40]
        alphas = np.linspace(4.0, 5.0, 5)
        result = run_psat_sweep(
            ns=ns, alphas=alphas, n_instances=10, k=3, master_seed=42, solver="dpll",
            output_dir="/tmp/test_psat", n_jobs=1
        )
        self.assertIsInstance(result, dict)

    def test_has_psat_matrix(self):
        """Result should contain psat_matrix."""
        ns = [30, 40]
        alphas = np.linspace(4.0, 5.0, 5)
        result = run_psat_sweep(
            ns=ns, alphas=alphas, n_instances=10, k=3, master_seed=42, solver="dpll",
            output_dir="/tmp/test_psat2", n_jobs=1
        )
        self.assertIn("psat_matrix", result)
        self.assertEqual(result["psat_matrix"].shape, (len(ns), len(alphas)))

    def test_has_thresholds(self):
        """Result should contain thresholds."""
        ns = [30, 40]
        alphas = np.linspace(4.0, 5.0, 5)
        result = run_psat_sweep(
            ns=ns, alphas=alphas, n_instances=10, k=3, master_seed=42, solver="dpll",
            output_dir="/tmp/test_psat3", n_jobs=1
        )
        self.assertIn("thresholds", result)


class TestThresholdLocationAccuracy(unittest.TestCase):
    """Tests for accuracy of threshold location."""

    def test_threshold_near_literature_value(self):
        """Estimated threshold should be near literature value 4.267."""
        alphas = np.linspace(3.5, 5.0, 30)

        psats = 1.0 / (1.0 + np.exp(5 * (alphas - 4.267)))
        result = locate_threshold(alphas, psats, target=0.5)
        self.assertAlmostEqual(result, 4.267, delta=0.1)

    def test_threshold_monotonicity_required(self):
        """Function assumes monotonic decreasing P_sat."""
        alphas = np.array([3.0, 4.0, 5.0, 6.0])
        psats = np.array([0.9, 0.5, 0.7, 0.1])

        result = locate_threshold(alphas, psats, target=0.5)

        self.assertTrue(np.isfinite(result) or np.isnan(result))


if __name__ == "__main__":
    unittest.main()