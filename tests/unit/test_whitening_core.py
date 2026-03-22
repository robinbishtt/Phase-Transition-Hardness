"""Unit tests for src/whitening_core.py.

The whitening core of a K-SAT instance is the set of variables that survive
all rounds of iterative unit propagation (leaf removal / peeling).  At
densities below the rigidity threshold α_r ≈ 3.86 the core is empty with
high probability; above it a positive fraction of variables persists.  The
manuscript documents this transition in Supplementary Section 6.4.

All tests call functions with their exact signatures and check return dicts
against the keys that the source code actually produces.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance
from src.whitening_core import (
    analyze_whitening_transition,
    compute_clause_whitening_levels,
    compute_core_fraction,
    compute_core_size,
    compute_residual_formula,
    compute_whitening_core,
    compute_whitening_distribution,
    estimate_core_size_distribution,
    is_in_whitening_core,
)


class TestComputeWhiteningCore(unittest.TestCase):

    def setUp(self):
        self.inst_easy = generate_ksat_instance(20, 2.5, k=3, seed=42)
        self.inst_hard = generate_ksat_instance(20, 4.20, k=3, seed=42)

    def test_returns_set(self):
        self.assertIsInstance(compute_whitening_core(self.inst_easy), set)

    def test_core_is_subset_of_variables(self):
        core = compute_whitening_core(self.inst_easy)
        all_vars = set(range(1, self.inst_easy["n"] + 1))
        self.assertTrue(core.issubset(all_vars))

    def test_core_size_bounded(self):
        for inst in [self.inst_easy, self.inst_hard]:
            core = compute_whitening_core(inst)
            self.assertGreaterEqual(len(core), 0)
            self.assertLessEqual(len(core), inst["n"])

    def test_deterministic(self):
        inst = generate_ksat_instance(15, 4.0, k=3, seed=7)
        self.assertEqual(compute_whitening_core(inst), compute_whitening_core(inst))

    def test_empty_clause_list_gives_empty_core(self):
        inst = {"n": 5, "m": 0, "k": 3, "alpha": 0.0, "seed": 0, "clauses": []}
        self.assertEqual(len(compute_whitening_core(inst)), 0)


class TestComputeCoreSize(unittest.TestCase):

    def test_returns_non_negative_int(self):
        inst = generate_ksat_instance(20, 3.0, k=3, seed=42)
        size = compute_core_size(inst)
        self.assertIsInstance(size, int)
        self.assertGreaterEqual(size, 0)

    def test_at_most_n(self):
        inst = generate_ksat_instance(20, 4.0, k=3, seed=42)
        self.assertLessEqual(compute_core_size(inst), inst["n"])

    def test_consistent_with_whitening_core(self):
        inst = generate_ksat_instance(15, 4.0, k=3, seed=42)
        self.assertEqual(compute_core_size(inst), len(compute_whitening_core(inst)))


class TestComputeCoreFraction(unittest.TestCase):

    def test_float_in_unit_interval(self):
        inst = generate_ksat_instance(20, 3.5, k=3, seed=42)
        frac = compute_core_fraction(inst)
        self.assertIsInstance(frac, float)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_equals_core_size_over_n(self):
        inst = generate_ksat_instance(20, 4.0, k=3, seed=42)
        self.assertAlmostEqual(
            compute_core_fraction(inst),
            compute_core_size(inst) / inst["n"],
            places=10,
        )


class TestIsInWhiteningCore(unittest.TestCase):

    def test_returns_bool(self):
        inst = generate_ksat_instance(15, 4.0, k=3, seed=42)
        self.assertIsInstance(is_in_whitening_core(inst, 1), bool)

    def test_consistent_with_core_set(self):
        inst = generate_ksat_instance(15, 4.0, k=3, seed=42)
        core = compute_whitening_core(inst)
        for vi in range(1, inst["n"] + 1):
            self.assertEqual(is_in_whitening_core(inst, vi), vi in core)


class TestComputeClauseWhiteningLevels(unittest.TestCase):

    def test_returns_dict_keyed_by_clause_index(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        levels = compute_clause_whitening_levels(inst)
        self.assertIsInstance(levels, dict)
        for ci in range(inst["m"]):
            self.assertIn(ci, levels)

    def test_level_values_are_integers(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        for level in compute_clause_whitening_levels(inst).values():
            self.assertIsInstance(level, int)

    def test_core_clauses_use_level_minus_one(self):
        # Clauses that survive all peeling rounds (permanently in the core) get
        # level = -1.  All other levels must be >= 0.
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        for level in compute_clause_whitening_levels(inst).values():
            self.assertGreaterEqual(level, -1)


class TestComputeWhiteningDistribution(unittest.TestCase):

    def test_returns_dict(self):
        self.assertIsInstance(
            compute_whitening_distribution(generate_ksat_instance(20, 3.5, k=3, seed=42)),
            dict,
        )

    def test_total_counts_equal_m(self):
        inst = generate_ksat_instance(20, 3.5, k=3, seed=42)
        self.assertEqual(sum(compute_whitening_distribution(inst).values()), inst["m"])

    def test_all_counts_non_negative(self):
        inst = generate_ksat_instance(20, 3.5, k=3, seed=42)
        for count in compute_whitening_distribution(inst).values():
            self.assertGreaterEqual(count, 0)


class TestEstimateCoreSizeDistribution(unittest.TestCase):
    """Return keys: core_sizes, core_fractions, mean_size, std_size,
    mean_fraction, std_fraction, n, alpha, k."""

    def setUp(self):
        self.result = estimate_core_size_distribution(
            n=15, alpha=4.0, n_instances=10, k=3, seed=42
        )

    def test_required_keys(self):
        for key in ["core_sizes", "core_fractions", "mean_size", "std_size",
                    "mean_fraction", "std_fraction", "n", "alpha", "k"]:
            self.assertIn(key, self.result)

    def test_mean_fraction_in_unit_interval(self):
        self.assertGreaterEqual(self.result["mean_fraction"], 0.0)
        self.assertLessEqual(self.result["mean_fraction"],   1.0)

    def test_std_fraction_non_negative(self):
        self.assertGreaterEqual(self.result["std_fraction"], 0.0)

    def test_mean_fraction_consistent_with_mean_size(self):
        self.assertAlmostEqual(
            self.result["mean_fraction"],
            self.result["mean_size"] / self.result["n"],
            delta=1e-6,
        )

    def test_core_fractions_length(self):
        self.assertEqual(len(self.result["core_fractions"]), 10)

    def test_metadata_preserved(self):
        self.assertEqual(self.result["n"], 15)
        self.assertEqual(self.result["alpha"], 4.0)
        self.assertEqual(self.result["k"], 3)


class TestAnalyzeWhiteningTransition(unittest.TestCase):
    """Return keys: alphas, mean_fractions, std_fractions, n, k."""

    def setUp(self):
        self.alphas = [3.0, 3.5, 4.0]
        self.result = analyze_whitening_transition(
            n=15, alphas=self.alphas, n_instances=5, k=3, seed=42
        )

    def test_required_keys(self):
        for key in ["alphas", "mean_fractions", "std_fractions", "n", "k"]:
            self.assertIn(key, self.result)

    def test_array_lengths_match(self):
        self.assertEqual(len(self.result["alphas"]),         3)
        self.assertEqual(len(self.result["mean_fractions"]), 3)
        self.assertEqual(len(self.result["std_fractions"]),  3)

    def test_mean_fractions_in_unit_interval(self):
        for frac in self.result["mean_fractions"]:
            self.assertGreaterEqual(frac, 0.0)
            self.assertLessEqual(frac,   1.0)

    def test_std_fractions_non_negative(self):
        for std in self.result["std_fractions"]:
            self.assertGreaterEqual(std, 0.0)


class TestComputeResidualFormula(unittest.TestCase):
    """Return keys: n, m, clauses, vars."""

    def setUp(self):
        self.inst   = generate_ksat_instance(15, 3.5, k=3, seed=42)
        self.result = compute_residual_formula(self.inst)

    def test_required_keys(self):
        for key in ["n", "m", "clauses", "vars"]:
            self.assertIn(key, self.result)

    def test_residual_n_leq_original(self):
        self.assertLessEqual(self.result["n"], self.inst["n"])

    def test_residual_m_leq_original(self):
        self.assertLessEqual(self.result["m"], self.inst["m"])

    def test_m_matches_clauses_list(self):
        self.assertEqual(self.result["m"], len(self.result["clauses"]))

    def test_n_matches_vars_list(self):
        self.assertEqual(self.result["n"], len(self.result["vars"]))

    def test_vars_positive_integers(self):
        original = set(range(1, self.inst["n"] + 1))
        for vi in self.result["vars"]:
            self.assertIsInstance(vi, int)
            self.assertIn(vi, original)


if __name__ == "__main__":
    unittest.main()
