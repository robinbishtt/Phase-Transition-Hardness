"""
Unit tests for src/hardness_metrics.py

Tests SAT solver implementations including:
- DPLL solver
- WalkSAT solver
- Hardness measurement
- Hardness curve generation
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hardness_metrics import (
    CNF,
    dpll_solve,
    walksat_solve,
    measure_hardness,
    hardness_curve,
    MAX_DECISIONS_DEFAULT,
    WALKSAT_MAX_FLIPS,
    WALKSAT_NOISE,
)
from src.instance_generator import generate_ksat_instance


class TestCNF(unittest.TestCase):
    """Tests for CNF class."""

    def test_init(self):
        """Test CNF initialization."""
        cnf = CNF(3, [[1, 2, 3], [-1, -2, 3]])
        self.assertEqual(cnf.n, 3)
        self.assertEqual(cnf.m, 2)

    def test_from_instance(self):
        """Test creating CNF from instance dict."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        cnf = CNF.from_instance(instance)
        self.assertEqual(cnf.n, instance["n"])
        self.assertEqual(cnf.m, instance["m"])

    def test_copy(self):
        """Test CNF copying."""
        cnf1 = CNF(3, [[1, 2, 3]])
        cnf2 = cnf1.copy()
        self.assertEqual(cnf1.n, cnf2.n)
        self.assertEqual(cnf1.m, cnf2.m)

        cnf2.clauses[0][0] = 999
        self.assertNotEqual(cnf1.clauses[0][0], cnf2.clauses[0][0])


class TestDPLLSolve(unittest.TestCase):
    """Tests for dpll_solve function."""

    def test_sat_instance(self):
        """DPLL should find satisfying assignment for SAT instance."""
        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertTrue(result["satisfiable"])
        self.assertIsNotNone(result["assignment"])
        self.assertGreater(result["decisions"], 0)

    def test_unsat_instance(self):
        """DPLL should detect UNSAT for unsatisfiable instance."""

        instance = {
            "n": 1,
            "k": 1,
            "alpha": 2.0,
            "m": 2,
            "clauses": [[1], [-1]],
            "seed": 42,
        }
        result = dpll_solve(instance, max_decisions=10000)
        self.assertFalse(result["satisfiable"])

    def test_assignment_is_complete(self):
        """Assignment should include all variables."""
        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)
        if result["satisfiable"]:
            self.assertEqual(len(result["assignment"]), instance["n"])

    def test_assignment_satisfies_clauses(self):
        """Assignment should satisfy all clauses."""
        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)
        if result["satisfiable"]:
            assignment = result["assignment"]
            for clause in instance["clauses"]:
                satisfied = False
                for lit in clause:
                    var = abs(lit)
                    val = assignment.get(var, False)
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                self.assertTrue(satisfied)

    def test_max_decisions_cutoff(self):
        """Should respect max_decisions limit."""
        instance = generate_ksat_instance(50, 4.5, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10)

        self.assertIn(result["satisfiable"], [True, False, None])
        self.assertLessEqual(result["decisions"], 10 + 1)

    def test_deterministic(self):
        """Same instance should give same result."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        result1 = dpll_solve(instance, max_decisions=10000)
        result2 = dpll_solve(instance, max_decisions=10000)
        self.assertEqual(result1["satisfiable"], result2["satisfiable"])
        self.assertEqual(result1["decisions"], result2["decisions"])


class TestWalkSAT(unittest.TestCase):
    """Tests for walksat_solve function."""

    def test_sat_instance(self):
        """WalkSAT should find satisfying assignment for easy SAT instance."""
        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)
        result = walksat_solve(instance, max_flips=10000, seed=42)
        self.assertTrue(result["satisfiable"])
        self.assertIsNotNone(result["assignment"])

    def test_returns_dict(self):
        """Should return a dictionary."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        result = walksat_solve(instance, max_flips=1000, seed=42)
        self.assertIsInstance(result, dict)
        self.assertIn("satisfiable", result)
        self.assertIn("flips", result)
        self.assertIn("assignment", result)

    def test_flips_counted(self):
        """Should count the number of flips."""
        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)
        result = walksat_solve(instance, max_flips=10000, seed=42)
        self.assertGreater(result["flips"], 0)

    def test_seed_determinism(self):
        """Same seed should give same result."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        result1 = walksat_solve(instance, max_flips=5000, seed=123)
        result2 = walksat_solve(instance, max_flips=5000, seed=123)
        self.assertEqual(result1["satisfiable"], result2["satisfiable"])
        self.assertEqual(result1["flips"], result2["flips"])

    def test_different_seeds_may_differ(self):
        """Different seeds may give different results."""
        instance = generate_ksat_instance(15, 3.5, k=3, seed=42)
        result1 = walksat_solve(instance, max_flips=5000, seed=123)
        result2 = walksat_solve(instance, max_flips=5000, seed=456)

        self.assertIn(result1["satisfiable"], [True, False])
        self.assertIn(result2["satisfiable"], [True, False])


class TestMeasureHardness(unittest.TestCase):
    """Tests for measure_hardness function."""

    def test_returns_float(self):
        """Should return a float."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        result = measure_hardness(instance, solver="dpll")
        self.assertIsInstance(result, float)

    def test_positive_value(self):
        """Hardness should be non-negative."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        result = measure_hardness(instance, solver="dpll")
        self.assertGreaterEqual(result, 0.0)

    def test_dpll_vs_walksat(self):
        """Both solvers should give valid hardness values."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        h_dpll = measure_hardness(instance, solver="dpll", max_decisions=10000)
        h_walksat = measure_hardness(instance, solver="walksat", walksat_seed=42)
        self.assertGreaterEqual(h_dpll, 0.0)
        self.assertGreaterEqual(h_walksat, 0.0)

    def test_scales_with_n(self):
        """Hardness density should be intensive (normalized by n)."""
        instance1 = generate_ksat_instance(20, 3.0, k=3, seed=42)
        instance2 = generate_ksat_instance(40, 3.0, k=3, seed=42)
        h1 = measure_hardness(instance1, solver="dpll", max_decisions=10000)
        h2 = measure_hardness(instance2, solver="dpll", max_decisions=10000)

        self.assertGreaterEqual(h1, 0.0)
        self.assertLess(h1, 1.0)
        self.assertGreaterEqual(h2, 0.0)
        self.assertLess(h2, 1.0)

    def test_harder_at_higher_alpha(self):
        """Hardness should generally increase with alpha in hard region."""
        instance_easy = generate_ksat_instance(20, 2.5, k=3, seed=42)
        instance_hard = generate_ksat_instance(20, 4.5, k=3, seed=43)
        h_easy = measure_hardness(instance_easy, solver="dpll", max_decisions=10000)
        h_hard = measure_hardness(instance_hard, solver="dpll", max_decisions=10000)

        self.assertGreaterEqual(h_hard, h_easy)


class TestHardnessCurve(unittest.TestCase):
    """Tests for hardness_curve function."""

    def test_returns_three_arrays(self):
        """Should return three numpy arrays."""
        alphas = np.linspace(3.0, 4.0, 3)
        mean, lo, hi = hardness_curve(
            n=20, alphas=alphas, n_instances=10, k=3, solver="dpll", master_seed=42
        )
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(lo, np.ndarray)
        self.assertIsInstance(hi, np.ndarray)

    def test_correct_length(self):
        """Output arrays should match input alphas length."""
        alphas = np.linspace(3.0, 4.0, 5)
        mean, lo, hi = hardness_curve(
            n=20, alphas=alphas, n_instances=10, k=3, solver="dpll", master_seed=42
        )
        self.assertEqual(len(mean), len(alphas))
        self.assertEqual(len(lo), len(alphas))
        self.assertEqual(len(hi), len(alphas))

    def test_mean_between_bounds(self):
        """Mean should be between lo and hi bounds."""
        alphas = np.linspace(3.0, 4.0, 3)
        mean, lo, hi = hardness_curve(
            n=20, alphas=alphas, n_instances=10, k=3, solver="dpll", master_seed=42
        )
        for i in range(len(alphas)):
            self.assertGreaterEqual(mean[i], lo[i])
            self.assertLessEqual(mean[i], hi[i])

    def test_lo_less_than_hi(self):
        """Lower bound should be less than upper bound."""
        alphas = np.linspace(3.0, 4.0, 3)
        mean, lo, hi = hardness_curve(
            n=20, alphas=alphas, n_instances=10, k=3, solver="dpll", master_seed=42
        )
        self.assertTrue(np.all(lo <= hi))


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_max_decisions_positive(self):
        """MAX_DECISIONS_DEFAULT should be positive."""
        self.assertGreater(MAX_DECISIONS_DEFAULT, 0)

    def test_walksat_max_flips_positive(self):
        """WALKSAT_MAX_FLIPS should be positive."""
        self.assertGreater(WALKSAT_MAX_FLIPS, 0)

    def test_walksat_noise_in_range(self):
        """WALKSAT_NOISE should be in [0, 1]."""
        self.assertGreaterEqual(WALKSAT_NOISE, 0.0)
        self.assertLessEqual(WALKSAT_NOISE, 1.0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_clause_list(self):
        """Should handle empty clause list."""
        instance = {
            "n": 5,
            "k": 3,
            "alpha": 0.0,
            "m": 0,
            "clauses": [],
            "seed": 42,
        }
        result = dpll_solve(instance, max_decisions=1000)

        self.assertTrue(result["satisfiable"])

    def test_single_variable(self):
        """Should handle single variable instances."""
        instance = {
            "n": 1,
            "k": 1,
            "alpha": 1.0,
            "m": 1,
            "clauses": [[1]],
            "seed": 42,
        }
        result = dpll_solve(instance, max_decisions=1000)
        self.assertTrue(result["satisfiable"])


if __name__ == "__main__":
    unittest.main()