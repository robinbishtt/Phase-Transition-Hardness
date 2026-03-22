"""
Unit tests for src/instance_generator.py

Tests random k-SAT instance generation including:
- Instance structure validation
- Batch generation
- Clause properties
- Seed determinism
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import (
    generate_ksat_instance,
    generate_instance_batch,
    instance_to_adjacency,
    count_violated_clauses,
    is_satisfying,
)


class TestGenerateKSATInstance(unittest.TestCase):
    """Tests for generate_ksat_instance function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        """Result should have required keys."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        required_keys = ["n", "k", "alpha", "m", "clauses", "seed"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_correct_n(self):
        """n should match input."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(result["n"], 10)

    def test_correct_k(self):
        """k should match input."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(result["k"], 3)

    def test_correct_alpha(self):
        """alpha should match input."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(result["alpha"], 3.0)

    def test_correct_m(self):
        """m should be approximately alpha * n."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        expected_m = int(round(3.0 * 10))
        self.assertEqual(result["m"], expected_m)

    def test_correct_number_of_clauses(self):
        """Number of clauses should match m."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(len(result["clauses"]), result["m"])

    def test_each_clause_has_k_literals(self):
        """Each clause should have exactly k literals."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        for clause in result["clauses"]:
            self.assertEqual(len(clause), 3)

    def test_literals_are_integers(self):
        """All literals should be integers."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        for clause in result["clauses"]:
            for lit in clause:
                self.assertIsInstance(lit, int)

    def test_variable_indices_in_range(self):
        """Variable indices should be in [1, n]."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        for clause in result["clauses"]:
            for lit in clause:
                var = abs(lit)
                self.assertGreaterEqual(var, 1)
                self.assertLessEqual(var, 10)

    def test_no_duplicate_variables_in_clause(self):
        """No clause should have duplicate variables."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        for clause in result["clauses"]:
            vars_in_clause = [abs(lit) for lit in clause]
            self.assertEqual(len(vars_in_clause), len(set(vars_in_clause)))

    def test_seed_determinism(self):
        """Same seed should give same instance."""
        result1 = generate_ksat_instance(10, 3.0, k=3, seed=42)
        result2 = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(result1["clauses"], result2["clauses"])

    def test_different_seeds_different_instances(self):
        """Different seeds should likely give different instances."""
        result1 = generate_ksat_instance(10, 3.0, k=3, seed=42)
        result2 = generate_ksat_instance(10, 3.0, k=3, seed=43)

        self.assertNotEqual(result1["clauses"], result2["clauses"])

    def test_seed_recorded(self):
        """Seed should be recorded in result."""
        result = generate_ksat_instance(10, 3.0, k=3, seed=42)
        self.assertEqual(result["seed"], 42)

    def test_n_less_than_k_raises_error(self):
        """Should raise error if n < k."""
        with self.assertRaises(ValueError):
            generate_ksat_instance(2, 3.0, k=3, seed=42)

    def test_non_positive_alpha_raises_error(self):
        """Should raise error if alpha <= 0."""
        with self.assertRaises(ValueError):
            generate_ksat_instance(10, 0.0, k=3, seed=42)


class TestGenerateInstanceBatch(unittest.TestCase):
    """Tests for generate_instance_batch function."""

    def test_returns_list(self):
        """Should return a list."""
        result = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        self.assertIsInstance(result, list)

    def test_correct_length(self):
        """Should return n_instances instances."""
        result = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        self.assertEqual(len(result), 5)

    def test_all_instances_different(self):
        """All instances in batch should be different."""
        result = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        clauses_list = [tuple(tuple(c) for c in inst["clauses"]) for inst in result]
        self.assertEqual(len(clauses_list), len(set(clauses_list)))

    def test_all_same_n(self):
        """All instances should have same n."""
        result = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        for inst in result:
            self.assertEqual(inst["n"], 10)

    def test_all_same_alpha(self):
        """All instances should have same alpha."""
        result = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        for inst in result:
            self.assertEqual(inst["alpha"], 3.0)

    def test_deterministic_with_same_seed(self):
        """Same master_seed should give same batch."""
        result1 = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        result2 = generate_instance_batch(10, 3.0, 5, k=3, master_seed=42)
        for i in range(5):
            self.assertEqual(result1[i]["clauses"], result2[i]["clauses"])


class TestInstanceToAdjacency(unittest.TestCase):
    """Tests for instance_to_adjacency function."""

    def test_returns_tuple(self):
        """Should return a tuple of two lists."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        var_to_clauses, clause_to_vars = instance_to_adjacency(instance)
        self.assertIsInstance(var_to_clauses, list)
        self.assertIsInstance(clause_to_vars, list)

    def test_correct_sizes(self):
        """Output sizes should match n and m."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        var_to_clauses, clause_to_vars = instance_to_adjacency(instance)
        self.assertEqual(len(var_to_clauses), instance["n"] + 1)
        self.assertEqual(len(clause_to_vars), instance["m"])

    def test_consistency(self):
        """Adjacency should be consistent."""
        instance = generate_ksat_instance(10, 3.0, k=3, seed=42)
        var_to_clauses, clause_to_vars = instance_to_adjacency(instance)

        for ci, cvars in enumerate(clause_to_vars):
            for vi in cvars:
                self.assertIn(ci, var_to_clauses[vi])


class TestCountViolatedClauses(unittest.TestCase):
    """Tests for count_violated_clauses function."""

    def test_all_satisfied(self):
        """Should return 0 for satisfying assignment."""
        instance = {
            "n": 3,
            "clauses": [[1, 2], [-1, 3]],
        }
        assignment = {1: True, 2: True, 3: True}
        result = count_violated_clauses(instance, assignment)
        self.assertEqual(result, 0)

    def test_one_violated(self):
        """Should count one violated clause."""
        instance = {
            "n": 2,
            "clauses": [[1], [-1]],
        }
        assignment = {1: True, 2: True}
        result = count_violated_clauses(instance, assignment)
        self.assertEqual(result, 1)

    def test_all_violated(self):
        """Should count all violated clauses."""
        instance = {
            "n": 2,
            "clauses": [[1], [2]],
        }
        assignment = {1: False, 2: False}
        result = count_violated_clauses(instance, assignment)
        self.assertEqual(result, 2)

    def test_partial_assignment(self):
        """Should work with partial assignments."""
        instance = {
            "n": 3,
            "clauses": [[1, 2, 3]],
        }
        assignment = {1: False, 2: False}
        result = count_violated_clauses(instance, assignment)


        self.assertEqual(result, 1)


class TestIsSatisfying(unittest.TestCase):
    """Tests for is_satisfying function."""

    def test_satisfying_assignment(self):
        """Should return True for satisfying assignment."""
        instance = {
            "n": 2,
            "clauses": [[1, 2]],
        }
        assignment = {1: True, 2: False}
        result = is_satisfying(instance, assignment)
        self.assertTrue(result)

    def test_unsatisfying_assignment(self):
        """Should return False for unsatisfying assignment."""
        instance = {
            "n": 2,
            "clauses": [[1]],
        }
        assignment = {1: False, 2: True}
        result = is_satisfying(instance, assignment)
        self.assertFalse(result)

    def test_empty_instance(self):
        """Empty instance should be satisfied by any assignment."""
        instance = {
            "n": 2,
            "clauses": [],
        }
        assignment = {1: True, 2: True}
        result = is_satisfying(instance, assignment)
        self.assertTrue(result)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_large_n(self):
        """Should handle large n."""
        result = generate_ksat_instance(1000, 4.0, k=3, seed=42)
        self.assertEqual(result["n"], 1000)
        self.assertEqual(len(result["clauses"]), result["m"])

    def test_k_equals_n(self):
        """Should handle k = n case."""
        result = generate_ksat_instance(5, 1.0, k=5, seed=42)
        self.assertEqual(result["k"], 5)
        for clause in result["clauses"]:
            self.assertEqual(len(clause), 5)

    def test_very_low_alpha(self):
        """Should handle very low alpha."""
        result = generate_ksat_instance(100, 0.1, k=3, seed=42)
        self.assertEqual(result["m"], 10)


if __name__ == "__main__":
    unittest.main()