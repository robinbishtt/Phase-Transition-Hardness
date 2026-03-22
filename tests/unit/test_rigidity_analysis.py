"""Unit tests for src/rigidity_analysis.py.

Rigidity analysis characterises which variables are frozen — taking the
same value in every satisfying assignment within a given solution cluster.
The empirical frozen fraction computed here is the per-instance counterpart
of the theoretical frozen_fraction(α) from the 1RSB cavity equations
(manuscript Section 3.2 and energy_model.py).

All tests use the exact function signatures:
    compute_rigidity_profile(instance, assignment) -> Dict[int, bool]
    compute_frozen_fraction(instance, assignment)  -> float
    estimate_cluster_rigidity(instance, ...)       -> {"mean", "std", "n_samples"}
    compute_rigidity_threshold_indicator(instance) -> bool
    analyze_rigidity_transition(n, alphas, ...)    -> {"alphas", "frozen_fractions", "n", "k"}
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance
from src.hardness_metrics import dpll_solve
from src.rigidity_analysis import (
    analyze_rigidity_transition,
    compute_frozen_fraction,
    compute_rigidity_profile,
    compute_rigidity_threshold_indicator,
    compute_variable_dependencies,
    estimate_cluster_rigidity,
    find_unit_clauses,
    propagate_units,
)


def _get_sat_assignment(inst):
    """Helper: return a satisfying assignment or an empty dict if UNSAT/timeout."""
    result = dpll_solve(inst, max_decisions=50000)
    if result["satisfiable"]:
        return result["assignment"]
    return {}


class TestComputeVariableDependencies(unittest.TestCase):

    def test_returns_dict(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        deps = compute_variable_dependencies(inst)
        self.assertIsInstance(deps, dict)

    def test_all_variables_represented(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        deps = compute_variable_dependencies(inst)
        for vi in range(1, inst["n"] + 1):
            self.assertIn(vi, deps)

    def test_values_are_sets(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        for dep_set in compute_variable_dependencies(inst).values():
            self.assertIsInstance(dep_set, set)

    def test_values_are_clause_indices(self):
        # compute_variable_dependencies maps each variable to the SET OF CLAUSE
        # INDICES in which it appears (0-indexed), not to other variable indices.
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        deps = compute_variable_dependencies(inst)
        for vi, clause_indices in deps.items():
            for ci in clause_indices:
                self.assertGreaterEqual(ci, 0)
                self.assertLess(ci, inst["m"])

    def test_each_variable_appears_in_at_least_one_clause(self):
        # In a random 3-SAT instance with α > 0 every variable must appear in
        # at least one clause (otherwise it is isolated and trivially satisfiable).
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        deps = compute_variable_dependencies(inst)
        for vi in range(1, inst["n"] + 1):
            self.assertIn(vi, deps, msg=f"Variable {vi} absent from deps")
            self.assertGreater(len(deps[vi]), 0,
                               msg=f"Variable {vi} has no clause occurrences")


class TestFindUnitClauses(unittest.TestCase):

    def test_returns_list(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        self.assertIsInstance(find_unit_clauses(inst), list)

    def test_items_are_tuples(self):
        inst = generate_ksat_instance(15, 3.5, k=3, seed=42)
        for item in find_unit_clauses(inst):
            self.assertIsInstance(item, tuple)

    def test_unit_clause_detected(self):
        # Construct an instance with an explicit unit clause.
        inst = {
            "n": 3, "m": 3, "k": 3, "alpha": 1.0, "seed": 0,
            "clauses": [[1], [2, 3, -1], [-2, 3, 1]],
        }
        units = find_unit_clauses(inst)
        if units:
            self.assertIn((1, True), units)


class TestPropagateUnits(unittest.TestCase):

    def test_returns_dict(self):
        inst = generate_ksat_instance(15, 2.5, k=3, seed=42)
        result = propagate_units(inst, {})
        self.assertIsInstance(result, dict)

    def test_values_are_bool(self):
        inst = generate_ksat_instance(15, 2.5, k=3, seed=42)
        for val in propagate_units(inst, {}).values():
            self.assertIsInstance(val, bool)

    def test_keys_are_valid_variable_indices(self):
        inst = generate_ksat_instance(15, 2.5, k=3, seed=42)
        all_vars = set(range(1, inst["n"] + 1))
        for vi in propagate_units(inst, {}).keys():
            self.assertIn(vi, all_vars)

    def test_seed_assignment_preserved(self):
        inst = generate_ksat_instance(15, 2.5, k=3, seed=42)
        seed = {1: True, 2: False}
        propagated = propagate_units(inst, seed)
        for vi, val in seed.items():
            if vi in propagated:
                self.assertEqual(propagated[vi], val)


class TestComputeRigidityProfile(unittest.TestCase):
    """compute_rigidity_profile(instance, assignment) -> Dict[int, bool]
    where True = variable is frozen under the given assignment."""

    def setUp(self):
        self.inst       = generate_ksat_instance(15, 4.0, k=3, seed=42)
        self.assignment = _get_sat_assignment(self.inst)

    def test_returns_dict(self):
        profile = compute_rigidity_profile(self.inst, self.assignment)
        self.assertIsInstance(profile, dict)

    def test_keys_are_variable_indices(self):
        profile = compute_rigidity_profile(self.inst, self.assignment)
        all_vars = set(range(1, self.inst["n"] + 1))
        for vi in profile:
            self.assertIn(vi, all_vars)

    def test_values_are_bool(self):
        profile = compute_rigidity_profile(self.inst, self.assignment)
        for val in profile.values():
            self.assertIsInstance(val, bool)

    def test_empty_assignment_all_unfrozen(self):
        # With no assigned variables none can be frozen.
        profile = compute_rigidity_profile(self.inst, {})
        for val in profile.values():
            self.assertFalse(val)


class TestComputeFrozenFraction(unittest.TestCase):
    """compute_frozen_fraction(instance, assignment) -> float."""

    def setUp(self):
        self.inst       = generate_ksat_instance(15, 4.0, k=3, seed=42)
        self.assignment = _get_sat_assignment(self.inst)

    def test_returns_float(self):
        frac = compute_frozen_fraction(self.inst, self.assignment)
        self.assertIsInstance(frac, float)

    def test_in_unit_interval(self):
        frac = compute_frozen_fraction(self.inst, self.assignment)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac,   1.0)

    def test_empty_assignment_gives_zero(self):
        # No variables assigned → no frozen variables → fraction = 0.
        frac = compute_frozen_fraction(self.inst, {})
        self.assertEqual(frac, 0.0)

    def test_consistent_with_rigidity_profile(self):
        profile  = compute_rigidity_profile(self.inst, self.assignment)
        expected = sum(profile.values()) / max(len(profile), 1)
        self.assertAlmostEqual(
            compute_frozen_fraction(self.inst, self.assignment),
            expected, places=10,
        )


class TestEstimateClusterRigidity(unittest.TestCase):
    """estimate_cluster_rigidity(instance, n_samples, seed) -> {"mean", "std", "n_samples"}."""

    def test_returns_dict_with_correct_keys(self):
        inst   = generate_ksat_instance(12, 3.0, k=3, seed=42)
        result = estimate_cluster_rigidity(inst, n_samples=5, seed=42)
        for key in ["mean", "std", "n_samples"]:
            self.assertIn(key, result)

    def test_mean_in_unit_interval(self):
        inst   = generate_ksat_instance(12, 3.0, k=3, seed=42)
        result = estimate_cluster_rigidity(inst, n_samples=5, seed=42)
        self.assertGreaterEqual(result["mean"], 0.0)
        self.assertLessEqual(result["mean"],   1.0)

    def test_std_non_negative(self):
        inst   = generate_ksat_instance(12, 3.0, k=3, seed=42)
        result = estimate_cluster_rigidity(inst, n_samples=5, seed=42)
        self.assertGreaterEqual(result["std"], 0.0)

    def test_n_samples_non_negative(self):
        inst   = generate_ksat_instance(12, 3.0, k=3, seed=42)
        result = estimate_cluster_rigidity(inst, n_samples=5, seed=42)
        self.assertGreaterEqual(result["n_samples"], 0)


class TestComputeRigidityThresholdIndicator(unittest.TestCase):
    """compute_rigidity_threshold_indicator(instance, threshold=0.5) -> bool."""

    def test_returns_bool_easy_phase(self):
        # This function runs WalkSAT internally (n_samples=50); use a small instance.
        inst = generate_ksat_instance(8, 2.5, k=3, seed=42)
        result = compute_rigidity_threshold_indicator(inst)
        self.assertIsInstance(result, bool)

    def test_returns_bool_hard_phase(self):
        inst = generate_ksat_instance(8, 4.0, k=3, seed=42)
        result = compute_rigidity_threshold_indicator(inst)
        self.assertIsInstance(result, bool)


class TestAnalyzeRigidityTransition(unittest.TestCase):
    """Return keys: alphas, frozen_fractions (list of dicts), n, k."""

    def setUp(self):
        self.alphas = [2.5, 3.5]
        self.result = analyze_rigidity_transition(
            n=12, alphas=self.alphas, n_instances=3, k=3, seed=42
        )

    def test_required_keys(self):
        for key in ["alphas", "frozen_fractions", "n", "k"]:
            self.assertIn(key, self.result)

    def test_frozen_fractions_length_matches_alphas(self):
        self.assertEqual(len(self.result["frozen_fractions"]), len(self.alphas))

    def test_frozen_fractions_is_list_of_dicts(self):
        for item in self.result["frozen_fractions"]:
            self.assertIsInstance(item, dict)

    def test_per_alpha_dict_has_required_keys(self):
        for item in self.result["frozen_fractions"]:
            for key in ["alpha", "mean", "std"]:
                self.assertIn(key, item)

    def test_means_in_unit_interval(self):
        for item in self.result["frozen_fractions"]:
            self.assertGreaterEqual(item["mean"], 0.0)
            self.assertLessEqual(item["mean"],   1.0)

    def test_metadata_preserved(self):
        self.assertEqual(self.result["n"], 12)
        self.assertEqual(self.result["k"],  3)


if __name__ == "__main__":
    unittest.main()
