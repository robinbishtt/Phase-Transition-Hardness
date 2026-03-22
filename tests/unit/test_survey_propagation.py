"""Unit tests for src/survey_propagation/.

Covers:
    bp_equations.py      — BeliefPropagation, BPResult
    sp_equations.py      — SurveyPropagation, SPResult
    warning_propagation.py — WarningPropagation, WPResult
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance
from src.survey_propagation.bp_equations import BeliefPropagation, BPResult
from src.survey_propagation.sp_equations import SurveyPropagation, SPResult
from src.survey_propagation.warning_propagation import WarningPropagation, WPResult


class TestBeliefPropagation(unittest.TestCase):

    def _easy_instance(self):
        return generate_ksat_instance(15, 3.0, k=3, seed=42)

    def _hard_instance(self):
        return generate_ksat_instance(15, 4.20, k=3, seed=42)

    def test_run_returns_bp_result(self):
        inst   = self._easy_instance()
        bp     = BeliefPropagation(inst, beta=2.0)
        result = bp.run()
        self.assertIsInstance(result, BPResult)

    def test_result_has_all_fields(self):
        inst   = self._easy_instance()
        result = BeliefPropagation(inst, beta=2.0).run()
        self.assertIsInstance(result.converged, bool)
        self.assertIsInstance(result.n_iterations, int)
        self.assertGreater(result.n_iterations, 0)
        self.assertIsInstance(result.magnetisations, dict)
        self.assertIsInstance(result.cavity_fields, dict)
        self.assertTrue(np.isfinite(result.free_energy))

    def test_magnetisations_cover_all_variables(self):
        inst   = self._easy_instance()
        result = BeliefPropagation(inst, beta=2.0).run()
        self.assertEqual(set(result.magnetisations.keys()), set(range(1, inst["n"] + 1)))

    def test_magnetisations_in_minus1_plus1(self):
        inst   = self._easy_instance()
        result = BeliefPropagation(inst, beta=2.0).run()
        for vi, m in result.magnetisations.items():
            self.assertGreaterEqual(m, -1.0 - 1e-6, msg=f"m[{vi}] = {m}")
            self.assertLessEqual(m,   1.0 + 1e-6, msg=f"m[{vi}] = {m}")

    def test_same_instance_same_result_different_objects(self):
        inst    = self._easy_instance()
        result1 = BeliefPropagation(inst, beta=2.0, damping=0.5).run()
        result2 = BeliefPropagation(inst, beta=2.0, damping=0.5).run()
        for vi in range(1, inst["n"] + 1):
            self.assertAlmostEqual(
                result1.magnetisations[vi], result2.magnetisations[vi], places=10
            )

    def test_low_beta_low_magnetisation(self):
        inst   = self._easy_instance()
        result = BeliefPropagation(inst, beta=0.1).run()
        mean_abs_m = np.mean([abs(m) for m in result.magnetisations.values()])
        self.assertLess(mean_abs_m, 0.5)

    def test_cavity_fields_key_format(self):
        inst   = self._easy_instance()
        result = BeliefPropagation(inst, beta=2.0).run()
        for (ci, vi) in result.cavity_fields:
            self.assertIsInstance(ci, int)
            self.assertIsInstance(vi, int)

    def test_hard_instance_runs_without_error(self):
        inst   = self._hard_instance()
        result = BeliefPropagation(inst, beta=5.0, max_iter=50).run()
        self.assertIsInstance(result, BPResult)


class TestSurveyPropagation(unittest.TestCase):

    def _instance(self, alpha=4.20):
        return generate_ksat_instance(12, alpha, k=3, seed=42)

    def test_run_returns_sp_result(self):
        inst   = self._instance(4.20)
        result = SurveyPropagation(inst).run()
        self.assertIsInstance(result, SPResult)

    def test_result_has_all_fields(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        self.assertIsInstance(result.converged,         bool)
        self.assertIsInstance(result.n_iterations,      int)
        self.assertIsInstance(result.eta_surveys,       dict)
        self.assertIsInstance(result.complexity,        float)
        self.assertIsInstance(result.frozen_fraction,   float)
        self.assertIsInstance(result.biases,            dict)

    def test_eta_surveys_in_unit_interval(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        for (ci, vi), eta in result.eta_surveys.items():
            self.assertGreaterEqual(eta, 0.0 - 1e-9)
            self.assertLessEqual(eta,   1.0 + 1e-9)

    def test_frozen_fraction_in_unit_interval(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        self.assertGreaterEqual(result.frozen_fraction, 0.0)
        self.assertLessEqual(result.frozen_fraction,   1.0)

    def test_biases_cover_all_variables(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        self.assertEqual(set(result.biases.keys()), set(range(1, inst["n"] + 1)))

    def test_biases_in_minus1_plus1(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        for vi, b in result.biases.items():
            self.assertGreaterEqual(b, -1.0 - 1e-6)
            self.assertLessEqual(b,   1.0 + 1e-6)

    def test_decimation_assignment_subset_of_variables(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        sp     = SurveyPropagation(inst)
        assign = sp.decimation_assignment(result, threshold=0.7)
        for vi in assign:
            self.assertIn(vi, range(1, inst["n"] + 1))
        for vi, val in assign.items():
            self.assertIsInstance(val, bool)

    def test_complexity_finite_and_nonnegative(self):
        inst   = self._instance()
        result = SurveyPropagation(inst).run()
        self.assertGreaterEqual(result.complexity, 0.0)
        self.assertTrue(np.isfinite(result.complexity))

    def test_easy_instance_runs(self):
        inst   = self._instance(alpha=3.0)
        result = SurveyPropagation(inst).run()
        self.assertIsInstance(result, SPResult)


class TestWarningPropagation(unittest.TestCase):

    def _trivial_sat_instance(self):
        return {
            "n": 3, "m": 3, "k": 3, "alpha": 1.0,
            "clauses": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            "seed": 0,
        }

    def _easy_instance(self):
        return generate_ksat_instance(15, 2.5, k=3, seed=42)

    def test_run_returns_wp_result(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        self.assertIsInstance(result, WPResult)

    def test_result_fields(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        self.assertIsInstance(result.converged,         bool)
        self.assertIsInstance(result.n_iterations,      int)
        self.assertIsInstance(result.warnings,          dict)
        self.assertIsInstance(result.forced_vars,       dict)
        self.assertIsInstance(result.n_contradictions,  int)

    def test_warning_values_in_valid_set(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        for (ci, vi), u in result.warnings.items():
            self.assertIn(u, [-1, 0, 1])

    def test_forced_vars_subset_of_variables(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        for vi in result.forced_vars:
            self.assertIn(vi, range(1, inst["n"] + 1))

    def test_forced_vars_bool_values(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        for vi, val in result.forced_vars.items():
            self.assertIsInstance(val, bool)

    def test_contradictions_nonnegative(self):
        inst   = self._easy_instance()
        result = WarningPropagation(inst).run()
        self.assertGreaterEqual(result.n_contradictions, 0)

    def test_trivial_sat_no_contradictions(self):
        inst   = self._trivial_sat_instance()
        result = WarningPropagation(inst).run()
        self.assertEqual(result.n_contradictions, 0)

    def test_hard_instance_runs(self):
        inst   = generate_ksat_instance(12, 4.20, k=3, seed=42)
        result = WarningPropagation(inst).run()
        self.assertIsInstance(result, WPResult)


if __name__ == "__main__":
    unittest.main()
