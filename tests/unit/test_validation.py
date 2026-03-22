"""Unit tests for src/validation.py.

The validation module implements the 8 automated checks that verify computed
experimental results against the manuscript's quantitative claims.  Each
check returns (bool, str) from a results directory.  run_all_checks runs
all 8 and returns a summary dict with keys {"passed", "failed", "total",
"details"}.

Tests are structured as follows:
  - Each check function is tested with (a) a minimally valid results
    directory that should pass, (b) a missing-file scenario that should fail,
    and (c) an out-of-range value that should fail.
  - run_all_checks is tested end-to-end.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import save_json
from src.validation import (
    ALPHA_S_LO,
    ALPHA_S_HI,
    ALPHA_STAR_LO,
    ALPHA_STAR_HI,
    EXP_R2_MIN,
    FSS_RESIDUAL_MAX,
    GAMMA_MAX_HI,
    GAMMA_MAX_LO,
    NU_HI,
    NU_LO,
    check_1_alpha_s,
    check_2_alpha_star,
    check_3_gamma_max,
    check_4_exponential_scaling,
    check_5_fss_residual,
    check_6_nu,
    check_7_barrier_positivity,
    check_8_psat_monotone,
    run_all_checks,
)


def _write_full_passing_results(tmp: str) -> None:
    """Write the minimal set of result files needed for all 8 checks to pass."""
    save_json(
        {"thresholds": {"100": 4.267, "200": 4.267}},
        f"{tmp}/phase_transition_summary.json",
    )
    save_json(
        {"alpha_star_inf": 4.20, "alpha_stars": [4.19, 4.20],
         "gamma_maxima": [0.030]},
        f"{tmp}/alpha_sweep_summary.json",
    )
    save_json(
        {"mean_r2": 0.90, "max_gamma": 0.015},
        f"{tmp}/exponential_scaling_summary.json",
    )
    save_json(
        {"residual": 0.05, "nu": 2.30, "converged": True,
         "alpha_s": 4.267, "n_system_sizes": 4, "alpha_step": 0.05},
        f"{tmp}/fss_result.json",
    )
    alphas = np.linspace(3.5, 5.0, 10)
    ns     = np.array([100, 200])
    from scipy.special import expit
    psat   = np.array([expit(-3.0 * (alphas - 4.267)) for _ in ns])
    np.savez(f"{tmp}/phase_transition.npz", alphas=alphas, ns=ns, psat_matrix=psat)


class TestCheck1AlphaS(unittest.TestCase):

    def test_passes_with_valid_alpha_s(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"thresholds": {"100": 4.267}},
                      f"{tmp}/phase_transition_summary.json")
            ok, msg = check_1_alpha_s(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_below_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"thresholds": {"100": ALPHA_S_LO - 0.05}},
                      f"{tmp}/phase_transition_summary.json")
            ok, _ = check_1_alpha_s(tmp)
        self.assertFalse(ok)

    def test_fails_above_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"thresholds": {"100": ALPHA_S_HI + 0.05}},
                      f"{tmp}/phase_transition_summary.json")
            ok, _ = check_1_alpha_s(tmp)
        self.assertFalse(ok)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_1_alpha_s(tmp)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_returns_tuple_of_bool_and_str(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"thresholds": {"100": 4.267}},
                      f"{tmp}/phase_transition_summary.json")
            result = check_1_alpha_s(tmp)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)


class TestCheck2AlphaStar(unittest.TestCase):

    def test_passes_with_valid_alpha_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"alpha_star_inf": 4.20, "gamma_maxima": [0.02]},
                      f"{tmp}/alpha_sweep_summary.json")
            ok, msg = check_2_alpha_star(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_outside_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"alpha_star_inf": ALPHA_STAR_HI + 0.1, "gamma_maxima": [0.02]},
                      f"{tmp}/alpha_sweep_summary.json")
            ok, _ = check_2_alpha_star(tmp)
        self.assertFalse(ok)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_2_alpha_star(tmp)
        self.assertFalse(ok)
        self.assertIn("not found", msg)


class TestCheck3GammaMax(unittest.TestCase):

    def test_passes_with_valid_gamma_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"gamma_maxima": [0.030]},
                      f"{tmp}/alpha_sweep_summary.json")
            ok, msg = check_3_gamma_max(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_below_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"gamma_maxima": [GAMMA_MAX_LO / 2.0]},
                      f"{tmp}/alpha_sweep_summary.json")
            ok, _ = check_3_gamma_max(tmp)
        self.assertFalse(ok)

    def test_fails_above_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"gamma_maxima": [GAMMA_MAX_HI + 0.05]},
                      f"{tmp}/alpha_sweep_summary.json")
            ok, _ = check_3_gamma_max(tmp)
        self.assertFalse(ok)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, _ = check_3_gamma_max(tmp)
        self.assertFalse(ok)


class TestCheck4ExponentialScaling(unittest.TestCase):

    def test_passes_with_valid_r2(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"mean_r2": 0.92, "max_gamma": 0.015},
                      f"{tmp}/exponential_scaling_summary.json")
            ok, msg = check_4_exponential_scaling(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_with_low_r2(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"mean_r2": EXP_R2_MIN - 0.10, "max_gamma": 0.015},
                      f"{tmp}/exponential_scaling_summary.json")
            ok, _ = check_4_exponential_scaling(tmp)
        self.assertFalse(ok)

    def test_fails_with_zero_gamma(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"mean_r2": 0.95, "max_gamma": 0.0},
                      f"{tmp}/exponential_scaling_summary.json")
            ok, _ = check_4_exponential_scaling(tmp)
        self.assertFalse(ok)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, _ = check_4_exponential_scaling(tmp)
        self.assertFalse(ok)


class TestCheck5FSSResidual(unittest.TestCase):

    def test_passes_with_small_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"residual": 0.05, "nu": 2.30, "converged": True},
                      f"{tmp}/fss_result.json")
            ok, msg = check_5_fss_residual(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_with_large_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"residual": FSS_RESIDUAL_MAX + 0.05, "nu": 2.30},
                      f"{tmp}/fss_result.json")
            ok, _ = check_5_fss_residual(tmp)
        self.assertFalse(ok)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_5_fss_residual(tmp)
        self.assertFalse(ok)
        self.assertIn("not found", msg)


class TestCheck6Nu(unittest.TestCase):

    def test_passes_with_valid_nu(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"nu": 2.30, "residual": 0.05, "converged": True,
                       "n_system_sizes": 4, "alpha_step": 0.05},
                      f"{tmp}/fss_result.json")
            ok, msg = check_6_nu(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_with_nu_outside_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"nu": NU_HI + 0.5, "residual": 0.05,
                       "n_system_sizes": 4, "alpha_step": 0.05},
                      f"{tmp}/fss_result.json")
            ok, _ = check_6_nu(tmp)
        self.assertFalse(ok)

    def test_fails_with_degenerate_nu(self):
        # ν outside [0.5, 6.0] is physically implausible — check flags it.
        with tempfile.TemporaryDirectory() as tmp:
            save_json({"nu": 0.3, "residual": 0.05},
                      f"{tmp}/fss_result.json")
            ok, msg = check_6_nu(tmp)
        self.assertFalse(ok)
        self.assertIn("degenerate", msg)

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_6_nu(tmp)
        self.assertFalse(ok)
        self.assertIn("not found", msg)


class TestCheck7BarrierPositivity(unittest.TestCase):
    """Check 7 requires no files — it evaluates barrier_density analytically."""

    def test_passes_always(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_7_barrier_positivity(tmp)
        self.assertTrue(ok, msg=f"Should always pass: {msg}")

    def test_message_contains_min_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, msg = check_7_barrier_positivity(tmp)
        self.assertIn("min b", msg)


class TestCheck8PsatMonotonicity(unittest.TestCase):

    def test_passes_with_monotone_psat(self):
        with tempfile.TemporaryDirectory() as tmp:
            from scipy.special import expit
            alphas = np.linspace(3.5, 5.0, 10)
            ns     = np.array([100, 200])
            psat   = np.array([expit(-3.0 * (alphas - 4.267)) for _ in ns])
            np.savez(f"{tmp}/phase_transition.npz",
                     alphas=alphas, ns=ns, psat_matrix=psat)
            ok, msg = check_8_psat_monotone(tmp)
        self.assertTrue(ok, msg=f"Should pass: {msg}")

    def test_fails_with_strongly_non_monotone_psat(self):
        # Inject a large increase (> 0.05) in P_sat to trigger the violation flag.
        with tempfile.TemporaryDirectory() as tmp:
            alphas = np.linspace(3.5, 5.0, 10)
            ns     = np.array([100])
            psat   = np.array([[0.9, 0.7, 0.6, 0.4, 0.8, 0.3, 0.2, 0.1, 0.05, 0.0]])
            np.savez(f"{tmp}/phase_transition.npz",
                     alphas=alphas, ns=ns, psat_matrix=psat)
            ok, msg = check_8_psat_monotone(tmp)
        # The value jumps from 0.4 → 0.8 at index 4, which is a +0.4 increase.
        self.assertFalse(ok, msg=f"Should fail due to +0.4 jump: {msg}")

    def test_fails_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = check_8_psat_monotone(tmp)
        self.assertFalse(ok)
        self.assertIn("not found", msg)


class TestRunAllChecks(unittest.TestCase):

    def test_all_8_pass_with_valid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            summary = run_all_checks(tmp)
        self.assertEqual(summary["passed"], 8)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["total"],  8)

    def test_returns_correct_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            summary = run_all_checks(tmp)
        for key in ["passed", "failed", "total", "details"]:
            self.assertIn(key, summary)

    def test_details_is_list_of_8(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            summary = run_all_checks(tmp)
        self.assertIsInstance(summary["details"], list)
        self.assertEqual(len(summary["details"]), 8)

    def test_detail_dict_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            summary = run_all_checks(tmp)
        for detail in summary["details"]:
            for key in ["check", "name", "passed", "message", "manuscript_val"]:
                self.assertIn(key, detail)

    def test_passed_plus_failed_equals_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            summary = run_all_checks(tmp)
        self.assertEqual(
            summary["passed"] + summary["failed"], summary["total"]
        )

    def test_empty_dir_gives_6_failures(self):
        # Checks 1, 2, 3, 4, 5, 6, 8 require files; check 7 is analytical.
        # With an empty directory all 7 file-dependent checks should fail.
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_all_checks(tmp)
        self.assertEqual(summary["passed"], 1)   # only check 7 passes
        self.assertEqual(summary["failed"], 7)

    def test_validation_json_written(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            run_all_checks(tmp)
            self.assertTrue(os.path.exists(f"{tmp}/validation.json"))

    def test_validation_json_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_full_passing_results(tmp)
            run_all_checks(tmp)
            with open(f"{tmp}/validation.json") as f:
                data = json.load(f)
            self.assertIn("passed", data)


if __name__ == "__main__":
    unittest.main()
