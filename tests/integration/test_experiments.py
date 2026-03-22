"""Integration tests for experiments/.

Tests each of the four experiment scripts by invoking their main() with
minimal arguments (tiny n, few instances, coarse alpha grid) and
verifying that (a) they run to completion with exit code 0, (b) they write
the expected output files to a temporary directory, and (c) the written
JSON / NPZ files contain the required keys and numerically sensible values.

These tests do not validate the scientific accuracy of results — that is
the role of the manuscript validation suite (tests/validation/).  They
verify that the pipeline machinery works end-to-end without errors.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _run_main(script_module, argv):
    """Invoke a script's main() with patched sys.argv, suppressing all output.

    All stdout, stderr, and Python logging is suppressed so that INFO messages
    from the validation suite (e.g. "3/8 checks passed") do not appear in CI
    logs and are not mistaken for actual test failures.
    """
    import io, logging
    from contextlib import redirect_stdout, redirect_stderr

    orig_argv = sys.argv[:]
    sys.argv = [script_module.__file__] + argv
    # logging.disable() globally disables all log records at or below CRITICAL,
    # bypassing individual logger levels and their handlers entirely.
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = script_module.main()
    finally:
        sys.argv = orig_argv
        logging.disable(logging.NOTSET)  # restore normal logging
    return rc if rc is not None else 0


class TestAlphaSweepExperiment(unittest.TestCase):
    """experiments/alpha_sweep.py — minimal smoke run."""

    def setUp(self):
        import experiments.alpha_sweep as m
        self.mod = m
        self.tmp = tempfile.mkdtemp()

    def test_main_runs_and_returns_zero(self):
        rc = _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        self.assertEqual(rc, 0)

    def test_output_files_written(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        self.assertTrue(os.path.exists(f"{self.tmp}/alpha_sweep.npz"))
        self.assertTrue(os.path.exists(f"{self.tmp}/alpha_sweep_summary.json"))

    def test_alpha_sweep_summary_json_structure(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        with open(f"{self.tmp}/alpha_sweep_summary.json") as f:
            s = json.load(f)
        for key in ["ns", "alpha_stars", "gamma_maxima", "alpha_star_inf"]:
            self.assertIn(key, s, msg=f"Key '{key}' missing from alpha_sweep_summary.json")
        self.assertTrue(np.isfinite(s["alpha_star_inf"]))

    def test_phase_transition_npz_loads_without_pickle(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        d = np.load(f"{self.tmp}/phase_transition.npz", allow_pickle=False)
        self.assertIn("psat_matrix", d)
        self.assertIn("alphas",      d)

    def test_validation_json_written(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        self.assertTrue(os.path.exists(f"{self.tmp}/validation.json"))
        with open(f"{self.tmp}/validation.json") as f:
            v = json.load(f)
        self.assertIn("passed", v)
        self.assertIn("total",  v)


class TestFiniteSizeScalingExperiment(unittest.TestCase):
    """experiments/finite_size_scaling.py — minimal smoke run."""

    def setUp(self):
        import experiments.finite_size_scaling as m
        self.mod = m
        self.tmp = tempfile.mkdtemp()

    def test_main_runs_and_returns_zero(self):
        # Requires phase_transition.npz from alpha_sweep first
        import experiments.alpha_sweep as asm
        _run_main(asm, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        rc = _run_main(self.mod, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        self.assertEqual(rc, 0)

    def test_fss_result_json_written(self):
        import experiments.alpha_sweep as asm
        _run_main(asm, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        _run_main(self.mod, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll",
            "--seed", "42", "--output_dir", self.tmp, "--n_jobs", "1",
        ])
        self.assertTrue(os.path.exists(f"{self.tmp}/fss_result.json"))
        with open(f"{self.tmp}/fss_result.json") as f:
            r = json.load(f)
        for key in ["alpha_s", "nu", "residual", "converged",
                    "n_system_sizes", "alpha_step"]:
            self.assertIn(key, r)
        self.assertGreater(r["nu"], 0.0)
        self.assertLessEqual(r["nu"], 6.0)
        self.assertGreaterEqual(r["residual"], 0.0)


class TestHardnessPeakExperiment(unittest.TestCase):
    """experiments/hardness_peak.py — minimal smoke run."""

    def setUp(self):
        import experiments.hardness_peak as m
        self.mod = m
        self.tmp = tempfile.mkdtemp()

    def test_main_runs_and_returns_zero(self):
        rc = _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_center", "4.20", "--alpha_width", "0.30",
            "--n_alpha_points", "4",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        self.assertEqual(rc, 0)

    def test_hardness_peak_summary_written(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_center", "4.20", "--alpha_width", "0.30",
            "--n_alpha_points", "4",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        self.assertTrue(os.path.exists(f"{self.tmp}/hardness_peak_summary.json"))
        with open(f"{self.tmp}/hardness_peak_summary.json") as f:
            s = json.load(f)
        self.assertIn("alpha_stars",    s)
        self.assertIn("gamma_maxima",   s)
        self.assertIn("alpha_star_inf", s)

    def test_alpha_star_inf_within_search_range(self):
        _run_main(self.mod, [
            "--n", "10",
            "--n_instances", "5",
            "--alpha_center", "4.20", "--alpha_width", "0.30",
            "--n_alpha_points", "4",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        with open(f"{self.tmp}/hardness_peak_summary.json") as f:
            s = json.load(f)
        a_inf = float(s["alpha_star_inf"])
        self.assertGreaterEqual(a_inf, 4.20 - 0.30 - 0.10)
        self.assertLessEqual(a_inf,    4.20 + 0.30 + 0.10)


class TestScalingLawVerificationExperiment(unittest.TestCase):
    """experiments/scaling_law_verification.py — minimal smoke run."""

    def setUp(self):
        import experiments.scaling_law_verification as m
        self.mod = m
        self.tmp = tempfile.mkdtemp()

    def test_main_runs_and_returns_zero(self):
        rc = _run_main(self.mod, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        self.assertEqual(rc, 0)

    def test_exponential_scaling_summary_written(self):
        _run_main(self.mod, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        self.assertTrue(os.path.exists(f"{self.tmp}/exponential_scaling_summary.json"))
        with open(f"{self.tmp}/exponential_scaling_summary.json") as f:
            s = json.load(f)
        self.assertIn("mean_r2",   s)
        self.assertIn("max_gamma", s)

    def test_r2_in_unit_interval(self):
        _run_main(self.mod, [
            "--n", "10", "15",
            "--n_instances", "5",
            "--alpha_min", "3.5", "--alpha_max", "4.5", "--alpha_step", "0.5",
            "--k", "3", "--solver", "dpll", "--max_decisions", "500",
            "--seed", "42", "--output_dir", self.tmp,
        ])
        with open(f"{self.tmp}/exponential_scaling_summary.json") as f:
            s = json.load(f)
        r2 = float(s["mean_r2"])
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2,   1.0)


if __name__ == "__main__":
    unittest.main()
