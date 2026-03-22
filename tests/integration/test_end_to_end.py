"""
Integration tests for end-to-end workflows

Tests complete experimental pipelines to ensure all components work together.
"""

import unittest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_instance_batch
from src.hardness_metrics import measure_hardness, dpll_solve
from src.phase_transition import run_psat_sweep
from src.runtime_measurement import alpha_sweep
from src.scaling_analysis import run_exponential_scaling, run_fss_analysis
from src.barrier_analysis import run_barrier_scaling_sweep, barrier_hardness_correlation
from src.validation import run_all_checks
from src.energy_model import ALPHA_D, ALPHA_S


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_instance_generation_and_solving(self):
        """Test generating instances and solving them."""
        instances = generate_instance_batch(n=20, alpha=3.0, n_instances=10, k=3, master_seed=42)
        self.assertEqual(len(instances), 10)

        for inst in instances:
            result = dpll_solve(inst, max_decisions=10000)
            self.assertIn(result["satisfiable"], [True, False, None])

    def test_hardness_measurement_pipeline(self):
        """Test complete hardness measurement pipeline."""
        instances = generate_instance_batch(n=20, alpha=3.0, n_instances=5, k=3, master_seed=42)
        hardness_values = []
        for inst in instances:
            h = measure_hardness(inst, solver="dpll", max_decisions=10000)
            hardness_values.append(h)

        self.assertEqual(len(hardness_values), 5)
        self.assertTrue(all(h >= 0 for h in hardness_values))

    def test_alpha_sweep_small_scale(self):
        """Test alpha sweep at small scale."""
        ns = [15, 20]
        alphas = np.linspace(3.0, 4.0, 3)

        result = alpha_sweep(
            ns=ns,
            alphas=alphas,
            n_instances=5,
            k=3,
            solver="dpll",
            master_seed=42,
            max_decisions=5000,
            output_dir=self.test_dir,
        )

        self.assertIn("gamma_mean_matrix", result)
        self.assertEqual(result["gamma_mean_matrix"].shape, (len(ns), len(alphas)))

    def test_psat_sweep_small_scale(self):
        """Test P_sat sweep at small scale."""
        ns = [15, 20]
        alphas = np.linspace(3.5, 5.0, 4)

        result = run_psat_sweep(
            ns=ns,
            alphas=alphas,
            n_instances=5,
            k=3,
            master_seed=42,
            solver="dpll",
            output_dir=self.test_dir,
            n_jobs=1,
        )

        self.assertIn("psat_matrix", result)
        self.assertEqual(result["psat_matrix"].shape, (len(ns), len(alphas)))

    def test_exponential_scaling_small_scale(self):
        """Test exponential scaling analysis."""
        ns = [15, 20, 25]
        alphas = np.linspace(3.0, 4.0, 3)


        gamma_matrix = np.random.rand(len(ns), len(alphas)) * 0.01

        result = run_exponential_scaling(
            ns=ns,
            alphas=alphas,
            gamma_matrix=gamma_matrix,
            output_dir=self.test_dir,
        )

        self.assertIn("mean_r2", result)
        self.assertIn("gamma_slope", result)

    def test_fss_collapse_small_scale(self):
        """Test FSS collapse at small scale."""
        alphas = np.linspace(3.5, 5.0, 10)
        ns = [15, 20]


        psat_matrix = np.zeros((len(ns), len(alphas)))
        for i, n in enumerate(ns):
            psat_matrix[i] = 1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (n ** 0.4)))

        result = run_fss_analysis(
            alphas=alphas,
            ns=ns,
            psat_matrix=psat_matrix,
            output_dir=self.test_dir,
        )

        self.assertIn("alpha_s", result)
        self.assertIn("nu", result)
        self.assertGreater(result["nu"], 0.0)

    def test_barrier_scaling_small_scale(self):
        """Test barrier scaling analysis."""
        ns = [10, 15, 20]
        alphas = np.linspace(3.5, 5.0, 10)

        result = run_barrier_scaling_sweep(
            ns=ns,
            alphas=alphas,
            k=3,
            output_dir=self.test_dir,
        )

        self.assertIn("b_curve", result)
        self.assertIn("alpha_peak", result)
        self.assertGreater(result["b_peak"], 0.0)

    def test_barrier_hardness_correlation_pipeline(self):
        """Test barrier-hardness correlation computation."""
        alphas = np.linspace(ALPHA_D + 0.1, ALPHA_S - 0.1, 15)
        gamma_mean = np.random.rand(len(alphas)) * 0.01

        result = barrier_hardness_correlation(alphas, gamma_mean, k=3)

        self.assertIn("correlation", result)
        self.assertIn("p_value", result)
        self.assertGreaterEqual(result["correlation"], -1.0)
        self.assertLessEqual(result["correlation"], 1.0)


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation suite."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()


        import json
        from src.utils import save_json, save_npz


        save_json({
            "thresholds": {"20": 4.25, "30": 4.27},
            "literature_alpha_s": ALPHA_S,
            "literature_alpha_d": ALPHA_D,
        }, f"{self.test_dir}/phase_transition_summary.json")


        save_json({
            "ns": [20, 30],
            "alpha_stars": [4.15, 4.18],
            "gamma_maxima": [0.01, 0.012],
            "alpha_star_inf": 4.20,
            "extrap_r2": 0.95,
        }, f"{self.test_dir}/alpha_sweep_summary.json")


        save_json({
            "mean_r2": 0.90,
            "min_r2": 0.85,
            "max_gamma": 0.015,
        }, f"{self.test_dir}/exponential_scaling_summary.json")


        save_json({
            "alpha_s": 4.27,
            "nu": 2.30,
            "residual": 0.05,
            "converged": True,
        }, f"{self.test_dir}/fss_result.json")


        alphas = np.linspace(3.5, 5.0, 10)
        psat_matrix = np.array([
            1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (20 ** 0.4))),
            1.0 / (1.0 + np.exp(2 * (alphas - 4.2) * (30 ** 0.4))),
        ])
        save_npz(
            f"{self.test_dir}/phase_transition.npz",
            alphas=alphas,
            psat_matrix=psat_matrix,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validation_runs(self):
        """Validation suite should run without errors."""
        result = run_all_checks(results_dir=self.test_dir)

        self.assertIn("passed", result)
        self.assertIn("failed", result)
        self.assertIn("total", result)
        self.assertIn("details", result)

    def test_validation_counts_consistent(self):
        """Passed + failed should equal total."""
        result = run_all_checks(results_dir=self.test_dir)
        self.assertEqual(result["passed"] + result["failed"], result["total"])


class TestDeterminism(unittest.TestCase):
    """Tests for reproducibility and determinism."""

    def test_instance_generation_deterministic(self):
        """Instance generation should be deterministic."""
        instances1 = generate_instance_batch(n=20, alpha=3.0, n_instances=5, k=3, master_seed=42)
        instances2 = generate_instance_batch(n=20, alpha=3.0, n_instances=5, k=3, master_seed=42)

        for i in range(5):
            self.assertEqual(instances1[i]["clauses"], instances2[i]["clauses"])

    def test_dpll_deterministic(self):
        """DPLL solver should be deterministic."""
        from src.instance_generator import generate_ksat_instance
        instance = generate_ksat_instance(20, 3.0, k=3, seed=42)

        result1 = dpll_solve(instance, max_decisions=10000)
        result2 = dpll_solve(instance, max_decisions=10000)

        self.assertEqual(result1["satisfiable"], result2["satisfiable"])
        self.assertEqual(result1["decisions"], result2["decisions"])


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests."""

    def test_small_instance_fast(self):
        """Small instances should solve quickly."""
        import time
        from src.instance_generator import generate_ksat_instance

        instance = generate_ksat_instance(10, 2.0, k=3, seed=42)

        start = time.time()
        result = dpll_solve(instance, max_decisions=10000)
        elapsed = time.time() - start

        self.assertLess(elapsed, 1.0)

    def test_batch_generation_fast(self):
        """Batch generation should be fast."""
        import time

        start = time.time()
        instances = generate_instance_batch(n=15, alpha=3.0, n_instances=20, k=3, master_seed=42)
        elapsed = time.time() - start

        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()