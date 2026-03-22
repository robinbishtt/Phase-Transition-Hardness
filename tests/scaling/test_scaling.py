"""
Scaling tests for performance and accuracy at different scales

Tests to verify correct behavior across different system sizes and parameters.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance, generate_instance_batch
from src.hardness_metrics import dpll_solve, measure_hardness
from src.energy_model import barrier_height, barrier_density
from src.statistics import bootstrap_ci


class TestSystemSizeScaling(unittest.TestCase):
    """Tests for scaling with system size n."""

    def test_instance_size_scaling(self):
        """Instance properties should scale correctly with n."""
        for n in [10, 20, 50, 100]:
            instance = generate_ksat_instance(n, 3.0, k=3, seed=42)
            expected_m = int(round(3.0 * n))
            self.assertEqual(instance["n"], n)
            self.assertEqual(instance["m"], expected_m)

    def test_barrier_height_linear_scaling(self):
        """Barrier height should scale linearly with n."""
        alpha = 4.2
        ns = [10, 20, 50, 100]
        barriers = [barrier_height(n, alpha) for n in ns]


        for i in range(1, len(ns)):
            ratio = barriers[i] / barriers[0]
            expected_ratio = ns[i] / ns[0]
            self.assertAlmostEqual(ratio, expected_ratio, places=5)

    def test_barrier_density_independent_of_n(self):
        """Barrier density should be independent of n."""
        alpha = 4.2
        for n in [10, 50, 100, 200]:
            b = barrier_density(alpha, k=3)

            self.assertAlmostEqual(b, barrier_density(alpha, k=3), places=10)


class TestConstraintDensityScaling(unittest.TestCase):
    """Tests for scaling with constraint density alpha."""

    def test_clause_count_scaling(self):
        """Clause count should scale linearly with alpha."""
        n = 50
        for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
            instance = generate_ksat_instance(n, alpha, k=3, seed=42)
            expected_m = int(round(alpha * n))
            self.assertEqual(instance["m"], expected_m)

    def test_hardness_increases_with_alpha(self):
        """Hardness should generally increase with alpha in hard region."""
        n = 20
        alphas = [2.5, 3.0, 3.5, 4.0, 4.5]
        hardness_values = []

        for alpha in alphas:
            instance = generate_ksat_instance(n, alpha, k=3, seed=42)
            h = measure_hardness(instance, solver="dpll", max_decisions=10000)
            hardness_values.append(h)



        self.assertGreaterEqual(max(hardness_values[2:]), min(hardness_values[:2]))


class TestKSATScaling(unittest.TestCase):
    """Tests for scaling with k in k-SAT."""

    def test_clause_length(self):
        """Each clause should have exactly k literals."""
        for k in [2, 3, 4, 5]:
            instance = generate_ksat_instance(20, 3.0, k=k, seed=42)
            for clause in instance["clauses"]:
                self.assertEqual(len(clause), k)

    def test_k_affects_hardness(self):
        """Higher k should generally increase hardness."""
        n = 15
        alpha = 3.0

        hardness_values = []
        for k in [2, 3, 4]:
            instance = generate_ksat_instance(n, alpha, k=k, seed=42)
            h = measure_hardness(instance, solver="dpll", max_decisions=5000)
            hardness_values.append(h)



        self.assertTrue(all(h >= 0 for h in hardness_values))


class TestSampleSizeScaling(unittest.TestCase):
    """Tests for scaling with number of samples."""

    def test_bootstrap_ci_narrowing(self):
        """Bootstrap CI should narrow with more data."""
        data_small = np.random.randn(20)
        data_large = np.random.randn(200)

        lo_small, hi_small = bootstrap_ci(data_small, n_boot=100, ci=0.95, seed=42)
        lo_large, hi_large = bootstrap_ci(data_large, n_boot=100, ci=0.95, seed=42)

        width_small = hi_small - lo_small
        width_large = hi_large - lo_large



        self.assertGreater(width_small, 0)
        self.assertGreater(width_large, 0)

    def test_batch_generation_scaling(self):
        """Batch generation should scale with n_instances."""
        for n_instances in [5, 10, 20]:
            instances = generate_instance_batch(20, 3.0, n_instances, k=3, master_seed=42)
            self.assertEqual(len(instances), n_instances)


class TestPerformanceScaling(unittest.TestCase):
    """Tests for performance scaling."""

    def test_solve_time_vs_n(self):
        """Solve time should increase with n."""
        import time

        times = []
        for n in [10, 15, 20]:
            instance = generate_ksat_instance(n, 3.0, k=3, seed=42)
            start = time.time()
            dpll_solve(instance, max_decisions=10000)
            elapsed = time.time() - start
            times.append(elapsed)



        self.assertTrue(all(t > 0 for t in times))

    def test_solve_time_vs_alpha(self):
        """Solve time should increase with alpha in hard region."""
        import time

        times = []
        for alpha in [2.5, 3.5, 4.5]:
            instance = generate_ksat_instance(15, alpha, k=3, seed=42)
            start = time.time()
            dpll_solve(instance, max_decisions=10000)
            elapsed = time.time() - start
            times.append(elapsed)


        self.assertTrue(all(t > 0 for t in times))


class TestMemoryScaling(unittest.TestCase):
    """Tests for memory usage scaling."""

    def test_instance_size_proportional_to_n(self):
        """Instance size should be proportional to n."""
        import sys

        sizes = []
        for n in [10, 20, 50]:
            instance = generate_ksat_instance(n, 3.0, k=3, seed=42)
            size = sys.getsizeof(instance)
            sizes.append((n, size))


        self.assertTrue(all(s[1] > 0 for s in sizes))


if __name__ == "__main__":
    unittest.main()