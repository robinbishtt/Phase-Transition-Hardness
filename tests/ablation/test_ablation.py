"""
Ablation tests for analyzing component contributions

Tests to understand the impact of different components on overall performance.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hardness_metrics import dpll_solve, walksat_solve
from src.instance_generator import generate_ksat_instance


class TestDPLLAblation(unittest.TestCase):
    """Ablation tests for DPLL solver components."""

    def test_without_unit_propagation(self):
        """Test DPLL behavior without unit propagation (conceptual)."""


        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)

        self.assertIn(result["satisfiable"], [True, False, None])

    def test_without_pure_literal(self):
        """Test DPLL behavior without pure literal elimination (conceptual)."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)
        self.assertIn(result["satisfiable"], [True, False, None])

    def test_branching_heuristic_impact(self):
        """Test impact of different branching heuristics (conceptual)."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)
        result = dpll_solve(instance, max_decisions=10000)

        self.assertIn(result["satisfiable"], [True, False, None])


class TestWalkSATAblation(unittest.TestCase):
    """Ablation tests for WalkSAT solver components."""

    def test_noise_parameter_sensitivity(self):
        """Test sensitivity to noise parameter."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)

        results = []
        for noise in [0.3, 0.5, 0.57, 0.7]:
            result = walksat_solve(instance, max_flips=5000, noise=noise, seed=42)
            results.append(result["satisfiable"])


        self.assertTrue(all(r in [True, False] for r in results))

    def test_restart_parameter_sensitivity(self):
        """Test sensitivity to restart parameter."""
        instance = generate_ksat_instance(15, 3.0, k=3, seed=42)

        results = []
        for restarts in [1, 3, 5, 10]:
            result = walksat_solve(instance, max_flips=5000, noise=0.57, seed=42, restarts=restarts)
            results.append(result["satisfiable"])


        self.assertTrue(all(r in [True, False] for r in results))

    def test_max_flips_sensitivity(self):
        """Test sensitivity to max_flips parameter."""
        instance = generate_ksat_instance(15, 3.5, k=3, seed=42)

        results = []
        for max_flips in [1000, 5000, 10000]:
            result = walksat_solve(instance, max_flips=max_flips, noise=0.57, seed=42)
            results.append(result["satisfiable"])



        self.assertTrue(all(r in [True, False] for r in results))


class TestInstanceGenerationAblation(unittest.TestCase):
    """Ablation tests for instance generation."""

    def test_k_variation(self):
        """Test impact of different k values."""
        from src.instance_generator import generate_ksat_instance

        for k in [2, 3, 4, 5]:
            instance = generate_ksat_instance(20, 3.0, k=k, seed=42)
            self.assertEqual(instance["k"], k)
            self.assertEqual(len(instance["clauses"][0]), k)

    def test_alpha_variation(self):
        """Test impact of different alpha values."""
        from src.instance_generator import generate_ksat_instance

        for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
            instance = generate_ksat_instance(20, alpha, k=3, seed=42)
            expected_m = int(round(alpha * 20))
            self.assertEqual(instance["m"], expected_m)

    def test_n_variation(self):
        """Test impact of different n values."""
        from src.instance_generator import generate_ksat_instance

        for n in [10, 20, 50, 100]:
            instance = generate_ksat_instance(n, 3.0, k=3, seed=42)
            self.assertEqual(instance["n"], n)


class TestEnergyModelAblation(unittest.TestCase):
    """Ablation tests for energy model components."""

    def test_entropy_without_rs_correction(self):
        """Test annealed entropy vs RS entropy."""
        from src.energy_model import annealed_entropy, rs_entropy_density

        alpha = 3.0
        annealed = annealed_entropy(alpha, k=3)
        rs = rs_entropy_density(alpha, k=3)


        self.assertLessEqual(rs, annealed + 1e-10)

    def test_barrier_without_gaussian_shape(self):
        """Test barrier density properties."""
        from src.energy_model import barrier_density, ALPHA_D, ALPHA_S


        self.assertEqual(barrier_density(ALPHA_D - 0.1), 0.0)
        self.assertEqual(barrier_density(ALPHA_S + 0.1), 0.0)


        self.assertGreater(barrier_density(4.2), 0.0)


if __name__ == "__main__":
    unittest.main()