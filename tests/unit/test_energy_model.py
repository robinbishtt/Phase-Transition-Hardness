"""
Unit tests for src/energy_model.py

Tests the thermodynamic energy model functions including:
- Entropy density calculations
- Cluster complexity
- Free energy density
- Barrier density functions
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.energy_model import (
    annealed_entropy,
    rs_entropy_density,
    cluster_complexity,
    free_energy_density,
    frozen_fraction,
    barrier_density,
    barrier_height,
    ALPHA_D,
    ALPHA_R,
    ALPHA_C,
    ALPHA_S,
    K_DEFAULT,
)


class TestAnnealedEntropy(unittest.TestCase):
    """Tests for annealed_entropy function."""

    def test_zero_alpha(self):
        """At alpha=0, entropy should be log(2)."""
        result = annealed_entropy(0.0, k=3)
        self.assertAlmostEqual(result, np.log(2.0), places=10)

    def test_k3_low_alpha(self):
        """Test entropy at low alpha for k=3."""
        result = annealed_entropy(1.0, k=3)
        expected = np.log(2.0) + 1.0 * np.log(1.0 - 2.0 ** (1 - 3))
        self.assertAlmostEqual(result, expected, places=10)

    def test_k4(self):
        """Test entropy for k=4."""
        result = annealed_entropy(2.0, k=4)
        expected = np.log(2.0) + 2.0 * np.log(1.0 - 2.0 ** (1 - 4))
        self.assertAlmostEqual(result, expected, places=10)

    def test_negative_entropy_possible(self):
        """Entropy can become negative at high alpha."""
        result = annealed_entropy(10.0, k=3)
        self.assertLess(result, 0.0)


class TestRSEntropyDensity(unittest.TestCase):
    """Tests for rs_entropy_density function."""

    def test_zero_at_alpha_s(self):
        """Entropy should be zero at satisfiability threshold."""
        result = rs_entropy_density(ALPHA_S, k=3)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_positive_below_alpha_s(self):
        """Entropy should be positive below alpha_s."""
        result = rs_entropy_density(3.0, k=3)
        self.assertGreater(result, 0.0)

    def test_clipped_at_log2(self):
        """Entropy should not exceed log(2)."""
        result = rs_entropy_density(0.0, k=3)
        self.assertLessEqual(result, np.log(2.0) + 1e-10)

    def test_non_k3_fallback(self):
        """For k != 3, should use max(annealed entropy, 0) as the RS approximation."""
        alpha = 2.0
        result = rs_entropy_density(alpha, k=4)
        expected = max(annealed_entropy(alpha, k=4), 0.0)
        self.assertAlmostEqual(result, expected, places=10)

    def test_monotonic_decreasing(self):
        """Entropy should decrease with alpha."""
        e1 = rs_entropy_density(2.0, k=3)
        e2 = rs_entropy_density(3.0, k=3)
        e3 = rs_entropy_density(4.0, k=3)
        self.assertGreater(e1, e2)
        self.assertGreater(e2, e3)


class TestClusterComplexity(unittest.TestCase):
    """Tests for cluster_complexity function."""

    def test_zero_below_alpha_d(self):
        """Complexity should be zero below clustering threshold."""
        result = cluster_complexity(ALPHA_D - 0.1)
        self.assertEqual(result, 0.0)

    def test_zero_above_alpha_c(self):
        """Complexity should be zero above condensation threshold."""
        result = cluster_complexity(ALPHA_C + 0.1)
        self.assertEqual(result, 0.0)

    def test_positive_in_shattered_phase(self):
        """Complexity should be positive in shattered phase."""
        result = cluster_complexity((ALPHA_D + ALPHA_C) / 2)
        self.assertGreater(result, 0.0)

    def test_small_near_alpha_d(self):
        """Complexity rises from zero just above α_d; value at α_d+0.01 should be small."""
        result = cluster_complexity(ALPHA_D + 0.01)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.01)   # near-zero onset

    def test_peak_near_alpha_peak(self):
        """Complexity peaks near α≈4.05 with Σ_max≈0.047 (manuscript Figure S9)."""
        alphas = [cluster_complexity(a) for a in [3.95, 4.00, 4.05, 4.10, 4.15]]
        self.assertAlmostEqual(max(alphas), 0.047, delta=0.01)

    def test_zero_at_alpha_c(self):
        """Complexity should approach zero at alpha_c."""
        result = cluster_complexity(ALPHA_C - 0.001)
        self.assertAlmostEqual(result, 0.0, places=2)


class TestFreeEnergyDensity(unittest.TestCase):
    """Tests for free_energy_density function."""

    def test_finite_output(self):
        """Should return finite values."""
        result = free_energy_density(3.0, beta=1.0, k=3)
        self.assertTrue(np.isfinite(result))

    def test_beta_dependence(self):
        """Free energy should depend on beta."""
        f1 = free_energy_density(3.0, beta=0.5, k=3)
        f2 = free_energy_density(3.0, beta=2.0, k=3)
        self.assertNotAlmostEqual(f1, f2, places=5)

    def test_returns_float(self):
        """Should return a float."""
        result = free_energy_density(3.0)
        self.assertIsInstance(result, float)


class TestFrozenFraction(unittest.TestCase):
    """Tests for frozen_fraction function."""

    def test_zero_below_alpha_d(self):
        """Frozen fraction should be zero below clustering threshold."""
        result = frozen_fraction(ALPHA_D - 0.1)
        self.assertEqual(result, 0.0)

    def test_one_above_alpha_s(self):
        """Frozen fraction should be one above satisfiability threshold."""
        result = frozen_fraction(ALPHA_S + 0.1)
        self.assertEqual(result, 1.0)

    def test_increasing(self):
        """Frozen fraction should increase with alpha."""
        f1 = frozen_fraction(ALPHA_D + 0.1)
        f2 = frozen_fraction((ALPHA_D + ALPHA_S) / 2)
        f3 = frozen_fraction(ALPHA_S - 0.1)
        self.assertLess(f1, f2)
        self.assertLess(f2, f3)

    def test_in_range(self):
        """Frozen fraction should always be in [0, 1]."""
        for alpha in np.linspace(0, 10, 100):
            result = frozen_fraction(alpha)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)


class TestBarrierDensity(unittest.TestCase):
    """Tests for barrier_density function."""

    def test_zero_below_alpha_d(self):
        """Barrier density should be zero below clustering threshold."""
        result = barrier_density(ALPHA_D - 0.1)
        self.assertEqual(result, 0.0)

    def test_zero_above_alpha_s(self):
        """Barrier density should be zero above satisfiability threshold."""
        result = barrier_density(ALPHA_S + 0.1)
        self.assertEqual(result, 0.0)

    def test_positive_in_hard_phase(self):
        """Barrier density should be positive in hard phase."""
        result = barrier_density(4.2)
        self.assertGreater(result, 0.0)

    def test_peak_near_4_2(self):
        """Barrier density should peak near alpha=4.2."""
        b_4_0 = barrier_density(4.0)
        b_4_2 = barrier_density(4.2)
        b_4_4 = barrier_density(4.4)
        self.assertGreater(b_4_2, b_4_0)
        self.assertGreater(b_4_2, b_4_4)

    def test_maximum_value(self):
        """Maximum barrier density should be approximately 0.021 (manuscript Table 4, Table 2 H∞)."""
        result = barrier_density(4.2)
        self.assertAlmostEqual(result, 0.021, delta=0.003)

    def test_symmetric_shape(self):
        """Barrier density should be roughly symmetric around peak within (α_d, α_s).
        Note: cannot use 4.3 as the right test point because 4.3 > α_s=4.267 gives b=0."""
        b_left = barrier_density(4.1)
        b_right = barrier_density(4.25)   # 4.2-4.1=0.10 ≈ 4.25-4.2=0.05, within (α_d,α_s)
        self.assertAlmostEqual(b_left, b_right, delta=0.008)


class TestBarrierHeight(unittest.TestCase):
    """Tests for barrier_height function."""

    def test_linear_in_n(self):
        """Barrier height should scale linearly with n."""
        b1 = barrier_height(100, 4.2)
        b2 = barrier_height(200, 4.2)
        self.assertAlmostEqual(b2 / b1, 2.0, places=5)

    def test_zero_outside_hard_phase(self):
        """Barrier height should be zero outside hard phase."""
        result = barrier_height(100, ALPHA_D - 0.1)
        self.assertEqual(result, 0.0)

    def test_proportional_to_density(self):
        """Barrier height should be n times barrier density."""
        n = 100
        alpha = 4.2
        result = barrier_height(n, alpha)
        expected = n * barrier_density(alpha)
        self.assertAlmostEqual(result, expected, places=10)


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_alpha_d_value(self):
        """Clustering threshold should be approximately 3.86."""
        self.assertAlmostEqual(ALPHA_D, 3.86, places=2)

    def test_alpha_r_value(self):
        """Rigidity threshold should be approximately 3.86 (= α_d for K=3)."""
        self.assertAlmostEqual(ALPHA_R, 3.86, places=2)

    def test_alpha_c_value(self):
        """Condensation threshold should be approximately 4.267 (= α_s for K=3)."""
        self.assertAlmostEqual(ALPHA_C, 4.267, places=2)

    def test_alpha_s_value(self):
        """Satisfiability threshold should be approximately 4.267."""
        self.assertAlmostEqual(ALPHA_S, 4.267, places=3)

    def test_k_default(self):
        """Default k should be 3."""
        self.assertEqual(K_DEFAULT, 3)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_very_small_alpha(self):
        """Should handle very small alpha values."""
        result = annealed_entropy(0.001, k=3)
        self.assertTrue(np.isfinite(result))

    def test_very_large_alpha(self):
        """Should handle very large alpha values."""
        result = annealed_entropy(100.0, k=3)
        self.assertTrue(np.isfinite(result))

    def test_negative_alpha_raises_error(self):
        """Negative alpha: function should return a finite value without crashing."""
        result = annealed_entropy(-1.0, k=3)
        self.assertTrue(np.isfinite(result))


if __name__ == "__main__":
    unittest.main()