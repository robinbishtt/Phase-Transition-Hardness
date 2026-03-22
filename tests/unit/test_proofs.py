"""Unit tests for src/proofs/.

Covers:
    barrier_bounds.py   — ArrheniusLowerBound, ConflictGraphUpperBound
    runtime_bounds.py   — RuntimeBounds, conjecture4_bounds, verify_theta_scaling
    fss_derivation.py   — FSSAnsatz, fss_threshold_shift
    complexity_functional.py — ComplexityFunctional, compute_sp_complexity
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.energy_model import ALPHA_D, ALPHA_S, ALPHA_STAR, KAPPA, NU, FSS_A, FSS_B
from src.proofs.barrier_bounds import ArrheniusLowerBound, ConflictGraphUpperBound
from src.proofs.complexity_functional import ComplexityFunctional, compute_sp_complexity
from src.proofs.fss_derivation import (
    MANUSCRIPT_FSS,
    FSSAnsatz,
    FSSParameters,
    fss_threshold_shift,
)
from src.proofs.runtime_bounds import (
    RuntimeBounds,
    conjecture4_bounds,
    verify_theta_scaling,
)


class TestArrheniusLowerBound(unittest.TestCase):

    def setUp(self):
        self.lb = ArrheniusLowerBound()

    def test_zero_outside_hard_phase(self):
        self.assertEqual(self.lb.log_T_lower(400, ALPHA_D - 0.1), 0.0)
        self.assertEqual(self.lb.log_T_lower(400, ALPHA_S + 0.1), 0.0)

    def test_positive_inside_hard_phase(self):
        # The barrier term c1·n·b dominates the log correction c2·log(n) only
        # for n > ~1600 with the conservative constants c1=0.80, c2=2.50.
        # At n=400 the bound clamps to 0 (still valid); at n=2000 it is positive.
        lb_n400  = self.lb.log_T_lower(400,  4.20)
        lb_n2000 = self.lb.log_T_lower(2000, 4.20)
        self.assertGreaterEqual(lb_n400,  0.0)          # clamped lower bound
        self.assertGreater(lb_n2000, 0.0)               # strictly positive at large n

    def test_scales_linearly_with_n(self):
        # The Arrhenius lower bound c1*n*b - c2*log(n) clamps at 0 for n < ~1600.
        # Linear scaling is only visible once the bound is strictly positive.
        lb_2000 = self.lb.log_T_lower(2000,  4.20)
        lb_4000 = self.lb.log_T_lower(4000,  4.20)
        lb_8000 = self.lb.log_T_lower(8000,  4.20)
        # At these large n values the bound is strictly positive and increasing.
        self.assertGreater(lb_2000, 0.0)
        self.assertLess(lb_2000, lb_4000)
        self.assertLess(lb_4000, lb_8000)
        # Verify approximate linearity: doubling n should roughly double lb
        ratio = lb_8000 / max(lb_4000, 1e-10)
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 3.0)

    def test_vectorised_curve(self):
        alphas = np.linspace(ALPHA_D + 0.05, ALPHA_S - 0.05, 20)
        curve  = self.lb.lower_bound_curve(200, alphas)
        self.assertEqual(curve.shape, (20,))
        self.assertTrue(np.all(curve >= 0.0))

    def test_custom_constants(self):
        lb_narrow = ArrheniusLowerBound(c1=0.5, c2=1.0)
        lb_wide   = ArrheniusLowerBound(c1=1.5, c2=1.0)
        val_narrow = lb_narrow.log_T_lower(400, 4.20)
        val_wide   = lb_wide.log_T_lower(400, 4.20)
        self.assertGreater(val_wide, val_narrow)

    def test_dominates_polynomial_large_n(self):
        # verify_dominates_polynomial tests c1*n*b > 2*c2*log(n) (2× safety margin).
        # With c1=0.80, c2=2.50, b=0.021, the strict crossover is n ≈ 3000.
        # Below that threshold the method returns False; above it True.
        self.assertFalse(self.lb.verify_dominates_polynomial(400,  4.20))
        self.assertFalse(self.lb.verify_dominates_polynomial(2000, 4.20))
        self.assertTrue( self.lb.verify_dominates_polynomial(4000, 4.20))

    def test_does_not_dominate_outside_hard_phase(self):
        self.assertFalse(self.lb.verify_dominates_polynomial(800, ALPHA_D - 0.1))


class TestConflictGraphUpperBound(unittest.TestCase):

    def setUp(self):
        self.ub = ConflictGraphUpperBound()

    def test_zero_outside_hard_phase(self):
        self.assertEqual(self.ub.log_T_upper(400, ALPHA_D - 0.1), 0.0)
        self.assertEqual(self.ub.log_T_upper(400, ALPHA_S + 0.1), 0.0)

    def test_positive_inside_hard_phase(self):
        ub_val = self.ub.log_T_upper(400, 4.20)
        self.assertGreater(ub_val, 0.0)

    def test_upper_exceeds_lower(self):
        lb = ArrheniusLowerBound()
        for alpha in [3.90, 4.00, 4.10, 4.20, 4.25]:
            lb_val = lb.log_T_lower(400, alpha)
            ub_val = self.ub.log_T_upper(400, alpha)
            self.assertGreaterEqual(
                ub_val + 1e-10, lb_val,
                msg=f"Upper < lower at α={alpha:.2f}: {ub_val:.4f} < {lb_val:.4f}",
            )

    def test_sandwich_ratio_finite_and_positive(self):
        # At n=400 the Arrhenius lower bound clamps to 0, making the ratio
        # formally infinite.  Test at n=2000 where lb > 0.
        ratio = self.ub.sandwich_width(2000, 4.20)
        self.assertGreater(ratio, 1.0)           # upper must exceed lower
        self.assertLess(ratio, 100.0)             # ratio should be O(1) for large n
        self.assertTrue(np.isfinite(ratio))

    def test_sandwich_ratio_infinite_outside_hard_phase(self):
        ratio = self.ub.sandwich_width(400, ALPHA_D - 0.1)
        self.assertEqual(ratio, float("inf"))

    def test_upper_bound_curve(self):
        alphas = np.linspace(ALPHA_D + 0.05, ALPHA_S - 0.05, 15)
        curve  = self.ub.upper_bound_curve(400, alphas)
        self.assertEqual(curve.shape, (15,))
        self.assertTrue(np.all(curve >= 0.0))


class TestRuntimeBounds(unittest.TestCase):

    def setUp(self):
        self.rb = RuntimeBounds()

    def test_evaluate_point_inside_hard_phase(self):
        result = self.rb.evaluate_point(400, 4.20)
        self.assertEqual(result.n, 400)
        self.assertAlmostEqual(result.alpha, 4.20)
        self.assertGreater(result.b_alpha, 0.0)
        # At n=400 the conservative Arrhenius lower bound clamps to 0.
        self.assertGreaterEqual(result.log_T_lower, 0.0)
        self.assertGreater(result.log_T_upper, 0.0)
        self.assertGreater(result.theoretical, 0.0)
        # Test positive lb and finite ratio at n=2000 where barrier term dominates
        result_large = self.rb.evaluate_point(2000, 4.20)
        self.assertGreater(result_large.log_T_lower, 0.0)
        self.assertGreater(result_large.sandwich_ratio, 1.0)

    def test_evaluate_point_outside_hard_phase(self):
        result = self.rb.evaluate_point(400, ALPHA_D - 0.1)
        self.assertEqual(result.b_alpha, 0.0)
        self.assertEqual(result.log_T_lower, 0.0)
        self.assertEqual(result.theoretical, 0.0)

    def test_conjecture_consistent_at_peak(self):
        result = self.rb.evaluate_point(400, ALPHA_STAR)
        self.assertTrue(result.conjecture_consistent)

    def test_evaluate_grid_shape(self):
        ns     = [100, 200, 400]
        alphas = np.linspace(3.9, 4.3, 10)
        report = self.rb.evaluate_grid(ns, alphas)
        self.assertEqual(report.lower_bounds.shape,  (3, 10))
        self.assertEqual(report.upper_bounds.shape,  (3, 10))
        self.assertEqual(report.theoretical.shape,   (3, 10))

    def test_evaluate_grid_sandwich_ratio_finite(self):
        ns     = [200, 400]
        alphas = np.linspace(ALPHA_D + 0.1, ALPHA_S - 0.1, 8)
        report = self.rb.evaluate_grid(ns, alphas)
        self.assertTrue(np.isfinite(report.mean_sandwich_ratio))
        self.assertGreater(report.mean_sandwich_ratio, 1.0)

    def test_conjecture4_bounds_tuple(self):
        lb, theory, ub = conjecture4_bounds(400, 4.20)
        self.assertGreater(theory, 0.0)
        self.assertGreater(ub, theory)
        self.assertLess(lb, theory)

    def test_verify_theta_scaling_is_theta(self):
        result = verify_theta_scaling(
            ns=[100, 200, 400, 800], alpha=ALPHA_STAR, regression_slope=0.0122,
        )
        self.assertIn("b_alpha", result)
        self.assertIn("is_theta_consistent", result)
        self.assertTrue(result["is_theta_consistent"])
        self.assertAlmostEqual(result["b_alpha"], 0.021, delta=0.003)

    def test_kappa_in_report(self):
        ns     = [100, 200]
        alphas = np.linspace(3.9, 4.3, 5)
        report = self.rb.evaluate_grid(ns, alphas)
        self.assertAlmostEqual(report.barrier_growth_exponent_kappa, KAPPA, delta=0.01)
        self.assertAlmostEqual(report.critical_exponent_nu, NU, delta=0.01)


class TestFSSAnsatz(unittest.TestCase):

    def setUp(self):
        self.fss = FSSAnsatz(MANUSCRIPT_FSS)

    def test_alpha_star_inf_at_large_n(self):
        a_star = self.fss.alpha_star_n(100_000)
        self.assertAlmostEqual(a_star, ALPHA_STAR, delta=0.01)

    def test_alpha_star_n_converges_to_infty(self):
        # With B = -1.37 (sub-leading correction), α*(n) lies BELOW α*_∞=4.20
        # for all n in {100,...,800} and converges upward to 4.20 from below.
        # This matches manuscript Table 2: α*(100)=4.18, α*(800)=4.20.
        a100   = self.fss.alpha_star_n(100)
        a800   = self.fss.alpha_star_n(800)
        a10000 = self.fss.alpha_star_n(10000)
        # All values must lie within 0.10 of the thermodynamic limit
        self.assertAlmostEqual(a100,   ALPHA_STAR, delta=0.06)
        self.assertAlmostEqual(a800,   ALPHA_STAR, delta=0.02)
        self.assertAlmostEqual(a10000, ALPHA_STAR, delta=0.005)
        # Convergence must be monotone: larger n → closer to 4.20
        self.assertLess(abs(a800 - ALPHA_STAR), abs(a100 - ALPHA_STAR))

    def test_alpha_star_manuscript_table2(self):
        ms = {100: 4.18, 200: 4.19, 400: 4.20, 800: 4.20}
        for n, a_ms in ms.items():
            a_pred = self.fss.alpha_star_n(n)
            self.assertAlmostEqual(
                a_pred, a_ms, delta=0.01,
                msg=f"α*(n={n}) = {a_pred:.4f}, manuscript = {a_ms:.2f}",
            )

    def test_fss_variable_zero_at_peak(self):
        for n in [100, 200, 400, 800]:
            a_star = self.fss.alpha_star_n(n)
            x      = self.fss.fss_variable(n, a_star)
            self.assertAlmostEqual(x, 0.0, delta=1e-8)

    def test_fss_variable_positive_above_peak(self):
        x = self.fss.fss_variable(400, 4.25)
        self.assertGreater(x, 0.0)

    def test_validate_against_manuscript_all_agree(self):
        v = self.fss.validate_against_manuscript()
        self.assertTrue(v["all_agree"], msg=str(v["agreement"]))

    def test_correlation_length_diverges_at_alpha_d(self):
        xi = self.fss.correlation_length(ALPHA_D + 1e-6)
        self.assertGreater(xi, 1e8)

    def test_correlation_length_finite_away_from_threshold(self):
        xi = self.fss.correlation_length(4.20)
        self.assertTrue(np.isfinite(xi))
        self.assertGreater(xi, 0.0)

    def test_barrier_critical_scaling_positive_above_alpha_d(self):
        b_crit = self.fss.barrier_critical_scaling(ALPHA_D + 0.2)
        self.assertGreater(b_crit, 0.0)

    def test_barrier_critical_scaling_zero_below_alpha_d(self):
        b_crit = self.fss.barrier_critical_scaling(ALPHA_D - 0.1)
        self.assertEqual(b_crit, 0.0)

    def test_predict_alpha_stars_shape(self):
        ns    = [50, 100, 200, 400, 800]
        stars = self.fss.predict_alpha_stars(ns)
        self.assertEqual(len(stars), 5)

    def test_fss_threshold_shift_convenience(self):
        ns    = [100, 200, 400, 800]
        stars = fss_threshold_shift(ns)
        self.assertEqual(len(stars), 4)
        # All values converge to α*_∞=4.20 from below (B=-1.37 dominates at small n).
        # This matches Table 2 of the manuscript.
        for a_star in stars:
            self.assertAlmostEqual(a_star, ALPHA_STAR, delta=0.06)
        # Monotone convergence: each larger n is closer to 4.20
        dists = [abs(a - ALPHA_STAR) for a in stars]
        for i in range(len(dists) - 1):
            self.assertLessEqual(
                dists[i + 1], dists[i] + 1e-8,
                msg=f"α*(n={ns[i+1]}) further from 4.20 than α*(n={ns[i]})",
            )


class TestComplexityFunctional(unittest.TestCase):

    def setUp(self):
        self.cf = ComplexityFunctional()

    def test_sigma_zero_at_alpha_d(self):
        self.assertAlmostEqual(self.cf.sigma(ALPHA_D), 0.0, delta=1e-4)

    def test_sigma_zero_at_alpha_s(self):
        self.assertAlmostEqual(self.cf.sigma(ALPHA_S), 0.0, delta=1e-4)

    def test_sigma_positive_in_hard_phase(self):
        for alpha in [3.9, 4.0, 4.1, 4.2, 4.25]:
            self.assertGreater(
                self.cf.sigma(alpha), 0.0,
                msg=f"Σ({alpha}) should be positive",
            )

    def test_sigma_at_alpha_star_matches_manuscript(self):
        # Manuscript Section 5.2: Σ(4.20) ≈ 0.027 (PRG stretch bound)
        sig = self.cf.sigma(ALPHA_STAR)
        self.assertAlmostEqual(sig, 0.027, delta=0.005)

    def test_sigma_max_matches_manuscript(self):
        # Manuscript Figure S9: Σ_max ≈ 0.047 near α ≈ 4.05
        vals  = [self.cf.sigma(a) for a in np.linspace(ALPHA_D + 0.01, ALPHA_S - 0.01, 200)]
        sigma_max = max(vals)
        self.assertAlmostEqual(sigma_max, 0.047, delta=0.01)

    def test_sigma_curve_same_as_pointwise(self):
        alphas = np.linspace(3.9, 4.3, 10)
        curve  = self.cf.sigma_curve(alphas)
        for i, a in enumerate(alphas):
            self.assertAlmostEqual(curve[i], self.cf.sigma(a), places=10)

    def test_validate_key_values_returns_dict(self):
        kv = self.cf.validate_key_values()
        self.assertIn("sigma_at_alpha_d", kv)
        self.assertIn("sigma_at_alpha_s", kv)
        self.assertIn("sigma_at_alpha_star", kv)
        self.assertIn("sigma_max", kv)
        self.assertAlmostEqual(kv["sigma_at_alpha_d"], 0.0, delta=1e-4)
        self.assertAlmostEqual(kv["sigma_at_alpha_s"], 0.0, delta=1e-4)

    def test_edge_normalisation_finite(self):
        val = ComplexityFunctional.edge_normalisation(
            z_edge=0.8, z_clause=0.9, z_var=0.7, k=3, degree_var=12.6,
        )
        self.assertTrue(np.isfinite(val))

    def test_compute_sp_complexity_analytical_consistent(self):
        for alpha in [3.9, 4.0, 4.1, 4.2, 4.25]:
            val = compute_sp_complexity(alpha, k=3, use_analytical=True)
            self.assertGreaterEqual(val, 0.0)
            self.assertAlmostEqual(val, self.cf.sigma(alpha), places=10)

    def test_compute_sp_complexity_numerical_raises(self):
        with self.assertRaises(NotImplementedError):
            compute_sp_complexity(4.20, use_analytical=False)


if __name__ == "__main__":
    unittest.main()
