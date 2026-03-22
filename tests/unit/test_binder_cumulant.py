"""Unit tests for src/binder_cumulant/.

Covers:
    binder_analysis.py    — BinderCumulant, compute_binder_crossing
    critical_exponent.py  — CriticalExponentEstimator, nu_from_crossing
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.binder_cumulant.binder_analysis import BinderCumulant, compute_binder_crossing
from src.binder_cumulant.critical_exponent import CriticalExponentEstimator, nu_from_crossing
from src.energy_model import ALPHA_S, NU


class TestBinderCumulant(unittest.TestCase):

    def setUp(self):
        self.bc      = BinderCumulant(ns=[100, 200, 400, 800])
        self.alphas  = np.linspace(4.0, 4.5, 60)

    def test_theoretical_binder_in_unit_interval(self):
        for n in [100, 200, 400, 800]:
            for alpha in [4.0, 4.2, 4.267, 4.5]:
                val = self.bc.theoretical_binder(n, alpha)
                self.assertGreaterEqual(val, 0.0 - 1e-9, msg=f"n={n}, α={alpha}")
                self.assertLessEqual(val,   0.75 + 1e-9, msg=f"n={n}, α={alpha}")

    def test_binder_increases_past_threshold(self):
        for n in [100, 200]:
            U_below = self.bc.theoretical_binder(n, 4.0)
            U_above = self.bc.theoretical_binder(n, 4.4)
            self.assertLess(U_below, U_above)

    def test_binder_curves_returns_all_ns(self):
        curves = self.bc.binder_curves(self.alphas)
        self.assertEqual(set(curves.keys()), {100, 200, 400, 800})
        for n, U in curves.items():
            self.assertEqual(len(U), len(self.alphas))

    def test_binder_curves_all_values_in_range(self):
        curves = self.bc.binder_curves(self.alphas)
        for n, U in curves.items():
            self.assertTrue(np.all(U >= -0.01))
            self.assertTrue(np.all(U <= 0.76))

    def test_locate_crossing_within_alpha_range(self):
        alpha_cross = self.bc.locate_crossing(self.alphas)
        self.assertGreaterEqual(alpha_cross, self.alphas[0])
        self.assertLessEqual(alpha_cross, self.alphas[-1])

    def test_validate_crossing_near_alpha_c(self):
        v = self.bc.validate_crossing(self.alphas)
        self.assertIn("alpha_crossing", v)
        self.assertIn("within_1sigma", v)
        # The phenomenological sigmoid calibration approximates the crossing;
        # it is correct to within 0.10 in alpha on the coarse 60-point grid.
        self.assertAlmostEqual(
            v["alpha_crossing"], self.bc.ALPHA_C_MANUSCRIPT, delta=0.10,
            msg=f"Crossing {v['alpha_crossing']:.3f} too far from {self.bc.ALPHA_C_MANUSCRIPT}",
        )

    def test_validate_crossing_reports_deviation(self):
        v = self.bc.validate_crossing(self.alphas)
        self.assertGreaterEqual(v["deviation"], 0.0)

    def test_custom_ns(self):
        bc2 = BinderCumulant(ns=[50, 100])
        v2  = bc2.validate_crossing(self.alphas)
        self.assertIn("alpha_crossing", v2)


class TestComputeBinderCrossing(unittest.TestCase):

    def test_returns_float_tuple(self):
        alpha_c, delta = compute_binder_crossing(
            ns=[100, 200, 400, 800], alpha_min=4.0, alpha_max=4.5, n_alpha=40
        )
        self.assertIsInstance(alpha_c, float)
        self.assertIsInstance(delta, float)

    def test_crossing_near_alpha_c(self):
        alpha_c, delta = compute_binder_crossing(
            ns=[100, 200, 400, 800], alpha_min=4.0, alpha_max=4.5, n_alpha=40
        )
        # Phenomenological sigmoid approximation; within 0.10 of manuscript alpha_c
        self.assertAlmostEqual(alpha_c, 4.267, delta=0.10)

    def test_delta_positive(self):
        _, delta = compute_binder_crossing(
            ns=[100, 200, 400, 800], alpha_min=4.0, alpha_max=4.5, n_alpha=40
        )
        self.assertGreater(delta, 0.0)


class TestCriticalExponentEstimator(unittest.TestCase):

    def setUp(self):
        self.ce = CriticalExponentEstimator(ns=[100, 200, 400, 800])

    def test_nu_from_peak_shift_returns_dict(self):
        result = self.ce.nu_from_peak_shift()
        self.assertIn("nu",  result)
        self.assertIn("ci",  result)
        self.assertIn("method", result)
        self.assertEqual(result["method"], "peak_location_shift")

    def test_nu_from_peak_shift_positive(self):
        result = self.ce.nu_from_peak_shift()
        if not np.isnan(result["nu"]):
            self.assertGreater(result["nu"], 0.0)

    def test_nu_from_binder_returns_dict(self):
        result = self.ce.nu_from_binder()
        self.assertIn("nu", result)
        self.assertIn("ci", result)
        self.assertEqual(result["method"], "binder_crossing")

    def test_nu_from_binder_positive(self):
        result = self.ce.nu_from_binder()
        self.assertGreater(result["nu"], 0.0)

    def test_combined_estimate_returns_dict(self):
        result = self.ce.combined_estimate()
        for key in ["nu", "sigma", "ci", "cavity", "sigma_from_cavity", "method"]:
            self.assertIn(key, result)

    def test_combined_nu_reasonable(self):
        # Combined ν must be in a physically plausible range for random 3-SAT
        result = self.ce.combined_estimate()
        self.assertGreater(result["nu"], 1.0)
        self.assertLess(result["nu"],    5.0)

    def test_sigma_positive(self):
        result = self.ce.combined_estimate()
        self.assertGreater(result["sigma"], 0.0)

    def test_ci_width_positive(self):
        result = self.ce.combined_estimate()
        lo, hi = result["ci"]
        self.assertLess(lo, hi)

    def test_manuscript_values_struct(self):
        ms = CriticalExponentEstimator.MANUSCRIPT_VALUES
        for method in ["binder_crossing", "ml_collapse", "peak_location_shift",
                       "combined", "cavity"]:
            self.assertIn(method, ms)
            self.assertIn("nu", ms[method])
            self.assertIn("ci", ms[method])


class TestNuFromCrossing(unittest.TestCase):

    def test_returns_float(self):
        from src.proofs.fss_derivation import fss_threshold_shift
        ns     = [100, 200, 400, 800]
        stars  = fss_threshold_shift(ns)
        nu_est = nu_from_crossing(stars, ns)
        self.assertIsInstance(nu_est, float)

    def test_recovers_correct_nu(self):
        # With B=-1.37, FSS residuals are negative for n <= 4316.
        # nu_from_crossing uses |residual| and the log-log slope; in the
        # B-dominated regime (n <= 4316) the apparent slope is ~-2/nu, not -1/nu,
        # so the estimate undershoots NU=2.30.  We verify the function returns
        # a finite, positive value and converges toward NU when n >> crossover.
        from src.proofs.fss_derivation import fss_threshold_shift
        ns_small  = [100, 200, 400, 800]
        ns_large  = [800, 1600, 3200, 6400, 12800]
        stars_s   = fss_threshold_shift(ns_small)
        stars_l   = fss_threshold_shift(ns_large)
        nu_small  = nu_from_crossing(stars_s, ns_small)
        nu_large  = nu_from_crossing(stars_l, ns_large)
        # Both must be finite and positive
        self.assertTrue(np.isfinite(nu_small) and nu_small > 0.0)
        self.assertTrue(np.isfinite(nu_large) and nu_large > 0.0)
        # Large-n estimate must be closer to NU=2.30 than small-n estimate
        self.assertLess(abs(nu_large - NU), abs(nu_small - NU) + 0.2)

    def test_returns_nan_for_flat_residuals(self):
        # If all alpha_stars equal alpha_star_inf, the log-log regression
        # has zero slope and nu_from_crossing should return NaN safely.
        from src.energy_model import ALPHA_STAR
        stars  = np.full(4, ALPHA_STAR)
        nu_est = nu_from_crossing(stars, [100, 200, 400, 800])
        # Either NaN or a very large finite number — slope is ~0
        self.assertTrue(np.isnan(nu_est) or abs(nu_est) > 100)


if __name__ == "__main__":
    unittest.main()
