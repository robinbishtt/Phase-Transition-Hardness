"""Unit tests for src/cryptography/.

Covers:
    one_way_function.py   — GoldreichOWF, owf_security_analysis
    proof_of_work.py      — KSATProofOfWork, pow_difficulty_parameter
    prg_construction.py   — APKPseudoRandomGenerator
    security_parameters.py — SecurityParameterTable, compute_security_bits
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cryptography.one_way_function import GoldreichOWF, owf_security_analysis
from src.cryptography.proof_of_work import KSATProofOfWork, pow_difficulty_parameter
from src.cryptography.prg_construction import APKPseudoRandomGenerator
from src.cryptography.security_parameters import SecurityParameterTable, compute_security_bits
from src.energy_model import ALPHA_D, ALPHA_S, ALPHA_STAR


class TestGoldreichOWF(unittest.TestCase):

    def setUp(self):
        self.owf = GoldreichOWF(n=20, alpha=4.20, seed=42)

    def test_rejects_easy_alpha(self):
        with self.assertRaises(ValueError):
            GoldreichOWF(n=20, alpha=ALPHA_D - 0.1)

    def test_rejects_unsat_alpha(self):
        with self.assertRaises(ValueError):
            GoldreichOWF(n=20, alpha=ALPHA_S + 0.1)

    def test_evaluate_returns_binary_list(self):
        x = {i: bool(i % 2) for i in range(1, 21)}
        y = self.owf.evaluate(x)
        self.assertEqual(len(y), self.owf.m)
        for bit in y:
            self.assertIn(bit, [0, 1])

    def test_evaluate_deterministic(self):
        x = {i: (i % 3 == 0) for i in range(1, 21)}
        y1 = self.owf.evaluate(x)
        y2 = self.owf.evaluate(x)
        self.assertEqual(y1, y2)

    def test_preimage_check_correct(self):
        x = {i: True for i in range(1, 21)}
        y = self.owf.evaluate(x)
        self.assertTrue(self.owf.is_preimage(x, y))

    def test_preimage_check_rejects_wrong(self):
        x = {i: True  for i in range(1, 21)}
        z = {i: False for i in range(1, 21)}
        y = self.owf.evaluate(x)
        self.assertFalse(self.owf.is_preimage(z, y))

    def test_security_bits_match_table6(self):
        # n=400: Table 6 reports ~40 bits
        owf_400 = GoldreichOWF(n=400, alpha=4.20, seed=42)
        self.assertAlmostEqual(owf_400.security_bits(), 40.0, delta=2.0)

    def test_security_bits_scale_with_n(self):
        s_200 = GoldreichOWF(n=200, alpha=4.20, seed=42).security_bits()
        s_400 = GoldreichOWF(n=400, alpha=4.20, seed=42).security_bits()
        s_800 = GoldreichOWF(n=800, alpha=4.20, seed=42).security_bits()
        self.assertLess(s_200, s_400)
        self.assertLess(s_400, s_800)

    def test_security_analysis_fields(self):
        analysis = self.owf.security_analysis()
        required = {"n", "alpha", "b_alpha", "cluster_complexity",
                    "security_bits", "is_in_hard_window", "hardness_maximised"}
        for field in required:
            self.assertIn(field, analysis)

    def test_security_analysis_in_hard_window(self):
        analysis = self.owf.security_analysis()
        self.assertTrue(analysis["is_in_hard_window"])

    def test_security_analysis_hardness_maximised_at_alpha_star(self):
        owf_peak = GoldreichOWF(n=20, alpha=ALPHA_STAR, seed=42)
        self.assertTrue(owf_peak.security_analysis()["hardness_maximised"])


class TestOWFSecurityAnalysis(unittest.TestCase):

    def test_returns_list(self):
        results = owf_security_analysis(ns=[400, 600, 800])
        self.assertEqual(len(results), 3)

    def test_table6_values(self):
        results = owf_security_analysis(ns=[400, 600, 800], alpha=4.20)
        targets = {400: 40, 600: 60, 800: 80}
        for row in results:
            n   = row["n"]
            exp = targets[n]
            self.assertAlmostEqual(row["security_bits"], exp, delta=2.0,
                                   msg=f"n={n}: {row['security_bits']:.1f} != {exp}")

    def test_labels_correct(self):
        results = owf_security_analysis(ns=[400, 600, 800])
        labels  = {r["n"]: r["security_label"] for r in results}
        self.assertEqual(labels[400], "Basic")
        self.assertEqual(labels[600], "Standard")
        self.assertEqual(labels[800], "High")


class TestKSATProofOfWork(unittest.TestCase):

    def setUp(self):
        self.pow = KSATProofOfWork(n=20, alpha=4.20, k=3, master_seed=42)

    def test_rejects_easy_alpha(self):
        with self.assertRaises(ValueError):
            KSATProofOfWork(n=20, alpha=ALPHA_D - 0.1)

    def test_generate_puzzle_fields(self):
        puzzle = self.pow.generate_puzzle(nonce=0)
        for field in ["instance", "challenge", "nonce", "n", "alpha", "difficulty"]:
            self.assertIn(field, puzzle)

    def test_challenge_is_hex_string(self):
        puzzle = self.pow.generate_puzzle(nonce=0)
        self.assertEqual(len(puzzle["challenge"]), 64)
        int(puzzle["challenge"], 16)  # must be valid hex

    def test_different_nonces_give_different_challenges(self):
        c0 = self.pow.generate_puzzle(nonce=0)["challenge"]
        c1 = self.pow.generate_puzzle(nonce=1)["challenge"]
        self.assertNotEqual(c0, c1)

    def test_difficulty_positive(self):
        self.assertGreater(self.pow.expected_difficulty(), 0.0)

    def test_security_bits_positive(self):
        self.assertGreater(self.pow.security_bits(), 0.0)

    def test_security_bits_n400_near_40(self):
        pow_400 = KSATProofOfWork(n=400, alpha=4.20, master_seed=42)
        self.assertAlmostEqual(pow_400.security_bits(), 40.0, delta=2.0)


class TestPowDifficultyParameter(unittest.TestCase):

    def test_returns_positive_integer(self):
        n = pow_difficulty_parameter(40)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_scales_with_target_bits(self):
        n40 = pow_difficulty_parameter(40)
        n80 = pow_difficulty_parameter(80)
        self.assertLess(n40, n80)

    def test_n_for_40_bits_near_400(self):
        n = pow_difficulty_parameter(40, alpha=ALPHA_STAR)
        self.assertAlmostEqual(n, 400, delta=30)

    def test_n_for_80_bits_near_800(self):
        n = pow_difficulty_parameter(80, alpha=ALPHA_STAR)
        self.assertAlmostEqual(n, 800, delta=30)

    def test_raises_for_zero_barrier_alpha(self):
        with self.assertRaises(ValueError):
            pow_difficulty_parameter(40, alpha=ALPHA_D - 0.1)


class TestAPKPseudoRandomGenerator(unittest.TestCase):

    def setUp(self):
        self.prg = APKPseudoRandomGenerator(n=400, epsilon=0.020, alpha=4.20)

    def test_rejects_hard_phase_violation(self):
        with self.assertRaises(ValueError):
            APKPseudoRandomGenerator(n=100, epsilon=0.01, alpha=ALPHA_D - 0.5)

    def test_rejects_epsilon_above_sigma(self):
        # Σ(4.20) ≈ 0.027; epsilon=0.030 must be rejected
        with self.assertRaises(ValueError):
            APKPseudoRandomGenerator(n=400, epsilon=0.030, alpha=4.20)

    def test_seed_length(self):
        self.assertEqual(self.prg.seed_length(), 400)

    def test_output_length_exceeds_seed(self):
        self.assertGreater(self.prg.output_length(), self.prg.seed_length())

    def test_stretch_fraction_matches_epsilon(self):
        self.assertAlmostEqual(self.prg.stretch_fraction(), 0.020, places=6)

    def test_stretch_absolute_positive(self):
        self.assertGreater(self.prg.stretch(), 0.0)

    def test_is_nc0_computable(self):
        self.assertTrue(self.prg.is_nc0_computable())

    def test_security_level_positive(self):
        self.assertGreater(self.prg.security_level(), 0.0)

    def test_aik_conditions_all_satisfied(self):
        conds = self.prg.aik_conditions()
        for name, cond in conds.items():
            self.assertTrue(
                cond["satisfied"],
                msg=f"AIK condition '{name}' violated: {cond['reason']}",
            )

    def test_prg_parameters_complete(self):
        params = self.prg.prg_parameters()
        for key in ["seed_length_n", "output_length", "stretch_epsilon",
                    "security_bits", "is_nc0_computable"]:
            self.assertIn(key, params)


class TestSecurityParameterTable(unittest.TestCase):

    def setUp(self):
        self.spt = SecurityParameterTable()

    def test_reproduce_table6_correct_labels(self):
        rows   = self.spt.reproduce_table6()
        labels = [r["label"] for r in rows]
        self.assertIn("Basic",    labels)
        self.assertIn("Standard", labels)
        self.assertIn("High",     labels)

    def test_table6_bits_match_manuscript(self):
        # Manuscript Table 6: ~40/60/80 bits for n=400/600/800
        targets = {"Basic": 40, "Standard": 60, "High": 80}
        for row in self.spt.reproduce_table6():
            exp = targets[row["label"]]
            self.assertAlmostEqual(
                row["security_bits"], exp, delta=1.5,
                msg=f"{row['label']}: {row['security_bits']:.2f} bits, expected ~{exp}",
            )

    def test_validate_table6_passes(self):
        self.assertTrue(self.spt.validate_table6())

    def test_compute_row_scales_with_n(self):
        r200 = self.spt.compute_row(200)
        r400 = self.spt.compute_row(400)
        r800 = self.spt.compute_row(800)
        self.assertLess(r200["security_bits"], r400["security_bits"])
        self.assertLess(r400["security_bits"], r800["security_bits"])

    def test_compute_security_bits_n400(self):
        s = compute_security_bits(400, alpha=4.20)
        self.assertAlmostEqual(s, 40.0, delta=1.5)

    def test_compute_security_bits_n800(self):
        s = compute_security_bits(800, alpha=4.20)
        self.assertAlmostEqual(s, 80.0, delta=1.5)

    def test_n_for_target_invertible(self):
        n = self.spt.n_for_target(40)
        s = compute_security_bits(n, alpha=self.spt.alpha)
        self.assertAlmostEqual(s, 40.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
