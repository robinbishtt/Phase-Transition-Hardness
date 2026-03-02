"""
Unit tests for src/utils.py

Tests utility functions including:
- Random number generation
- Seed derivation
- Timer functionality
- File I/O operations
- Mathematical utilities
"""

import unittest
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import (
    make_rng,
    derive_seed,
    Timer,
    ensure_dir,
    save_json,
    load_json,
    save_npz,
    load_npz,
    log_sum_exp,
    safe_log,
    binary_entropy,
    interpolate_threshold,
    exponential_fit,
    get_logger,
)


class TestMakeRNG(unittest.TestCase):
    """Tests for make_rng function."""

    def test_returns_random_state(self):
        """Should return numpy RandomState."""
        result = make_rng(42)
        self.assertIsInstance(result, np.random.RandomState)

    def test_deterministic_with_seed(self):
        """Same seed should give same sequence."""
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        self.assertEqual(rng1.randint(100), rng2.randint(100))

    def test_different_seeds_different_sequences(self):
        """Different seeds should give different sequences."""
        rng1 = make_rng(42)
        rng2 = make_rng(43)

        self.assertNotEqual(rng1.randint(1000000), rng2.randint(1000000))

    def test_none_seed(self):
        """None seed should work (non-deterministic)."""
        rng = make_rng(None)
        self.assertIsInstance(rng, np.random.RandomState)


class TestDeriveSeed(unittest.TestCase):
    """Tests for derive_seed function."""

    def test_deterministic(self):
        """Same inputs should give same output."""
        result1 = derive_seed(42, 100, 4.2, 0)
        result2 = derive_seed(42, 100, 4.2, 0)
        self.assertEqual(result1, result2)

    def test_different_inputs_different_outputs(self):
        """Different inputs should likely give different outputs."""
        result1 = derive_seed(42, 100, 4.2, 0)
        result2 = derive_seed(42, 100, 4.2, 1)
        self.assertNotEqual(result1, result2)

    def test_returns_int(self):
        """Should return an integer."""
        result = derive_seed(42, 100, 4.2, 0)
        self.assertIsInstance(result, int)

    def test_positive(self):
        """Should return non-negative integer."""
        result = derive_seed(42, 100, 4.2, 0)
        self.assertGreaterEqual(result, 0)

    def test_multiple_identifiers(self):
        """Should work with multiple identifiers."""
        result = derive_seed(42, 100, 4.2, 0, "extra", 123)
        self.assertIsInstance(result, int)


class TestTimer(unittest.TestCase):
    """Tests for Timer class."""

    def test_elapsed_zero_before_use(self):
        """Elapsed should be zero before timing."""
        timer = Timer()
        self.assertEqual(timer.elapsed, 0.0)

    def test_measures_time(self):
        """Should measure elapsed time."""
        import time
        timer = Timer()
        with timer:
            time.sleep(0.01)
        self.assertGreater(timer.elapsed, 0.0)

    def test_context_manager(self):
        """Should work as context manager."""
        timer = Timer()
        with timer:
            pass
        self.assertGreaterEqual(timer.elapsed, 0.0)


class TestEnsureDir(unittest.TestCase):
    """Tests for ensure_dir function."""

    def test_creates_directory(self):
        """Should create directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dir"
            result = ensure_dir(path)
            self.assertTrue(path.exists())
            self.assertTrue(path.is_dir())

    def test_returns_path(self):
        """Should return Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dir"
            result = ensure_dir(path)
            self.assertIsInstance(result, Path)

    def test_existing_directory(self):
        """Should not fail if directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            result = ensure_dir(path)
            self.assertTrue(path.exists())


class TestSaveLoadJSON(unittest.TestCase):
    """Tests for save_json and load_json functions."""

    def test_save_and_load(self):
        """Should save and load JSON correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            path = Path(tmpdir) / "test.json"
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(data, loaded)

    def test_numpy_array(self):
        """Should handle numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"array": np.array([1, 2, 3])}
            path = Path(tmpdir) / "test.json"
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded["array"], [1, 2, 3])

    def test_numpy_scalar(self):
        """Should handle numpy scalars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"int": np.int64(42), "float": np.float64(3.14)}
            path = Path(tmpdir) / "test.json"
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded["int"], 42)
            self.assertEqual(loaded["float"], 3.14)


class TestSaveLoadNPZ(unittest.TestCase):
    """Tests for save_npz and load_npz functions."""

    def test_save_and_load(self):
        """Should save and load NPZ correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr1 = np.array([1, 2, 3])
            arr2 = np.array([[4, 5], [6, 7]])
            path = Path(tmpdir) / "test.npz"
            save_npz(path, arr1=arr1, arr2=arr2)
            loaded = load_npz(path)
            np.testing.assert_array_equal(loaded["arr1"], arr1)
            np.testing.assert_array_equal(loaded["arr2"], arr2)

    def test_returns_dict(self):
        """Should return dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            save_npz(path, arr=np.array([1, 2, 3]))
            loaded = load_npz(path)
            self.assertIsInstance(loaded, dict)


class TestLogSumExp(unittest.TestCase):
    """Tests for log_sum_exp function."""

    def test_basic(self):
        """Should compute log(sum(exp(x))) correctly."""
        arr = np.array([1.0, 2.0, 3.0])
        result = log_sum_exp(arr)
        expected = np.log(np.sum(np.exp(arr)))
        self.assertAlmostEqual(result, expected, places=10)

    def test_large_values(self):
        """Should handle large values without overflow."""
        arr = np.array([1000.0, 1001.0, 1002.0])
        result = log_sum_exp(arr)
        self.assertTrue(np.isfinite(result))

    def test_single_element(self):
        """Should work with single element."""
        arr = np.array([5.0])
        result = log_sum_exp(arr)
        self.assertAlmostEqual(result, 5.0, places=10)


class TestSafeLog(unittest.TestCase):
    """Tests for safe_log function."""

    def test_positive_values(self):
        """Should compute log correctly for positive values."""
        result = safe_log(2.0)
        self.assertAlmostEqual(result, np.log(2.0), places=10)

    def test_zero(self):
        """Should handle zero without error."""
        result = safe_log(0.0)
        self.assertTrue(np.isfinite(result))
        self.assertLess(result, 0.0)

    def test_negative(self):
        """Should handle negative values gracefully."""
        result = safe_log(-1.0)

        self.assertTrue(np.isfinite(result))

    def test_array_input(self):
        """Should work with arrays."""
        arr = np.array([1.0, 0.0, 2.0])
        result = safe_log(arr)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(np.isfinite(result)))


class TestBinaryEntropy(unittest.TestCase):
    """Tests for binary_entropy function."""

    def test_zero(self):
        """H(0) should be 0."""
        result = binary_entropy(0.0)
        self.assertEqual(result, 0.0)

    def test_one(self):
        """H(1) should be 0."""
        result = binary_entropy(1.0)
        self.assertEqual(result, 0.0)

    def test_half(self):
        """H(0.5) should be log(2)."""
        result = binary_entropy(0.5)
        self.assertAlmostEqual(result, np.log(2.0), places=10)

    def test_symmetric(self):
        """H(p) should equal H(1-p)."""
        for p in [0.1, 0.25, 0.4]:
            self.assertAlmostEqual(binary_entropy(p), binary_entropy(1 - p), places=10)

    def test_maximum_at_half(self):
        """Maximum should be at p=0.5."""
        h_half = binary_entropy(0.5)
        for p in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            self.assertGreater(h_half, binary_entropy(p))

    def test_out_of_range(self):
        """Should handle out-of-range values."""
        self.assertEqual(binary_entropy(-0.1), 0.0)
        self.assertEqual(binary_entropy(1.1), 0.0)


class TestInterpolateThreshold(unittest.TestCase):
    """Tests for interpolate_threshold function."""

    def test_linear_interpolation(self):
        """Should linearly interpolate."""
        alphas = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 0.5, 0.0])
        result = interpolate_threshold(alphas, values, target=0.5)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_finds_crossing(self):
        """Should find where values cross target."""
        alphas = np.array([3.0, 4.0, 5.0, 6.0])
        values = np.array([0.9, 0.6, 0.4, 0.1])
        result = interpolate_threshold(alphas, values, target=0.5)
        self.assertGreater(result, 4.0)
        self.assertLess(result, 5.0)


class TestExponentialFit(unittest.TestCase):
    """Tests for exponential_fit function."""

    def test_linear_fit(self):
        """Should fit linear relationship."""
        ns = np.array([1, 2, 3, 4, 5])
        log_means = 2.0 * ns + 1.0
        slope, intercept, r2 = exponential_fit(ns, log_means)
        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(intercept, 1.0, places=5)
        self.assertAlmostEqual(r2, 1.0, places=5)

    def test_returns_floats(self):
        """Should return floats."""
        ns = np.array([1, 2, 3])
        log_means = np.array([1.0, 2.0, 3.0])
        slope, intercept, r2 = exponential_fit(ns, log_means)
        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(r2, float)


class TestGetLogger(unittest.TestCase):
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Should return a logger."""
        logger = get_logger("test")
        self.assertIsInstance(logger, logging.Logger)

    def test_same_name_same_logger(self):
        """Same name should return same logger."""
        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        self.assertIs(logger1, logger2)


import logging


if __name__ == "__main__":
    unittest.main()