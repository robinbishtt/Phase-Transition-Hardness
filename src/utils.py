"""Utility functions for Phase-Transition-Hardness experiments.

Key reproducibility guarantee: derive_seed() uses SHA-256 (not Python's
built-in hash()) so that instance seeds are deterministic across Python
versions, platforms, and PYTHONHASHSEED values.
"""

import os
import json
import time
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union


# =============================================================================
# Logging
# =============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a single StreamHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =============================================================================
# Random-number generation
# =============================================================================

def make_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Return a seeded RandomState (legacy API, kept for compatibility)."""
    return np.random.RandomState(seed)


def derive_seed(master_seed: int, *identifiers) -> int:
    """Derive a deterministic per-instance seed via SHA-256.

    Replaces the previous implementation that used Python's built-in hash(),
    which is non-deterministic (hash randomisation is enabled by default in
    Python 3.3+) and therefore broke cross-run reproducibility.

    Algorithm
    ---------
    key = "<master_seed>:<id0>:<id1>:..."
    seed = int(SHA-256(key), 16) & 0x7FFF_FFFF

    This matches the per-instance seeding strategy described in the paper
    (Supplementary Section 5.1, "Deterministic Instance Generation").

    Parameters
    ----------
    master_seed : int
        Global master seed (paper uses 20240223).
    *identifiers : int | float | str
        Ordered tuple of values that uniquely identify the instance
        (e.g., n, alpha, instance_index).

    Returns
    -------
    int
        A 31-bit non-negative integer suitable as a NumPy / Python seed.
    """
    key = str(master_seed) + ":" + ":".join(str(x) for x in identifiers)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


# =============================================================================
# Timing context manager
# =============================================================================

class Timer:
    """Context manager that records elapsed wall-clock seconds."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# =============================================================================
# File I/O helpers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (including parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialise *data* to a JSON file, converting NumPy types automatically."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r") as fh:
        return json.load(fh)


def save_npz(path: Union[str, Path], **arrays) -> None:
    """Save keyword arrays to a compressed .npz archive."""
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load a .npz archive and return a plain dict of arrays."""
    data = np.load(str(path), allow_pickle=False)
    return dict(data)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# =============================================================================
# Mathematical helpers
# =============================================================================

def log_sum_exp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def safe_log(x: Union[float, np.ndarray], eps: float = 1e-300) -> Union[float, np.ndarray]:
    """log(max(x, eps)) to avoid log(0)."""
    return np.log(np.maximum(x, eps))


def binary_entropy(p: float) -> float:
    """Binary entropy H(p) = -p log p - (1-p) log(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def interpolate_threshold(
    alphas: np.ndarray,
    values: np.ndarray,
    target: float = 0.5,
) -> float:
    """Find α such that values(α) = target via linear interpolation."""
    from scipy.interpolate import interp1d
    try:
        f = interp1d(values[::-1], alphas[::-1], kind="linear",
                     bounds_error=False, fill_value=float("nan"))
        return float(f(target))
    except Exception:
        return float("nan")


def exponential_fit(ns: np.ndarray, log_means: np.ndarray):
    """OLS fit of log_mean = slope·n + intercept; returns (slope, intercept, R²)."""
    from scipy.stats import linregress
    slope, intercept, r, _, _ = linregress(ns, log_means)
    return float(slope), float(intercept), float(r ** 2)


# =============================================================================
# Progress bar
# =============================================================================

def progress(iterable, desc: str = "", total: Optional[int] = None):
    """Wrap *iterable* with tqdm if available, otherwise return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=True)
    except ImportError:
        return iterable
