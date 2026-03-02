import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union





def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    
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






def make_rng(seed: Optional[int] = None) -> np.random.RandomState:
    
    return np.random.RandomState(seed)


def derive_seed(master_seed: int, *identifiers) -> int:
    
    h = master_seed
    for ident in identifiers:
        h = hash((h, ident)) & 0x7FFFFFFF
    return int(h)






class Timer:
    

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start






def ensure_dir(path: Union[str, Path]) -> Path:
    
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)


def load_json(path: Union[str, Path]) -> Any:
    
    with open(path, "r") as fh:
        return json.load(fh)


def save_npz(path: Union[str, Path], **arrays) -> None:
    
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    
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






def log_sum_exp(a: np.ndarray) -> float:
    
    a_max = np.max(a)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def safe_log(x: Union[float, np.ndarray], eps: float = 1e-300) -> Union[float, np.ndarray]:
    
    return np.log(np.maximum(x, eps))


def binary_entropy(p: float) -> float:
    
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def interpolate_threshold(alphas: np.ndarray, values: np.ndarray,
                           target: float = 0.5) -> float:
    
    from scipy.interpolate import interp1d
    try:
        f = interp1d(values[::-1], alphas[::-1], kind="linear",
                     bounds_error=False, fill_value=float("nan"))
        return float(f(target))
    except Exception:
        return float("nan")


def exponential_fit(ns: np.ndarray, log_means: np.ndarray):
    
    from scipy.stats import linregress
    slope, intercept, r, _, _ = linregress(ns, log_means)
    return float(slope), float(intercept), float(r ** 2)






def progress(iterable, desc: str = "", total: Optional[int] = None):
    
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=True)
    except ImportError:
        return iterable