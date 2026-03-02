# API Documentation

Complete API reference for the Phase-Transition-Hardness codebase.

## Table of Contents

- [src.utils](#srcutils)
- [src.instance_generator](#srcinstance_generator)
- [src.energy_model](#srcenergy_model)
- [src.phase_transition](#srcphase_transition)
- [src.hardness_metrics](#srchardness_metrics)
- [src.barrier_analysis](#srcbarrier_analysis)
- [src.scaling_analysis](#srcscaling_analysis)
- [src.statistics](#srcstatistics)
- [src.runtime_measurement](#srcruntime_measurement)
- [src.validation](#srcvalidation)

---

## src.utils

Utility functions for random number generation, I/O operations, and mathematical helpers.

### Functions

#### `make_rng(seed: Optional[int] = None) -> np.random.RandomState`

Creates a reproducible random number generator.

**Parameters:**
- `seed`: Integer seed for reproducibility. If None, uses random seed.

**Returns:**
- `RandomState` instance

**Example:**
```python
from src.utils import make_rng

rng = make_rng(42)
random_numbers = rng.rand(10)
```

#### `derive_seed(master_seed: int, *identifiers) -> int`

Derives a deterministic child seed from a master seed.

**Parameters:**
- `master_seed`: Parent seed value
- `*identifiers`: Additional identifiers for unique derivation

**Returns:**
- Derived seed as integer

**Example:**
```python
from src.utils import derive_seed

instance_seed = derive_seed(42, 100, 4.2, 0)
```

#### `Timer`

Context manager for timing code execution.

**Example:**
```python
from src.utils import Timer

with Timer() as timer:
    # Your code here
    pass

print(f"Elapsed: {timer.elapsed:.4f} seconds")
```

#### `save_json(data: Any, path: Union[str, Path]) -> None`

Saves data to JSON file with numpy type handling.

**Parameters:**
- `data`: Data to save
- `path`: Output file path

#### `load_json(path: Union[str, Path]) -> Any`

Loads data from JSON file.

**Parameters:**
- `path`: Input file path

**Returns:**
- Loaded data

#### `save_npz(path: Union[str, Path], **arrays) -> None`

Saves numpy arrays to compressed NPZ file.

**Parameters:**
- `path`: Output file path
- `**arrays`: Named arrays to save

#### `load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]`

Loads numpy arrays from NPZ file.

**Parameters:**
- `path`: Input file path

**Returns:**
- Dictionary of loaded arrays

#### `log_sum_exp(a: np.ndarray) -> float`

Numerically stable computation of log(sum(exp(a))).

**Parameters:**
- `a`: Input array

**Returns:**
- log(sum(exp(a)))

#### `safe_log(x: Union[float, np.ndarray], eps: float = 1e-300) -> Union[float, np.ndarray]`

Safe logarithm that handles zero and negative values.

**Parameters:**
- `x`: Input value or array
- `eps`: Minimum value for clipping

**Returns:**
- log(max(x, eps))

#### `binary_entropy(p: float) -> float`

Computes binary entropy H(p) = -p*log(p) - (1-p)*log(1-p).

**Parameters:**
- `p`: Probability value in [0, 1]

**Returns:**
- Binary entropy value

#### `interpolate_threshold(alphas: np.ndarray, values: np.ndarray, target: float = 0.5) -> float`

Interpolates to find alpha where values cross target.

**Parameters:**
- `alphas`: Array of alpha values
- `values`: Array of corresponding values
- `target`: Target value to find crossing

**Returns:**
- Interpolated alpha value

---

## src.instance_generator

Random k-SAT instance generation functions.

### Functions

#### `generate_ksat_instance(n: int, alpha: float, k: int = 3, seed: Optional[int] = None) -> dict`

Generates a random k-SAT instance.

**Parameters:**
- `n`: Number of variables
- `alpha`: Constraint density (m/n)
- `k`: Clause length (default: 3)
- `seed`: Random seed for reproducibility

**Returns:**
- Dictionary with keys: `n`, `k`, `alpha`, `m`, `clauses`, `seed`

**Example:**
```python
from src.instance_generator import generate_ksat_instance

instance = generate_ksat_instance(n=100, alpha=4.2, k=3, seed=42)
print(f"Generated instance with {instance['m']} clauses")
```

#### `generate_instance_batch(n: int, alpha: float, n_instances: int, k: int = 3, master_seed: int = 42) -> List[dict]`

Generates a batch of independent k-SAT instances.

**Parameters:**
- `n`: Number of variables
- `alpha`: Constraint density
- `n_instances`: Number of instances to generate
- `k`: Clause length
- `master_seed`: Master seed for reproducibility

**Returns:**
- List of instance dictionaries

#### `instance_to_adjacency(instance: dict) -> Tuple[List[List[int]], List[List[int]]]`

Converts instance to adjacency representation.

**Parameters:**
- `instance`: k-SAT instance dictionary

**Returns:**
- Tuple of (var_to_clauses, clause_to_vars) adjacency lists

#### `count_violated_clauses(instance: dict, assignment: dict) -> int`

Counts number of violated clauses for an assignment.

**Parameters:**
- `instance`: k-SAT instance
- `assignment`: Variable assignment dictionary

**Returns:**
- Number of violated clauses

#### `is_satisfying(instance: dict, assignment: dict) -> bool`

Checks if assignment satisfies all clauses.

**Parameters:**
- `instance`: k-SAT instance
- `assignment`: Variable assignment dictionary

**Returns:**
- True if satisfying, False otherwise

---

## src.energy_model

Thermodynamic energy model functions for phase transition analysis.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ALPHA_D` | 3.86 | Clustering threshold |
| `ALPHA_R` | 4.00 | Rigidity threshold |
| `ALPHA_C` | 4.10 | Condensation threshold |
| `ALPHA_S` | 4.267 | Satisfiability threshold |
| `K_DEFAULT` | 3 | Default clause length |

### Functions

#### `annealed_entropy(alpha: float, k: int = 3) -> float`

Computes annealed entropy density.

**Formula:** s = log(2) + α·log(1 - 2^(1-k))

**Parameters:**
- `alpha`: Constraint density
- `k`: Clause length

**Returns:**
- Annealed entropy density

#### `rs_entropy_density(alpha: float, k: int = 3) -> float`

Computes replica symmetric entropy density.

**Parameters:**
- `alpha`: Constraint density
- `k`: Clause length

**Returns:**
- RS entropy density (clipped to [0, log(2)])

#### `cluster_complexity(alpha: float) -> float`

Computes cluster complexity (configurational entropy).

**Parameters:**
- `alpha`: Constraint density

**Returns:**
- Cluster complexity Σ(α)

#### `free_energy_density(alpha: float, beta: float = 1.0, k: int = 3) -> float`

Computes free energy density at inverse temperature β.

**Parameters:**
- `alpha`: Constraint density
- `beta`: Inverse temperature
- `k`: Clause length

**Returns:**
- Free energy density f(β, α)

#### `frozen_fraction(alpha: float) -> float`

Computes fraction of frozen variables.

**Parameters:**
- `alpha`: Constraint density

**Returns:**
- Fraction of frozen variables in [0, 1]

#### `barrier_density(alpha: float, k: int = 3) -> float`

Computes intensive free-energy barrier density.

**Formula:** Gaussian centered at α=4.2 with amplitude 0.015

**Parameters:**
- `alpha`: Constraint density
- `k`: Clause length

**Returns:**
- Barrier density b(α)

#### `barrier_height(n: int, alpha: float, k: int = 3) -> float`

Computes total barrier height B(n, α) = n · b(α).

**Parameters:**
- `n`: System size
- `alpha`: Constraint density
- `k`: Clause length

**Returns:**
- Total barrier height

---

## src.phase_transition

Phase transition detection and analysis functions.

### Functions

#### `estimate_psat_single(n: int, alpha: float, n_instances: int = 200, k: int = 3, master_seed: int = 42, solver: str = "dpll") -> float`

Estimates P_sat(n, α) by sampling instances.

**Parameters:**
- `n`: System size
- `alpha`: Constraint density
- `n_instances`: Number of instances to sample
- `k`: Clause length
- `master_seed`: Random seed
- `solver`: Solver to use ("dpll" or "walksat")

**Returns:**
- Estimated P_sat value in [0, 1]

#### `psat_curve(n: int, alphas: np.ndarray, n_instances: int = 200, k: int = 3, master_seed: int = 42, solver: str = "dpll", n_jobs: int = 1) -> np.ndarray`

Computes P_sat curve across alpha values.

**Parameters:**
- `n`: System size
- `alphas`: Array of alpha values
- `n_instances`: Instances per alpha
- `k`: Clause length
- `master_seed`: Random seed
- `solver`: Solver to use
- `n_jobs`: Number of parallel jobs

**Returns:**
- Array of P_sat values

#### `locate_threshold(alphas: np.ndarray, psats: np.ndarray, target: float = 0.5) -> float`

Locates threshold alpha where P_sat crosses target.

**Parameters:**
- `alphas`: Array of alpha values
- `psats`: Array of P_sat values
- `target`: Target P_sat value (default: 0.5)

**Returns:**
- Interpolated threshold alpha

#### `theoretical_order_parameters(alphas: np.ndarray) -> Dict[str, np.ndarray]`

Computes theoretical order parameters.

**Parameters:**
- `alphas`: Array of alpha values

**Returns:**
- Dictionary with keys: `entropy`, `cluster_complexity`, `frozen_fraction`

#### `run_psat_sweep(ns: List[int], alphas: np.ndarray, n_instances: int = 200, k: int = 3, master_seed: int = 42, solver: str = "dpll", output_dir: str = "results", n_jobs: int = 1) -> Dict`

Runs complete P_sat sweep across system sizes.

**Parameters:**
- `ns`: List of system sizes
- `alphas`: Array of alpha values
- `n_instances`: Instances per (n, α)
- `k`: Clause length
- `master_seed`: Random seed
- `solver`: Solver to use
- `output_dir`: Output directory
- `n_jobs`: Number of parallel jobs

**Returns:**
- Dictionary with results and saves to files

---

## src.hardness_metrics

SAT solver implementations and hardness measurement.

### Classes

#### `CNF`

Represents a CNF formula.

**Methods:**
- `from_instance(instance: dict) -> CNF`: Create CNF from instance
- `copy() -> CNF`: Create a copy

### Functions

#### `dpll_solve(instance: dict, max_decisions: int = MAX_DECISIONS_DEFAULT) -> Dict`

DPLL solver with unit propagation and pure literal elimination.

**Parameters:**
- `instance`: k-SAT instance
- `max_decisions`: Maximum branching decisions

**Returns:**
- Dictionary with keys:
  - `satisfiable`: True/False/None (None = cutoff)
  - `decisions`: Number of decisions made
  - `assignment`: Satisfying assignment (if found)

#### `walksat_solve(instance: dict, max_flips: int = WALKSAT_MAX_FLIPS, noise: float = WALKSAT_NOISE, seed: Optional[int] = None, restarts: int = 5) -> Dict`

WalkSAT local search solver.

**Parameters:**
- `instance`: k-SAT instance
- `max_flips`: Maximum flips per restart
- `noise`: Noise parameter (probability of random walk)
- `seed`: Random seed
- `restarts`: Number of restarts

**Returns:**
- Dictionary with keys:
  - `satisfiable`: True/False
  - `flips`: Number of flips performed
  - `assignment`: Satisfying assignment (if found)

#### `measure_hardness(instance: dict, solver: str = "dpll", max_decisions: int = MAX_DECISIONS_DEFAULT, walksat_seed: Optional[int] = None) -> float`

Measures hardness density γ = log(T) / n.

**Parameters:**
- `instance`: k-SAT instance
- `solver`: Solver to use ("dpll" or "walksat")
- `max_decisions`: Maximum decisions for DPLL
- `walksat_seed`: Seed for WalkSAT

**Returns:**
- Hardness density γ

#### `hardness_curve(n: int, alphas: np.ndarray, n_instances: int = 200, k: int = 3, solver: str = "dpll", master_seed: int = 42, max_decisions: int = MAX_DECISIONS_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Computes hardness curve with confidence intervals.

**Parameters:**
- `n`: System size
- `alphas`: Array of alpha values
- `n_instances`: Instances per alpha
- `k`: Clause length
- `solver`: Solver to use
- `master_seed`: Random seed
- `max_decisions`: Maximum decisions

**Returns:**
- Tuple of (gamma_mean, gamma_lo, gamma_hi) arrays

---

## src.barrier_analysis

Free-energy barrier analysis functions.

### Functions

#### `path_barrier(instance: dict, assign1: dict, assign2: dict, n_samples: int = 500, rng: Optional[np.random.RandomState] = None) -> float`

Estimates energy barrier between two assignments.

**Parameters:**
- `instance`: k-SAT instance
- `assign1`: First assignment
- `assign2`: Second assignment
- `n_samples`: Number of path samples
- `rng`: Random number generator

**Returns:**
- Estimated barrier height

#### `theoretical_barrier_curve(alphas: np.ndarray, k: int = 3) -> np.ndarray`

Computes theoretical barrier density curve.

**Parameters:**
- `alphas`: Array of alpha values
- `k`: Clause length

**Returns:**
- Array of barrier densities

#### `barrier_scaling_data(ns: List[int], alpha: float, k: int = 3) -> Dict`

Generates barrier scaling data for given alpha.

**Parameters:**
- `ns`: List of system sizes
- `alpha`: Constraint density
- `k`: Clause length

**Returns:**
- Dictionary with scaling data

#### `run_barrier_scaling_sweep(ns: List[int], alphas: np.ndarray, k: int = 3, output_dir: str = "results") -> Dict`

Runs complete barrier scaling sweep.

**Parameters:**
- `ns`: List of system sizes
- `alphas`: Array of alpha values
- `k`: Clause length
- `output_dir`: Output directory

**Returns:**
- Dictionary with results

#### `barrier_hardness_correlation(alphas: np.ndarray, gamma_mean: np.ndarray, k: int = 3) -> Dict`

Computes correlation between barrier density and hardness.

**Parameters:**
- `alphas`: Array of alpha values
- `gamma_mean`: Array of hardness values
- `k`: Clause length

**Returns:**
- Dictionary with correlation, p-value, and curves

---

## src.scaling_analysis

Finite-size scaling and exponential scaling analysis.

### Functions

#### `run_fss_analysis(alphas: np.ndarray, ns: List[int], psat_matrix: np.ndarray, output_dir: str = "results") -> Dict`

Performs finite-size scaling collapse analysis.

**Parameters:**
- `alphas`: Array of alpha values
- `ns`: List of system sizes
- `psat_matrix`: P_sat values matrix (n_sizes × n_alphas)
- `output_dir`: Output directory

**Returns:**
- Dictionary with keys:
  - `alpha_s`: Estimated threshold
  - `nu`: Critical exponent
  - `residual`: Collapse quality
  - `converged`: Whether optimization converged

#### `run_exponential_scaling(ns: List[int], alphas: np.ndarray, gamma_matrix: np.ndarray, output_dir: str = "results") -> Dict`

Fits exponential scaling log T̄ = γ·n.

**Parameters:**
- `ns`: List of system sizes
- `alphas`: Array of alpha values
- `gamma_matrix`: Hardness values matrix
- `output_dir`: Output directory

**Returns:**
- Dictionary with fit results

#### `locate_hardness_peak(alphas: np.ndarray, gamma_mean: np.ndarray) -> Tuple[float, float]`

Locates hardness peak position and value.

**Parameters:**
- `alphas`: Array of alpha values
- `gamma_mean`: Array of hardness values

**Returns:**
- Tuple of (alpha_star, gamma_max)

#### `finite_size_peak_extrapolation(ns: List[int], alpha_stars: np.ndarray) -> dict`

Extrapolates peak location to infinite system size.

**Parameters:**
- `ns`: List of system sizes
- `alpha_stars`: Array of peak locations per size

**Returns:**
- Dictionary with extrapolated value and fit quality

---

## src.statistics

Statistical analysis functions.

### Functions

#### `bootstrap_ci(data: np.ndarray, statistic=np.mean, n_boot: int = 1000, ci: float = 0.95, seed: Optional[int] = None) -> Tuple[float, float]`

Computes bootstrap confidence interval.

**Parameters:**
- `data`: Input data array
- `statistic`: Statistic function (default: mean)
- `n_boot`: Number of bootstrap samples
- `ci`: Confidence level
- `seed`: Random seed

**Returns:**
- Tuple of (lower_bound, upper_bound)

#### `lognormal_mean_ci(data: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]`

Computes confidence interval for lognormal mean.

**Parameters:**
- `data`: Input data array
- `ci`: Confidence level

**Returns:**
- Tuple of (mean, lower_bound, upper_bound)

#### `exponential_scaling_fit(ns: np.ndarray, log_mean_runtimes: np.ndarray) -> dict`

Fits exponential scaling model.

**Parameters:**
- `ns`: System sizes
- `log_mean_runtimes`: Log of mean runtimes

**Returns:**
- Dictionary with gamma, intercept, r2, p_value, stderr, residuals

#### `fss_collapse(alphas: np.ndarray, ns: np.ndarray, psat_matrix: np.ndarray, alpha_s_init: float = 4.267, nu_init: float = 2.3) -> dict`

Performs finite-size scaling collapse optimization.

**Parameters:**
- `alphas`: Array of alpha values
- `ns`: Array of system sizes
- `psat_matrix`: P_sat values matrix
- `alpha_s_init`: Initial alpha_s guess
- `nu_init`: Initial nu guess

**Returns:**
- Dictionary with optimized parameters and collapse data

#### `fit_lognormal(data: np.ndarray) -> dict`

Fits lognormal distribution to data.

**Parameters:**
- `data`: Input data array

**Returns:**
- Dictionary with mu, sigma, ks_stat, ks_pvalue

#### `fit_exponential_tail(data: np.ndarray, tail_quantile: float = 0.90) -> dict`

Fits exponential tail to data.

**Parameters:**
- `data`: Input data array
- `tail_quantile`: Quantile defining tail

**Returns:**
- Dictionary with lambda, tail_threshold, n_tail

---

## src.runtime_measurement

Experiment orchestration and runtime measurement.

### Functions

#### `measure_runtime_distribution(n: int, alpha: float, n_instances: int = 200, k: int = 3, solver: str = "dpll", master_seed: int = 42, max_decisions: int = 100_000) -> Dict`

Measures runtime distribution for given parameters.

**Parameters:**
- `n`: System size
- `alpha`: Constraint density
- `n_instances`: Number of instances
- `k`: Clause length
- `solver`: Solver to use
- `master_seed`: Random seed
- `max_decisions`: Maximum decisions

**Returns:**
- Dictionary with runtime statistics

#### `alpha_sweep(ns: List[int], alphas: np.ndarray, n_instances: int = 200, k: int = 3, solver: str = "dpll", master_seed: int = 42, max_decisions: int = 100_000, output_dir: str = "results") -> Dict`

Runs complete alpha sweep experiment.

**Parameters:**
- `ns`: List of system sizes
- `alphas`: Array of alpha values
- `n_instances`: Instances per (n, α)
- `k`: Clause length
- `solver`: Solver to use
- `master_seed`: Random seed
- `max_decisions`: Maximum decisions
- `output_dir`: Output directory

**Returns:**
- Dictionary with complete sweep results

#### `localise_hardness_peak(ns: List[int], alpha_center: float = 4.20, width: float = 0.30, n_points: int = 30, n_instances: int = 500, k: int = 3, solver: str = "dpll", master_seed: int = 42, max_decisions: int = 100_000, output_dir: str = "results") -> Dict`

Runs fine-resolution hardness peak localization.

**Parameters:**
- `ns`: List of system sizes
- `alpha_center`: Center of sweep
- `width`: Half-width of sweep
- `n_points`: Number of alpha points
- `n_instances`: Instances per point
- `k`: Clause length
- `solver`: Solver to use
- `master_seed`: Random seed
- `max_decisions`: Maximum decisions
- `output_dir`: Output directory

**Returns:**
- Dictionary with peak localization results

---

## src.validation

Automated validation checks against manuscript predictions.

### Functions

#### `run_all_checks(results_dir: str = "results") -> Dict`

Runs all eight validation checks.

**Parameters:**
- `results_dir`: Directory containing result files

**Returns:**
- Dictionary with:
  - `passed`: Number of passed checks
  - `failed`: Number of failed checks
  - `total`: Total number of checks
  - `details`: List of individual check results

**Validation Checks:**

| # | Check | Criterion | Expected |
|---|-------|-----------|----------|
| 1 | Satisfiability threshold | α_s ∈ [4.20, 4.35] | ~4.267 |
| 2 | Hardness peak location | α* ∈ [4.10, 4.40] | ~4.20 |
| 3 | Peak hardness density | γ_max ∈ [0.005, 0.05] | ~0.015 |
| 4 | Exponential scaling | R² ≥ 0.85, γ > 0 | Conjecture 1 |
| 5 | FSS collapse quality | Residual < 0.10 | Figure 4 |
| 6 | FSS critical exponent | ν ∈ [1.5, 3.5] | ~2.30 |
| 7 | Barrier density | b(α) > 0 in hard phase | Conjecture 1 |
| 8 | P_sat monotonicity | Non-increasing | Threshold structure |

**Example:**
```python
from src.validation import run_all_checks

result = run_all_checks("results")
print(f"Passed: {result['passed']}/{result['total']}")

if result['failed'] > 0:
    for detail in result['details']:
        if not detail['passed']:
            print(f"Failed: {detail['name']} - {detail['message']}")
```

---

## Type Hints

The codebase uses Python type hints throughout. Common type annotations:

- `np.ndarray`: NumPy array
- `Dict[str, Any]`: Dictionary with string keys
- `List[int]`: List of integers
- `Tuple[float, float]`: Tuple of two floats
- `Optional[int]`: Optional integer (can be None)
- `Union[str, Path]`: Either string or Path

---

## Error Handling

Functions generally handle errors gracefully:

- Invalid inputs may raise `ValueError`
- File operations may raise `FileNotFoundError`
- Mathematical operations return `NaN` or `inf` for invalid results
- Solver cutoffs return `None` for satisfiability

---

## Performance Considerations

- Use `n_jobs > 1` for parallel execution where available
- Set appropriate `max_decisions` for DPLL to prevent excessive runtime
- Use smaller `n_instances` for quick tests, larger for publication results
- Batch instance generation for memory efficiency
