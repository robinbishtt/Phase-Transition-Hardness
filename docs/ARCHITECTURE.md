# Architecture Documentation

System architecture and design decisions for Phase-Transition-Hardness.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Module Dependencies](#module-dependencies)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Key Design Decisions](#key-design-decisions)
- [Extensibility](#extensibility)

---

## Overview

The Phase-Transition-Hardness codebase implements a research artifact for studying the correspondence between computational hardness in random constraint satisfaction problems (CSPs) and phase-transition phenomena in statistical mechanics. The architecture prioritizes:

1. **Reproducibility**: All experiments are deterministic from a single master seed
2. **Modularity**: Clear separation of concerns between components
3. **Testability**: Comprehensive test coverage at multiple levels
4. **Extensibility**: Easy to add new solvers, ensembles, or analysis methods

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Phase-Transition-Hardness                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Instance   │───▶│   Solver    │───▶│  Hardness   │───▶│  Analysis   │  │
│  │  Generator  │    │   (DPLL/    │    │  Metrics    │    │   & Stats   │  │
│  │             │    │  WalkSAT)   │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Energy Model                                 │   │
│  │  (Entropy, Complexity, Barriers, Free Energy)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Experiment Orchestration                         │   │
│  │  (Alpha Sweep, FSS, Peak Localization, Scaling Laws)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Validation Suite                                │   │
│  │  (8 Automated Checks Against Manuscript Predictions)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
src/utils.py
    │
    ├──▶ src/instance_generator.py
    │       │
    │       ├──▶ src/energy_model.py
    │       │       │
    │       │       ├──▶ src/barrier_analysis.py
    │       │       │
    │       │       └──▶ src/hardness_metrics.py
    │       │               │
    │       │               ├──▶ src/phase_transition.py
    │       │               │
    │       │               └──▶ src/runtime_measurement.py
    │       │                       │
    │       │                       ├──▶ src/scaling_analysis.py
    │       │                       │
    │       │                       └──▶ src/statistics.py
    │       │                               │
    │       └──▶ src/validation.py ◀────────┘
    │
    └──▶ experiments/*.py
```

### Dependency Rules

1. **No circular dependencies**: The import graph is a DAG
2. **Low-level to high-level**: Utilities → Generators → Models → Analysis
3. **Experiment scripts import from src/**: Not vice versa
4. **Validation imports from all modules**: At the top of the hierarchy

---

## Layered Architecture

### Layer 1: Foundation (src/utils.py)

**Purpose**: Low-level utilities used throughout the codebase

**Components**:
- Random number generation with reproducibility
- Seed derivation for hierarchical randomness
- Timer context manager
- JSON/NPZ I/O with numpy type handling
- Mathematical utilities (log-sum-exp, binary entropy)

**Design Pattern**: Utility module with pure functions

### Layer 2: Data Generation (src/instance_generator.py)

**Purpose**: Generate random k-SAT instances from the ensemble

**Components**:
- `generate_ksat_instance()`: Single instance generation
- `generate_instance_batch()`: Batch generation with derived seeds
- `instance_to_adjacency()`: Graph representation
- `count_violated_clauses()`: Energy function

**Design Pattern**: Factory pattern for instance creation

**Key Feature**: Deterministic instance generation via SHA-256-based seed hierarchy

### Layer 3: Physical Models (src/energy_model.py)

**Purpose**: Implement thermodynamic energy landscape models

**Components**:
- Entropy density functions (annealed, RS)
- Cluster complexity Σ(α)
- Free energy density f(β, α)
- Frozen variable fraction
- Barrier density b(α) and height B(n, α)

**Design Pattern**: Mathematical model with threshold constants

**Key Feature**: Analytical functions based on cavity method predictions

### Layer 4: Solvers (src/hardness_metrics.py)

**Purpose**: SAT solver implementations instrumented for hardness measurement

**Components**:
- `CNF` class: Formula representation
- `dpll_solve()`: Complete solver with unit propagation
- `walksat_solve()`: Incomplete local search solver
- `measure_hardness()`: Hardness density γ = log(T)/n

**Design Pattern**: Strategy pattern for different solvers

**Key Feature**: Pure Python for complete instrumentation and reproducibility

### Layer 5: Phase Transition Analysis (src/phase_transition.py)

**Purpose**: Detect and analyze phase transitions

**Components**:
- `estimate_psat_single()`: P_sat estimation
- `psat_curve()`: P_sat across alpha values
- `locate_threshold()`: Threshold detection
- `theoretical_order_parameters()`: Theoretical predictions
- `run_psat_sweep()`: Complete sweep experiment

**Design Pattern**: Analysis pipeline with configurable parameters

### Layer 6: Barrier Analysis (src/barrier_analysis.py)

**Purpose**: Analyze free-energy barriers and their correspondence to hardness

**Components**:
- `path_barrier()`: Energy barrier between assignments
- `theoretical_barrier_curve()`: b(α) curve
- `barrier_scaling_data()`: Scaling with system size
- `barrier_hardness_correlation()`: Test Conjecture 1

**Design Pattern**: Correlation analysis between theoretical and empirical quantities

### Layer 7: Scaling Analysis (src/scaling_analysis.py)

**Purpose**: Finite-size scaling and exponential scaling analysis

**Components**:
- `run_fss_analysis()`: FSS collapse optimization
- `run_exponential_scaling()`: Exponential fit
- `locate_hardness_peak()`: Peak detection
- `finite_size_peak_extrapolation()`: Infinite-size extrapolation

**Design Pattern**: Optimization and curve fitting

### Layer 8: Statistics (src/statistics.py)

**Purpose**: Statistical analysis and confidence estimation

**Components**:
- `bootstrap_ci()`: Bootstrap confidence intervals
- `lognormal_mean_ci()`: Lognormal parameter estimation
- `exponential_scaling_fit()`: Linear regression for scaling
- `fss_collapse()`: FSS optimization
- Distribution fitting (lognormal, exponential tail)

**Design Pattern**: Statistical utilities with robust implementations

### Layer 9: Experiment Orchestration (src/runtime_measurement.py)

**Purpose**: Coordinate complete experimental workflows

**Components**:
- `measure_runtime_distribution()`: Runtime statistics per parameter
- `alpha_sweep()`: Full alpha sweep with multiple system sizes
- `localise_hardness_peak()`: High-resolution peak localization

**Design Pattern**: Orchestrator with progress tracking

### Layer 10: Validation (src/validation.py)

**Purpose**: Verify computational outputs against manuscript predictions

**Components**:
- 8 individual check functions
- `run_all_checks()`: Complete validation suite

**Design Pattern**: Assertion-based validation with detailed reporting

---

## Data Flow

### Experiment Execution Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Configuration  │────▶│   Experiment    │────▶│   Raw Results   │
│   (CLI args)    │     │   Execution     │     │   (.npz files)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Figure Gen     │◀────│   Validation    │◀────│   Summary JSON  │
│  (Publication)  │     │   (8 Checks)    │     │   (Scalars)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Data Lifecycle

1. **Generation**: Instances generated from master seed
2. **Measurement**: Solvers instrumented for runtime/decisions
3. **Aggregation**: Statistics computed with bootstrap CIs
4. **Analysis**: Scaling laws and FSS applied
5. **Validation**: Outputs checked against predictions
6. **Visualization**: Publication-quality figures generated

---

## Design Patterns

### 1. Factory Pattern

**Used in**: `instance_generator.py`

```python
# Create instances through factory functions
def generate_ksat_instance(n, alpha, k=3, seed=None) -> dict:
    # Factory implementation
    return instance_dict
```

**Benefit**: Centralized instance creation with consistent structure

### 2. Strategy Pattern

**Used in**: `hardness_metrics.py`

```python
# Different solver strategies
result = dpll_solve(instance, max_decisions=10000)
result = walksat_solve(instance, max_flips=10000, noise=0.57)
```

**Benefit**: Interchangeable algorithms with common interface

### 3. Context Manager Pattern

**Used in**: `utils.py`

```python
# Timer context manager
with Timer() as timer:
    # Code to time
    pass
print(timer.elapsed)
```

**Benefit**: Clean resource management and timing

### 4. Pipeline Pattern

**Used in**: `experiments/*.py`

```python
# Sequential pipeline execution
psat_result = run_psat_sweep(...)
hardness_result = alpha_sweep(...)
scaling_result = run_exponential_scaling(...)
```

**Benefit**: Clear data flow and modular stages

### 5. Template Method Pattern

**Used in**: `validation.py`

```python
# Common check structure
def check_function(results_dir) -> Tuple[bool, str]:
    # Load data
    # Perform check
    # Return (passed, message)
```

**Benefit**: Consistent validation interface

---

## Key Design Decisions

### 1. Pure Python Solvers

**Decision**: Implement DPLL and WalkSAT in pure Python rather than using optimized C++ libraries.

**Rationale**:
- Complete instrumentation of internal state
- Deterministic reproducibility across platforms
- Educational value and transparency
- No external dependencies beyond standard scientific Python

**Trade-off**: Accept slower execution for reproducibility and transparency

### 2. Hierarchical Seed Derivation

**Decision**: Use SHA-256-based seed derivation for instance generation.

```python
def derive_seed(master_seed, *identifiers):
    h = master_seed
    for ident in identifiers:
        h = hash((h, ident)) & 0x7FFFFFFF
    return h
```

**Rationale**:
- Deterministic instance generation
- Independent random streams per parameter setting
- Reproducibility from single master seed

### 3. NPZ + JSON Output Format

**Decision**: Save arrays as NPZ and scalars as JSON.

**Rationale**:
- NPZ: Efficient binary storage for large arrays
- JSON: Human-readable metadata and summary statistics
- Separation allows quick inspection of results without loading large files

### 4. Function-Based Architecture

**Decision**: Use functions rather than classes where possible.

**Rationale**:
- Simpler testing and debugging
- Clear input-output relationships
- Easier to understand and modify

**Exceptions**: CNF class for solver state, Timer context manager

### 5. Defensive Programming

**Decision**: Include extensive input validation and error handling.

**Examples**:
- Check n >= k in instance generation
- Clip entropy to [0, log(2)]
- Handle timeout/cutoff gracefully in solvers

**Rationale**: Robustness in long-running experiments

---

## Extensibility

### Adding a New Solver

1. Create solver function in `src/hardness_metrics.py`:
```python
def my_solver(instance, **kwargs) -> Dict:
    # Implementation
    return {
        "satisfiable": True/False/None,
        "decisions": count,  # or "flips": count
        "assignment": {...} or None
    }
```

2. Update `measure_hardness()` to support new solver

3. Add tests in `tests/unit/test_hardness_metrics.py`

### Adding a New Ensemble

1. Create generator in `src/instance_generator.py`:
```python
def generate_my_ensemble(n, alpha, **kwargs) -> dict:
    # Implementation
    return instance_dict
```

2. Add energy model functions in `src/energy_model.py`

3. Update experiments to support new ensemble

### Adding a New Analysis

1. Create analysis function in appropriate module
2. Add to experiment orchestration if needed
3. Add validation check if making quantitative predictions
4. Add tests and documentation

---

## Performance Considerations

### Memory Management

- Instances are generated on-demand, not stored in memory
- Results saved incrementally to disk
- NPZ compression for large arrays

### Parallelization

- `n_jobs` parameter for parallel instance generation
- `joblib.Parallel` for embarrassingly parallel workloads
- Thread-safe random number generation

### Caching

- No explicit caching (deterministic regeneration preferred)
- Results saved to disk for reuse

---

## Security Considerations

- No network operations
- No execution of user-provided code
- File I/O limited to specified directories
- Random seeds are user-controlled

---

## Future Architecture Directions

1. **Plugin System**: Formalize solver/ensemble plugins
2. **Database Backend**: Store results in SQLite for complex queries
3. **Distributed Execution**: Support for cluster computing
4. **Web Interface**: Browser-based experiment configuration
5. **Real-time Monitoring**: Live progress dashboards
