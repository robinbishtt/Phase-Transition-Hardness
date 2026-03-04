# Phase-Transition Structure as Foundation for Cryptographic Hardness

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> **Formalizing the link between statistical physics and computational complexity to establish a structural foundation for cryptographic hardness.**

This repository contains the complete computational artifact for the research manuscript on phase-transition phenomena in random constraint satisfaction problems and their correspondence to computational hardness.

## Overview

The Phase-Transition-Hardness codebase implements a comprehensive framework for studying the relationship between:

- **Phase transitions** in random k-SAT (sharp changes in solution space structure)
- **Computational hardness** (exponential growth of solver runtime)
- **Free-energy barriers** (energy landscape features in statistical mechanics)

### Key Scientific Contribution

**Conjecture 1 (Barrier-Hardness Correspondence):** For random k-SAT with k ≥ 3, the typical solver runtime T(n, α) satisfies:

```
log T(n, α) = Θ(n · b(α))
```

where b(α) is the free-energy barrier density computed from the cavity method.

### Critical Thresholds (3-SAT)

| Threshold | Symbol | Value | Significance |
|-----------|--------|-------|--------------|
| Clustering | α_d | ≈ 3.86 | Solution space shatters into clusters |
| Rigidity | α_r | ≈ 4.00 | Variables become frozen within clusters |
| Condensation | α_c | ≈ 4.10 | Gibbs measure concentrates on few clusters |
| Satisfiability | α_s | ≈ 4.267 | Transition from SAT to UNSAT |
| Hardness Peak | α* | ≈ 4.20 | Maximum computational difficulty |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/robinbishtt/Phase-Transition-Hardness.git
cd Phase-Transition-Hardness

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### 5-Minute Demo

```python
from src.instance_generator import generate_ksat_instance
from src.hardness_metrics import measure_hardness
from src.energy_model import barrier_density

# Generate a random 3-SAT instance
instance = generate_ksat_instance(n=50, alpha=4.2, k=3, seed=42)

# Measure computational hardness
hardness = measure_hardness(instance, solver='dpll')
print(f"Hardness density γ = {hardness:.4f}")

# Compare with theoretical barrier
barrier = barrier_density(4.2)
print(f"Barrier density b(α) = {barrier:.4f}")
```

### Run Experiments

```bash
# Quick test (5 minutes)
./scripts/quick_test.sh

# Full reproduction (4.5 hours for n=800)
./scripts/reproduce_all_figures.sh
```

### Validate Results

```bash
python src/validation.py --results_dir results
```

Expected output: 8/8 validation checks passing.

## Repository Structure

```
Phase-Transition-Hardness/
├── src/                      # Core computational library
│   ├── __init__.py
│   ├── utils.py              # Utilities (RNG, I/O, math)
│   ├── instance_generator.py # Random k-SAT generation
│   ├── energy_model.py       # Thermodynamic models
│   ├── hardness_metrics.py   # SAT solvers (DPLL, WalkSAT)
│   ├── phase_transition.py   # P_sat estimation
│   ├── barrier_analysis.py   # Free-energy barriers
│   ├── scaling_analysis.py   # FSS and exponential scaling
│   ├── statistics.py         # Statistical analysis
│   ├── runtime_measurement.py # Experiment orchestration
│   └── validation.py         # Automated validation
│
├── experiments/              # Experiment scripts
│   ├── alpha_sweep.py        # Hardness density curve
│   ├── hardness_peak.py      # Peak localization
│   ├── finite_size_scaling.py # FSS analysis
│   └── scaling_law_verification.py # Conjecture 1 test
│
├── tests/                    # Comprehensive test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── ablation/             # Ablation tests
│   ├── robustness/           # Robustness tests
│   ├── scaling/              # Scaling tests
│   └── validation/           # Validation tests
│
├── docs/                     # Documentation
│   ├── API.md                # Complete API reference
│   ├── ARCHITECTURE.md       # System architecture
│   └── REPRODUCIBILITY.md    # Reproduction guide
│
├── notebooks/                # Tutorial notebooks
│   ├── 01_introduction.ipynb
│   ├── 02_alpha_sweep.ipynb
│   └── 03_finite_size_scaling.ipynb
│
├── scripts/                  # Automation scripts
│   ├── reproduce_all_figures.sh
│   ├── quick_test.sh
│   └── setup_dev.sh
│
├── config/                   # Configuration files
│   ├── experiment_config.yaml
│   └── validation_config.yaml
│
├── .github/workflows/        # CI/CD workflows
│   ├── run-tests.yml
│   ├── lint-and-format.yml
│   └── validate-notebooks.yml
│
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Dependencies
├── Dockerfile                # Container image
├── README.md                 # This file
├── LICENSE                   # MIT License
└── CITATION.cff              # Citation metadata
```

## Features

### Solvers

- **DPLL**: Complete solver with unit propagation and pure literal elimination
- **WalkSAT**: Incomplete local search solver with noise parameter

### Analysis Tools

- **Phase Transition Detection**: P_sat estimation and threshold location
- **Hardness Measurement**: Runtime/decision counting with normalization
- **Finite-Size Scaling**: Data collapse for critical exponent estimation
- **Barrier Analysis**: Free-energy barrier computation and correlation
- **Statistical Analysis**: Bootstrap confidence intervals, distribution fitting

### Validation

- 8 automated checks against manuscript predictions
- Comprehensive test suite (100+ tests)
- CI/CD integration for continuous validation

## Documentation

- **[API Reference](docs/API.md)**: Complete documentation of all functions
- **[Architecture](docs/ARCHITECTURE.md)**: System design and module dependencies
- **[Reproducibility](docs/REPRODUCIBILITY.md)**: Step-by-step reproduction guide

## Hardware Requirements

### Minimum

- CPU: 4 cores
- RAM: 8 GB
- Python: 3.10+

### Recommended for Full Reproduction

- CPU: 8+ cores
- RAM: 16+ GB
- Python: 3.11

### Manuscript Experiments

- CPU: Intel Xeon Gold 6248R @ 3.0 GHz
- Cores: 24 physical per node
- RAM: 256 GB per node
- Total CPU-hours: ~450,000

## Computational Budget

| Experiment | Time (n=800) | CPU-hours | Cost (AWS) |
|------------|--------------|-----------|------------|
| Alpha sweep | ~4.5 hours | ~450,000 | ~$2,400 |
| FSS analysis | ~3 hours | ~300,000 | ~$1,600 |
| Peak localization | ~2 hours | ~200,000 | ~$1,100 |


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This research builds on decades of work in statistical mechanics of disordered systems, constraint satisfaction problems, and computational complexity theory. Key theoretical foundations include:

- Replica symmetry breaking in spin glasses (Mézard-Parisi-Virasoro)
- Cavity method for random CSPs (Mézard-Mora-Zecchina)
- Phase transitions in k-SAT (Monasson-Zecchina-Kirkpatrick-Troyansky)
- Algorithmic barriers in random CSPs (Achlioptas-Coja-Oghlan)

## Support

For questions or issues:
1. Check the [documentation](docs/)
2. Review [existing issues](https://github.com/robinbishtt/Phase-Transition-Hardness/issues)
3. Open a new issue with:
   - Command used
   - Error message
   - System information
   - Expected vs actual behavior

---

**Note**: This is a research artifact. For production use, consider performance optimizations and additional error handling.
