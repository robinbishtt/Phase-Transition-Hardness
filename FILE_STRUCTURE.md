# File Structure

Complete directory structure for the Phase-Transition-Hardness repository.

## Directory Tree

```
Phase-Transition-Hardness/
├── .github/
│   └── workflows/              # CI/CD workflows
│       ├── experiment-tracker.yml
│       ├── lint-and-format.yml
│       ├── run-tests.yml
│       └── validate-notebooks.yml
│
├── config/                     # Configuration files
│   ├── experiment_config.yaml  # Experiment parameters
│   └── validation_config.yaml  # Validation criteria
│
├── docs/                       # Documentation
│   ├── API.md                  # Complete API reference
│   ├── ARCHITECTURE.md         # System architecture
│   └── REPRODUCIBILITY.md      # Reproduction guide
│
├── experiments/                # Experiment scripts
│   ├── __init__.py
│   ├── alpha_sweep.py          # Hardness density curve
│   ├── finite_size_scaling.py  # FSS analysis
│   ├── hardness_peak.py        # Peak localization
│   └── scaling_law_verification.py # Conjecture 1 test
│
├── figures/                    # Generated figures (gitignored)
│
├── notebooks/                  # Tutorial notebooks
│   ├── 01_introduction.ipynb   # Getting started
│   ├── 02_alpha_sweep.ipynb    # Alpha sweep demo
│   └── 03_finite_size_scaling.ipynb # FSS demo
│
├── results/                    # Experiment results (gitignored)
│
├── scripts/                    # Automation scripts
│   ├── quick_test.sh           # Quick validation
│   ├── reproduce_all_figures.sh # Full reproduction
│   └── setup_dev.sh            # Dev environment setup
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── barrier_analysis.py     # Free-energy barriers
│   ├── energy_model.py         # Thermodynamic models
│   ├── hardness_metrics.py     # SAT solvers
│   ├── instance_generator.py   # Random k-SAT generation
│   ├── phase_transition.py     # P_sat estimation
│   ├── py.typed                # Type hints marker
│   ├── runtime_measurement.py  # Experiment orchestration
│   ├── scaling_analysis.py     # FSS and scaling
│   ├── statistics.py           # Statistical analysis
│   ├── utils.py                # Utilities
│   └── validation.py           # Automated validation
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── ablation/               # Ablation tests
│   │   ├── __init__.py
│   │   └── test_ablation.py
│   ├── integration/            # Integration tests
│   │   ├── __init__.py
│   │   └── test_end_to_end.py
│   ├── robustness/             # Robustness tests
│   │   ├── __init__.py
│   │   └── test_robustness.py
│   ├── scaling/                # Scaling tests
│   │   ├── __init__.py
│   │   └── test_scaling.py
│   ├── unit/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── test_barrier_analysis.py
│   │   ├── test_energy_model.py
│   │   ├── test_hardness_metrics.py
│   │   ├── test_instance_generator.py
│   │   ├── test_phase_transition.py
│   │   ├── test_statistics.py
│   │   └── test_utils.py
│   └── validation/             # Validation tests
│       ├── __init__.py
│       └── test_validation.py
│
├── .dockerignore               # Docker ignore rules
├── .gitignore                  # Git ignore rules
├── CHANGELOG.md                # Version history
├── CITATION.cff                # Citation metadata
├── CONTRIBUTING.md             # Contribution guidelines
├── Dockerfile                  # Container image
├── FILE_STRUCTURE.md           # This file
├── LICENSE                     # MIT License
├── MANIFEST.in                 # Package manifest
├── README.md                   # Project overview
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
└── requirements-dev.txt        # Development dependencies
```

## File Count Summary

| Category | Files | Description |
|----------|-------|-------------|
| Source Code | 11 | Core computational library |
| Experiments | 4 | Experiment orchestration scripts |
| Tests | 12 | Comprehensive test suite |
| Documentation | 4 | API, architecture, reproducibility |
| Notebooks | 3 | Tutorial Jupyter notebooks |
| Scripts | 3 | Automation and setup scripts |
| Config | 2 | YAML configuration files |
| CI/CD | 4 | GitHub Actions workflows |
| Packaging | 6 | pyproject.toml, Dockerfile, etc. |
| **Total** | **49+** | Complete production-ready codebase |

## Key Files

### Entry Points

- `experiments/alpha_sweep.py` - Main experiment script
- `src/validation.py` - Validation suite
- `scripts/reproduce_all_figures.sh` - Full reproduction

### Configuration

- `pyproject.toml` - Package metadata and tool configuration
- `config/experiment_config.yaml` - Experiment parameters
- `config/validation_config.yaml` - Validation thresholds

### Documentation

- `README.md` - Quick start and overview
- `docs/API.md` - Complete API reference
- `docs/REPRODUCIBILITY.md` - Reproduction guide

### Testing

- `tests/unit/` - Unit tests for all modules
- `tests/integration/` - End-to-end integration tests
- `tests/validation/` - Tests against manuscript claims
