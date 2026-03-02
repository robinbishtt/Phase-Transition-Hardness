# Changelog

All notable changes to the Phase-Transition-Hardness project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-23

### Added
- Initial release of Phase-Transition-Hardness codebase
- Complete implementation of DPLL and WalkSAT solvers
- Energy model with entropy, complexity, and barrier functions
- Phase transition detection and threshold location
- Hardness measurement and exponential scaling analysis
- Finite-size scaling (FSS) collapse analysis
- Barrier-hardness correlation analysis
- Comprehensive test suite (unit, integration, ablation, robustness)
- Automated validation suite with 8 checks
- Experiment orchestration scripts
- Jupyter notebook tutorials
- Docker containerization
- GitHub Actions CI/CD workflows

### Features
- **Reproducibility**: All experiments deterministic from single master seed
- **Modularity**: Clean separation of concerns between components
- **Extensibility**: Easy to add new solvers, ensembles, or analysis methods
- **Validation**: Automated checks against manuscript predictions
- **Documentation**: Comprehensive API, architecture, and reproducibility guides

### Scientific Claims
- Conjecture 1: Barrier-Hardness Correspondence (log T = Θ(n·b(α)))
- Critical threshold: α_s ≈ 4.267
- Critical exponent: ν ≈ 2.30 ± 0.18
- Hardness peak: α* ≈ 4.20

### Dependencies
- numpy >= 1.24.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- networkx >= 3.1.0
- tqdm >= 4.65.0
- joblib >= 1.3.0
- pandas >= 2.0.0
- seaborn >= 0.12.0

### Testing
- 100+ unit tests covering all modules
- Integration tests for end-to-end workflows
- Ablation tests for component analysis
- Robustness tests for stability verification
- Scaling tests for performance validation
- Validation tests against manuscript claims

### Documentation
- README.md: Project overview and quick start
- docs/API.md: Complete API reference
- docs/ARCHITECTURE.md: System architecture and design
- docs/REPRODUCIBILITY.md: Reproduction guide
- notebooks/: Tutorial notebooks
- scripts/: Automation scripts

## [Unreleased]

### Planned
- GPU acceleration for large-scale experiments
- Additional CSP ensembles (XORSAT, NAE-SAT)
- Interactive web dashboard for results
- Database backend for result storage
- Distributed execution support

---

## Version History

- v1.0.0 (2026-02-23): Initial public release
  - Manuscript publication date
  - Zenodo DOI: 10.5281/zenodo.18764848
  - GitHub release: robinbishtt/Phase-Transition-Hardness@v1.0.0
