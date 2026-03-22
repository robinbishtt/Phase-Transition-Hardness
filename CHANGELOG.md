# Changelog

All notable changes to the Phase-Transition-Hardness project are documented here.

---

## 2026-02-23 - Initial public release

### Added
- Complete implementation of DPLL and WalkSAT solvers
- Energy model with entropy, complexity, and barrier functions
- Phase transition detection and threshold location
- Hardness measurement and exponential scaling analysis
- Finite-size scaling (FSS) collapse analysis
- Barrier-hardness correlation analysis
- Comprehensive test suite - 737 tests across unit, integration, validation, ablation, robustness, scaling suites
- Automated validation suite with 8 manuscript checks
- Experiment orchestration scripts
- Jupyter notebook tutorials (18 notebooks)
- Docker containerisation
- GitHub Actions CI workflows

### Scientific Claims
- Conjecture 4: Barrier-Hardness Correspondence - log T = Θ(n·b(α))
- Satisfiability threshold: α_s ≈ 4.267 (Ding-Sly-Sun 2015)
- Critical exponent: ν = 2.30 ± 0.18
- Hardness peak: α* ≈ 4.20

### Dependencies
- numpy ≥ 1.24.0
- scipy ≥ 1.11.0
- matplotlib ≥ 3.7.0
- networkx ≥ 3.1.0
- joblib ≥ 1.3.0
- pandas ≥ 2.0.0
- seaborn ≥ 0.12.0
- pyyaml ≥ 6.0

---

## Planned
- GPU acceleration for large-scale experiments
- Additional CSP ensembles (XORSAT, NAE-SAT)
- Distributed execution support
