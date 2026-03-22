# File Structure

Complete directory structure for the Phase-Transition-Hardness repository.

## Directory Tree

```
Phase-Transition-Hardness/
├── .github/
│   └── workflows/
│       ├── experiment-tracker.yml
│       ├── lint-and-format.yml
│       ├── reproduce.yml
│       ├── run-tests.yml
│       ├── validate-figures.yml
│       └── validate-notebooks.yml
│
├── ablation/
│   ├── 01_finite_n_correction.py
│   ├── 02_off_critical_hardness.py
│   ├── 03_k_variation.py
│   ├── 04_solver_comparison.py
│   ├── 05_censoring_sensitivity.py
│   ├── 06_bp_convergence_threshold.py
│   ├── 07_sample_size_sensitivity.py
│   └── 08_complexity_functional_correction.py
│
├── config/
│   ├── experiment_config.yaml
│   └── validation_config.yaml
│
├── data/
│   ├── barrier_function_table4.json
│   ├── fss_result_table3.json
│   ├── hardness_peak_table2.json
│   ├── phase_constants_table1.json
│   ├── security_parameters_table6.json
│   └── synthetic_alpha_sweep.npz
│
├── docs/
│   ├── API.md
│   ├── APPENDIX_MAPPING.md
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── LIMITATIONS.md
│   ├── MATHEMATICAL_PROOFS.md
│   ├── REPRODUCIBILITY.md
│   ├── REPRODUCIBILITY_CHECKLIST.md
│   ├── SYMBOL_LIST.md
│   └── TROUBLESHOOTING.md
│
├── experiments/
│   ├── alpha_sweep.py
│   ├── finite_size_scaling.py
│   ├── hardness_peak.py
│   └── scaling_law_verification.py
│
├── figures/
│   ├── extended_data_figures.py
│   ├── generate_all_figures.py
│   ├── hardness_plots.py
│   ├── landscape_visuals.py
│   ├── phase_transition_plots.py
│   └── scaling_collapse.py
│
├── notebooks/
│   ├── 01_introduction_and_setup.ipynb
│   ├── 02_instance_generation.ipynb
│   ├── 03_phase_transition_psat.ipynb
│   ├── 04_hardness_peak.ipynb
│   ├── 05_finite_size_scaling.ipynb
│   ├── 06_critical_exponent_nu.ipynb
│   ├── 07_barrier_hardness_correspondence.ipynb
│   ├── 08_self_averaging.ipynb
│   ├── 09_cross_solver_validation.ipynb
│   ├── 10_bp_equations.ipynb
│   ├── 11_survey_propagation.ipynb
│   ├── 12_cryptographic_owf.ipynb
│   ├── 13_proof_of_work.ipynb
│   ├── 14_intensive_barrier_function.ipynb
│   ├── 15_fss_conjecture_validation.ipynb
│   ├── 16_self_averaging_verification.ipynb
│   ├── 17_cryptographic_security_analysis.ipynb
│   └── 18_complete_pipeline_reproduction.ipynb
│
├── results/
│   ├── figures/        (15 PNG manuscript figures)
│   └── tables/         (6 CSV tables - Tables 1–6)
│
├── scripts/
│   ├── generate_figures.sh
│   ├── generate_tables.py
│   ├── quick_test.sh
│   ├── reproduce_all_figures.sh
│   ├── run_ablations.sh
│   ├── run_full_experiment.sh
│   └── setup_dev.sh
│
├── src/
│   ├── barrier_analysis.py
│   ├── binder_cumulant/
│   │   ├── binder_analysis.py
│   │   └── critical_exponent.py
│   ├── cryptography/
│   │   ├── one_way_function.py
│   │   ├── prg_construction.py
│   │   ├── proof_of_work.py
│   │   └── security_parameters.py
│   ├── data_management/
│   │   ├── database.py
│   │   ├── export.py
│   │   └── import_.py
│   ├── energy_model.py
│   ├── hardness_metrics.py
│   ├── instance_generator.py
│   ├── phase_transition.py
│   ├── proofs/
│   │   ├── barrier_bounds.py
│   │   ├── complexity_functional.py
│   │   ├── fss_derivation.py
│   │   └── runtime_bounds.py
│   ├── rigidity_analysis.py
│   ├── runtime_measurement.py
│   ├── scaling_analysis.py
│   ├── solver_wrappers/
│   │   ├── cadical_wrapper.py
│   │   └── kissat_wrapper.py
│   ├── statistics.py
│   ├── survey_propagation/
│   │   ├── bp_equations.py
│   │   ├── sp_equations.py
│   │   └── warning_propagation.py
│   ├── utils.py
│   ├── validation.py
│   └── whitening_core.py
│
└── tests/
    ├── ablation/
    │   └── test_ablation.py
    ├── integration/
    │   └── test_end_to_end.py
    ├── robustness/
    │   └── test_robustness.py
    ├── scaling/
    │   └── test_scaling.py
    ├── unit/
    │   ├── test_barrier_analysis.py
    │   ├── test_binder_cumulant.py
    │   ├── test_cryptography.py
    │   ├── test_data_management.py
    │   ├── test_energy_model.py
    │   ├── test_hardness_metrics.py
    │   ├── test_instance_generator.py
    │   ├── test_phase_transition.py
    │   ├── test_proofs.py
    │   ├── test_rigidity_analysis.py
    │   ├── test_runtime_measurement.py
    │   ├── test_scaling_analysis.py
    │   ├── test_solver_wrappers.py
    │   ├── test_statistics.py
    │   ├── test_survey_propagation.py
    │   ├── test_utils.py
    │   ├── test_validation.py
    │   └── test_whitening_core.py
    └── validation/
        ├── test_manuscript_claims.py
        └── test_validation.py
```

## Key Files

| File | Purpose |
|---|---|
| `reproduce.sh` | Single-command reproduction of all results |
| `pyproject.toml` | PEP 517 build configuration and tool settings |
| `setup.cfg` | Legacy setuptools metadata |
| `requirements.txt` | Pinned runtime dependencies (numpy 2.4.2, scipy 1.17.0, etc.) |
| `requirements-dev.txt` | Development and testing dependencies |
| `environment.yml` | Conda environment with exact package versions |
| `CITATION.cff` | Machine-readable citation metadata |
| `.pre-commit-config.yaml` | Black, isort, ruff, mypy hooks |
| `Makefile` | Convenience targets for common tasks |
