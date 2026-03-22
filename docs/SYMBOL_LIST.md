# Symbol and Notation Reference

This document defines every mathematical symbol used in the manuscript
and maps each to its implementation in the codebase.

---

## Problem Definition and Factor Graph

| Symbol | Definition | Code |
|---|---|---|
| n | Number of Boolean variables | `instance['n']` |
| m | Number of clauses | `instance['m']` |
| K | Clause length (this paper: K=3) | `instance['k']` |
| α = m/n | Constraint density | `instance['alpha']` |
| x_i ∈ {0,1} | Boolean variable | keys of assignment dict |
| s_i = (−1)^{1−x_i} ∈ {−1,+1} | Ising spin representation | - |
| σ = (s₁,…,s_n) | Spin configuration | assignment dict |
| Σ_n = {−1,+1}^n | Configuration space | - |
| J_{a,j} ∈ {−1,+1} | Literal sign | sign of literal in clause |
| G = (V∪C, E) | Factor graph | `instance_to_adjacency()` |
| ∂i | Clause nodes adjacent to variable i | `var_to_clauses[vi]` |
| ∂a | Variable nodes adjacent to clause a | `clause_to_vars[ci]` |

---

## Energy and Thermodynamics

| Symbol | Definition | Code |
|---|---|---|
| H_a(σ) | Clause Hamiltonian (Eq. 4) | `count_violated_clauses()` |
| E(σ) = Σ_a H_a(σ) | Total energy (violated clauses) | `count_violated_clauses()` |
| β = 1/T | Inverse temperature | `beta` parameter |
| Z(β,α) = Σ_σ e^{−βE(σ)} | Partition function | `compute_partition_function_log()` |
| f(β,α) = −(βn)^{−1} log Z | Free energy density | `free_energy_density()` |
| s(α) = lim n^{−1} log N_SAT | Entropy density | `rs_entropy_density()` |
| q_0 | Inter-cluster overlap (1RSB) | - |
| q_1 = q_EA | Intra-cluster / Edwards-Anderson overlap | - |
| m | Parisi breaking parameter ∈ [0,1] | - |

---

## Phase Transitions and Critical Phenomena

| Symbol | Value | Definition | Code |
|---|---|---|---|
| α_d = α_r | 3.86 | Clustering / rigidity threshold | `ALPHA_D`, `ALPHA_R` |
| α_AT | 3.92 | de Almeida-Thouless instability | `ALPHA_AT` |
| α_c = α_s | 4.267 | Condensation = SAT/UNSAT threshold | `ALPHA_C`, `ALPHA_S` |
| α* | 4.20 | Peak-hardness density | `ALPHA_STAR` |
| ν | 2.30 ± 0.18 | Correlation-length exponent | `NU` |
| η | ≈ 0.22 | Fisher anomalous dimension | `ETA` |
| κ = ν(1−η) | ≈ 1.80 | Barrier-growth exponent | `KAPPA` |
| ξ ~ |α−α_d|^{−ν} | - | Correlation length | `FSSAnsatz.correlation_length()` |
| U_n(α) = 1 − ⟨q⁴⟩/(3⟨q²⟩²) | - | Binder cumulant | `BinderCumulant.theoretical_binder()` |

---

## Complexity, Barriers, and Hardness

| Symbol | Definition | Code |
|---|---|---|
| Σ(α) = lim n^{−1} log N_clusters | 1RSB complexity | `cluster_complexity()` |
| B(σ₁,σ₂) | Point-to-point barrier (Eq. 10) | `path_barrier()` |
| b(α) | Intensive barrier function | `barrier_density()` |
| B(n,α) = n·b(α) | Extensive barrier height | `barrier_height()` |
| T(n,α) | Geometric mean CDCL runtime | `measure_cdcl_hardness()` |
| H(n,α) = E[log T]/n | Hardness density (Eq. 16) | `measure_hardness()` |
| H_∞ | Thermodynamic-limit barrier ≈ 0.021 | `barrier_density(ALPHA_STAR)` |
| ρ_s | Spearman rank correlation | `scipy.stats.spearmanr` |

---

## Cavity Method and Message Passing

| Symbol | Definition | Code |
|---|---|---|
| m_{i→a} | BP cavity magnetisation (Eq. 6) | `BeliefPropagation._m` |
| u_{a→i} | BP cavity field (Eq. 7) | `BeliefPropagation._u` |
| Z_{i→a}, Z_a, Z_i | BP normalisation factors | - |
| P_{i→a}(u) | SP distribution over cavity fields (Eq. 8) | `SurveyPropagation._eta` |
| η_{a→i} | SP survey (fraction of non-trivial surveys) | `SPResult.eta_surveys` |
| Σ[{P}] | Complexity functional (Eq. 9, corrected) | `ComplexityFunctional` |

---

## Finite-Size Scaling (Eq. 14–15)

| Symbol | Definition | Code |
|---|---|---|
| F(x) | Universal FSS master function | `FSSAnsatz` |
| α*(n) | Finite-size pseudo-critical density | `FSSAnsatz.alpha_star_n()` |
| A = +0.036 | Leading FSS shift coefficient | `FSS_A` |
| B = −1.37 | Sub-leading FSS shift coefficient | `FSS_B` |
| x = n^{1/ν}(α−α*(n)) | FSS scaling variable | `FSSAnsatz.fss_variable()` |

---

## Experimental and Statistical Notation

| Symbol | Definition | Code |
|---|---|---|
| N ∈ {100,200,400,800} | System sizes | `--n` argument |
| I(N,α,ω) | Random 3-SAT instance | `generate_ksat_instance()` |
| ω | Per-instance seed (SHA-256) | `derive_seed()` |
| T_max = 3600 s | Per-instance timeout | `PAPER_TIMEOUT_SECS` |
| R² = 0.9997 | FSS collapse quality | `fss_collapse()` |
| χ²/dof = 0.89 | Reduced chi-squared | - |
| F = 33.77 | ANOVA F-statistic | - |
| S ≈ n·H_∞/ln 2 | Security bits | `compute_security_bits()` |
