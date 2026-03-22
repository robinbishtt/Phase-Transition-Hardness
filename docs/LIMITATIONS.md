# Limitations and Scope Boundaries

This document explicitly states the theoretical, experimental, and practical
limitations of the work, as required for scientific rigour.  These limitations
are also stated in the manuscript (Section 11.4 and the Discussion).

---

## Theoretical Limitations

**Conjecture 4 is not a theorem.**  The barrier-hardness correspondence
`log T(n,α) = Θ(n·b(α))` is a conjecture supported by a 20-page proof
sketch (Supplementary Section 4) that closes the bound for DPLL solvers.
The extension to CDCL solvers requires new combinatorial tools to handle
the learned-clause dynamics; this gap remains open.

**The Fisher anomalous dimension η ≈ 0.22 is measured, not derived.**
The current value comes from fitting `κ = ν(1−η)` to the barrier-function
data (Section 4.5).  A first-principles derivation of η from loop corrections
to the tree-level cavity equations is an open problem (Section 6, item ii).

**The 1RSB ansatz may not be exact near the SAT-UNSAT threshold.**  For
random 3-SAT, full replica symmetry breaking (full-RSB) corrections become
important as α → α_s = 4.267.  The manuscript notes this limitation in
Supplementary Section 5.5.

**The proof covers average-case hardness only.**  There is no known
average-case-to-worst-case reduction for random K-SAT, so the security
guarantees of the cryptographic constructions (Section 5) cannot be elevated
to worst-case hardness without a major new result.

---

## Experimental Limitations

**Solvers: Kissat 3.1.0 and CaDiCaL 1.9.4 only.**  Results may differ
for other CDCL implementations.  The cross-solver Spearman correlations
(ρ_s ≥ 0.86) partially address this, but the quantitative constants c₁, c₃
in the runtime bounds are not universal.

**System sizes n ≤ 800.**  At n = 800 the censoring rate reaches 15.6%
at peak hardness.  Larger n would improve the thermodynamic-limit
extrapolation of H_∞ but require proportionally more compute (∼450,000
CPU-hours for n=800 alone).

**Censored-runtime estimation uses a conservative lower bound.**  The full
Kaplan-Meier + Tobit regression pipeline described in Supplementary Section 5.3
is documented in `src/statistics.py` but the Tobit regression itself uses
only the conservative lower bound `censored_log_mean()`.  The relative bias
is bounded by the censoring fraction (≤ 15.6% at peak hardness).

**Non-Gaussian runtime tails.**  The hardness metric H = E[log T]/n uses
geometric mean runtime.  The distribution of log T has heavier tails than
Gaussian (see Supplementary Figure S17), so the standard error of H is
larger than the Gaussian prediction n^{−1/2} (Table 5 and Section 4.6).

---

## Cryptographic Limitations

**Security is average-case, not worst-case.**  Instances that deviate from
the uniform random ensemble (planted solutions, near-duplicate clauses,
biased literal signs) invalidate the hardness argument entirely.  Instance
generation must be publicly verifiable and demonstrably uniform.

**Quantum algorithms are not covered.**  Conjecture 4 applies to classical
CDCL solvers.  Quantum annealing and fault-tolerant quantum tunnelling
through the classical energy barriers are not analysed, and their impact
on the security parameters of Table 6 is an open question.

**Security estimates are conservative and hardware-dependent.**  The formula
S ≈ n·H_∞/ln 2 with H_∞ = 0.021 provides a lower bound.  Actual security
depends on the constant c₁ in the Arrhenius lower bound (Proposition 5),
which is α-dependent and not tightly characterised for all α ∈ (α_d, α_s).

---

## Code Limitations

**DPLL and WalkSAT are proxy solvers.**  `src/hardness_metrics.py` documents
this explicitly: `measure_hardness()` returns `log(decisions+1)/n`, not the
paper's metric `E[log T]/n` in wall-clock seconds.  Quantitative comparison
with Table 2 requires installing Kissat/CaDiCaL binaries and using
`measure_cdcl_hardness()`.

**BP / SP implementations are pedagogical.**  The Belief Propagation and
Survey Propagation implementations in `src/survey_propagation/` are intended
to illustrate the cavity equations, not for large-scale production use.
Population-dynamics SP (the numerically accurate approach for large-n
complexity evaluation) is not implemented; only the analytical calibration
from `src/energy_model.py` is used.

**The analytical b(α) formula is calibrated, not exact.**  The barrier
density `barrier_density()` uses a two-sided power law fitted to match the
experimental data.  The exact 1RSB solution from the SP fixed point (which
would require population dynamics) is not computed numerically.
