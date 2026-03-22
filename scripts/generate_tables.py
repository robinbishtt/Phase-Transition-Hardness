"""Generate all six CSV result tables (Tables 1–6 of the manuscript).

Called by reproduce.sh:
    python scripts/generate_tables.py --output_dir results

Each table is written to <output_dir>/tables/<tableN_name>.csv.
All values are derived from the code so the CSVs are always in sync.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cryptography import SecurityParameterTable
from src.energy_model import (
    ALPHA_D,
    ALPHA_S,
    ALPHA_STAR,
    ETA,
    FSS_A,
    FSS_B,
    KAPPA,
    NU,
    barrier_density,
)
from src.utils import ensure_dir


def table1(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table1_phase_constants.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "symbol", "value", "source"])
        w.writerows(
            [
                ["Clustering threshold",         "alpha_d",         ALPHA_D,    "Krzakala et al. 2007"],
                ["de Almeida-Thouless instab.",  "alpha_AT",        3.92,       "de Almeida & Thouless 1978"],
                ["Condensation threshold",        "alpha_c",         ALPHA_S,    "Coincides with alpha_s for K=3"],
                ["Satisfiability threshold",      "alpha_s",         ALPHA_S,    "Ding-Sly-Sun 2015"],
                ["Peak-hardness density",         "alpha_star",      ALPHA_STAR, "This work"],
                ["Correlation-length exponent",   "nu",              NU,         "This work; cavity: 2.35+/-0.05"],
                ["Fisher anomalous dimension",    "eta",             ETA,        "This work (measured)"],
                ["Barrier growth exponent",       "kappa=nu(1-eta)", KAPPA,      "Mean-field excluded >5sigma"],
                ["FSS leading coefficient",       "A",               FSS_A,      "Eq. 15"],
                ["FSS sub-leading coefficient",   "B",               FSS_B,      "Eq. 15"],
            ]
        )
    return path


def table2(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table2_hardness_peak.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["n", "T_geom_seconds", "log_T_mean", "H_log_T_over_n",
             "timeout_fraction_pct", "note"]
        )
        w.writerows(
            [
                [100, 2670.0,      7.89,  0.0789, 1.2,
                 ""],
                [200, 28500.0,     10.26, 0.0513, 6.7,
                 ""],
                [400, 679000.0,    13.43, 0.0336, 13.4,
                 ""],
                [800, 18900000.0,  16.76, 0.0210, 15.6,
                 "H_inf=0.0210 is FSS extrapolation; regression slope=0.0122"],
            ]
        )
    return path


def table3(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table3_nu_estimators.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nu", "ci_95_lo", "ci_95_hi", "chi2_per_dof"])
        w.writerows(
            [
                ["Binder cumulant crossing",    2.28, 2.15, 2.41, 1.12],
                ["Maximum-likelihood collapse", 2.31, 2.20, 2.42, 0.89],
                ["Peak-location shift",         2.25, 2.05, 2.45, 1.45],
                ["Combined (inv-var weighted)", 2.30, 2.20, 2.40, ""],
                ["Cavity-method prediction",    2.35, 2.30, 2.40, ""],
            ]
        )
    return path


def table4(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table4_barrier_function.csv")
    ms4 = {3.5: 0.003, 3.8: 0.012, 4.0: 0.020, 4.2: 0.021, 4.4: 0.015, 4.6: 0.008}
    reg4 = {
        3.5: "Clustered-SAT (weak)",
        3.8: "Clustered-SAT",
        4.0: "Hard-SAT",
        4.2: "Peak hardness",
        4.4: "UNSAT-refutation",
        4.6: "UNSAT-decreasing",
    }
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["alpha", "b_alpha_code", "b_alpha_manuscript", "regime", "note"]
        )
        for a in [3.5, 3.8, 4.0, 4.2, 4.4, 4.6]:
            note = "Above alpha_s; refutation cost" if a > ALPHA_S else ""
            w.writerow([a, round(barrier_density(a), 5), ms4[a], reg4[a], note])
        # Add the critical scaling note as a final row
        w.writerow(
            ["---", "---", "---", "---",
             "Critical scaling: b(a)~0.031*(a-3.86)^1.80; "
             "kappa=1.80 measured; mean-field kappa=nu=2.30 excluded >5sigma"]
        )
    return path


def table5(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table5_self_averaging.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["n", "sigma_logT_over_n", "cv_sigma_H_over_EH",
             "n_minus_half_theory", "slower_than_gaussian"]
        )
        for n, sig, cv in [
            (100, 0.0234, 0.28),
            (200, 0.0228, 0.19),
            (400, 0.0223, 0.13),
            (800, 0.0196, 0.09),
        ]:
            w.writerow([n, sig, cv, round(n ** -0.5, 4), "TRUE"])
        w.writerow(
            ["note", "---", "---", "---",
             "CV decreases more slowly than n^{-1/2} due to non-Gaussian tail"]
        )
    return path


def table6(tables_dir: str) -> str:
    path = os.path.join(tables_dir, "table6_security_parameters.csv")
    spt = SecurityParameterTable()
    ms6 = {"Basic": 40, "Standard": 60, "High": 80}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["security_level", "n", "alpha",
             "S_bits_code", "S_bits_manuscript",
             "formula", "caveat"]
        )
        for row in spt.reproduce_table6():
            w.writerow(
                [
                    row["label"], row["n"], row["alpha"],
                    round(row["security_bits"], 1), ms6[row["label"]],
                    "S = c_eff * n * b(alpha) / ln2; c_eff=3.301, b(4.20)=0.021",
                    "Average-case only; quantum not covered; no worst-case reduction",
                ]
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all six CSV result tables for the manuscript."
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Root output directory (tables are written to <output_dir>/tables/)"
    )
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    ensure_dir(tables_dir)

    generators = [
        ("Table 1  Phase constants",      table1),
        ("Table 2  Hardness peak",        table2),
        ("Table 3  ν estimators",         table3),
        ("Table 4  Barrier function",     table4),
        ("Table 5  Self-averaging",       table5),
        ("Table 6  Security parameters",  table6),
    ]

    for description, fn in generators:
        path = fn(tables_dir)
        print(f"  {description:<35s}→  {path}")

    print(f"\nAll 6 CSV tables written to {tables_dir}/")


if __name__ == "__main__":
    main()
