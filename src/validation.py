from __future__ import annotations

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, load_npz, get_logger, save_json, ensure_dir

logger = get_logger(__name__)





ALPHA_S_LO       = 4.20
ALPHA_S_HI       = 4.35
ALPHA_STAR_LO    = 4.10
ALPHA_STAR_HI    = 4.40
GAMMA_MAX_LO     = 0.005
GAMMA_MAX_HI     = 0.05
EXP_R2_MIN       = 0.85
FSS_RESIDUAL_MAX = 0.10
NU_LO            = 1.5
NU_HI            = 3.5
ALPHA_D_REF      = 3.86






def check_1_alpha_s(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_json(f"{results_dir}/phase_transition_summary.json")
        thresholds = data.get("thresholds", {})
        if not thresholds:
            return False, "No threshold data found."

        ns = sorted(int(k) for k in thresholds)
        alpha_s = thresholds[str(ns[-1])]
        if np.isnan(alpha_s):
            return False, f"α_s = NaN (not estimated)"
        ok = ALPHA_S_LO <= alpha_s <= ALPHA_S_HI
        return ok, f"α_s = {alpha_s:.4f}  (expected [{ALPHA_S_LO}, {ALPHA_S_HI}])"
    except FileNotFoundError:
        return False, "phase_transition_summary.json not found  run alpha_sweep.py first."


def check_2_alpha_star(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_json(f"{results_dir}/alpha_sweep_summary.json")
        alpha_star = data.get("alpha_star_inf", float("nan"))
        if np.isnan(alpha_star):

            stars = data.get("alpha_stars", [])
            alpha_star = stars[-1] if stars else float("nan")
        ok = ALPHA_STAR_LO <= alpha_star <= ALPHA_STAR_HI
        return ok, f"α* = {alpha_star:.4f}  (expected [{ALPHA_STAR_LO}, {ALPHA_STAR_HI}])"
    except FileNotFoundError:
        return False, "alpha_sweep_summary.json not found  run alpha_sweep.py first."


def check_3_gamma_max(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_json(f"{results_dir}/alpha_sweep_summary.json")
        maxima = data.get("gamma_maxima", [])
        if not maxima:
            return False, "No γ_max data found."
        gamma_max = float(np.max(maxima))
        ok = GAMMA_MAX_LO <= gamma_max <= GAMMA_MAX_HI
        return ok, f"γ_max = {gamma_max:.5f}  (expected [{GAMMA_MAX_LO}, {GAMMA_MAX_HI}])"
    except FileNotFoundError:
        return False, "alpha_sweep_summary.json not found."


def check_4_exponential_scaling(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_json(f"{results_dir}/exponential_scaling_summary.json")
        r2   = data.get("mean_r2", 0.0)
        gmax = data.get("max_gamma", 0.0)
        ok   = (r2 >= EXP_R2_MIN) and (gmax > 0)
        return ok, f"R² = {r2:.4f}, γ_slope_max = {gmax:.5f}  (R² ≥ {EXP_R2_MIN}, γ > 0)"
    except FileNotFoundError:
        return False, "exponential_scaling_summary.json not found  run scaling_law_verification.py."


def check_5_fss_residual(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data     = load_json(f"{results_dir}/fss_result.json")
        residual = data.get("residual", float("inf"))
        ok       = residual < FSS_RESIDUAL_MAX
        return ok, f"FSS residual = {residual:.6f}  (expected < {FSS_RESIDUAL_MAX})"
    except FileNotFoundError:
        return False, "fss_result.json not found  run finite_size_scaling.py."


def check_6_nu(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_json(f"{results_dir}/fss_result.json")
        nu   = data.get("nu", float("nan"))
        ok   = NU_LO <= nu <= NU_HI
        return ok, f"ν = {nu:.4f}  (expected [{NU_LO}, {NU_HI}])"
    except FileNotFoundError:
        return False, "fss_result.json not found."


def check_7_barrier_positivity(results_dir: str) -> Tuple[bool, str]:
    
    from src.energy_model import barrier_density, ALPHA_D, ALPHA_S
    alphas_test = np.linspace(ALPHA_D + 0.05, ALPHA_S - 0.05, 20)
    b_values    = [barrier_density(a) for a in alphas_test]
    all_positive = all(b > 0 for b in b_values)
    min_b = min(b_values)
    return all_positive, (
        f"min b(α) in hard phase = {min_b:.6f}  (expected > 0)"
        if all_positive else
        f"b(α) ≤ 0 found (min = {min_b:.6f})"
    )


def check_8_psat_monotone(results_dir: str) -> Tuple[bool, str]:
    
    try:
        data = load_npz(f"{results_dir}/phase_transition.npz")
        psat_matrix = data["psat_matrix"]

        violations = 0
        for row in psat_matrix:
            diffs = np.diff(row)
            violations += int(np.sum(diffs > 0.05))
        ok = violations == 0
        return ok, (
            f"P_sat is monotone non-increasing (violations = 0)"
            if ok else
            f"P_sat monotonicity violated: {violations} points with increase > 0.05"
        )
    except FileNotFoundError:
        return False, "phase_transition.npz not found  run alpha_sweep.py first."






CHECKS = [
    (1, "Satisfiability threshold",      check_1_alpha_s,           "≈ 4.267"),
    (2, "Hardness peak location",        check_2_alpha_star,        "≈ 4.20"),
    (3, "Peak hardness density",         check_3_gamma_max,         "≈ 0.015"),
    (4, "Exponential scaling fit",       check_4_exponential_scaling, "Conjecture 1"),
    (5, "FSS collapse quality",          check_5_fss_residual,      "Figure 4"),
    (6, "FSS critical exponent",         check_6_nu,                "≈ 2.3"),
    (7, "Barrier density positivity",    check_7_barrier_positivity,"Conjecture 1"),
    (8, "P_sat monotonicity",            check_8_psat_monotone,     "Threshold structure"),
]


def run_all_checks(results_dir: str = "results") -> Dict:
    
    ensure_dir(results_dir)

    print("\n" + "=" * 70)
    print("  Phase-Transition-Hardness  Validation Suite")
    print("=" * 70)
    print(f"  Results directory: {results_dir}\n")

    passed = 0
    failed = 0
    details = []

    for idx, name, fn, manuscript_val in CHECKS:
        try:
            ok, message = fn(results_dir)
        except Exception as exc:
            ok      = False
            message = f"Exception: {exc}"

        status = "✓ PASS" if ok else "✗ FAIL"
        colour = "\033[92m" if ok else "\033[91m"
        reset  = "\033[0m"

        print(f"  [{idx}] {colour}{status}{reset}  {name}")
        print(f"        {message}")
        print(f"        Manuscript value: {manuscript_val}\n")

        details.append({
            "check":          idx,
            "name":           name,
            "passed":         ok,
            "message":        message,
            "manuscript_val": manuscript_val,
        })

        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"  Total: {passed}/{passed + failed} checks passed")
    print("=" * 70 + "\n")

    summary = {
        "passed": passed,
        "failed": failed,
        "total":  passed + failed,
        "details": details,
    }

    save_json(summary, f"{results_dir}/validation.json")
    return summary






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run validation checks for Phase-Transition-Hardness."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing experiment result files.")
    args = parser.parse_args()

    summary = run_all_checks(args.results_dir)
    sys.exit(0 if summary["failed"] == 0 else 1)