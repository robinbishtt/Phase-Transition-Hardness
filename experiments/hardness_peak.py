from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir
from src.runtime_measurement import localise_hardness_peak
from src.scaling_analysis import finite_size_peak_extrapolation
from src.utils import save_json
from src.validation import run_all_checks

logger = get_logger("hardness_peak")


def main():
    parser = argparse.ArgumentParser(description="Fine-resolution hardness peak localisation.")
    parser.add_argument("--n",              type=int, nargs="+", default=[100, 200])
    parser.add_argument("--n_instances",    type=int, default=200)
    parser.add_argument("--alpha_center",   type=float, default=4.20,
                        help="Centre of the fine α sweep.")
    parser.add_argument("--alpha_width",    type=float, default=0.40,
                        help="Half-width of the fine sweep.")
    parser.add_argument("--n_alpha_points", type=int, default=40,
                        help="Number of α points in the fine sweep.")
    parser.add_argument("--k",             type=int, default=3)
    parser.add_argument("--solver",        choices=["dpll", "walksat"], default="dpll")
    parser.add_argument("--max_decisions", type=int, default=100_000)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output_dir",    default="results")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    logger.info(f"Hardness peak localisation: n={args.n}, "
                f"α∈[{args.alpha_center - args.alpha_width:.2f}, "
                f"{args.alpha_center + args.alpha_width:.2f}], "
                f"points={args.n_alpha_points}")

    result = localise_hardness_peak(
        ns=args.n,
        alpha_center=args.alpha_center,
        width=args.alpha_width,
        n_points=args.n_alpha_points,
        n_instances=args.n_instances,
        k=args.k,
        solver=args.solver,
        master_seed=args.seed,
        max_decisions=args.max_decisions,
        output_dir=args.output_dir,
    )

    alpha_stars   = result["alpha_stars"]
    gamma_maxima  = result["gamma_maxima"]
    alpha_star_inf = result["alpha_star_inf"]

    logger.info("Per-size peak estimates:")
    for n, a_star, g_max in zip(args.n, alpha_stars, gamma_maxima):
        logger.info(f"  n={n:4d}: α* = {a_star:.4f},  γ_max = {g_max:.5f}")
    logger.info(f"Extrapolated α*(∞) = {alpha_star_inf:.4f}")

    save_json(
        {
            "ns":            args.n,
            "alpha_stars":   alpha_stars.tolist(),
            "gamma_maxima":  gamma_maxima.tolist(),
            "alpha_star_inf": float(alpha_star_inf),
        },
        f"{args.output_dir}/hardness_peak_summary.json",
    )

    logger.info("Done. Hardness peak results saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())