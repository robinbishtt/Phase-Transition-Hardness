from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir, save_json, load_npz
from src.runtime_measurement import alpha_sweep
from src.scaling_analysis import run_exponential_scaling
from src.barrier_analysis import run_barrier_scaling_sweep, barrier_hardness_correlation

logger = get_logger("scaling_law_verification")


def main():
    parser = argparse.ArgumentParser(description="Exponential runtime scaling verification.")
    parser.add_argument("--n",             type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--n_instances",   type=int, default=1000)
    parser.add_argument("--alpha_min",     type=float, default=3.5)
    parser.add_argument("--alpha_max",     type=float, default=5.0)
    parser.add_argument("--alpha_step",    type=float, default=0.10)
    parser.add_argument("--k",             type=int,   default=3)
    parser.add_argument("--solver",        choices=["dpll", "walksat"], default="dpll")
    parser.add_argument("--max_decisions", type=int,   default=100_000)
    parser.add_argument("--seed",          type=int,   default=20240223)
    parser.add_argument("--output_dir",    default="results")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)

    logger.info(f"Scaling law verification: n={args.n}, "
                f"α=[{args.alpha_min}, {args.alpha_max}]")


    logger.info("Measuring γ(α, n) …")
    hardness_result = alpha_sweep(
        ns=args.n,
        alphas=alphas,
        n_instances=args.n_instances,
        k=args.k,
        solver=args.solver,
        master_seed=args.seed,
        max_decisions=args.max_decisions,
        output_dir=args.output_dir,
    )


    logger.info("Fitting exponential scaling log T̄ = γ·n …")
    scaling_result = run_exponential_scaling(
        ns=args.n,
        alphas=alphas,
        gamma_matrix=hardness_result["gamma_mean_matrix"],
        output_dir=args.output_dir,
    )

    logger.info(f"Mean R² = {scaling_result['mean_r2']:.4f}")


    logger.info("Computing theoretical barrier density b(α) …")
    barrier_result = run_barrier_scaling_sweep(
        ns=args.n, alphas=alphas, k=args.k, output_dir=args.output_dir
    )



    largest_n_idx = -1
    gamma_mean_largest = hardness_result["gamma_mean_matrix"][largest_n_idx]
    corr = barrier_hardness_correlation(alphas, gamma_mean_largest, k=args.k)

    logger.info(f"Barrier–hardness correlation r = {corr['correlation']:.4f} "
                f"(p = {corr['p_value']:.4e})")

    save_json(
        {
            "mean_r2":           scaling_result["mean_r2"],
            "barrier_hardness_r": corr["correlation"],
            "barrier_hardness_p": corr["p_value"],
            "conjecture_1":      "SUPPORTED" if scaling_result["mean_r2"] >= 0.85 else "NOT SUPPORTED",
        },
        f"{args.output_dir}/scaling_law_summary.json",
    )

    logger.info("Scaling law verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())