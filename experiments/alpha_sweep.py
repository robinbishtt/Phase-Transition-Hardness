from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir, save_json
from src.phase_transition import run_psat_sweep
from src.runtime_measurement import alpha_sweep
from src.scaling_analysis import run_exponential_scaling
from src.validation import run_all_checks

logger = get_logger("alpha_sweep")


def main():
    parser = argparse.ArgumentParser(description="Full α-sweep experiment.")
    parser.add_argument("--n",           type=int, nargs="+", default=[100, 200],
                        help="System sizes (number of variables).")
    parser.add_argument("--n_instances", type=int, default=200,
                        help="Instances per (n, α) measurement point.")
    parser.add_argument("--alpha_min",   type=float, default=3.0,
                        help="Lower bound of the α sweep.")
    parser.add_argument("--alpha_max",   type=float, default=5.0,
                        help="Upper bound of the α sweep.")
    parser.add_argument("--alpha_step",  type=float, default=0.10,
                        help="Grid resolution of the α sweep.")
    parser.add_argument("--k",           type=int,   default=3,
                        help="Clause length (default: 3-SAT).")
    parser.add_argument("--solver",      choices=["dpll", "walksat"], default="dpll",
                        help="SAT algorithm.")
    parser.add_argument("--max_decisions", type=int, default=100_000,
                        help="DPLL branching cutoff per instance.")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Master random seed.")
    parser.add_argument("--output_dir",  default="results",
                        help="Directory for result files.")
    parser.add_argument("--n_jobs",      type=int,   default=1,
                        help="Parallel workers (1 = sequential).")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2,
                       args.alpha_step)

    logger.info(f"Alpha sweep: n={args.n}, α=[{args.alpha_min}, {args.alpha_max}] "
                f"step={args.alpha_step}, instances={args.n_instances}, seed={args.seed}")


    logger.info("Step 1/3: P_sat sweep …")
    psat_result = run_psat_sweep(
        ns=args.n,
        alphas=alphas,
        n_instances=args.n_instances,
        k=args.k,
        master_seed=args.seed,
        solver=args.solver,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )


    logger.info("Step 2/3: Hardness γ sweep …")
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


    logger.info("Step 3/3: Exponential scaling fit …")
    run_exponential_scaling(
        ns=args.n,
        alphas=alphas,
        gamma_matrix=hardness_result["gamma_mean_matrix"],
        output_dir=args.output_dir,
    )


    logger.info("Running validation checks …")
    summary = run_all_checks(args.output_dir)

    logger.info(f"Sweep complete. {summary['passed']}/{summary['total']} checks passed.")
    logger.info(f"Results saved to: {args.output_dir}/")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())