from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir, load_npz
from src.phase_transition import run_psat_sweep
from src.scaling_analysis import run_fss_analysis
from src.utils import save_json

logger = get_logger("finite_size_scaling")


def main():
    parser = argparse.ArgumentParser(description="Finite-size scaling collapse.")
    parser.add_argument("--n",              type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--n_instances",    type=int, default=1000)
    parser.add_argument("--alpha_min",      type=float, default=3.5)
    parser.add_argument("--alpha_max",      type=float, default=5.0)
    parser.add_argument("--alpha_step",     type=float, default=0.05)
    parser.add_argument("--k",              type=int, default=3)
    parser.add_argument("--solver",         choices=["dpll", "walksat"], default="dpll")
    parser.add_argument("--seed",           type=int, default=20240223)
    parser.add_argument("--output_dir",     default="results")
    parser.add_argument("--n_jobs",         type=int, default=1)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)

    logger.info(f"FSS collapse: n={args.n}, α=[{args.alpha_min}, {args.alpha_max}]")


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


    fss_result = run_fss_analysis(
        alphas=alphas,
        ns=args.n,
        psat_matrix=psat_result["psat_matrix"],
        output_dir=args.output_dir,
    )

    logger.info(f"FSS result: α_s = {fss_result['alpha_s']:.4f}, "
                f"ν = {fss_result['nu']:.3f}, "
                f"residual = {fss_result['residual']:.6f}")

    save_json(
        {
            "alpha_s":        fss_result["alpha_s"],
            "nu":             fss_result["nu"],
            "residual":       fss_result["residual"],
            "converged":      fss_result["converged"],
            "n_system_sizes": len(args.n),
            "alpha_step":     args.alpha_step,
            "ns":             args.n,
        },
        f"{args.output_dir}/fss_result.json",
    )

    logger.info("FSS collapse complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())