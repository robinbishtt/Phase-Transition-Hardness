from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir

logger = get_logger("generate_all_figures")


def main():
    parser = argparse.ArgumentParser(description="Generate all manuscript figures.")
    parser.add_argument("--results_dir", default="results",
                        help="Directory with pre-computed result files.")
    parser.add_argument("--output_dir",  default="results/figures",
                        help="Directory for figure output.")
    parser.add_argument("--format",      default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output figure format.")
    parser.add_argument("--dpi",         type=int, default=300,
                        help="DPI for raster formats.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    from figures.phase_transition_plots  import generate_phase_transition_plots
    from figures.hardness_plots          import generate_hardness_plots
    from figures.scaling_collapse        import generate_scaling_collapse
    from figures.landscape_visuals       import generate_landscape_visuals
    from figures.extended_data_figures   import generate_extended_data_figures

    figure_fns = [
        ("Phase-transition plots (Fig. 3 left)",                generate_phase_transition_plots),
        ("Hardness plots (Fig. 3 right)",                        generate_hardness_plots),
        ("Scaling collapse (Fig. 4)",                            generate_scaling_collapse),
        ("Landscape visuals (Figs. 1, 5, Ext. 3, 4)",           generate_landscape_visuals),
        ("Extended Data Figures 1, 2, 5, 6, 7",                 generate_extended_data_figures),
    ]

    generated = []
    for description, fn in figure_fns:
        logger.info(f"Generating: {description} …")
        try:
            paths = fn(
                results_dir=args.results_dir,
                output_dir=args.output_dir,
                fmt=args.format,
                dpi=args.dpi,
            )
            for p in paths:
                logger.info(f"  → {p}")
            generated.extend(paths)
        except Exception as exc:
            logger.warning(f"  Could not generate {description}: {exc}")

    logger.info(f"\nGenerated {len(generated)} figures in {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())