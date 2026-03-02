from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, load_npz, load_json, ensure_dir
from src.energy_model import ALPHA_S

logger = get_logger("scaling_collapse")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

COLORS = ["
          "


def generate_scaling_collapse(
    results_dir: str = "results",
    output_dir:  str = "results/figures",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    ensure_dir(output_dir)


    try:
        pt_data     = load_npz(f"{results_dir}/phase_transition.npz")
        alphas      = pt_data["alphas"]
        psat_matrix = pt_data["psat_matrix"]
        ns          = pt_data["ns"].tolist()
    except FileNotFoundError:
        logger.warning("phase_transition.npz not found  using synthetic data.")
        alphas, psat_matrix, ns = _synthetic_psat_data()


    try:
        fss = load_json(f"{results_dir}/fss_result.json")
        alpha_s = fss["alpha_s"]
        nu      = fss["nu"]
    except FileNotFoundError:
        alpha_s = ALPHA_S
        nu      = 2.3
        logger.warning(f"fss_result.json not found  using literature values α_s={alpha_s}, ν={nu}.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))


    ax = axes[0]
    for i, n in enumerate(ns):
        color = COLORS[i % len(COLORS)]
        ax.plot(alphas, psat_matrix[i], color=color,
                lw=1.8, label=f"$n={int(n)}$")
    ax.axvline(alpha_s, color="gray", ls="--", lw=1.2, alpha=0.8,
               label=f"$\\alpha_s = {alpha_s:.3f}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$P_{\mathrm{sat}}(\alpha, n)$")
    ax.set_title("Raw satisfiability curves")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.3)


    ax = axes[1]
    for i, n in enumerate(ns):
        color  = COLORS[i % len(COLORS)]
        x_coll = (alphas - alpha_s) * n ** (1.0 / nu)
        ax.plot(x_coll, psat_matrix[i], "o", color=color,
                ms=3, alpha=0.7, label=f"$n={int(n)}$")


    all_x = np.concatenate(
        [(alphas - alpha_s) * n ** (1.0 / nu) for n in ns]
    )
    all_y = np.concatenate([psat_matrix[i] for i in range(len(ns))])
    sort_idx = np.argsort(all_x)
    x_sorted, y_sorted = all_x[sort_idx], all_y[sort_idx]

    from scipy.interpolate import UnivariateSpline
    try:
        spl = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * 0.01, k=4)
        x_plot = np.linspace(x_sorted[0], x_sorted[-1], 200)
        ax.plot(x_plot, np.clip(spl(x_plot), 0, 1), "k-", lw=2.0,
                label=r"Universal $F(x)$")
    except Exception:
        pass

    ax.set_xlabel(r"Scaled variable $(\alpha - \alpha_s)\,n^{1/\nu}$")
    ax.set_ylabel(r"$P_{\mathrm{sat}}$")
    ax.set_title(f"FSS collapse  ($\\nu = {nu:.2f}$,  $\\alpha_s = {alpha_s:.3f}$)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/fig4_fss_collapse.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return [path]


def _synthetic_psat_data():
    from scipy.special import expit
    alphas = np.linspace(3.5, 5.0, 31)
    ns     = [100, 200, 400, 800]
    psat   = np.array([
        expit(-(4.0 + n / 80.0) * (alphas - ALPHA_S))
        for n in ns
    ])
    return alphas, psat, ns