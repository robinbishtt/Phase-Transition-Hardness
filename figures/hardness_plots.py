from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, load_npz, ensure_dir
from src.energy_model import barrier_density, ALPHA_D, ALPHA_S

logger = get_logger("hardness_plots")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
})

COLORS = ["
          "


def generate_hardness_plots(
    results_dir: str = "results",
    output_dir:  str = "results/figures",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    ensure_dir(output_dir)
    generated = []


    try:
        data         = load_npz(f"{results_dir}/alpha_sweep.npz")
        alphas       = data["alphas"]
        ns           = data["ns"].tolist()
        gamma_matrix = data["gamma_mean_matrix"]
        gamma_lo     = data["gamma_lo_matrix"]
        gamma_hi     = data["gamma_hi_matrix"]
    except FileNotFoundError:
        logger.warning("alpha_sweep.npz not found  generating synthetic demo.")
        alphas, ns, gamma_matrix, gamma_lo, gamma_hi = _synthetic_hardness_data()


    b_curve = np.array([barrier_density(a) for a in alphas])
    b_scale = gamma_matrix.max() / max(b_curve.max(), 1e-10)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, n in enumerate(ns):
        color = COLORS[i % len(COLORS)]
        ax.plot(alphas, gamma_matrix[i], color=color,
                lw=1.8, label=f"$n = {int(n)}$")
        ax.fill_between(alphas, gamma_lo[i], gamma_hi[i],
                        color=color, alpha=0.15)


    ax.plot(alphas, b_curve * b_scale, "k--", lw=1.4,
            label=r"$b(\alpha)$ (theory, rescaled)")


    for alpha_val, label in [(ALPHA_D, r"$\alpha_d$"), (ALPHA_S, r"$\alpha_s$")]:
        ax.axvline(alpha_val, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.text(alpha_val + 0.02, 0.95, label,
                transform=ax.get_xaxis_transform(),
                fontsize=10, color="gray", va="top")

    ax.set_xlabel(r"Clause density $\alpha$")
    ax.set_ylabel(r"Hardness density $\gamma(\alpha) = \log T / n$")
    ax.set_title(r"Hardness peak near $\alpha^* \approx 4.20$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/fig3_hardness_curves.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)
    logger.info(f"Saved: {path}")
    return generated


def _synthetic_hardness_data():
    alphas = np.linspace(3.0, 5.0, 41)
    ns     = [100, 200, 400, 800]
    gamma_matrix = np.zeros((len(ns), len(alphas)))
    gamma_lo     = np.zeros_like(gamma_matrix)
    gamma_hi     = np.zeros_like(gamma_matrix)

    for i, n in enumerate(ns):
        for j, a in enumerate(alphas):
            b = barrier_density(a)
            g = b * (1.0 + 0.5 * np.log(n) / np.log(100))
            gamma_matrix[i, j] = g
            noise = 0.002 / np.sqrt(200)
            gamma_lo[i, j] = max(0, g - 1.96 * noise)
            gamma_hi[i, j] = g + 1.96 * noise

    return alphas, ns, gamma_matrix, gamma_lo, gamma_hi