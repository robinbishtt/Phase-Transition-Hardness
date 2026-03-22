from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir
from src.energy_model import (
    rs_entropy_density, cluster_complexity, frozen_fraction, barrier_density,
    ALPHA_D, ALPHA_R, ALPHA_C, ALPHA_S,
)

logger = get_logger("landscape_visuals")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})


def generate_landscape_visuals(
    results_dir: str = "results",
    output_dir:  str = "results/figures",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    ensure_dir(output_dir)
    generated = []


    path1 = _energy_landscape_schematic(output_dir, fmt, dpi)
    generated.append(path1)


    path2 = _phase_diagram(output_dir, fmt, dpi)
    generated.append(path2)


    path3 = _rigidity_complexity(output_dir, fmt, dpi)
    generated.append(path3)


    path4 = _barrier_height_scaling(output_dir, fmt, dpi)
    generated.append(path4)

    return generated






def _energy_landscape_schematic(output_dir, fmt, dpi):
    fig = plt.figure(figsize=(9, 5))
    ax  = fig.add_subplot(111, projection="3d")


    rng = np.random.RandomState(42)
    x   = np.linspace(-3, 3, 120)
    y   = np.linspace(-3, 3, 120)
    X, Y = np.meshgrid(x, y)


    Z = 0.3 * (X ** 2 + Y ** 2)


    centres = [(-1.8, -1.5), (1.7, -1.6), (-1.6, 1.8), (1.5, 1.7), (0.0, 0.0)]
    for (cx, cy) in centres:
        depth = 1.2 + 0.3 * rng.rand()
        width = 0.6 + 0.2 * rng.rand()
        Z -= depth * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))


    Z -= Z.min()
    Z = np.clip(Z, 0, 2.5)

    surf = ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.85,
                           linewidth=0, antialiased=True)

    ax.set_xlabel("Config. direction 1", labelpad=4)
    ax.set_ylabel("Config. direction 2", labelpad=4)
    ax.set_zlabel(r"Energy $\mathcal{H}(\sigma)$", labelpad=4)
    ax.set_title(r"Rugged energy landscape ($\alpha_d < \alpha < \alpha_s$)", pad=10)
    ax.view_init(elev=28, azim=-60)
    ax.tick_params(labelsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.05, label="Energy")

    fig.tight_layout()
    path = f"{output_dir}/fig1_energy_landscape.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _phase_diagram(output_dir, fmt, dpi):
    alphas = np.linspace(2.5, 4.8, 200)

    entropy    = np.array([rs_entropy_density(a)  for a in alphas])
    complexity = np.array([cluster_complexity(a)   for a in alphas])
    frozen     = np.array([frozen_fraction(a)       for a in alphas])
    b_curve    = np.array([barrier_density(a)        for a in alphas])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(alphas, entropy    / np.log(2), color="#1f77b4",#1f77b4",
            label=r"Entropy density $s(\alpha)/\log 2$")
    ax.plot(alphas, complexity * 2,          color="#2ca02c",#2ca02c",
            label=r"Cluster complexity $2\Sigma(\alpha)$")
    ax.plot(alphas, frozen,                  color="#d62728",#d62728",
            label=r"Frozen fraction $f_{\mathrm{frz}}(\alpha)$")
    ax.plot(alphas, b_curve / 0.021,         color="#ff7f0e",#ff7f0e",
            label=r"Barrier density $b(\alpha)/b_{\max}$")


    ax.axvspan(ALPHA_D, ALPHA_S, alpha=0.08, color="red",
               label="Hard phase")

    for alpha_val, label in [
        (ALPHA_D, r"$\alpha_d$"),
        (ALPHA_R, r"$\alpha_r$"),
        (ALPHA_C, r"$\alpha_c$"),
        (ALPHA_S, r"$\alpha_s$"),
    ]:
        ax.axvline(alpha_val, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.text(alpha_val + 0.02, 1.02, label, transform=ax.get_xaxis_transform(),
                fontsize=9.5, color="gray", va="bottom", ha="left")

    ax.set_xlabel(r"Clause density $\alpha$")
    ax.set_ylabel("Normalised order parameter")
    ax.set_title("Phase diagram of random 3-SAT")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/fig5_phase_diagram.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _rigidity_complexity(output_dir, fmt, dpi):
    alphas = np.linspace(3.5, 4.6, 200)
    frozen = np.array([frozen_fraction(a) for a in alphas])
    sigma  = np.array([cluster_complexity(a) for a in alphas])

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()

    ax1.plot(alphas, frozen, color="#d62728",
             label=r"Frozen fraction $f_{\mathrm{frz}}$")
    ax2.plot(alphas, sigma,  color="#2ca02c",
             label=r"Complexity $\Sigma(\alpha)$")

    for alpha_val, label in [(ALPHA_R, r"$\alpha_r$"), (ALPHA_C, r"$\alpha_c$")]:
        ax1.axvline(alpha_val, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax1.text(alpha_val + 0.01, 0.97, label,
                 transform=ax1.get_xaxis_transform(), fontsize=10, color="gray")

    ax1.set_xlabel(r"Clause density $\alpha$")
    ax1.set_ylabel(r"Frozen fraction", color="#d62728")
    ax2.set_ylabel(r"Cluster complexity $\Sigma$", color="#2ca02c")
    ax1.set_title("Rigidity and condensation transitions")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    fig.tight_layout()
    path = f"{output_dir}/ext_fig3_rigidity_complexity.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _barrier_height_scaling(output_dir, fmt, dpi):
    ns = np.array([100, 200, 400, 800, 1600])
    b_vals = [0.005, 0.012, 0.021]
    colors_bh = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6, 4.2))

    for b, color in zip(b_vals, colors_bh):
        barriers = b * ns
        ax.plot(ns, barriers, "o-", color=color, lw=1.8, ms=5,
                label=f"$b = {b:.3f}$")

    ax.set_xlabel("System size $n$")
    ax.set_ylabel("Free-energy barrier $B(n)$")
    ax.set_title(r"Extensive barrier scaling $B(n) \sim n \cdot b(\alpha)$")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/ext_fig4_barrier_scaling.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path