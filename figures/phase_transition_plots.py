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
from src.energy_model import ALPHA_D, ALPHA_R, ALPHA_C, ALPHA_S

logger = get_logger("phase_transition_plots")


plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  13,
    "legend.fontsize": 11,
    "figure.dpi":      150,
})

COLORS = ["
          "


def generate_phase_transition_plots(
    results_dir: str = "results",
    output_dir: str  = "results/figures",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    
    ensure_dir(output_dir)
    generated = []


    try:
        data = load_npz(f"{results_dir}/phase_transition.npz")
        alphas      = data["alphas"]
        psat_matrix = data["psat_matrix"]
        ns          = data["ns"].tolist()
    except FileNotFoundError:
        logger.warning("phase_transition.npz not found  generating synthetic demo data.")
        alphas, psat_matrix, ns = _synthetic_psat_data()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, n in enumerate(ns):
        color = COLORS[i % len(COLORS)]
        ax.plot(alphas, psat_matrix[i], color=color,
                lw=1.8, label=f"$n = {int(n)}$")


    for alpha_val, label, ls in [
        (ALPHA_D, r"$\alpha_d$",  "--"),
        (ALPHA_S, r"$\alpha_s$",  "-"),
    ]:
        ax.axvline(alpha_val, color="gray", ls=ls, lw=1.2, alpha=0.7)
        ax.text(alpha_val + 0.02, 0.95, label,
                transform=ax.get_xaxis_transform(),
                fontsize=11, color="gray", va="top")

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel(r"Clause density $\alpha = m/n$")
    ax.set_ylabel(r"Satisfiability probability $P_{\mathrm{sat}}(\alpha, n)$")
    ax.set_title(r"Phase transition in random 3-SAT")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/fig3_psat_curves.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)
    logger.info(f"Saved: {path}")
    return generated


def _synthetic_psat_data():
    
    from scipy.special import expit
    alphas = np.linspace(3.0, 5.0, 41)
    ns     = [100, 200, 400, 800]
    psat_matrix = np.zeros((len(ns), len(alphas)))
    for i, n in enumerate(ns):
        steepness = 4.0 + n / 80.0
        psat_matrix[i] = expit(-steepness * (alphas - ALPHA_S))
    return alphas, psat_matrix, ns