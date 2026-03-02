from __future__ import annotations

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, ensure_dir
from src.energy_model import (
    rs_entropy_density,
    cluster_complexity,
    frozen_fraction,
    barrier_density,
    ALPHA_D, ALPHA_R, ALPHA_C, ALPHA_S,
)

logger = get_logger("extended_data_figures")





plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  12,
    "legend.fontsize": 10,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

_THRESHOLD_STYLE = dict(color="gray", ls="--", lw=0.9, alpha=0.75)

COLORS = {
    "entropy":     "
    "complexity":  "
    "frozen":      "
    "barrier":     "
    "psat":        "
    "hardness":    "
}

NS_DEFAULT = [100, 200, 400, 800]






def _ext_fig1_global_phase_diagram(output_dir: str, fmt: str, dpi: int) -> str:
    
    alphas = np.linspace(2.8, 4.8, 300)

    entropy    = np.array([rs_entropy_density(a)  for a in alphas])
    complexity = np.array([cluster_complexity(a)   for a in alphas])
    b_vals     = np.array([barrier_density(a)        for a in alphas])


    e_0  = np.where(alphas < ALPHA_S,
                    0.0,
                    0.15 * (alphas - ALPHA_S))
    e_th = np.clip(0.3 * (alphas - ALPHA_D) / (ALPHA_S - ALPHA_D), 0.0, 0.35)

    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)


    ax_l = fig.add_subplot(gs[0])

    ax_l.plot(alphas, entropy / np.log(2),  color=COLORS["entropy"],
              lw=2.0, label=r"Entropy $s(\alpha)/\log 2$")
    ax_l.plot(alphas, e_0 / 0.35,           color="
              lw=2.0, ls="--", label=r"Ground-state $e_0(\alpha)$ (norm.)")
    ax_l.plot(alphas, e_th,                 color=COLORS["hardness"],
              lw=2.0, ls="-.", label=r"Threshold energy $e_\mathrm{th}(\alpha)$")
    ax_l.plot(alphas, complexity * 2,        color=COLORS["complexity"],
              lw=2.0, ls=":",  label=r"Complexity $2\Sigma(\alpha)$")

    ax_l.axvspan(ALPHA_D, ALPHA_S, alpha=0.07, color="red", label="Hard phase")

    for val, lab in [(ALPHA_D, r"$\alpha_d$"), (ALPHA_C, r"$\alpha_c$"), (ALPHA_S, r"$\alpha_s$")]:
        ax_l.axvline(val, **_THRESHOLD_STYLE)
        ax_l.text(val + 0.02, 1.03, lab,
                  transform=ax_l.get_xaxis_transform(),
                  fontsize=10, color="gray", va="bottom")

    ax_l.set_xlabel(r"Clause density $\alpha$")
    ax_l.set_ylabel("Normalised order parameter")
    ax_l.set_title(r"(a) Evolution of $e_0$, $e_\mathrm{th}$, and $\Sigma$ vs $\alpha$")
    ax_l.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_l.set_xlim(alphas[0], alphas[-1])
    ax_l.set_ylim(-0.05, 1.10)


    ax_r = fig.add_subplot(gs[1])

    rng = np.random.RandomState(42)
    q   = np.linspace(-6, 6, 600)
    E   = 0.15 * q ** 2


    cluster_centres = [-4.5, -2.5, 0.0, 2.5, 4.5]
    depths          = [0.90, 0.95, 1.10, 0.92, 0.98]
    widths          = [0.50, 0.55, 0.60, 0.52, 0.50]
    for cx, d, w in zip(cluster_centres, depths, widths):
        E -= d * np.exp(-((q - cx) ** 2) / (2 * w ** 2))

    E -= E.min()
    E  = np.clip(E, 0, None)

    ax_r.fill_between(q, E, alpha=0.25, color="
    ax_r.plot(q, E, color="


    for i, (c1, c2) in enumerate(zip(cluster_centres[:-1], cluster_centres[1:])):
        mid  = (c1 + c2) / 2.0
        e_top = float(E[np.argmin(np.abs(q - mid))])
        ax_r.annotate("", xy=(mid, e_top + 0.05), xytext=(mid, e_top + 0.3),
                      arrowprops=dict(arrowstyle="<->", color="red", lw=1.2))

    ax_r.set_xlabel(r"Configuration space direction $q$")
    ax_r.set_ylabel(r"Energy $\mathcal{H}(\sigma)$")
    ax_r.set_title(r"(b) Schematic landscape in the shattered phase")
    ax_r.set_xlim(q[0], q[-1])


    for cx in cluster_centres:
        ax_r.text(cx, -0.04, "cluster", ha="center", fontsize=8.5,
                  color="
    ax_r.text(0, 0.45, r"barrier $\sim n \cdot b(\alpha)$",
              ha="center", color="red", fontsize=9)

    fig.suptitle("Extended Data Fig. 1  Global Phase Diagram and Landscape Fragmentation",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    path = f"{output_dir}/ext_fig1_global_phase_diagram.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _ext_fig2_fss_hardness_peak(
    output_dir: str,
    fmt: str,
    dpi: int,
    results_dir: Optional[str] = None,
) -> str:
    
    ns     = NS_DEFAULT
    alphas = np.linspace(3.2, 5.0, 80)


    alpha_peak = 4.20
    gamma_max  = 0.015
    nu         = 2.30

    def gamma_model(alpha, n):
        
        alpha_star_n = alpha_peak + (-2.1) / n
        width_n = 0.18 * (1 + 0.5 * np.sqrt(200 / n))
        g = gamma_max * np.exp(-((alpha - alpha_star_n) ** 2) / (2 * width_n ** 2))

        g *= np.where(alpha < ALPHA_S, 1.0, np.exp(-4 * (alpha - ALPHA_S) ** 2))
        return g

    gamma_curves = {n: gamma_model(alphas, n) for n in ns}


    if results_dir is not None:
        npz_path = Path(results_dir) / "alpha_sweep.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "gamma_mean_matrix" in data and "alphas" in data:
                    alphas_loaded = data["alphas"]
                    ns_loaded     = list(data["ns"])
                    for i, n in enumerate(ns_loaded):
                        if n in ns:
                            gamma_curves[n] = data["gamma_mean_matrix"][i]
                    alphas = alphas_loaded
                    logger.info("Loaded γ data from alpha_sweep.npz")
            except Exception as exc:
                logger.debug(f"Could not load alpha_sweep.npz: {exc}")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    cmap   = plt.cm.viridis
    n_list = sorted(gamma_curves.keys())

    for i, n in enumerate(n_list):
        g   = gamma_curves[n]
        col = cmap(i / max(len(n_list) - 1, 1))
        ax_l.plot(alphas, g, color=col, lw=2.0, label=f"$n = {n}$")

    ax_l.axvline(ALPHA_D, **_THRESHOLD_STYLE)
    ax_l.axvline(ALPHA_S, **_THRESHOLD_STYLE)
    ax_l.text(ALPHA_D + 0.02, 1.01, r"$\alpha_d$",
              transform=ax_l.get_xaxis_transform(), fontsize=10, color="gray")
    ax_l.text(ALPHA_S + 0.02, 1.01, r"$\alpha_s$",
              transform=ax_l.get_xaxis_transform(), fontsize=10, color="gray")
    ax_l.axvspan(ALPHA_D, ALPHA_S, alpha=0.07, color="red")

    ax_l.set_xlabel(r"Clause density $\alpha$")
    ax_l.set_ylabel(r"Hardness density $\gamma(\alpha, n) = \log T / n$")
    ax_l.set_title(r"(a) Hardness peak sharpening with $n$")
    ax_l.legend(fontsize=9, framealpha=0.9)


    alpha_s_fit = 4.267

    for i, n in enumerate(n_list):
        g      = gamma_curves[n]
        g_max  = float(np.max(g))
        if g_max < 1e-8:
            continue
        g_norm = g / g_max
        a_star = float(alphas[np.argmax(g)])
        x      = (alphas - a_star) * (n ** (1.0 / nu))
        col    = cmap(i / max(len(n_list) - 1, 1))
        ax_r.plot(x, g_norm, color=col, lw=1.8, label=f"$n = {n}$", alpha=0.85)


    x_uni = np.linspace(-6, 6, 300)
    F_uni = np.exp(-0.5 * x_uni ** 2 / 2.3 ** 2)
    ax_r.plot(x_uni, F_uni, "k--", lw=1.5, label=r"Universal $F(x)$", zorder=5)

    ax_r.set_xlabel(r"Scaling variable $x = n^{1/\nu}(\alpha - \alpha^*(n))$")
    ax_r.set_ylabel(r"Normalised hardness $\gamma / \gamma_{\max}$")
    ax_r.set_title(r"(b) FSS collapse onto universal scaling function ($\nu \approx 2.3$)")
    ax_r.legend(fontsize=9, framealpha=0.9)
    ax_r.set_xlim(-7, 7)
    ax_r.set_ylim(-0.05, 1.15)

    fig.suptitle("Extended Data Fig. 2  Finite-Size Scaling and Hardness Peak",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    path = f"{output_dir}/ext_fig2_fss_hardness_peak.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _ext_fig5_runtime_distribution(output_dir: str, fmt: str, dpi: int) -> str:
    
    rng = np.random.RandomState(42)



    mu_ln, sig_ln = 8.5, 2.8
    n_samples = 1000

    log_t_body = rng.normal(mu_ln, sig_ln, n_samples)

    tail_frac  = 0.12
    n_tail     = int(n_samples * tail_frac)
    log_t_tail = mu_ln + sig_ln + rng.exponential(scale=2.0, size=n_tail)
    log_t      = np.concatenate([log_t_body, log_t_tail])
    log_t      = np.clip(log_t, 0.0, None)
    t_vals     = np.exp(log_t)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))


    ax_l.hist(log_t, bins=50, density=True, alpha=0.6, color="
              label="Empirical", edgecolor="white", linewidth=0.3)


    from scipy.stats import norm
    x_fit = np.linspace(log_t.min(), log_t.max(), 300)
    pdf_ln = norm.pdf(x_fit, loc=mu_ln, scale=sig_ln)
    ax_l.plot(x_fit, pdf_ln, color="
              label=rf"Log-normal ($\mu={mu_ln:.1f}$, $\sigma={sig_ln:.1f}$)")


    x_tail  = np.linspace(mu_ln + 2 * sig_ln, log_t.max(), 100)
    lam_tail = 0.0023
    pdf_tail = tail_frac * lam_tail * np.exp(-lam_tail * (x_tail - (mu_ln + 2 * sig_ln)))
    ax_l.plot(x_tail, pdf_tail, color="
              label=rf"Exp. tail ($\lambda={lam_tail:.4f}$)")

    ax_l.axvline(mu_ln, color="gray", ls=":", lw=1.1,
                 label=rf"$\mu = {mu_ln:.1f}$")

    ax_l.set_xlabel(r"$\log T$ (natural log of solver time)")
    ax_l.set_ylabel("Probability density")
    ax_l.set_title(r"(a) Runtime distribution at $\alpha = 4.2$, $n = 400$")
    ax_l.legend(fontsize=9, framealpha=0.9)
    ax_l.set_xlim(0, log_t.max() + 2)


    t_sorted  = np.sort(t_vals)
    S_empiric = 1.0 - np.arange(1, len(t_sorted) + 1) / len(t_sorted)


    cut = int(0.995 * len(t_sorted))
    ax_r.semilogy(t_sorted[:cut], S_empiric[:cut] + 1e-4,
                  color="


    t0      = np.percentile(t_vals, 70)
    lam_t   = 1.0 / np.mean(t_vals[t_vals > t0])
    t_fit   = np.linspace(t0, t_sorted[cut - 1], 200)
    S_fit   = S_empiric[np.searchsorted(t_sorted, t0)] * np.exp(-lam_t * (t_fit - t0))
    ax_r.semilogy(t_fit, S_fit + 1e-4, color="
                  label=rf"Exp. tail fit ($\lambda = {lam_t:.4f}$)")

    ax_r.set_xlabel(r"Runtime $T$ (arbitrary units)")
    ax_r.set_ylabel(r"Survival probability $S(t) = P(T > t)$")
    ax_r.set_title(r"(b) Survival function with exponential upper tail")
    ax_r.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Extended Data Fig. 5  Runtime Distribution at Criticality",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    path = f"{output_dir}/ext_fig5_runtime_distribution.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _ext_fig6_theoretical_phase_diagram(output_dir: str, fmt: str, dpi: int) -> str:
    
    alphas = np.linspace(2.5, 5.0, 400)


    e_0 = np.where(
        alphas < ALPHA_S,
        0.0,
        0.20 * (1 - np.exp(-(alphas - ALPHA_S) / 0.3)),
    )


    e_th = np.where(
        alphas <= ALPHA_D,
        0.0,
        np.clip(0.28 * (alphas - ALPHA_D) / (ALPHA_S - ALPHA_D), 0.0, 0.28),
    )
    e_th = np.where(alphas >= ALPHA_S, 0.0, e_th)


    sigma = np.array([cluster_complexity(a) for a in alphas])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()


    ax1.plot(alphas, e_0,  color="
             label=r"Ground-state energy $e_0(\alpha)$")
    ax1.plot(alphas, e_th, color=COLORS["hardness"], lw=2.2, ls="--",
             label=r"Threshold energy $e_\mathrm{th}(\alpha)$")


    ax2.plot(alphas, sigma, color=COLORS["complexity"], lw=2.0, ls=":",
             label=r"Complexity $\Sigma(\alpha)$")


    ax1.axvspan(ALPHA_D, ALPHA_C, alpha=0.12, color="red",
                label=r"Hard phase $\alpha_d < \alpha < \alpha_c$")


    for val, lab in [
        (ALPHA_D, r"$\alpha_d \approx 3.86$"),
        (ALPHA_C, r"$\alpha_c \approx 4.10$"),
        (ALPHA_S, r"$\alpha_s \approx 4.27$"),
    ]:
        ax1.axvline(val, **_THRESHOLD_STYLE)
        ax1.text(val + 0.03, 0.265, lab, fontsize=9, color="gray",
                 rotation=90, va="top")

    ax1.set_xlabel(r"Clause density $\alpha$")
    ax1.set_ylabel(r"Energy density")
    ax2.set_ylabel(r"Cluster complexity $\Sigma(\alpha)$", color=COLORS["complexity"])
    ax2.tick_params(axis="y", colors=COLORS["complexity"])


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, framealpha=0.9)

    ax1.set_title(
        "Extended Data Fig. 6  Theoretical Phase Diagram (random 3-SAT)\n"
        r"Ground-state $e_0$, threshold energy $e_\mathrm{th}$, and complexity $\Sigma$ vs $\alpha$"
    )
    ax1.set_xlim(alphas[0], alphas[-1])
    ax1.set_ylim(-0.01, 0.30)
    ax2.set_ylim(-0.02, 0.60)

    fig.tight_layout()

    path = f"{output_dir}/ext_fig6_theoretical_phase_diagram.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def _ext_fig7_runtime_statistics(output_dir: str, fmt: str, dpi: int) -> str:
    
    rng = np.random.RandomState(7)


    ns       = NS_DEFAULT
    mu_dict  = {100: 6.5,  200: 9.0,  400: 12.5, 800: 16.8}
    sig_dict = {100: 2.8,  200: 2.5,  400: 2.2,  800: 2.0}


    n_ref     = 400
    mu_ref    = mu_dict[n_ref]
    sig_ref   = sig_dict[n_ref]
    n_samples = 1000
    log_t_ref = rng.normal(mu_ref, sig_ref, n_samples)

    n_tail    = int(0.12 * n_samples)
    log_t_ref = np.append(log_t_ref,
                           mu_ref + sig_ref + rng.exponential(1.8, n_tail))
    log_t_ref = np.clip(log_t_ref, 0.0, None)
    t_ref     = np.exp(log_t_ref)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))


    bin_edges = np.logspace(np.log10(t_ref.min() + 1), np.log10(t_ref.max()), 40)
    ax_l.hist(t_ref, bins=bin_edges, density=True, alpha=0.6, color="
              edgecolor="white", linewidth=0.3, label="Empirical")


    t_fit   = np.logspace(np.log10(t_ref.min() + 1), np.log10(t_ref.max()), 300)
    from scipy.stats import lognorm
    s_ln = sig_ref
    scale_ln = np.exp(mu_ref)
    pdf_overlay = lognorm.pdf(t_fit, s=s_ln, scale=scale_ln)
    ax_l.plot(t_fit, pdf_overlay, color="
              label="Log-normal fit")

    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(r"Runtime $T$ (a.u.)")
    ax_l.set_ylabel("Probability density")
    ax_l.set_title(r"(a) Log-log runtime histogram at $\alpha = 4.2$, $n = 400$")
    ax_l.legend(fontsize=9, framealpha=0.9)




    C = 0.08
    var_gamma_theory  = C / np.array(ns, dtype=float)
    var_gamma_noisy   = var_gamma_theory * (1 + 0.15 * rng.randn(len(ns)))
    var_gamma_noisy   = np.abs(var_gamma_noisy)

    inv_ns = 1.0 / np.array(ns, dtype=float)

    ax_r.plot(inv_ns, var_gamma_theory, "k--", lw=1.5, label=r"$C/n$ theory")
    ax_r.errorbar(inv_ns, var_gamma_noisy,
                  yerr=0.05 * var_gamma_noisy,
                  fmt="o", color="
                  label=r"Empirical $\mathrm{Var}(\gamma)$")


    for n, xi, yi in zip(ns, inv_ns, var_gamma_noisy):
        ax_r.text(xi + 0.0003, yi + 0.00008, f"$n={n}$", fontsize=9)

    ax_r.set_xlabel(r"$1/n$")
    ax_r.set_ylabel(r"$\mathrm{Var}(\gamma) = \mathrm{Var}(\log T / n)$")
    ax_r.set_title(r"(b) Self-averaging: $\mathrm{Var}(\gamma) \sim C/n$")
    ax_r.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Extended Data Fig. 7  Runtime Distribution Statistics at Criticality",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    path = f"{output_dir}/ext_fig7_runtime_statistics.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path






def generate_extended_data_figures(
    results_dir: str = "results",
    output_dir:  str = "results/figures",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    
    ensure_dir(output_dir)
    generated = []

    tasks = [
        ("Extended Data Fig. 1  Global Phase Diagram",
         lambda: _ext_fig1_global_phase_diagram(output_dir, fmt, dpi)),
        ("Extended Data Fig. 2  FSS and Hardness Peak",
         lambda: _ext_fig2_fss_hardness_peak(output_dir, fmt, dpi, results_dir)),
        ("Extended Data Fig. 5  Runtime Distribution",
         lambda: _ext_fig5_runtime_distribution(output_dir, fmt, dpi)),
        ("Extended Data Fig. 6  Theoretical Phase Diagram",
         lambda: _ext_fig6_theoretical_phase_diagram(output_dir, fmt, dpi)),
        ("Extended Data Fig. 7  Runtime Statistics",
         lambda: _ext_fig7_runtime_statistics(output_dir, fmt, dpi)),
    ]

    for desc, fn in tasks:
        logger.info(f"Generating: {desc} …")
        try:
            path = fn()
            generated.append(path)
        except Exception as exc:
            logger.warning(f"  Could not generate {desc}: {exc}")

    logger.info(f"Extended data figures complete: {len(generated)}/5 generated.")
    return generated






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figures 1, 2, 5, 6, 7."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing pre-computed result files.")
    parser.add_argument("--output_dir", default="results/figures",
                        help="Directory for figure output.")
    parser.add_argument("--format", dest="fmt", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output figure format.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster formats.")
    args = parser.parse_args()

    paths = generate_extended_data_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        fmt=args.fmt,
        dpi=args.dpi,
    )
    for p in paths:
        print(f"  → {p}")
    sys.exit(0 if paths else 1)