"""Generates every figure in results/figures from the saved result files."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PINNConfig
from setup_problem import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# palette (validated categorical slots 1-3) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
COL = {"uniform": BLUE, "causal": ORANGE, "fd": AQUA}
LBL = {"uniform": "Uniform weighting", "causal": "Causal weighting", "fd": "FD model (reference)"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK,
})


def kind(name):
    return "causal" if name.startswith("causal") else "uniform"


def fig_plant():
    plant, basis, *_ = build(PINNConfig())
    us = [0.95, 1.0, 1.05, 1.10, 1.15]
    shades = ["#9ec5f0", "#6aa6e8", "#2a78d6", "#1d5aa6", "#123d73"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for u, c in zip(us, shades):
        C, T = plant.split(plant.steady_state(u))
        ax[0].plot(plant.z, C, color=c, label=f"u = {u:.2f}")
        ax[1].plot(plant.z, T, color=c)
    ug = np.linspace(plant.p.u_min, plant.p.u_max, 41)
    cout = [plant.split(plant.steady_state(u))[0][-1] for u in ug]
    ax[2].plot(ug, cout, color=BLUE)
    ax[0].set(title="Concentration C(z)", xlabel="axial position z", ylabel="C")
    ax[1].set(title="Temperature T(z)", xlabel="axial position z", ylabel="T")
    ax[2].set(title="Steady-state gain: outlet C vs u", xlabel="jacket temperature u", ylabel="C(1)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.36, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_plant.png"))
    plt.close(fig)


def fig_training(names):
    hs = {n: json.load(open(os.path.join(ROOT, "results", "models", f"{n}_history.json"))) for n in names}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for n, d in hs.items():
        h = d["hist"]
        ax[0].semilogy(h["it"], h["L_pde"], color=COL[kind(n)], label=LBL[kind(n)])
        ax[1].semilogy(h["it"], h["L_ic"], color=COL[kind(n)])
        ax[1].semilogy(h["it"], h["L_bc"], color=COL[kind(n)], ls="--")
    n_adam = hs[names[0]]["meta"]["config"]["TrainConfig"]["adam_iters"]
    for a in ax[:2]:
        a.axvline(n_adam, color=MUTED, lw=1, ls=":")
    ax[0].text(n_adam, ax[0].get_ylim()[1], " L-BFGS", color=INK2, fontsize=8, va="top")
    ax[0].set(title="Unweighted PDE residual", xlabel="iteration", ylabel="mean squared residual")
    ax[1].set(title="IC (solid) and BC (dashed) losses", xlabel="iteration")
    ax[0].legend(fontsize=8)
    causal = [n for n in names if kind(n) == "causal"]
    if causal:
        h = hs[causal[0]]["hist"]
        W = np.array(h["w_slices"]).T
        im = ax[2].imshow(W, aspect="auto", origin="lower", cmap="Oranges", vmin=0, vmax=1,
                          extent=[h["it"][0], h["it"][-1], 0.5, W.shape[0] + 0.5])
        for it, e in h.get("eps_change", []):
            ax[2].axvline(it, color=INK, lw=0.8, ls=":")
            ax[2].text(it, W.shape[0] + 0.3, f" ε={e:g}", fontsize=7, color=INK, va="top")
        ax[2].set(title="Causal weights per time slice", xlabel="iteration", ylabel="time slice (early → late)")
        ax[2].grid(False)
        cb = fig.colorbar(im, ax=ax[2], fraction=0.046)
        cb.set_label("mean weight wᵢ", color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_training.png"))
    plt.close(fig)


def fig_rollout(rfile):
    R = json.load(open(rfile))
    taus, H = np.array(R["taus"]), R["H"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    hops = np.arange(1, H + 1)
    # one line per arm: mean over seeds, shaded band = min..max over seeds
    for k in ("uniform", "causal"):
        ms = [m for n, m in R["models"].items() if kind(n) == k]
        if not ms:
            continue
        lbl = LBL[k] + (f" (mean of {len(ms)} seeds)" if len(ms) > 1 else "")
        for a, x, key, mk in ((ax[0], taus, "within_rmse_C", 4), (ax[1], hops, "rollout_cout_abs_err_mean", 3.5)):
            Y = np.array([m[key] for m in ms])
            a.plot(x, Y.mean(0), color=COL[k], marker="o", ms=mk, label=lbl)
            if len(ms) > 1:
                a.fill_between(x, Y.min(0), Y.max(0), color=COL[k], alpha=0.18, lw=0)
    ax[1].set_yscale("log")
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax[1].semilogy(np.arange(1, H + 1), R["encoding_only"]["cout_abs_err_mean"], color=MUTED, ls="--",
                   label="Encoding-only floor")
    ax[0].set(title="Error inside one hop", xlabel="normalised time within hop  τ = t/Δt",
              ylabel="RMSE of C(z) vs FD")
    ax[1].set(title="Chained rollout drift", xlabel="number of chained hops (rollout length)",
              ylabel="mean |outlet C error|")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_rollout.png"))
    plt.close(fig)


def fig_closed_loop(labels, out):
    runs = {l: json.load(open(os.path.join(ROOT, "results", "closed_loop", f"{l}.json"))) for l in labels}
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    first = next(iter(runs.values()))["log"]
    t = np.array(first["t"])
    ax[0].step(np.concatenate([[0], t]), np.concatenate([[first["r"][0]], first["r"]]), where="pre",
               color=INK2, lw=1.2, ls="--", label="Setpoint")
    for l, d in runs.items():
        k = "fd" if l.startswith("fd") else kind(l.replace("pinn_", ""))
        name = LBL[k] if k == "fd" else f"PINN-MPC, {LBL[k].lower()}"
        lg = d["log"]
        ax[0].plot(np.concatenate([[0], t]), np.concatenate([[0.354], lg["y"]]), color=COL[k],
                   label=f"{name}  (IAE {d['summary']['IAE']:.3f})")
        ax[1].step(np.concatenate([[0], t]), np.concatenate([[1.0], lg["u"]]), where="pre", color=COL[k])
    ax[0].set(ylabel="outlet concentration C(1,t)", title="Closed-loop setpoint tracking on the FD plant")
    ax[1].set(ylabel="jacket temp. u", xlabel="time (residence times)")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "plant":
        fig_plant()
    elif what == "training":
        fig_training(sys.argv[2:])
    elif what == "rollout":
        fig_rollout(sys.argv[2])
    elif what == "closed":
        fig_closed_loop(sys.argv[3:], sys.argv[2])
