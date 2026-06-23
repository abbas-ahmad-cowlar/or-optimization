"""
Solution visualizations for P1/P2/P3
====================================
Reads the logged results in output/results/ and renders publication-quality
figures to output/figures/. Run after the experiments:

    python -m visualization.plot_solutions
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config

RESULTS = config.RESULTS_DIR
FIGS = config.FIGURES_DIR


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.family": "sans-serif",
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


# ============================================================ P2
def plot_p2_alpha_sweep(sweep):
    mult = [r["alpha_mult"] for r in sweep]
    n_open = [r["n_open"] for r in sweep]
    constr = [r["construction"] or np.nan for r in sweep]
    trans = [r["transmission"] or np.nan for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(mult, n_open, "o-", color="#2563eb", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"transmission-rate multiplier ($\alpha/\alpha_0$)")
    ax1.set_ylabel("substations built")
    ax1.set_title("Optimal #substations vs transmission rate")

    ax2.plot(mult, constr, "s-", label="construction", color="#16a34a")
    ax2.plot(mult, trans, "^-", label="transmission", color="#dc2626")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"transmission-rate multiplier ($\alpha/\alpha_0$)")
    ax2.set_ylabel("cost component")
    ax2.set_title("Cost split vs transmission rate")
    ax2.legend()
    fig.suptitle("P2 - Substation siting: transmission-rate sensitivity", fontweight="bold")
    _save(fig, "p2_alpha_sweep.png")


def plot_p2_comparison(exact, meta):
    methods = ["exact-MILP"] + list(meta.keys())
    best = [exact["objective"]] + [meta[m]["best_cost"] for m in meta]
    means = [exact["objective"]] + [meta[m]["mean_cost"] for m in meta]
    stds = [0] + [meta[m]["std_cost"] for m in meta]
    colors = ["#1e293b", "#2563eb", "#16a34a", "#f59e0b"][:len(methods)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax1.axhline(exact["objective"], ls="--", color="#dc2626", lw=1,
                label="exact optimum")
    ax1.set_ylabel("total cost"); ax1.set_title("Mean cost by method (10 seeds)")
    ax1.legend()
    ax1.set_ylim(min(best) * 0.97, max(means) * 1.03)

    for m in meta:
        h = meta[m]["best_history"]
        x = np.linspace(0, 1, len(h))
        ax2.plot(x, h, label=m, lw=1.8)
    ax2.axhline(exact["objective"], ls="--", color="#dc2626", lw=1, label="exact")
    ax2.set_xlabel("search progress (normalized)")
    ax2.set_ylabel("best-so-far cost")
    ax2.set_title("Convergence (best seed)")
    ax2.legend()
    fig.suptitle("P2 - Exact vs metaheuristic", fontweight="bold")
    _save(fig, "p2_method_comparison.png")


def plot_p2_network(solution):
    """Classical-MDS embedding of zones; edges = zone -> serving substation."""
    from models.facility_location import build_instance
    inst = build_instance()
    D = inst.distance.copy()
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = finite.max() * 1.3
    D = 0.5 * (D + D.T)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    XY = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))

    assign = np.array(solution["assign"])
    open_mask = np.array(solution["open_mask"], dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 7))
    for z in range(n):
        s = assign[z]
        if s >= 0 and s != z:
            ax.plot([XY[z, 0], XY[s, 0]], [XY[z, 1], XY[s, 1]],
                    color="#cbd5e1", lw=0.4, zorder=1)
    ax.scatter(XY[~open_mask, 0], XY[~open_mask, 1], s=18, color="#94a3b8",
               label="served zone", zorder=2)
    ax.scatter(XY[open_mask, 0], XY[open_mask, 1], s=90, marker="*",
               color="#dc2626", edgecolor="k", linewidth=0.4,
               label=f"substation ({open_mask.sum()})", zorder=3)
    ax.set_title("P2 - Substation network (MDS embedding of zone distances)",
                 fontweight="bold")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    _save(fig, "p2_network.png")


# ============================================================ P1
def plot_p1_layout(solution):
    from models.qap import build_instance
    inst = build_instance(mode="all")
    coords = np.array(solution["coords"])
    assign = np.array(solution["assign"])
    flow = inst.flow

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="#e2e8f0",
               label="candidate site", zorder=1)
    placed = coords[assign]
    # top flow pairs as ribbons
    fpairs = [(i, j, flow[i, j]) for i in range(inst.n_dept)
              for j in range(inst.n_dept) if i != j]
    fpairs.sort(key=lambda t: -t[2])
    fmax = fpairs[0][2]
    for i, j, w in fpairs[:40]:
        ax.plot([placed[i, 0], placed[j, 0]], [placed[i, 1], placed[j, 1]],
                color="#2563eb", alpha=0.18 + 0.5 * w / fmax,
                lw=0.5 + 2.5 * w / fmax, zorder=2)
    ax.scatter(placed[:, 0], placed[:, 1], s=70, color="#dc2626",
               edgecolor="k", linewidth=0.4, zorder=3, label="placed department")
    for i, (x, y) in enumerate(placed):
        ax.annotate(str(i + 1), (x, y), fontsize=6, ha="center", va="center",
                    color="white", zorder=4)
    ax.set_title(f"P1 - Department layout ({solution['method']}, "
                 f"cost={solution['objective']:,.0f})", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper right")
    _save(fig, "p1_layout.png")


def plot_p1_convergence(full, bounds=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, m in full.items():
        h = m["best_history"]
        ax.plot(np.linspace(0, 1, len(h)), h,
                label=f"{method} (best={m['best_cost']:,.0f})", lw=1.8)
    if bounds:
        ax.axhline(bounds["random_mean"], ls=":", color="#94a3b8", lw=1.2,
                   label=f"random mean ({bounds['random_mean']:,.0f})")
        ax.axhline(bounds["glb"], ls="--", color="#16a34a", lw=1.4,
                   label=f"GL lower bound ({bounds['glb']:,.0f})")
        ax.set_title(f"P1 - QAP convergence (full 30->197): SA certified within "
                     f"{bounds['gap_to_glb_pct']:.0f}% of optimum", fontweight="bold")
    else:
        ax.set_title("P1 - QAP convergence: GA vs SA (full 30->197)", fontweight="bold")
    ax.set_xlabel("search progress (normalized)")
    ax.set_ylabel("best-so-far cost")
    ax.legend(fontsize=9)
    _save(fig, "p1_convergence.png")


# ============================================================ P3
def plot_p3(sol, results):
    t = np.array(sol["throughput"]); q = np.array(sol["storage"])
    coi = np.array(sol["coi"]); order = np.array(sol["order"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    sc = ax1.scatter(t, q, c=coi, s=80, cmap="viridis_r", edgecolor="k", linewidth=0.4)
    for i in range(len(t)):
        ax1.annotate(str(i + 1), (t[i], q[i]), fontsize=6, ha="center", va="center")
    fig.colorbar(sc, ax=ax1, label="COI = storage / throughput")
    ax1.set_xlabel("throughput (ops/period)"); ax1.set_ylabel("storage (slots)")
    ax1.set_title("Products by throughput, storage, COI")

    # placement order: distance band per product (rank)
    ranks = np.argsort(np.argsort(coi))   # 0 = nearest
    ax2.barh(range(len(order)), q[order], color=plt.cm.viridis_r(coi[order] / coi.max()))
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"P{p+1}" for p in order], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("slots (block width)")
    ax2.set_title("COI ordering: nearest (top) -> farthest")
    fig.suptitle(f"P3 - Storage layout (COI optimal travel = {sol['objective']:,.0f})",
                 fontweight="bold")
    _save(fig, "p3_storage.png")

    if results:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = ["random\n(mean)", "popularity\n(throughput-only)", "COI\n(optimal)"]
        vals = [results["random_mean"],
                results.get("popularity_cost", results["random_mean"]),
                results["coi_optimal"]]
        ax.bar(labels, vals, color=["#94a3b8", "#f59e0b", "#16a34a"], alpha=0.9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        s_pop = results.get("savings_vs_popularity_pct",
                            100 * (1 - results["coi_optimal"] / results.get("popularity_cost", 1)))
        ax.set_ylabel("expected travel")
        ax.set_title(f"P3 - COI saves {s_pop:.1f}% vs a realistic popularity policy",
                     fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.12)
        _save(fig, "p3_savings.png")


# ============================================================ cross-cutting
def plot_gap_summary():
    comp = _load("comparison.json")
    if not comp:
        return
    items = [(f"{r['problem']}\n{r['method']}", r["gap_pct"])
             for r in comp
             if isinstance(r.get("gap_pct"), (int, float)) and r["gap_pct"] > 0
             and "random" not in r["method"]]
    if not items:
        return
    labels, gaps = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, gaps, color="#6366f1", alpha=0.85)
    ax.set_ylabel("optimality gap (%)")
    ax.set_title("Metaheuristic optimality gap vs exact/optimal", fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    _save(fig, "comparison_gaps.png")


def main():
    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    print("[viz] rendering solution figures...")

    sweep = _load("p2_alpha_sweep.json")
    p2e = _load("p2_exact.json")
    p2m = _load("p2_metaheuristics.json")
    p2sol = _load("p2_solution.json")
    if sweep:
        plot_p2_alpha_sweep(sweep)
    if p2e and p2m:
        plot_p2_comparison(p2e, p2m)
    if p2sol:
        plot_p2_network(p2sol)

    p1sol = _load("p1_solution.json")
    p1full = _load("p1_full_metaheuristics.json")
    p1bounds = _load("p1_bounds.json")
    if p1sol:
        plot_p1_layout(p1sol)
    if p1full:
        plot_p1_convergence(p1full, p1bounds)

    p3sol = _load("p3_solution.json")
    p3res = _load("p3_results.json")
    if p3sol:
        plot_p3(p3sol, p3res)

    plot_gap_summary()
    print("[viz] done. figures in output/figures/")


if __name__ == "__main__":
    main()
