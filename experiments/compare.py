"""
Cross-problem comparison: exact vs metaheuristic
================================================
Aggregates the logged results from P1/P2/P3 into a single tidy table
(output/results/comparison.csv) plus a JSON summary, quantifying solution
quality, optimality gap, and runtime for each method on each problem.

Run after run_p1, run_p2, run_p3. Usage: python -m experiments.compare
"""

import csv
import json
import os

from experiments import config

RESULTS = config.RESULTS_DIR


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_rows():
    rows = []

    # ---------------- P2 (CFLP) ----------------
    p2e = _load("p2_exact.json")
    p2m = _load("p2_metaheuristics.json")
    if p2e:
        rows.append(dict(problem="P2-CFLP", scope="full(137)", method="exact-MILP",
                         objective=p2e.get("objective"), gap_pct=0.0,
                         mean=p2e.get("objective"), std=0.0,
                         runtime_s=p2e.get("runtime"),
                         status=p2e.get("status"),
                         notes=f"n_open={p2e.get('n_open')}"))
    if p2m:
        for method, m in p2m.items():
            rows.append(dict(problem="P2-CFLP", scope="full(137)", method=method,
                             objective=m["best_cost"], gap_pct=m.get("gap_to_exact_pct"),
                             mean=m["mean_cost"], std=m["std_cost"],
                             runtime_s=m["mean_runtime"], status="",
                             notes=f"{m['n_seeds']} seeds"))

    # ---------------- P1 (QAP) ----------------
    p1r = _load("p1_reduced.json")
    if p1r:
        K = p1r["K"]
        ex = p1r["exact"]
        rows.append(dict(problem="P1-QAP", scope=f"reduced({K}x{K})",
                         method="exact-enum", objective=ex["objective"], gap_pct=0.0,
                         mean=ex["objective"], std=0.0, runtime_s="",
                         status="Optimal", notes="full enumeration (true optimum)"))
        if ex.get("milp_objective") is not None:
            rows.append(dict(problem="P1-QAP", scope=f"reduced({K}x{K})",
                             method="MILP(CBC)", objective=ex["milp_objective"],
                             gap_pct=ex.get("milp_gap_pct"), mean=ex["milp_objective"],
                             std=0.0, runtime_s=ex.get("milp_runtime"),
                             status=ex.get("milp_status"),
                             notes="generic linearization stalls on QAP"))
        for method, m in p1r["metaheuristics"].items():
            rows.append(dict(problem="P1-QAP", scope=f"reduced({K}x{K})", method=method,
                             objective=m["best_cost"], gap_pct=m.get("gap_to_exact_pct"),
                             mean=m["mean_cost"], std=m["std_cost"],
                             runtime_s=m["mean_runtime"], status="",
                             notes=f"{m['n_seeds']} seeds"))
    p1f = _load("p1_full_metaheuristics.json")
    p1b = _load("p1_bounds.json")
    if p1f:
        best_full = min(m["best_cost"] for m in p1f.values())
        for method, m in p1f.items():
            gap = (p1b["gap_to_glb_pct"] if (p1b and m["best_cost"] == best_full)
                   else 100.0 * (m["best_cost"] - best_full) / best_full)
            note = (f"certified within {p1b['gap_to_glb_pct']:.1f}% of optimum (GL bound)"
                    if (p1b and m["best_cost"] == best_full) else "vs best heuristic")
            rows.append(dict(problem="P1-QAP", scope="full(30->197)", method=method,
                             objective=m["best_cost"], gap_pct=gap,
                             mean=m["mean_cost"], std=m["std_cost"],
                             runtime_s=m["mean_runtime"], status="", notes=note))
    if p1b:
        rows.append(dict(problem="P1-QAP", scope="full(30->197)",
                         method="GL lower bound", objective=p1b["glb"], gap_pct=0.0,
                         mean=p1b["glb"], std=0.0, runtime_s="", status="valid bound",
                         notes=f"random mean {p1b['random_mean']:,.0f}; "
                               f"seed std {p1b['seed_std_pct']:.1f}%"))

    # ---------------- P3 (storage) ----------------
    p3 = _load("p3_results.json")
    if p3:
        rows.append(dict(problem="P3-storage", scope="full(25189 slots)",
                         method="COI/exact-LP", objective=p3["coi_optimal"], gap_pct=0.0,
                         mean=p3["coi_optimal"], std=0.0, runtime_s="",
                         status="optimal",
                         notes=f"LP verified={p3['exact_lp']['matches']}"))
        for method, m in p3["metaheuristics"].items():
            rows.append(dict(problem="P3-storage", scope="full(25189 slots)", method=method,
                             objective=m["best_cost"], gap_pct=m.get("gap_to_coi_pct"),
                             mean=m["mean_cost"], std=m["std_cost"], runtime_s="",
                             status="", notes="ordering of 20 products"))
        if p3.get("popularity_cost"):
            rows.append(dict(problem="P3-storage", scope="full(25189 slots)",
                             method="popularity", objective=p3["popularity_cost"],
                             gap_pct=100.0 * (p3["popularity_cost"] - p3["coi_optimal"]) / p3["coi_optimal"],
                             mean=p3["popularity_cost"], std="", runtime_s="", status="",
                             notes="throughput-only naive policy"))
        rows.append(dict(problem="P3-storage", scope="full(25189 slots)",
                         method="random(mean)", objective=p3["random_mean"],
                         gap_pct=100.0 * (p3["random_mean"] - p3["coi_optimal"]) / p3["coi_optimal"],
                         mean=p3["random_mean"], std="", runtime_s="", status="",
                         notes="weak baseline"))
    return rows


def main():
    rows = build_rows()
    if not rows:
        print("[compare] no result files found - run the experiments first.")
        return

    cols = ["problem", "scope", "method", "objective", "gap_pct", "mean", "std",
            "runtime_s", "status", "notes"]
    out_csv = os.path.join(RESULTS, "comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(RESULTS, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # pretty print
    print(f"{'problem':12} {'scope':18} {'method':14} {'objective':>16} {'gap%':>8} {'runtime':>9}")
    print("-" * 82)
    for r in rows:
        obj = f"{r['objective']:,.0f}" if isinstance(r['objective'], (int, float)) else "n/a"
        gap = f"{r['gap_pct']:+.2f}" if isinstance(r['gap_pct'], (int, float)) else ""
        rt = f"{r['runtime_s']:.1f}s" if isinstance(r['runtime_s'], (int, float)) else ""
        print(f"{r['problem']:12} {r['scope']:18} {r['method']:14} {obj:>16} {gap:>8} {rt:>9}")
    print(f"\n[compare] wrote {out_csv}")


if __name__ == "__main__":
    main()
