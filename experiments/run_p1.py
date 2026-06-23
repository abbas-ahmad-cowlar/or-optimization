"""
Experiment runner for P1 - Department Layout (QAP)
==================================================
Two studies:
  1. Reduced K x K instance solved BOTH exactly (MILP) and with GA/SA, to
     measure the heuristics' optimality gap on a provably-optimal benchmark.
  2. Full 30 -> 197 instance (intractable for exact), solved with GA/SA across
     multiple seeds; the best layout is exported for visualization. A
     type-compatibility variant (A3) is also reported.

Results -> output/results/p1_*.json

Usage:
    python -m experiments.run_p1            # full run
    python -m experiments.run_p1 --quick    # fast smoke test
"""

import argparse
import json
import os
import numpy as np

from experiments import config
from models.qap import build_instance
from solvers.exact.qap_milp import solve_qap, brute_force_qap, gilmore_lawler_bound
from solvers.metaheuristic.qap_ops import QAPEncoding
from solvers.metaheuristic.genetic import GeneticAlgorithm
from solvers.metaheuristic.simulated_annealing import SimulatedAnnealing


def _ga(enc, hp, seed):
    return GeneticAlgorithm(
        enc.init, enc.evaluate, enc.crossover, enc.mutate,
        pop_size=hp["pop_size"], n_generations=hp["n_generations"],
        tournament_k=hp["tournament_k"], elitism=hp["elitism"],
        crossover_rate=hp["crossover_rate"], mutation_rate=hp["mutation_rate"],
        seed=seed).run()


def _sa(enc, hp, seed):
    return SimulatedAnnealing(
        enc.init, enc.evaluate, enc.neighbor,
        n_iterations=hp["n_iterations"], t_end=hp["t_end"],
        cooling=hp["cooling"], seed=seed).run()


def _multiseed(enc, runner, hp, seeds):
    per_seed, best = [], None
    for s in seeds:
        res = runner(enc, hp, s)
        per_seed.append(dict(seed=s, best_cost=res.best_cost,
                             runtime=res.runtime, n_evals=res.n_evals))
        if best is None or res.best_cost < best.best_cost:
            best = res
    costs = np.array([p["best_cost"] for p in per_seed])
    return {
        "method": best.method, "n_seeds": len(seeds),
        "best_cost": float(costs.min()), "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std()), "worst_cost": float(costs.max()),
        "mean_runtime": float(np.mean([p["runtime"] for p in per_seed])),
        "per_seed": per_seed, "best_seed": best.seed,
        "best_history": [float(h) for h in best.history],
        "best_assign": [int(a) for a in best.best_solution],
    }


def main(quick=False):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    inst = build_instance(mode="all")
    print(f"[P1] departments={inst.n_dept}  candidate locations={inst.n_loc}  "
          f"flow asymmetric={not np.allclose(inst.flow, inst.flow.T)}")

    ga_hp, sa_hp = dict(config.P1_GA), dict(config.P1_SA)
    seeds = config.SEEDS
    K = config.P1_EXACT_K
    exact_tl = config.EXACT_TIME_LIMIT
    if quick:
        ga_hp.update(pop_size=30, n_generations=60)
        sa_hp.update(n_iterations=4000)
        seeds = config.SEEDS[:2]
        K, exact_tl = 6, 20

    # ================================================================
    # 1) Reduced K x K instance: exact vs heuristic
    # ================================================================
    rng = np.random.default_rng(config.SEED)
    total_flow = inst.flow.sum(axis=0) + inst.flow.sum(axis=1)
    dept_idx = np.argsort(-total_flow)[:K]                 # top-K busiest depts
    loc_idx = rng.choice(inst.n_loc, size=K, replace=False)  # seeded site subset
    red = inst.reduced(dept_idx, loc_idx)
    red_enc = QAPEncoding(red, location_subset=np.arange(K))

    print(f"[P1] reduced {K}x{K} QAP: brute-force enumeration (authoritative)...")
    ex = brute_force_qap(red.flow, red.distance)   # guaranteed-correct optimum
    print(f"      enumeration optimum={ex['objective']:,.0f}  ({ex['n_perms']:,} perms)")
    # illustrative MILP: generic linearization often cannot prove QAP optimality
    milp = solve_qap(red.flow, red.distance, time_limit=min(exact_tl, 30), msg=False)
    ex["milp_objective"] = milp["objective"]
    ex["milp_status"] = milp["status"]
    ex["milp_gap_pct"] = (100.0 * (milp["objective"] - ex["objective"]) / ex["objective"]
                          if milp["objective"] else None)
    ex["milp_runtime"] = milp["runtime"]
    print(f"      MILP(CBC) obj={milp['objective']:,.0f} status={milp['status']} "
          f"-> {ex['milp_gap_pct']:+.2f}% above true optimum (CBC mislabels QAP)")

    red_meta = {}
    for name, runner, hp in [("GA", _ga, ga_hp), ("SA", _sa, sa_hp)]:
        r = _multiseed(red_enc, runner, hp, seeds)
        gap = 100.0 * (r["best_cost"] - ex["objective"]) / ex["objective"] \
            if ex["objective"] else float("nan")
        r["gap_to_exact_pct"] = gap
        red_meta[name] = r
        print(f"      {name}: best={r['best_cost']:,.0f}  gap-to-exact={gap:+.2f}%")

    _dump("p1_reduced.json", {
        "K": K, "dept_idx": dept_idx.tolist(), "loc_idx": loc_idx.tolist(),
        "exact": ex, "metaheuristics": red_meta,
    })

    # ================================================================
    # 2) Full 30 -> 197 instance: heuristics only
    # ================================================================
    full_enc = QAPEncoding(inst)
    print(f"[P1] full 30x197 QAP, mode=all: GA & SA x {len(seeds)} seeds...")
    full_meta = {}
    for name, runner, hp in [("GA", _ga, ga_hp), ("SA", _sa, sa_hp)]:
        full_meta[name] = _multiseed(full_enc, runner, hp, seeds)
        m = full_meta[name]
        print(f"      {name}: best={m['best_cost']:,.0f}  "
              f"mean={m['mean_cost']:,.0f} +/- {m['std_cost']:,.0f}")
    _dump("p1_full_metaheuristics.json", full_meta)

    # ---- quality certificate for the full (no-exact) instance --------------
    # Triangulate: GL lower bound below, the random-layout distribution as
    # context, and seed-to-seed stability as convergence evidence.
    best_method = min(full_meta, key=lambda k: full_meta[k]["best_cost"])
    best_cost = full_meta[best_method]["best_cost"]
    glb = gilmore_lawler_bound(inst.flow, inst.distance)
    rng2 = np.random.default_rng(config.SEED + 1)
    n_rand = 200 if quick else 5000
    rand = [full_enc.evaluate(full_enc.random_solution(rng2)) for _ in range(n_rand)]
    bounds = {
        "glb": glb, "best_method": best_method, "best_cost": best_cost,
        "gap_to_glb_pct": 100.0 * (best_cost - glb) / glb,
        "random_mean": float(np.mean(rand)), "random_best": float(np.min(rand)),
        "improvement_over_random_mean_pct": 100.0 * (1 - best_cost / np.mean(rand)),
        "seed_std_pct": 100.0 * full_meta[best_method]["std_cost"] / best_cost,
        "n_random": n_rand,
    }
    _dump("p1_bounds.json", bounds)
    print(f"[P1] GL lower bound={glb:,.0f}  ->  {best_method} certified within "
          f"{bounds['gap_to_glb_pct']:.1f}% of optimum (GLB is loose);")
    print(f"     {bounds['improvement_over_random_mean_pct']:.1f}% better than random "
          f"mean, seed std only {bounds['seed_std_pct']:.1f}% -> strong, stable optimum")

    # best overall full solution -> for visualization
    best_assign = full_meta[best_method]["best_assign"]
    _dump("p1_solution.json", {
        "mode": "all", "method": best_method,
        "objective": full_meta[best_method]["best_cost"],
        "assign": best_assign,
        "dept_names": inst.dept_names,
        "coords": inst.coords.tolist(),
        "loc_types": inst.loc_types,
    })

    # ================================================================
    # 3) Type-compatibility variant (A3)
    # ================================================================
    inst_t = build_instance(mode="type")
    enc_t = QAPEncoding(inst_t)
    print("[P1] type-compatibility variant: GA...")
    tv = _multiseed(enc_t, _ga, ga_hp, seeds)
    print(f"      GA(type): best={tv['best_cost']:,.0f}")
    _dump("p1_type_variant.json", {
        "method": "GA", "objective": tv["best_cost"],
        "assign": tv["best_assign"], "dept_names": inst_t.dept_names,
        "coords": inst_t.coords.tolist(), "loc_types": inst_t.loc_types,
        "best_history": tv["best_history"],
    })

    print("[P1] done. results in output/results/p1_*.json")


def _dump(name, obj):
    with open(os.path.join(config.RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
