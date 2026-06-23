"""
Experiment runner for P3 - Storage Layout (COI assignment)
==========================================================
  * COI rule (provably optimal ordering) at full scale  -> headline result
  * exact transportation LP on a downscaled instance     -> verifies optimality
  * GA & SA over the 20-product ordering                 -> confirm they reach COI
  * random-ordering baseline                             -> shows the savings

Results -> output/results/p3_*.json

Usage:
    python -m experiments.run_p3 [--quick]
"""

import argparse
import json
import os
import numpy as np

from experiments import config
from models.storage import build_instance
from solvers.exact.storage_lp import solve_storage_lp
from solvers.metaheuristic.permutation_ops import PermutationEncoding
from solvers.metaheuristic.genetic import GeneticAlgorithm
from solvers.metaheuristic.simulated_annealing import SimulatedAnnealing


def _multiseed(enc, kind, hp, seeds):
    per_seed, best = [], None
    for s in seeds:
        if kind == "GA":
            res = GeneticAlgorithm(
                enc.init, enc.evaluate, enc.crossover, enc.mutate,
                pop_size=hp["pop_size"], n_generations=hp["n_generations"],
                tournament_k=hp["tournament_k"], elitism=hp["elitism"],
                crossover_rate=hp["crossover_rate"],
                mutation_rate=hp["mutation_rate"], seed=s).run()
        else:
            res = SimulatedAnnealing(
                enc.init, enc.evaluate, enc.neighbor,
                n_iterations=hp["n_iterations"], t_end=hp["t_end"],
                cooling=hp["cooling"], seed=s).run()
        per_seed.append(dict(seed=s, best_cost=res.best_cost, runtime=res.runtime))
        if best is None or res.best_cost < best.best_cost:
            best = res
    costs = np.array([p["best_cost"] for p in per_seed])
    return {"method": best.method, "best_cost": float(costs.min()),
            "mean_cost": float(costs.mean()), "std_cost": float(costs.std()),
            "best_order": [int(i) for i in best.best_solution],
            "best_history": [float(h) for h in best.history],
            "per_seed": per_seed}


def main(quick=False):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    inst = build_instance()
    P = inst.n_products
    print(f"[P3] products={P}  total slots={inst.total_slots:,}  "
          f"total throughput={int(inst.throughput.sum()):,}")

    # ---- COI optimum (full scale) ----
    coi_order = inst.coi_order()
    coi_cost = inst.evaluate(coi_order)
    print(f"[P3] COI optimal travel = {coi_cost:,.1f}")

    # ---- baselines: random AND a realistic naive policy ----
    rng = np.random.default_rng(config.SEED)
    rand_costs = [inst.evaluate(rng.permutation(P)) for _ in range(2000)]
    pop_cost = inst.evaluate(inst.popularity_order())   # throughput-only (ignores cube)
    print(f"[P3] random ordering: mean={np.mean(rand_costs):,.0f}  "
          f"(COI saves {100*(1-coi_cost/np.mean(rand_costs)):.1f}% vs random mean)")
    print(f"[P3] popularity (throughput-only) policy: {pop_cost:,.0f}  "
          f"(COI saves {100*(1-coi_cost/pop_cost):.1f}% vs popularity)")

    # ---- metaheuristics over the 20-product ordering ----
    enc = PermutationEncoding(P, inst.evaluate)
    ga_hp = dict(config.P1_GA)
    sa_hp = dict(config.P1_SA)
    seeds = config.SEEDS if not quick else config.SEEDS[:2]
    if quick:
        ga_hp.update(n_generations=80)
        sa_hp.update(n_iterations=5000)

    meta = {}
    for kind, hp in [("GA", ga_hp), ("SA", sa_hp)]:
        m = _multiseed(enc, kind, hp, seeds)
        m["gap_to_coi_pct"] = 100.0 * (m["best_cost"] - coi_cost) / coi_cost
        meta[kind] = m
        print(f"[P3] {kind}: best={m['best_cost']:,.1f}  "
              f"gap-to-COI={m['gap_to_coi_pct']:+.3f}%")

    # ---- exact transportation LP on a downscaled instance ----
    small = inst.downscaled(target_slots=60 if quick else 240)
    lp = solve_storage_lp(small, msg=False)
    coi_small = small.evaluate(small.coi_order())
    lp_matches = (lp["objective"] is not None
                  and abs(lp["objective"] - coi_small) / coi_small < 1e-6)
    print(f"[P3] exact LP (downscaled N={small.total_slots}): "
          f"obj={lp['objective']:,.2f}  COI={coi_small:,.2f}  match={lp_matches}")

    # ---- dump ----
    _dump("p3_results.json", {
        "coi_optimal": coi_cost,
        "coi_order": [int(i) for i in coi_order],
        "random_mean": float(np.mean(rand_costs)),
        "random_best": float(np.min(rand_costs)),
        "popularity_cost": float(pop_cost),
        "savings_vs_popularity_pct": 100.0 * (1 - coi_cost / pop_cost),
        "savings_vs_random_pct": 100.0 * (1 - coi_cost / float(np.mean(rand_costs))),
        "metaheuristics": meta,
        "exact_lp": {"objective": lp["objective"], "coi_downscaled": coi_small,
                     "matches": bool(lp_matches), "n_slots": small.total_slots,
                     "status": lp["status"]},
    })
    blocks = inst.assignment(coi_order)
    _dump("p3_solution.json", {
        "method": "COI", "objective": coi_cost,
        "order": [int(i) for i in coi_order],
        "product_names": inst.product_names,
        "throughput": inst.throughput.tolist(),
        "storage": inst.storage.tolist(),
        "coi": inst.coi().tolist(),
        "blocks": {str(p): blocks[p] for p in blocks},
        "total_slots": inst.total_slots,
    })
    print("[P3] done. results in output/results/p3_*.json")


def _dump(name, obj):
    with open(os.path.join(config.RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
