"""
Experiment runner for P2 - Substation Siting (CFLP)
===================================================
Solves the capacitated facility-location problem with:
  * exact MILP (PuLP/CBC) at the baseline transmission rate
  * LP relaxation (lower bound)
  * GA, SA, PSO metaheuristics across multiple seeds (mean +/- std)
  * an alpha-sweep (transmission-rate sensitivity) via the exact solver

Results are written to output/results/p2_*.json so figures and the report
regenerate from data.

Usage:
    python -m experiments.run_p2            # full run
    python -m experiments.run_p2 --quick    # fast smoke test
"""

import argparse
import json
import os
import sys
import numpy as np

from experiments import config
from models.facility_location import build_instance
from solvers.exact.cflp_milp import solve_cflp
from solvers.metaheuristic.cflp_ops import CFLPEncoding
from solvers.metaheuristic.genetic import GeneticAlgorithm
from solvers.metaheuristic.simulated_annealing import SimulatedAnnealing
from solvers.metaheuristic.pso import ParticleSwarm


def _ga(enc, hp, seed):
    return GeneticAlgorithm(
        enc.init, enc.evaluate, enc.crossover, enc.mutate,
        pop_size=hp["pop_size"], n_generations=hp["n_generations"],
        tournament_k=hp["tournament_k"], elitism=hp["elitism"],
        crossover_rate=hp["crossover_rate"], mutation_rate=hp["mutation_rate"],
        seed=seed,
    ).run()


def _sa(enc, hp, seed):
    return SimulatedAnnealing(
        enc.init, enc.evaluate, enc.neighbor,
        n_iterations=hp["n_iterations"], t_end=hp["t_end"],
        cooling=hp["cooling"], seed=seed,
    ).run()


def _pso(enc, hp, seed):
    return ParticleSwarm(
        enc.n, enc.evaluate_position,
        n_particles=hp["n_particles"], n_iterations=hp["n_iterations"],
        w=hp["w"], c1=hp["c1"], c2=hp["c2"], seed=seed,
    ).run()


def run_metaheuristic(enc, runner, hp, seeds):
    """Run one metaheuristic across seeds; return aggregate + best history."""
    per_seed = []
    best = None
    for s in seeds:
        res = runner(enc, hp, s)
        per_seed.append(dict(seed=s, best_cost=res.best_cost,
                             runtime=res.runtime, n_evals=res.n_evals))
        if best is None or res.best_cost < best.best_cost:
            best = res
    costs = np.array([p["best_cost"] for p in per_seed], dtype=float)
    return {
        "method": best.method,
        "n_seeds": len(seeds),
        "best_cost": float(costs.min()),
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std()),
        "worst_cost": float(costs.max()),
        "mean_runtime": float(np.mean([p["runtime"] for p in per_seed])),
        "per_seed": per_seed,
        "best_seed": best.seed,
        "best_history": [float(h) for h in best.history],
        "best_open_mask": np.asarray(best.best_solution, dtype=bool).tolist(),
    }


def main(quick=False, exact_only=False):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    inst = build_instance()
    alpha = inst.alpha0 * config.P2_ALPHA_BASELINE

    print(f"[P2] zones={inst.n}  total_demand={inst.total_demand:,.0f}  "
          f"total_capacity={inst.total_capacity:,.0f}  "
          f"min_open(capacity)={inst.min_substations_for_capacity()}")
    print(f"[P2] alpha_0={inst.alpha0:.6g}  baseline alpha={alpha:.6g}")

    # hyperparameters (smaller for --quick)
    ga_hp, sa_hp, pso_hp = dict(config.GA), dict(config.SA), dict(config.PSO)
    seeds = config.SEEDS
    exact_tl = config.EXACT_TIME_LIMIT
    sweep_tl = max(20, config.EXACT_TIME_LIMIT // 4)
    if quick:
        ga_hp.update(pop_size=20, n_generations=20)
        sa_hp.update(n_iterations=1000)
        pso_hp.update(n_particles=12, n_iterations=20)
        seeds = config.SEEDS[:2]
        exact_tl, sweep_tl = 15, 8

    enc = CFLPEncoding(inst, alpha)

    # ---- exact MILP + LP bound at baseline alpha ----
    print(f"[P2] solving exact MILP (time limit {exact_tl}s)...")
    exact = solve_cflp(inst, alpha=alpha, time_limit=exact_tl, msg=False)
    print(f"      status={exact['status']}  feasible={exact['feasible']}  "
          f"obj={_fmt(exact['objective'])}  n_open={exact['n_open']}  "
          f"({exact['runtime']:.1f}s)")
    lp = solve_cflp(inst, alpha=alpha, time_limit=exact_tl, msg=False, relax=True)
    exact["lp_lower_bound"] = lp["objective"]
    if exact["objective"] and lp["objective"]:
        exact["lp_gap_pct"] = 100.0 * (exact["objective"] - lp["objective"]) / exact["objective"]
    _dump("p2_exact.json", exact)
    _dump("p2_solution.json", {
        "alpha": alpha, "objective": exact["objective"],
        "construction": exact["construction"], "transmission": exact["transmission"],
        "n_open": exact["n_open"], "open_mask": exact["open_mask"],
        "assign": exact["assign"], "zone_names": inst.zone_names,
        "status": exact["status"],
    })

    # ---- metaheuristics ----
    if not exact_only:
        meta = {}
        for name, runner, hp in [("GA", _ga, ga_hp), ("SA", _sa, sa_hp), ("PSO", _pso, pso_hp)]:
            print(f"[P2] running {name} x {len(seeds)} seeds...")
            meta[name] = run_metaheuristic(enc, runner, hp, seeds)
            m = meta[name]
            gap = 100.0 * (m["best_cost"] - exact["objective"]) / exact["objective"] \
                if exact["objective"] else float("nan")
            print(f"      best={m['best_cost']:,.2f}  mean={m['mean_cost']:,.2f}"
                  f" +/- {m['std_cost']:,.2f}  gap-to-exact={gap:+.2f}%")
            m["gap_to_exact_pct"] = gap
        _dump("p2_metaheuristics.json", meta)
    else:
        meta = None
        print("[P2] exact-only: keeping existing p2_metaheuristics.json")

    # ---- alpha sweep (sensitivity) ----
    # CBC where it proves optimality; a quick GA guarantees a feasible value at
    # the harder (low-alpha) points where CBC finds no integer incumbent in time.
    print(f"[P2] alpha sweep ({len(config.P2_ALPHA_SWEEP)} points, "
          f"time limit {sweep_tl}s each + GA fallback)...")
    sweep = []
    for mult in config.P2_ALPHA_SWEEP:
        a = inst.alpha0 * mult
        r = solve_cflp(inst, alpha=a, time_limit=sweep_tl, msg=False)

        enc_a = CFLPEncoding(inst, a)
        ga = GeneticAlgorithm(enc_a.init, enc_a.evaluate, enc_a.crossover,
                              enc_a.mutate, pop_size=50, n_generations=80,
                              tournament_k=3, elitism=2, crossover_rate=0.9,
                              mutation_rate=0.15, seed=config.SEED).run()
        ga_mask = np.asarray(ga.best_solution, dtype=bool)
        ga_constr = float(inst.cost[ga_mask].sum())

        if r["feasible"] and r["objective"] is not None and r["objective"] <= ga.best_cost:
            pt = dict(source="exact", n_open=r["n_open"],
                      construction=r["construction"], transmission=r["transmission"],
                      objective=r["objective"], is_optimal=r["is_optimal"])
        else:
            pt = dict(source="heuristic-GA", n_open=int(ga_mask.sum()),
                      construction=ga_constr, transmission=ga.best_cost - ga_constr,
                      objective=ga.best_cost, is_optimal=False)
        pt.update(alpha_mult=mult, alpha=a, feasible=True, status=r["status"])
        sweep.append(pt)
        print(f"      mult={mult:<5} n_open={pt['n_open']:<4} "
              f"obj={_fmt(pt['objective'])}  ({pt['source']}, "
              f"opt={pt['is_optimal']})")
    _dump("p2_alpha_sweep.json", sweep)

    print("[P2] done. results in output/results/p2_*.json")
    return exact, meta, sweep


def _fmt(v):
    return f"{v:,.0f}" if v is not None else "n/a"


def _dump(name, obj):
    path = os.path.join(config.RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--exact-only", action="store_true",
                    help="re-run only exact baseline + alpha sweep")
    args = ap.parse_args()
    main(quick=args.quick, exact_only=args.exact_only)
