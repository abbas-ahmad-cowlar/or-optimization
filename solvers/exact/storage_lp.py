"""
Exact transportation LP for P3 (storage slot assignment)
========================================================
Assigns products (supply = storage slots required) to unit-capacity slots
minimizing total expected travel. The constraint matrix is totally unimodular,
so the LP relaxation is integral and its optimum equals the Cube-per-Order-Index
assignment. Run on a downscaled StorageInstance (see StorageInstance.downscaled)
to keep the LP small while verifying the heuristic's optimality.
"""

import time
import numpy as np
import pulp


def solve_storage_lp(inst, msg=False, time_limit=120):
    P = inst.n_products
    N = inst.total_slots
    delta = inst.slot_dist
    freq = inst.throughput / inst.storage          # visits per slot for product p

    prob = pulp.LpProblem("storage_assignment", pulp.LpMinimize)
    x = {(p, s): pulp.LpVariable(f"x_{p}_{s}", lowBound=0, upBound=1,
                                 cat="Continuous")
         for p in range(P) for s in range(N)}

    prob += pulp.lpSum(freq[p] * delta[s] * x[(p, s)]
                       for p in range(P) for s in range(N))

    for p in range(P):                              # supply: each product's slots
        prob += pulp.lpSum(x[(p, s)] for s in range(N)) == int(inst.storage[p])
    for s in range(N):                              # each slot holds <= 1 product
        prob += pulp.lpSum(x[(p, s)] for p in range(P)) <= 1

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    runtime = time.perf_counter() - t0

    obj = pulp.value(prob.objective)
    return {
        "method": "exact-LP",
        "status": pulp.LpStatus[prob.status],
        "objective": float(obj) if obj is not None else None,
        "n_slots": N,
        "runtime": runtime,
    }
