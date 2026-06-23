"""
Exact MILP solver for a (square) Quadratic Assignment Problem
=============================================================
Standard linearized QAP (McCormick / Adams-Johnson style): assign n facilities
to n locations minimizing sum_{i,j,k,l} f_ij d_kl x_ik x_jl. Used only on the
*reduced* P1 instance (small K), since exact QAP is intractable at full scale.

See docs/formulations.md (P1).
"""

import itertools
import math
import time
import numpy as np
import pulp


def brute_force_qap(flow, distance):
    """Authoritative QAP optimum by full enumeration (feasible for n <= ~9).

    The generic McCormick MILP linearization is loose, so CBC can stall and
    mislabel a timeout incumbent as 'Optimal' on QAP; exhaustive search gives a
    guaranteed-correct baseline for the (small) reduced instance.
    """
    n = flow.shape[0]
    best, best_assign = None, None
    for perm in itertools.permutations(range(n)):
        a = np.array(perm)
        v = float(np.sum(flow * distance[np.ix_(a, a)]))
        if best is None or v < best:
            best, best_assign = v, a
    return {"method": "exact-enumeration", "n": n, "status": "Optimal",
            "is_optimal": True, "objective": best,
            "assign": best_assign.tolist(), "n_perms": math.factorial(n)}


def solve_qap(flow, distance, time_limit=120, msg=False):
    """Solve a square QAP exactly. Returns a result dict.

    Parameters
    ----------
    flow : (n, n) array      department interaction (may be asymmetric)
    distance : (n, n) array  location distances (symmetric)
    """
    n = flow.shape[0]
    assert distance.shape == (n, n)

    prob = pulp.LpProblem("QAP", pulp.LpMinimize)

    # assignment variables
    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary")
         for i in range(n) for k in range(n)}

    # linearization variables y_ikjl = x_ik * x_jl, only where flow != 0 (i != j)
    pairs = [(i, j) for i in range(n) for j in range(n)
             if i != j and flow[i, j] != 0]
    y = {}
    for (i, j) in pairs:
        for k in range(n):
            for l in range(n):
                if k == l:
                    continue
                y[(i, k, j, l)] = pulp.LpVariable(
                    f"y_{i}_{k}_{j}_{l}", lowBound=0, upBound=1, cat="Continuous")

    # objective
    prob += pulp.lpSum(flow[i, j] * distance[k, l] * y[(i, k, j, l)]
                       for (i, k, j, l) in y)

    # assignment constraints
    for i in range(n):
        prob += pulp.lpSum(x[(i, k)] for k in range(n)) == 1, f"dept_{i}"
    for k in range(n):
        prob += pulp.lpSum(x[(i, k)] for i in range(n)) == 1, f"loc_{k}"

    # McCormick linkage
    for (i, k, j, l) in y:
        prob += y[(i, k, j, l)] <= x[(i, k)]
        prob += y[(i, k, j, l)] <= x[(j, l)]
        prob += y[(i, k, j, l)] >= x[(i, k)] + x[(j, l)] - 1

    t0 = time.perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    prob.solve(solver)
    runtime = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    assign = np.full(n, -1, dtype=int)
    for i in range(n):
        for k in range(n):
            v = pulp.value(x[(i, k)])
            if v is not None and v > 0.5:
                assign[i] = k

    # recompute objective directly from the assignment (robust to relaxed y)
    obj = float(np.sum(flow * distance[np.ix_(assign, assign)])) \
        if np.all(assign >= 0) else None

    return {
        "method": "exact-MILP",
        "n": n,
        "status": status,
        "is_optimal": status == "Optimal",
        "objective": obj,
        "assign": assign.tolist(),
        "runtime": runtime,
        "time_limit": time_limit,
    }
