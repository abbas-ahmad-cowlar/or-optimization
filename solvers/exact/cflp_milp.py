"""
Exact MILP solver for P2 (Capacitated Facility Location)
========================================================
Single-source CFLP solved with PuLP + CBC. Forbidden links (distance = inf)
are handled by simply not creating those assignment variables, which keeps the
model small and avoids big-M. Also provides the LP relaxation for a lower bound.

See docs/formulations.md (P2).
"""

import time
import numpy as np
import pulp


def _build_problem(inst, alpha, relax=False):
    """Construct the PuLP model. Returns (prob, y, x, allowed)."""
    n = inst.n
    cat = "Continuous" if relax else "Binary"

    prob = pulp.LpProblem("CFLP_substation_siting", pulp.LpMinimize)

    y = {s: pulp.LpVariable(f"y_{s}", lowBound=0, upBound=1, cat=cat)
         for s in range(n)}

    # only create x for permitted links (finite distance)
    allowed = {}
    for z in range(n):
        for s in range(n):
            if np.isfinite(inst.distance[z, s]):
                allowed[(z, s)] = pulp.LpVariable(
                    f"x_{z}_{s}", lowBound=0, upBound=1, cat=cat)

    # objective: construction + alpha * transmission
    construction = pulp.lpSum(inst.cost[s] * y[s] for s in range(n))
    transmission = pulp.lpSum(
        alpha * inst.distance[z, s] * inst.demand[z] * allowed[(z, s)]
        for (z, s) in allowed
    )
    prob += construction + transmission

    # each zone served exactly once
    for z in range(n):
        prob += pulp.lpSum(allowed[(z, s)] for s in range(n)
                           if (z, s) in allowed) == 1, f"serve_{z}"

    # serve only from open substations
    for (z, s) in allowed:
        prob += allowed[(z, s)] <= y[s], f"link_{z}_{s}"

    # capacity
    for s in range(n):
        prob += pulp.lpSum(inst.demand[z] * allowed[(z, s)]
                           for z in range(n) if (z, s) in allowed) \
            <= inst.capacity[s] * y[s], f"cap_{s}"

    return prob, y, allowed


def solve_cflp(inst, alpha=None, time_limit=120, msg=False, relax=False):
    """Solve the CFLP. Returns a result dict.

    Parameters
    ----------
    inst : CFLPInstance
    alpha : float or None
        Transmission rate; defaults to inst.alpha0.
    time_limit : int
        CBC time limit (seconds).
    relax : bool
        Solve the LP relaxation (lower bound) instead of the MILP.
    """
    if alpha is None:
        alpha = inst.alpha0

    t0 = time.perf_counter()
    prob, y, allowed = _build_problem(inst, alpha, relax=relax)
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    prob.solve(solver)
    runtime = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    cbc_obj = pulp.value(prob.objective)
    n = inst.n

    if relax:
        # LP relaxation: report the (possibly fractional) bound directly.
        return {
            "method": "exact-LP", "alpha": alpha, "status": status,
            "is_optimal": status == "Optimal",
            "objective": float(cbc_obj) if cbc_obj is not None else None,
            "runtime": runtime, "time_limit": time_limit,
        }

    # --- robust integer extraction ---------------------------------------
    # Assign each zone to its highest-x substation (argmax, not >0.5), so a
    # near-integer or fractional CBC result still yields a concrete solution
    # that we then re-cost and feasibility-check ourselves.
    xval = {(z, s): (pulp.value(v) or 0.0) for (z, s), v in allowed.items()}
    assign = np.full(n, -1, dtype=int)
    best_v = np.full(n, -1.0)
    for (z, s), v in xval.items():
        if v > best_v[z]:
            best_v[z], assign[z] = v, s

    open_mask = np.zeros(n, dtype=bool)
    for s in range(n):
        yv = pulp.value(y[s])
        if yv is not None and yv > 0.5:
            open_mask[s] = True
    open_mask[assign[assign >= 0]] = True   # any served site counts as open

    # re-cost from the rounded integer solution (authoritative)
    construction = float(inst.cost[open_mask].sum())
    served = assign >= 0
    transmission = float(alpha * np.sum(
        inst.distance[np.arange(n)[served], assign[served]] * inst.demand[served]))
    objective = construction + transmission

    # feasibility of the rounded solution
    feasible = bool(served.all())
    if feasible:
        load = np.zeros(n)
        for z in range(n):
            load[assign[z]] += inst.demand[z]
            if not np.isfinite(inst.distance[z, assign[z]]):
                feasible = False
        if (load > inst.capacity + 1e-6).any():
            feasible = False

    return {
        "method": "exact-MILP",
        "alpha": alpha,
        "status": status,
        "is_optimal": status == "Optimal",
        "feasible": feasible,
        "objective": objective if feasible else None,
        "cbc_objective": float(cbc_obj) if cbc_obj is not None else None,
        "construction": construction if feasible else None,
        "transmission": transmission if feasible else None,
        "n_open": int(open_mask.sum()),
        "open_mask": open_mask.tolist(),
        "assign": assign.tolist(),
        "runtime": runtime,
        "time_limit": time_limit,
    }
