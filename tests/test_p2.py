"""Feasibility and sanity tests for P2 (CFLP)."""

import numpy as np
import pytest

from models.facility_location import build_instance
from solvers.exact.cflp_milp import solve_cflp
from solvers.metaheuristic.cflp_ops import CFLPEncoding


@pytest.fixture(scope="module")
def inst():
    return build_instance()


def test_instance_shapes(inst):
    assert inst.n == 137
    assert inst.demand.shape == (137,)
    assert inst.distance.shape == (137, 137)
    # diagonal set to 0 (self-service, A8)
    assert np.allclose(np.diag(inst.distance), 0.0)
    # every zone can self-serve (capacity_z >= demand_z) -- guarantees feasibility
    assert np.all(inst.capacity >= inst.demand)


def test_common_unit_feasible(inst):
    # A5: raw common-unit reading must be feasible (capacity > demand)
    assert inst.total_capacity > inst.total_demand


def test_alpha0_positive(inst):
    assert inst.alpha0 > 0


def test_greedy_all_open_is_feasible(inst):
    mask = np.ones(inst.n, dtype=bool)
    cost, feasible, assign = inst.total_cost(mask, inst.alpha0)
    assert feasible
    assert np.all(assign >= 0)
    assert np.isfinite(cost)


def test_repair_makes_feasible(inst):
    enc = CFLPEncoding(inst, inst.alpha0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = rng.random(inst.n) < 0.3   # sparse -> likely infeasible pre-repair
        mask = enc.repair(mask)
        _, feasible, _ = inst.greedy_assign(mask, inst.alpha0)
        assert feasible


def test_capacity_respected_in_assignment(inst):
    mask = np.ones(inst.n, dtype=bool)
    assign, feasible, _ = inst.greedy_assign(mask, inst.alpha0)
    assert feasible
    load = np.zeros(inst.n)
    for z, s in enumerate(assign):
        load[s] += inst.demand[z]
    assert np.all(load <= inst.capacity + 1e-6)


def test_no_forbidden_links_used(inst):
    mask = np.ones(inst.n, dtype=bool)
    assign, feasible, _ = inst.greedy_assign(mask, inst.alpha0)
    for z, s in enumerate(assign):
        assert np.isfinite(inst.distance[z, s])


def test_exact_solver_feasible(inst):
    """Exact MILP returns a feasible, sensible solution.

    The time limit is a cap, not a fixed duration: CBC returns as soon as it
    proves optimality (~20s for this instance), so the generous 90s ceiling
    removes timing flakiness without slowing the normal case.
    """
    res = solve_cflp(inst, alpha=inst.alpha0, time_limit=90, msg=False)
    assert res["feasible"], f"expected a feasible solution, got status={res['status']}"
    assert res["objective"] > 0
    assert res["n_open"] >= inst.min_substations_for_capacity() - 1
    assert all(s >= 0 for s in res["assign"])    # every zone assigned
