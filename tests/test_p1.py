"""Feasibility and sanity tests for P1 (QAP)."""

import numpy as np
import pytest

from models.qap import build_instance
from solvers.metaheuristic.qap_ops import QAPEncoding
from solvers.exact.qap_milp import solve_qap


@pytest.fixture(scope="module")
def inst():
    return build_instance(mode="all")


def test_instance_shapes(inst):
    assert inst.n_dept == 30
    assert inst.n_loc == 197
    assert inst.flow.shape == (30, 30)
    assert inst.distance.shape == (197, 197)
    # distance symmetric, zero diagonal
    assert np.allclose(inst.distance, inst.distance.T)
    assert np.allclose(np.diag(inst.distance), 0.0)


def test_flow_asymmetric(inst):
    assert not np.allclose(inst.flow, inst.flow.T)


def test_random_solution_feasible(inst):
    enc = QAPEncoding(inst)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = enc.random_solution(rng)
        assert len(set(a.tolist())) == inst.n_dept       # distinct sites
        assert inst.is_feasible(a)


def test_operators_preserve_feasibility(inst):
    enc = QAPEncoding(inst)
    rng = np.random.default_rng(1)
    a = enc.random_solution(rng)
    b = enc.random_solution(rng)
    for _ in range(100):
        c = enc.crossover(a, b, rng)
        assert len(set(c.tolist())) == inst.n_dept
        m = enc.mutate(a, rng)
        assert len(set(m.tolist())) == inst.n_dept
        a = m


def test_type_mode_feasible():
    inst = build_instance(mode="type")
    enc = QAPEncoding(inst)
    rng = np.random.default_rng(2)
    a = enc.random_solution(rng)
    assert len(set(a.tolist())) == inst.n_dept
    assert inst.is_feasible(a)
    # specialized department restricted to few sites
    assert inst.compat.sum(axis=1).min() >= 1


def test_objective_matches_bruteforce_small():
    """On a tiny instance the vectorized objective matches an explicit loop."""
    inst = build_instance(mode="all")
    red = inst.reduced([0, 1, 2, 3], [10, 20, 30, 40])
    assign = np.array([2, 0, 3, 1])
    explicit = sum(red.flow[i, j] * red.distance[assign[i], assign[j]]
                   for i in range(4) for j in range(4))
    assert abs(red.objective(assign) - explicit) < 1e-9


def test_brute_force_qap_optimal():
    """Brute force gives the true QAP optimum; no permutation and no MILP beats it."""
    import itertools
    from solvers.exact.qap_milp import brute_force_qap
    inst = build_instance(mode="all")
    red = inst.reduced([0, 1, 2, 3, 4], [5, 15, 25, 35, 45])
    bf = brute_force_qap(red.flow, red.distance)
    assert len(set(bf["assign"])) == 5
    best = min(red.objective(np.array(p)) for p in itertools.permutations(range(5)))
    assert abs(bf["objective"] - best) < 1e-9
    # the MILP can never be *better* than the true optimum (it may be worse / unproven)
    milp = solve_qap(red.flow, red.distance, time_limit=20, msg=False)
    assert milp["objective"] >= bf["objective"] - 1e-6
