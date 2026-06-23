"""Optimality and sanity tests for P3 (storage assignment)."""

import numpy as np
import pytest

from models.storage import build_instance
from solvers.exact.storage_lp import solve_storage_lp


@pytest.fixture(scope="module")
def inst():
    return build_instance()


def test_instance(inst):
    assert inst.n_products == 20
    assert inst.total_slots == int(inst.storage.sum())
    assert len(inst.slot_dist) == inst.total_slots
    # slot distances sorted ascending
    assert np.all(np.diff(inst.slot_dist) >= 0)


def test_coi_beats_random(inst):
    coi = inst.evaluate(inst.coi_order())
    rng = np.random.default_rng(0)
    randvals = [inst.evaluate(rng.permutation(inst.n_products)) for _ in range(500)]
    assert coi <= min(randvals) + 1e-6           # COI is optimal -> no random beats it


def test_coi_is_optimal_bruteforce_small():
    """On a 6-product sub-instance, COI matches brute-force optimum."""
    from itertools import permutations
    inst = build_instance()
    # take first 6 products into a tiny instance
    from models.storage import StorageInstance, _rack_distances
    q = inst.storage[:6].copy()
    t = inst.throughput[:6].copy()
    small = StorageInstance(throughput=t, storage=q,
                            product_names=inst.product_names[:6],
                            slot_dist=_rack_distances(int(q.sum())))
    coi = small.evaluate(small.coi_order())
    best = min(small.evaluate(np.array(p)) for p in permutations(range(6)))
    assert abs(coi - best) < 1e-6


def test_assignment_covers_all_slots(inst):
    blocks = inst.assignment(inst.coi_order())
    covered = sum(e - s for s, e in blocks.values())
    assert covered == inst.total_slots
    # blocks are contiguous and non-overlapping
    spans = sorted(blocks.values())
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 == s2


def test_exact_lp_matches_coi_downscaled(inst):
    small = inst.downscaled(target_slots=120)
    lp = solve_storage_lp(small, msg=False)
    coi_small = small.evaluate(small.coi_order())
    assert lp["objective"] is not None
    assert abs(lp["objective"] - coi_small) / coi_small < 1e-6
