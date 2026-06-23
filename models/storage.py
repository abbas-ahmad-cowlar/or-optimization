"""
P3 - Storage Layout (slot assignment / Cube-per-Order Index)
============================================================
Assign products to storage slots to minimize expected material-handling travel.

The warehouse is modelled as a rectangular grid of unit slots
(1.5 m x 1.5 m x 1.5 m) with a single I/O point at one corner (assumption A6);
each slot's round-trip travel distance is its Manhattan distance to the I/O.
Only the *sorted* slot-distance profile matters for the assignment, so a
solution is encoded as an ordering of the 20 products: the first product in the
order receives the nearest block of slots, and so on. With prefix sums over the
sorted distances, evaluating an ordering is O(#products).

The optimal ordering is the Cube-per-Order-Index rule (products sorted by
ascending COI = storage/throughput); see docs/formulations.md (P3).
"""

import numpy as np
from dataclasses import dataclass, field

from data.loader import load_data

SLOT_PITCH = 1.5           # metres (slot edge length)
ROUND_TRIP = 2.0           # to the slot and back to I/O


@dataclass
class StorageInstance:
    throughput: np.ndarray         # t_p, shape (P,)
    storage: np.ndarray            # q_p slots, shape (P,)
    product_names: list
    slot_dist: np.ndarray          # sorted ascending, length = sum(q_p)
    _prefix: np.ndarray = field(init=False)   # prefix sums of slot_dist

    def __post_init__(self):
        self._prefix = np.concatenate([[0.0], np.cumsum(self.slot_dist)])

    @property
    def n_products(self):
        return len(self.throughput)

    @property
    def total_slots(self):
        return int(self.storage.sum())

    def coi(self):
        """Cube-per-Order Index per product (storage / throughput)."""
        return self.storage / self.throughput

    def coi_order(self):
        """Optimal product ordering: ascending COI (nearest slots first)."""
        return np.argsort(self.coi())

    def evaluate(self, order):
        """Total expected travel for a product ordering.

        Product p (with q_p slots, throughput t_p) is visited t_p/q_p times per
        slot; assigned a contiguous block of the sorted slots, its travel is
        (t_p/q_p) * (sum of slot distances in its block).
        """
        order = np.asarray(order, dtype=int)
        bounds = np.concatenate([[0], np.cumsum(self.storage[order])]).astype(int)
        total = 0.0
        for rank, p in enumerate(order):
            block_sum = self._prefix[bounds[rank + 1]] - self._prefix[bounds[rank]]
            total += (self.throughput[p] / self.storage[p]) * block_sum
        return float(total)

    def assignment(self, order):
        """Return (slot_start, slot_end) block per product for the ordering."""
        order = np.asarray(order, dtype=int)
        bounds = np.concatenate([[0], np.cumsum(self.storage[order])]).astype(int)
        return {int(p): (int(bounds[r]), int(bounds[r + 1]))
                for r, p in enumerate(order)}

    def downscaled(self, target_slots=240):
        """Return a smaller StorageInstance for the exact transportation LP.

        Storage requirements are divided by a common factor so the LP has
        ~target_slots slots; the COI structure is preserved.
        """
        factor = max(1, int(round(self.total_slots / target_slots)))
        small_q = np.maximum(1, np.round(self.storage / factor)).astype(int)
        N = int(small_q.sum())
        slot_dist = _rack_distances(N)
        return StorageInstance(
            throughput=self.throughput.copy(),
            storage=small_q,
            product_names=list(self.product_names),
            slot_dist=slot_dist,
        )


def _rack_distances(n_slots):
    """Sorted round-trip Manhattan distances for n_slots in a grid warehouse."""
    side = int(np.ceil(np.sqrt(n_slots))) + 1
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    dist = (rows + cols).ravel() * SLOT_PITCH * ROUND_TRIP
    dist.sort()
    return dist[:n_slots].astype(float)


def build_instance(filepath=None):
    if filepath is None:
        from experiments.config import DATA_PATH
        filepath = DATA_PATH
    data = load_data(filepath)
    p = data.products
    throughput = p["throughput"].to_numpy(float)
    storage = p["storage"].to_numpy(int)
    slot_dist = _rack_distances(int(storage.sum()))
    return StorageInstance(
        throughput=throughput,
        storage=storage,
        product_names=list(p["product_id"]),
        slot_dist=slot_dist,
    )
