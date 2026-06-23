"""
P2 (CFLP) metaheuristic encoding
================================
Glue between the generic GA/SA/PSO engines and the substation-siting instance.

A solution is a boolean "open" mask over the 137 candidate substations; the
zone->substation assignment is derived greedily (see CFLPInstance.greedy_assign).
Every operator returns a *feasible* mask via `repair`, which exploits the fact
that every zone can always self-serve (capacity_z >= demand_z for all z, with
diagonal distance 0), so opening an unassigned zone's own substation always
restores feasibility.
"""

import numpy as np
from .base import sigmoid


class CFLPEncoding:
    def __init__(self, inst, alpha):
        self.inst = inst
        self.alpha = alpha
        self.n = inst.n

    # ------------------------------------------------------------------
    def repair(self, mask):
        """Return a feasible mask by opening own-substations of unplaced zones.

        Unplaced zones are opened in a single batch (each can always self-serve,
        since capacity_z >= demand_z and the self-distance is 0), so feasibility
        is restored in at most a couple of passes -- far cheaper than opening one
        substation per greedy re-run.
        """
        mask = np.asarray(mask, dtype=bool).copy()
        for _ in range(3):
            assign, feasible, _ = self.inst.greedy_assign(mask, self.alpha)
            if feasible:
                return mask
            unassigned = np.flatnonzero(assign == -1)
            if unassigned.size == 0:        # no open site at all
                mask[:] = True
                return mask
            mask[unassigned] = True          # batch self-serve -> feasibility
        return mask

    # ------------------------------------------------------------------
    def init(self, rng):
        p = rng.uniform(0.6, 0.95)
        return self.repair(rng.random(self.n) < p)

    def evaluate(self, mask):
        cost, feasible, _ = self.inst.total_cost(mask, self.alpha)
        return cost if feasible else np.inf

    def mutate(self, mask, rng):
        child = mask.copy()
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, self.n, size=k)
        child[idx] = ~child[idx]
        return self.repair(child)

    def crossover(self, a, b, rng):
        pick = rng.random(self.n) < 0.5
        return self.repair(np.where(pick, a, b))

    def neighbor(self, mask, rng):
        child = mask.copy()
        i = int(rng.integers(0, self.n))
        child[i] = ~child[i]
        return self.repair(child)

    # ---- PSO continuous decoding ----
    def decode(self, position):
        return self.repair(sigmoid(position) > 0.5)

    def evaluate_position(self, position):
        return self.evaluate(self.decode(position))
