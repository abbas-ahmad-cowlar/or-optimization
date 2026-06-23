"""
Generic permutation encoding for metaheuristics
===============================================
Reusable init / evaluate / mutate / crossover (order crossover, OX1) /
neighbour callbacks for any problem whose solution is a permutation of n items
scored by a user-supplied ``eval_fn``.
"""

import numpy as np


class PermutationEncoding:
    def __init__(self, n, eval_fn):
        self.n = n
        self.eval_fn = eval_fn

    def init(self, rng):
        return rng.permutation(self.n)

    def evaluate(self, order):
        return self.eval_fn(order)

    def mutate(self, order, rng):
        child = np.array(order, dtype=int).copy()
        i, j = rng.choice(self.n, size=2, replace=False)
        child[i], child[j] = child[j], child[i]
        return child

    def neighbor(self, order, rng):
        return self.mutate(order, rng)

    def crossover(self, a, b, rng):
        a = np.asarray(a, dtype=int)
        b = np.asarray(b, dtype=int)
        n = self.n
        i, j = sorted(rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[i:j + 1] = a[i:j + 1]
        chosen = set(a[i:j + 1].tolist())
        fill = [x for x in b.tolist() if x not in chosen]
        holes = [k for k in range(n) if child[k] == -1]
        for k, x in zip(holes, fill):
            child[k] = x
        return child
