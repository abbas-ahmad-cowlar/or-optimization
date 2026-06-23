"""
Generic Simulated Annealing
===========================
Problem-agnostic SA driven by user-supplied callbacks. Minimizes the objective.

Callbacks:
    init_fn(rng)          -> individual
    eval_fn(individual)   -> float cost (lower is better)
    neighbor_fn(ind, rng) -> neighbouring individual
"""

import math
import time
import numpy as np

from .base import OptResult, EvalCounter


class SimulatedAnnealing:
    def __init__(self, init_fn, eval_fn, neighbor_fn, *,
                 n_iterations=20000, t_start=None, t_end=1e-3,
                 cooling="exponential", seed=0):
        self.init_fn = init_fn
        self.eval_fn = EvalCounter(eval_fn)
        self.neighbor_fn = neighbor_fn
        self.n_iterations = n_iterations
        self.t_start = t_start            # None -> auto-calibrate from initial scale
        self.t_end = t_end
        self.cooling = cooling
        self.seed = seed

    def _temperature(self, frac):
        """frac in [0,1] -> temperature, from t_start down to t_end."""
        if self.cooling == "linear":
            return self.t_start + (self.t_end - self.t_start) * frac
        # exponential (geometric) cooling
        return self.t_start * (self.t_end / self.t_start) ** frac

    def run(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        cur = self.init_fn(rng)
        cur_cost = self.eval_fn(cur)
        best, best_cost = cur, cur_cost

        # Auto-calibrate starting temperature so ~typical uphill moves are
        # accepted early: T0 ~ |cost| scale.
        if self.t_start is None:
            self.t_start = max(abs(cur_cost) * 0.10, 1.0)

        history = [best_cost]
        for it in range(self.n_iterations):
            frac = it / max(self.n_iterations - 1, 1)
            T = self._temperature(frac)

            cand = self.neighbor_fn(cur, rng)
            cand_cost = self.eval_fn(cand)
            delta = cand_cost - cur_cost

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                cur, cur_cost = cand, cand_cost
                if cur_cost < best_cost:
                    best, best_cost = cur, cur_cost

            # record best-so-far periodically to keep history compact
            if it % max(self.n_iterations // 500, 1) == 0:
                history.append(best_cost)

        history.append(best_cost)
        return OptResult(
            best_solution=best,
            best_cost=best_cost,
            history=history,
            runtime=time.perf_counter() - t0,
            n_evals=self.eval_fn.count,
            method="SA",
            seed=self.seed,
            meta={"t_start": self.t_start, "t_end": self.t_end},
        )
