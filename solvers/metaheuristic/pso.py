"""
Generic Particle Swarm Optimization (continuous)
================================================
Standard global-best PSO over a continuous box. For combinatorial problems the
caller supplies an `eval_fn` that decodes a continuous position vector into a
feasible solution and returns its cost (e.g. sigmoid-thresholding a position to
a binary open/closed substation vector). Minimizes the objective.
"""

import time
import numpy as np

from .base import OptResult, EvalCounter


class ParticleSwarm:
    def __init__(self, dim, eval_fn, *, n_particles=40, n_iterations=200,
                 w=0.72, c1=1.49, c2=1.49, bounds=(-4.0, 4.0), seed=0):
        self.dim = dim
        self.eval_fn = EvalCounter(eval_fn)   # eval_fn(position_vector) -> cost
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.lo, self.hi = bounds
        self.seed = seed

    def run(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        span = self.hi - self.lo
        pos = self.lo + span * rng.random((self.n_particles, self.dim))
        vel = (rng.random((self.n_particles, self.dim)) - 0.5) * span * 0.1

        pbest = pos.copy()
        pbest_cost = np.array([self.eval_fn(p) for p in pos], dtype=float)
        g = int(np.argmin(pbest_cost))
        gbest = pbest[g].copy()
        gbest_cost = float(pbest_cost[g])
        history = [gbest_cost]

        for _ in range(self.n_iterations):
            r1 = rng.random((self.n_particles, self.dim))
            r2 = rng.random((self.n_particles, self.dim))
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, self.lo, self.hi)

            costs = np.array([self.eval_fn(p) for p in pos], dtype=float)
            improved = costs < pbest_cost
            pbest[improved] = pos[improved]
            pbest_cost[improved] = costs[improved]

            g = int(np.argmin(pbest_cost))
            if pbest_cost[g] < gbest_cost:
                gbest_cost = float(pbest_cost[g])
                gbest = pbest[g].copy()
            history.append(gbest_cost)

        return OptResult(
            best_solution=gbest,
            best_cost=gbest_cost,
            history=history,
            runtime=time.perf_counter() - t0,
            n_evals=self.eval_fn.count,
            method="PSO",
            seed=self.seed,
        )
