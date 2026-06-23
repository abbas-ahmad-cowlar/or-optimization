"""
Shared metaheuristic infrastructure
===================================
Common result container, RNG seeding, and small numerical helpers used by the
genetic algorithm, simulated annealing, and particle-swarm engines. The engines
themselves are problem-agnostic: each problem supplies encoding-specific
callbacks (init / evaluate / mutate / crossover / neighbour / decode).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List
import numpy as np


@dataclass
class OptResult:
    """Outcome of a single metaheuristic run."""
    best_solution: Any
    best_cost: float
    history: List[float]          # best-so-far objective per iteration
    runtime: float                # wall-clock seconds
    n_evals: int                  # objective evaluations
    method: str                   # "GA" | "SA" | "PSO" | ...
    seed: int
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.method} (seed={self.seed}): best={self.best_cost:,.2f} "
                f"in {self.runtime:.2f}s, {self.n_evals:,} evals, "
                f"{len(self.history)} iters")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class EvalCounter:
    """Wraps an objective so we can count evaluations transparently."""
    def __init__(self, fn: Callable):
        self._fn = fn
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._fn(*args, **kwargs)
