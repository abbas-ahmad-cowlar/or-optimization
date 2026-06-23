"""
Generic Genetic Algorithm
=========================
Problem-agnostic GA driven by user-supplied callbacks. Minimizes the objective.

Callbacks (all receive a numpy Generator for randomness where needed):
    init_fn(rng)            -> individual
    eval_fn(individual)     -> float cost (lower is better)
    crossover_fn(a, b, rng) -> child individual
    mutate_fn(ind, rng)     -> (possibly new) individual
"""

import time
import numpy as np

from .base import OptResult, EvalCounter


class GeneticAlgorithm:
    def __init__(self, init_fn, eval_fn, crossover_fn, mutate_fn, *,
                 pop_size=80, n_generations=200, tournament_k=3, elitism=2,
                 crossover_rate=0.9, mutation_rate=0.15, seed=0):
        self.init_fn = init_fn
        self.eval_fn = EvalCounter(eval_fn)
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed

    def _tournament(self, fitness, rng):
        idx = rng.integers(0, len(fitness), size=self.tournament_k)
        return idx[np.argmin(fitness[idx])]

    def run(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        pop = [self.init_fn(rng) for _ in range(self.pop_size)]
        fitness = np.array([self.eval_fn(ind) for ind in pop], dtype=float)

        best_i = int(np.argmin(fitness))
        best_sol, best_cost = pop[best_i], float(fitness[best_i])
        history = [best_cost]

        for _ in range(self.n_generations):
            # elitism: carry the best `elitism` individuals forward
            order = np.argsort(fitness)
            new_pop = [pop[i] for i in order[:self.elitism]]

            while len(new_pop) < self.pop_size:
                pa = pop[self._tournament(fitness, rng)]
                pb = pop[self._tournament(fitness, rng)]
                child = (self.crossover_fn(pa, pb, rng)
                         if rng.random() < self.crossover_rate else pa)
                if rng.random() < self.mutation_rate:
                    child = self.mutate_fn(child, rng)
                new_pop.append(child)

            pop = new_pop
            fitness = np.array([self.eval_fn(ind) for ind in pop], dtype=float)

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_cost:
                best_cost = float(fitness[gen_best])
                best_sol = pop[gen_best]
            history.append(best_cost)

        return OptResult(
            best_solution=best_sol,
            best_cost=best_cost,
            history=history,
            runtime=time.perf_counter() - t0,
            n_evals=self.eval_fn.count,
            method="GA",
            seed=self.seed,
        )
