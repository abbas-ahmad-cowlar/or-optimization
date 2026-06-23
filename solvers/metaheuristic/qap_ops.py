"""
P1 (QAP) metaheuristic encoding
===============================
Glue between the generic GA/SA engines and the department-layout instance.

A solution is an array ``assign`` of length n_dept giving each department's
location index; entries must be distinct and type-compatible. Operators keep
solutions feasible (distinct + compatible) by construction / repair.
"""

import numpy as np


class QAPEncoding:
    def __init__(self, inst, location_subset=None):
        self.inst = inst
        self.n_dept = inst.n_dept
        if location_subset is None:
            self.locations = np.arange(inst.n_loc)
        else:
            self.locations = np.asarray(location_subset, dtype=int)
        self._locset = set(self.locations.tolist())
        # compatible candidate sites per department (restricted to the subset)
        self.compat_sites = []
        for i in range(self.n_dept):
            sites = [k for k in inst.compatible_sites(i) if k in self._locset]
            self.compat_sites.append(np.array(sites, dtype=int))

    # ------------------------------------------------------------------
    def random_solution(self, rng):
        order = rng.permutation(self.n_dept)
        used = set()
        assign = np.full(self.n_dept, -1, dtype=int)
        for i in order:
            choices = [k for k in self.compat_sites[i] if k not in used]
            if not choices:                                  # fallback: any free site
                choices = [k for k in self.locations.tolist() if k not in used]
            k = int(choices[rng.integers(len(choices))])
            assign[i] = k
            used.add(k)
        return assign

    # alias so the engines' init callback signature matches
    def init(self, rng):
        return self.random_solution(rng)

    def evaluate(self, assign):
        return self.inst.objective(assign)

    # ------------------------------------------------------------------
    def mutate(self, assign, rng):
        child = assign.copy()
        if rng.random() < 0.5:
            i, j = rng.choice(self.n_dept, size=2, replace=False)
            # swap only if both remain type-compatible
            if self.inst.compat[i, child[j]] and self.inst.compat[j, child[i]]:
                child[i], child[j] = child[j], child[i]
        else:
            i = int(rng.integers(self.n_dept))
            used = set(child.tolist())
            choices = [k for k in self.compat_sites[i] if k not in used]
            if choices:
                child[i] = int(choices[rng.integers(len(choices))])
        return child

    def neighbor(self, assign, rng):
        return self.mutate(assign, rng)

    def crossover(self, a, b, rng):
        pick = rng.random(self.n_dept) < 0.5
        child = np.where(pick, a, b)
        # repair duplicate locations (entries from parents are already compatible)
        used = set()
        dups = []
        for i, k in enumerate(child):
            k = int(k)
            if k in used:
                dups.append(i)
            else:
                used.add(k)
        for i in dups:
            choices = [k for k in self.compat_sites[i] if k not in used]
            if not choices:
                choices = [k for k in self.locations.tolist() if k not in used]
            k = int(choices[rng.integers(len(choices))])
            child[i] = k
            used.add(k)
        return child
