"""
P2 - Capacitated Facility Location (substation siting)
======================================================
Builds the CFLP instance from the parsed Excel data and provides feasibility
checking / cost evaluation used by both the exact and metaheuristic solvers.

See docs/formulations.md (P2) for the mathematical model and assumptions
A4, A5, A8 (common-unit demand/capacity, self-service diagonal = 0,
transmission rate alpha).
"""

import numpy as np
from dataclasses import dataclass, field

from data.loader import load_data


@dataclass
class CFLPInstance:
    """A capacitated facility-location instance.

    Candidate substation sites coincide with the demand zones (A4).
    """
    demand: np.ndarray            # q_z,  shape (n,)
    cost: np.ndarray              # c_s,  shape (n,)   construction cost
    capacity: np.ndarray          # kappa_s, shape (n,)
    distance: np.ndarray          # d[z, s], shape (n, n); inf = forbidden link
    zone_names: list
    alpha0: float = 1.0           # data-calibrated transmission rate
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.demand)

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())

    @property
    def total_capacity(self) -> float:
        return float(self.capacity.sum())

    def allowed_pairs(self):
        """Yield (z, s) index pairs whose link is permitted (finite distance)."""
        finite = np.isfinite(self.distance)
        zs = np.argwhere(finite)
        return [(int(z), int(s)) for z, s in zs]

    # ------------------------------------------------------------------
    # Cost evaluation (used by metaheuristics and to score any solution)
    # ------------------------------------------------------------------
    def assignment_cost(self, assign, alpha):
        """Transmission cost of an assignment array (assign[z] = substation s)."""
        d = self.distance[np.arange(self.n), assign]
        return float(alpha * np.sum(d * self.demand))

    def greedy_assign(self, open_mask, alpha):
        """Assign every zone to a feasible OPEN substation, respecting capacity.
        Returns (assign, feasible, transmission_cost).

        Every open substation first serves its own zone (self-distance 0 is the
        cheapest possible, and capacity_z >= demand_z always holds), so only the
        closed zones need the greedy remote-assignment loop -- processed in
        descending demand order. A solution is infeasible if any closed zone
        cannot be placed in a finite-link open substation with residual capacity.
        """
        open_mask = np.asarray(open_mask, dtype=bool)
        open_sites = np.flatnonzero(open_mask)
        if open_sites.size == 0:
            return np.full(self.n, -1, dtype=int), False, np.inf

        assign = np.full(self.n, -1, dtype=int)
        remaining = self.capacity.copy()

        # open zones self-serve (vectorized)
        assign[open_mask] = open_sites
        remaining[open_mask] -= self.demand[open_mask]

        # closed zones routed to the nearest feasible open substation
        closed = np.flatnonzero(~open_mask)
        closed = closed[np.argsort(-self.demand[closed])]   # hardest first
        for z in closed:
            d_row = self.distance[z, open_sites]
            feasible = np.isfinite(d_row) & (remaining[open_sites] >= self.demand[z])
            if not feasible.any():
                return assign, False, np.inf
            cand = open_sites[feasible]
            best = cand[np.argmin(self.distance[z, cand])]   # alpha,q monotone
            assign[z] = best
            remaining[best] -= self.demand[z]

        trans = self.assignment_cost(assign, alpha)
        return assign, True, trans

    def total_cost(self, open_mask, alpha):
        """Full objective for an open set: construction + transmission.

        Returns (cost, feasible, assign).
        """
        assign, feasible, trans = self.greedy_assign(open_mask, alpha)
        if not feasible:
            return np.inf, False, assign
        construction = float(self.cost[open_mask].sum())
        return construction + trans, True, assign

    def min_substations_for_capacity(self) -> int:
        """Lower bound on #open substations from the total-capacity constraint."""
        caps = np.sort(self.capacity)[::-1]
        cum = np.cumsum(caps)
        return int(np.searchsorted(cum, self.total_demand) + 1)


def build_instance(filepath=None, self_service=True):
    """Construct a CFLPInstance from the Excel data.

    Parameters
    ----------
    filepath : str or None
        Path to Group 1.xlsx (defaults to the project copy).
    self_service : bool
        If True (default, assumption A8) set the distance diagonal to 0 so a
        substation serves its own zone for free.
    """
    if filepath is None:
        from experiments.config import DATA_PATH
        filepath = DATA_PATH

    data = load_data(filepath)
    z = data.zones

    distance = data.zone_distance_matrix.copy()
    if self_service:
        np.fill_diagonal(distance, 0.0)   # A8: substation serves its own zone free

    demand = z["demand"].to_numpy(float)
    cost = z["cost"].to_numpy(float)
    capacity = z["capacity"].to_numpy(float)

    # ---- calibrate alpha_0 (A8) ----------------------------------------
    # alpha_0 makes mean construction cost ~ mean nearest-neighbour transmission,
    # so neither term dominates and the location trade-off is non-degenerate.
    n = len(demand)
    off = distance.copy()
    np.fill_diagonal(off, np.inf)                       # ignore self for "nearest neighbour"
    nearest = np.min(off, axis=1)                       # nearest finite OTHER substation
    nearest_trans = np.mean(nearest * demand)           # mean d_nn * q
    alpha0 = float(np.mean(cost) / nearest_trans) if nearest_trans > 0 else 1.0

    return CFLPInstance(
        demand=demand,
        cost=cost,
        capacity=capacity,
        distance=distance,
        zone_names=list(data.zone_names),
        alpha0=alpha0,
    )
