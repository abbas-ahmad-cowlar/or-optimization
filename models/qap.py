"""
P1 - Department Layout (Quadratic Assignment Problem)
=====================================================
Builds the QAP instance: assign 30 departments to a subset of 197 candidate
sites so that high-flow department pairs sit close together.

Distances are Euclidean from site coordinates (assumption A1). Type
compatibility (assumption A3) is provided as an optional, *illustrative* layer:
a handful of departments with an obvious physical counterpart are restricted to
matching site types, while administrative departments may occupy any
"office-capable" site. The headline results use ``mode="all"`` (every site
eligible -- the standard QAP-with-selection); ``mode="type"`` is reported as a
realism variant.

See docs/formulations.md (P1).
"""

import re
import numpy as np
from dataclasses import dataclass, field

from data.loader import load_data


# Location types that can host a general administrative department.
GENERAL_OFFICE_TYPES = {
    "Residential and Commercial Areas",
    "Governmental Facility",
    "Available Land Sites",
    "Industrial Zone",
    "Ministry of Energy Branch",
    "City Center Hotel",
}

# Departments with an obvious physical counterpart -> allowed site types.
# Illustrative mapping (documented as such); everything else is "general".
SPECIALIZED = {
    "Spare Parts Agent":            {"Spare Parts Storage"},
    "Supply Chain and Logistics":   {"Storage and Logistics Centers", "Port"},
    "Distribution":                 {"Storage and Logistics Centers", "Port"},
    "Inventory Control":            {"Spare Parts Storage", "Storage and Logistics Centers"},
    "Maintenance and Utilities":    {"Turbine and PV Maintenance Center",
                                     "Water Cooling Systems Facility"},
    "Research and Development":     {"Turbine and PV Maintenance Center",
                                     "Large-scale battery storage facilities"},
    "Health & Safety":             {"Centrlized Hospital"},          # data spelling
    "Training and Improve":         {"Training Facilities"},
}


def _clean(name: str) -> str:
    """Strip a leading 'N. ' index and surrounding whitespace."""
    return re.sub(r"^\s*\d+\.\s*", "", str(name)).strip()


@dataclass
class QAPInstance:
    dept_names: list              # 30 cleaned department names
    flow: np.ndarray              # 30x30 asymmetric flow matrix
    distance: np.ndarray          # 197x197 Euclidean distance matrix
    coords: np.ndarray            # 197x2 (x, y)
    loc_types: list               # 197 location type labels
    compat: np.ndarray            # 30x197 boolean compatibility
    n_dept: int = field(init=False)
    n_loc: int = field(init=False)

    def __post_init__(self):
        self.n_dept = len(self.dept_names)
        self.n_loc = self.distance.shape[0]

    def objective(self, assign):
        """Total weighted flow*distance for assignment array (len n_dept)."""
        locs = np.asarray(assign, dtype=int)
        D = self.distance[np.ix_(locs, locs)]
        return float(np.sum(self.flow * D))

    def is_feasible(self, assign):
        locs = np.asarray(assign, dtype=int)
        if len(set(locs.tolist())) != self.n_dept:
            return False                                  # sites must be distinct
        return all(self.compat[i, locs[i]] for i in range(self.n_dept))

    def compatible_sites(self, i):
        return np.flatnonzero(self.compat[i])

    def reduced(self, dept_idx, loc_idx):
        """Return a square, all-compatible sub-instance for the exact baseline.

        Assignments on the reduced instance use *local* location indices
        (0..K-1) into the selected ``loc_idx``.
        """
        dept_idx = np.asarray(dept_idx, dtype=int)
        loc_idx = np.asarray(loc_idx, dtype=int)
        return QAPInstance(
            dept_names=[self.dept_names[i] for i in dept_idx],
            flow=self.flow[np.ix_(dept_idx, dept_idx)].copy(),
            distance=self.distance[np.ix_(loc_idx, loc_idx)].copy(),
            coords=self.coords[loc_idx].copy(),
            loc_types=[self.loc_types[k] for k in loc_idx],
            compat=np.ones((len(dept_idx), len(loc_idx)), dtype=bool),
        )


def _build_compat(dept_names, loc_types, mode):
    n_d, n_l = len(dept_names), len(loc_types)
    compat = np.zeros((n_d, n_l), dtype=bool)
    if mode == "all":
        compat[:] = True
        return compat

    loc_types_arr = np.array([t.strip() for t in loc_types])
    for i, dept in enumerate(dept_names):
        if dept in SPECIALIZED:
            allowed = SPECIALIZED[dept]
        else:
            allowed = GENERAL_OFFICE_TYPES
        compat[i] = np.isin(loc_types_arr, list(allowed))
        if not compat[i].any():
            # safety net: never leave a department with zero options
            compat[i] = np.isin(loc_types_arr, list(GENERAL_OFFICE_TYPES))
    return compat


def build_instance(filepath=None, mode="all"):
    """Construct a QAPInstance.

    Parameters
    ----------
    mode : {"all", "type"}
        "all"  -> every site eligible (standard QAP-with-selection, headline).
        "type" -> illustrative type-compatibility (A3).
    """
    if filepath is None:
        from experiments.config import DATA_PATH
        filepath = DATA_PATH

    data = load_data(filepath)
    dept_names = [_clean(d) for d in data.departments]
    loc_types = [str(t).strip() for t in data.locations["type"].tolist()]
    coords = data.locations[["x", "y"]].to_numpy(float)

    compat = _build_compat(dept_names, loc_types, mode)

    return QAPInstance(
        dept_names=dept_names,
        flow=data.flow_matrix.copy(),
        distance=data.location_distance_matrix.copy(),
        coords=coords,
        loc_types=loc_types,
        compat=compat,
    )
