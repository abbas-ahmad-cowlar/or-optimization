# Facility & Infrastructure Optimization for a Renewable-Energy Company

Three classic **Operations Research** problems drawn from one real-world dataset, each
solved with **exact mixed-integer programming** and **metaheuristics**, then compared
head-to-head on solution quality, runtime, and optimality gap.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Solver](https://img.shields.io/badge/solver-PuLP%2FCBC-orange)

> The dataset (`Group 1.xlsx`) describes a renewable-energy company — turbines, PV
> maintenance, grid-connection points, substations, and battery storage. The three
> problems below are independent but share the same data pipeline.

---

## The three problems

| # | Problem | OR formulation | Size | Methods |
|---|---------|----------------|------|---------|
| **P1** | Department layout | Quadratic Assignment Problem (QAP) with location selection + type compatibility | 30 departments → 197 sites | exact MILP (reduced) · GA · SA |
| **P2** | Substation siting | Capacitated Facility Location (CFLP) | 137 zones | exact MILP (CBC) · GA · SA · PSO |
| **P3** | Storage layout | Slot assignment / Cube-per-Order Index | 20 products, 25,189 slots | exact transportation LP · COI rule · GA · SA |

Full mathematical models are in **[docs/formulations.md](docs/formulations.md)**.

---

## Results at a glance

| Problem | Exact / optimal | Best metaheuristic | Gap | Notes |
|---------|-----------------|--------------------|-----|-------|
| **P2** substations (137 zones) | **877,868** (CBC, 109 built) | GA / SA **888,504** | **+1.21%** | PSO trails at +16.67% |
| **P1** layout — reduced 8×8 | **2,874,111** (enumeration) | GA / SA **2,874,111** | **0.00%** | generic MILP stalls at +5.52% |
| **P1** layout — full 30→197 | — (intractable) | SA **4,928,976** · GA 5,038,900 | — | SA beats GA |
| **P3** storage (25,189 slots) | **40,480,413** (COI = LP) | GA / SA **40,480,413** | **0.00%** | 27.3% better than random |

_Regenerate with `python -m experiments.compare` → `output/results/comparison.csv`._

Key findings:

- **P2 (substations):** exact CBC solves the 137-zone CFLP to proven optimality; **GA and
  SA come within ~1.2%** of optimal, while **PSO trails at ~17%** — a clean illustration of
  which metaheuristics suit a binary facility-location problem. An **α-sweep** traces the
  classic construction-vs-transmission trade-off (number of substations grows from ~70 to
  ~134 as the transmission rate rises).
- **P1 (layout):** on a reduced instance with a provably-optimal MILP baseline, GA and SA
  both reach the optimum; on the full 30→197 QAP, **simulated annealing outperforms the GA**,
  consistent with the QAP literature.
- **P3 (storage):** the **Cube-per-Order-Index rule is provably optimal** (verified three
  ways — exact LP, GA, and SA all reach the same value) and saves **~27% travel** versus a
  random layout.

Figures are regenerated into `output/figures/` by `python -m visualization.plot_solutions`.

---

## Repository layout

```
data/            Excel parser + validation + CSV export   (loader.py)
models/          Problem definitions (qap, facility_location, storage)
solvers/
  exact/         PuLP/CBC MILP & LP solvers
  metaheuristic/ Generic GA / SA / PSO engines + per-problem encodings
experiments/     Reproducible runners (run_p1/2/3, compare) + config.py
visualization/   Input-data and solution figures
docs/            formulations.md (the math)
report/          LaTeX source -> report.pdf
tests/           pytest feasibility/optimality checks
output/          figures/ and results/ (logged JSON/CSV)
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. parse & validate the data, regenerate input figures
python -m data.loader "Group 1.xlsx"
python -m visualization.plot_input_data

# 2. solve each problem (exact + metaheuristics, logged to output/results/)
python -m experiments.run_p2      # substation siting
python -m experiments.run_p1      # department layout
python -m experiments.run_p3      # storage layout

# 3. build the comparison table and solution figures
python -m experiments.compare
python -m visualization.plot_solutions

# tests
pytest
```

Add `--quick` to any `run_p*` for a fast smoke test. All runs are seeded
(`experiments/config.py`) so results and figures reproduce exactly.

---

## Methods

- **Exact:** [PuLP](https://github.com/coin-or/pulp) modeling with the bundled **CBC**
  solver. CFLP and the storage LP solve directly; the QAP is linearized (McCormick) and
  solved exactly only on a reduced instance, since full-scale QAP is intractable.
- **Metaheuristics:** lightweight, dependency-free **Genetic Algorithm**, **Simulated
  Annealing**, and **Particle Swarm Optimization** engines (`solvers/metaheuristic/`),
  each driven by problem-specific encodings with feasibility-preserving operators.
- **Reproducibility:** every experiment is seeded and logs results to `output/results/*.json`,
  so the comparison table, figures, and report all regenerate from data.

---

## Notes on the data

Several modeling decisions were required where the original brief is silent or
inconsistent (the client is no longer reachable); all are documented as assumptions
**A1–A8** in [docs/formulations.md](docs/formulations.md#assumptions-to-validate). The most
important: the P2 demand/capacity unit labels are internally inconsistent (taken literally
the instance is infeasible), so raw values are treated in a common unit.

## License

MIT — see [LICENSE](LICENSE).
