# Project Plan — Facility & Infrastructure Optimization (Renewable-Energy Company)

> **Status:** Draft for review · **Type:** Portfolio piece (original client no longer in contact)
> **Author:** Syed Abbas Ahmad
> **Last updated:** 2026-06-23

---

## 1. Overview & Context

This project solves three classic **Operations Research** problems drawn from a single
real-world dataset (`Group 1.xlsx`) for what appears to be a **renewable-energy company**
(turbines, PV maintenance, grid connection points, battery storage, substations). The data
layer (parsing, validation, export, exploratory figures) is **already complete**. This plan
covers building the optimization models, solving them with both **exact** and **metaheuristic**
methods, comparing the approaches, and packaging everything as a polished, reviewable portfolio
repository with a formal report.

### The three problems

| # | Name | OR formulation | Key data |
|---|------|----------------|----------|
| **P1** | Department layout | Quadratic Assignment Problem (QAP) with location selection + type compatibility | 30 departments, 30×30 **asymmetric** flow matrix, 197 candidate sites with (x, y), fixed-travel-distance, and type |
| **P2** | Substation siting | Capacitated Facility Location Problem (CFLP) | 137 zones: demand (×1000 KWh), build cost (×\$10), capacity (×100 KWh); 137×137 distance matrix where `inf` = infeasible link |
| **P3** | Storage layout | Storage-slot assignment (Cube-per-Order Index) | 20 products: throughput (ops/period), storage (slots); slot = 1.5 m × 1.5 m × 1.5 m |

### Locked decisions (from review on 2026-06-23)

- **Scope:** All three problems (P1 + P2 + P3).
- **Methods:** Exact MILP **and** metaheuristics, with a head-to-head comparison.
- **Deliverables:** GitHub repo + README + LaTeX/PDF report. **Interactive dashboard is deferred**
  to a later stage (Phase 8), after the core repo and report are solid.

---

## 2. Portfolio Success Criteria

The finished project should demonstrate, to a reviewer skimming for 5 minutes:

1. **Breadth** — three different OR problem classes solved competently.
2. **Rigor** — clean mathematical formulations and provably-optimal baselines where tractable.
3. **Engineering** — modular, tested, reproducible code (seeded runs, logged results).
4. **Judgment** — exact vs metaheuristic trade-off analysis (quality, runtime, optimality gap).
5. **Communication** — a strong README, a formulations doc, result figures, and a formal report.

**Definition of done:** a public GitHub repo that a stranger can clone, `pip install -r requirements.txt`,
run end-to-end with a single command per problem, and reproduce every figure and number in the report.

---

## 3. Final Repository Structure

```
or-optimization/
├── data/                       # ✅ DONE
│   ├── loader.py               #    Excel parser + validation + CSV export
│   └── __init__.py
├── models/                     # Pure problem definitions (math → data structures)
│   ├── qap.py                  #    P1: objective + constraints, no solver
│   ├── facility_location.py    #    P2
│   ├── storage.py              #    P3
│   └── __init__.py
├── solvers/
│   ├── exact/
│   │   ├── qap_milp.py         #    P1 linearized MILP (PuLP/CBC)
│   │   ├── cflp_milp.py        #    P2 MILP (PuLP/CBC)
│   │   ├── storage_lp.py       #    P3 assignment LP baseline
│   │   └── __init__.py
│   ├── metaheuristic/
│   │   ├── base.py             #    Shared engine: population, logging, seeding, callbacks
│   │   ├── genetic.py          #    GA (selection/crossover/mutation hooks)
│   │   ├── simulated_annealing.py
│   │   ├── pso.py
│   │   └── __init__.py
│   └── __init__.py
├── experiments/
│   ├── run_p1.py               #    Solve P1 (exact + heuristic), log results
│   ├── run_p2.py               #    Solve P2
│   ├── run_p3.py               #    Solve P3
│   ├── compare.py              #    Aggregate comparison tables/plots
│   └── config.py               #    Seeds, time limits, hyperparameters
├── visualization/              # ✅ plot_input_data.py done
│   ├── plot_input_data.py
│   └── plot_solutions.py       #    NEW: layout maps, network graph, convergence
├── output/                     # ✅ figures/ + clean CSVs done
│   ├── figures/
│   └── results/                #    NEW: solution JSON/CSV per run (reproducible)
├── docs/
│   └── formulations.md         #    NEW: the math for all three problems
├── report/                     #    NEW: LaTeX source → PDF
│   ├── report.tex
│   ├── references.bib
│   └── figures/                #    Symlinked/copied from output/figures
├── tests/                      #    NEW: pytest sanity + correctness checks
├── Group 1.xlsx                # ✅ source data
├── requirements.txt            # ✅ (will add pytest)
├── PLAN.md                     #    this file
├── README.md                   #    NEW
└── LICENSE                     #    NEW (MIT)
```

---

## 4. Phase-by-Phase Plan

Each phase lists: **objective**, **step-by-step tasks**, **deliverables**, and **acceptance criteria**.
Phases are ordered so each unblocks the next. Recommended commit boundary = end of each phase.

---

### Phase 0 — Foundation & Repo Hygiene

**Objective:** Put the project under version control with a clean structure so all later work is tracked.

**Steps:**
1. `git init`; create `main` branch.
2. Add `.gitignore` (Python: `__pycache__/`, `*.pyc`, `.venv/`, `output/results/*.tmp`, LaTeX aux files).
3. Add `LICENSE` (MIT).
4. **Cleanup:**
   - Delete the duplicate `Group 1rrrrrrrrrrrrr.xlsx` (byte-identical to `Group 1.xlsx`).
   - Delete scratch exploration scripts `read_data.py`, `read_data2.py`, `read_data3.py` (superseded by `data/loader.py`).
5. Create empty package dirs with `__init__.py`: `models/`, `solvers/`, `solvers/exact/`,
   `solvers/metaheuristic/`, `experiments/`, `docs/`, `report/`, `tests/`, `output/results/`.
6. Add `pytest` to `requirements.txt`.
7. Commit `PLAN.md` and the scaffold as the initial commit.

**Deliverables:** initialized git repo, clean tree, `LICENSE`, `.gitignore`, scaffolded packages.

**Acceptance criteria:** `git log` shows an initial commit; `python -c "import data.loader"` still works;
no duplicate/scratch files remain.

---

### Phase 1 — Mathematical Formulations (`docs/formulations.md`)

**Objective:** Write the precise math for all three problems. This is the reference every solver implements.

**Steps (one subsection per problem):**

1. **P1 — Quadratic Assignment Problem**
   - **Sets:** departments `D` (|D| = 30), candidate locations `L` (|L| = 197), types `T`.
   - **Parameters:** flow `f[i,j]` (asymmetric), distance `d[k,l]` (Euclidean from coords; fixed-travel-distance
     as fallback), type-compatibility `compat[i,k] ∈ {0,1}`.
   - **Decision vars:** `x[i,k] = 1` if department `i` is placed at location `k`.
   - **Objective:** minimize `Σ_i Σ_j Σ_k Σ_l f[i,j] · d[k,l] · x[i,k] · x[j,l]`.
   - **Constraints:** each department to exactly one location; each location ≤ one department;
     `x[i,k] ≤ compat[i,k]`.
   - Note the linearization strategy for the exact MILP (introduce `y[i,j,k,l]` with standard QAP linearization,
     or Frieze–Yadegar). Flag that full 30×197 exact is intractable → subset for exact, metaheuristic for full.

2. **P2 — Capacitated Facility Location**
   - **Sets:** zones `Z` (|Z| = 137), candidate substation sites = `Z` (a substation may be built in any zone).
   - **Parameters:** demand `dem[z]`, build cost `c[s]`, capacity `cap[s]`, distance/transmission cost `d[z,s]`
     (`inf` ⇒ link forbidden).
   - **Decision vars:** `y[s] = 1` if substation built at `s`; `x[z,s] = 1` if zone `z` served by `s`.
   - **Objective:** minimize `Σ_s c[s]·y[s] + Σ_z Σ_s d[z,s]·dem[z]·x[z,s]`.
   - **Constraints:** each zone served by exactly one open substation; `x[z,s] ≤ y[s]`;
     `Σ_z dem[z]·x[z,s] ≤ cap[s]·y[s]`; `x[z,s] = 0` where `d[z,s] = inf`.

3. **P3 — Storage Slot Assignment**
   - **Sets:** products `P` (|P| = 20), storage slots/zones `S`.
   - **Parameters:** throughput `t[p]`, storage demand `q[p]` (slots), slot geometry, distance from I/O point.
   - **Decision vars:** `x[p,s]` slot assignment.
   - **Objective:** minimize expected travel ≈ `Σ_p t[p] · (assigned distance)`; introduce the
     **Cube-per-Order Index (COI)** `= storage / throughput` and the COI-ordering optimality argument.
   - **Constraints:** capacity per slot, all product storage demand met.
   - Document assumptions (single I/O point vs dock list) explicitly.

**Deliverables:** `docs/formulations.md` with all three fully specified, plus an **"Assumptions to validate"**
list (see §5).

**Acceptance criteria:** each problem has sets, parameters, decision variables, objective, and constraints
written unambiguously; notation is consistent across problems.

---

### Phase 2 — P2 Substation Siting (exact first) ⭐ start here

**Objective:** Deliver a provably-optimal solution to the cleanest problem early, establishing the
solver/experiment pattern the other phases reuse.

**Steps:**
1. `models/facility_location.py` — build the CFLP instance from `data.loader` (demand, cost, capacity,
   distance matrix with `inf` handling).
2. `solvers/exact/cflp_milp.py` — implement the MILP in PuLP, solve with CBC; extract open substations +
   assignment + objective breakdown (build vs transmission).
3. `solvers/metaheuristic/genetic.py` + `base.py` — GA encoding (which substations open; greedy/feasible
   zone assignment given open set), with feasibility repair for capacity.
4. `experiments/run_p2.py` — run exact and GA with a fixed seed and time limit; write results to
   `output/results/p2_*.json`.
5. Sanity tests in `tests/test_p2.py` (objective ≥ 0, all zones served, capacities respected, no `inf` links used).

**Deliverables:** exact + GA solvers for P2, logged results, basic tests.

**Acceptance criteria:** CBC returns an optimal (or bounded-gap) solution; GA gets within a reported gap of it;
all feasibility tests pass.

---

### Phase 3 — P1 Department Layout (QAP)

**Objective:** Solve the hardest combinatorial problem with an exact baseline on a reduced instance and a
metaheuristic on the full instance.

**Steps:**
1. `models/qap.py` — flow matrix, distance matrix (Euclidean from coords), type-compatibility matrix from
   location types ↔ department semantics (document the mapping rules).
2. `solvers/exact/qap_milp.py` — linearized QAP MILP; run on a **tractable subset** (e.g., restrict candidate
   locations per department via compatibility, or a smaller k) to obtain an optimal reference.
3. `solvers/metaheuristic/simulated_annealing.py` — permutation-based SA/GA over the full 30→197 assignment
   with type constraints; neighborhood = swap/relocate moves.
4. `experiments/run_p1.py` — exact (subset) + heuristic (full), seeded, logged to `output/results/p1_*.json`.
5. `tests/test_p1.py` — assignment validity (bijection on used locations, compatibility respected).

**Deliverables:** exact (subset) + metaheuristic (full) solvers, logged results, tests.

**Acceptance criteria:** heuristic produces a valid, compatibility-respecting layout; exact subset confirms
heuristic quality on the reduced instance.

---

### Phase 4 — P3 Storage Assignment

**Objective:** Solve the storage layout with an LP baseline and the COI heuristic; show they agree.

**Steps:**
1. `models/storage.py` — products, throughput, storage demand, slot/distance model (state the I/O assumption).
2. `solvers/exact/storage_lp.py` — assignment LP/MILP baseline.
3. COI heuristic (rank by storage/throughput, assign nearest slots first) in `solvers/metaheuristic/` or a
   dedicated `heuristics/` module.
4. `experiments/run_p3.py` — both methods, logged to `output/results/p3_*.json`.
5. `tests/test_p3.py` — all storage demand met, capacity respected.

**Deliverables:** LP baseline + COI heuristic, logged results, tests.

**Acceptance criteria:** COI heuristic matches/within-gap of the LP optimum; feasibility tests pass.

---

### Phase 5 — Exact vs Metaheuristic Comparison ⭐ centerpiece

**Objective:** The analytical heart of the portfolio — quantify the trade-offs.

**Steps:**
1. `experiments/compare.py` — load all `output/results/*.json`; build a comparison table:
   objective value, runtime, **optimality gap** (vs exact where available), iterations to converge.
2. Run each metaheuristic across **multiple seeds** (e.g., 10–30) to report mean ± std (variance matters).
3. Record convergence histories for plotting in Phase 6.
4. Write a short findings summary feeding the report.

**Deliverables:** `output/results/comparison.csv`, per-problem convergence logs, summary notes.

**Acceptance criteria:** a single command regenerates the comparison table; numbers are seed-reproducible.

---

### Phase 6 — Solution Visualizations (`visualization/plot_solutions.py`)

**Objective:** Turn solutions into clear figures matching the existing input-data figure style.

**Steps:**
1. **P1:** candidate-site map with placed departments + flow ribbons for top pairs.
2. **P2:** geographic/network graph of zones → chosen substations (color by substation, size by demand).
3. **P3:** storage layout heatmap (COI / travel intensity per slot).
4. **Comparison:** convergence curves (objective vs iteration) with seed bands; exact-vs-heuristic bar charts.
5. Save all to `output/figures/`, 150 DPI, consistent palette.

**Deliverables:** solution figures for all three problems + comparison plots.

**Acceptance criteria:** every figure referenced by the report exists and regenerates from logged results.

---

### Phase 7 — LaTeX/PDF Report + README

**Objective:** Package the work for human readers.

**Steps:**
1. `report/report.tex` sections: Intro & context → Data → Formulations (from `docs/formulations.md`) →
   Methods (exact + metaheuristic) → Results per problem → Comparison & discussion → Conclusion.
2. Pull figures from `output/figures/`; cite OR references in `references.bib`.
3. Build to `report/report.pdf` (document the build command, e.g., `latexmk -pdf`).
4. `README.md`: project pitch, the renewable-energy framing, problem table, repo map, **how to run**
   (one command per problem + how to rebuild the report), result highlights with embedded figures, license.

**Deliverables:** compiled `report.pdf`, polished `README.md`.

**Acceptance criteria:** report compiles from a clean checkout; README "how to run" steps work verbatim.

---

### Phase 8 — Interactive Dashboard *(deferred — later stage)*

**Objective:** Single-page, zero-dependency HTML to explore solutions interactively (per locked decision,
built only after Phases 0–7 land).

**Steps (sketch, to be detailed when we get here):**
1. Export solution JSON the page can load.
2. Single self-contained `dashboard/index.html` (offline): toggle problems, view assignments on a map,
   slide solver parameters / replay convergence.
3. Link from README.

**Deliverables:** `dashboard/index.html`.

**Acceptance criteria:** opens offline in a browser; reflects the logged solutions.

---

## 5. Assumptions to Validate (before/while modeling)

These are modeling judgment calls where the original brief is silent. Each will be **documented in
`docs/formulations.md`** and revisited if results look off:

1. **P1 distance metric** — Euclidean from (x, y) vs the provided "fixed-travel-distance" column. Plan:
   Euclidean primary, fixed-distance as documented fallback/variant.
2. **P1 location selection** — 30 departments vs 197 sites ⇒ select-and-assign. Confirm whether all 197 are
   eligible or only type-compatible subsets (the type column strongly implies compatibility filtering).
3. **P1 type compatibility** — the mapping from location types to departments (e.g., "Spare Parts Storage" ↔
   "Spare Parts Agent"). Will be encoded explicitly and listed.
4. **P2 candidate sites** — assume a substation may be built in any of the 137 zones (siting = subset of zones).
5. **P2 transmission cost** — objective uses `distance × demand`; confirm whether cost should weight by demand
   (assumed yes) or be pure distance.
6. **P3 I/O / travel model** — single I/O point assumed unless the data implies dock locations; slot distances
   derived from a simple rack layout model (documented).
7. **Unit scaling** — demand ×1000 KWh, cost ×\$10, capacity ×100 KWh; keep raw units internally, convert only
   for reporting.

---

## 6. Tooling & Conventions

- **Exact:** PuLP + CBC (bundled with PuLP; no license needed).
- **Metaheuristics:** lightweight custom GA / SA / PSO sharing `solvers/metaheuristic/base.py` (no heavy deps).
- **Reproducibility:** every experiment takes a seed and time limit from `experiments/config.py`; results are
  written to `output/results/` as JSON/CSV so figures and the report regenerate from data, not memory.
- **Testing:** `pytest` feasibility + sanity checks per problem.
- **Report:** LaTeX → PDF via `latexmk`.
- **Commits:** one logical commit per phase (or finer); descriptive messages.

---

## 7. Suggested Execution Order (critical path)

```
Phase 0  Foundation ─┐
Phase 1  Formulations ┘─→ Phase 2 (P2 exact+GA) ─→ Phase 3 (P1) ─→ Phase 4 (P3)
                                                          │
                          Phase 5 (comparison) ←──────────┘
                                   │
                          Phase 6 (viz) ─→ Phase 7 (report + README)
                                                   │
                                          Phase 8 (dashboard, later)
```

**Recommended kickoff:** Phase 0 + Phase 1 together (foundation + formulations doc), since they unblock all
solver work and produce the docs reviewers read first.

---

## 8. Open Questions for Reviewer

1. Any preference on the **P1 distance metric** (Euclidean vs fixed-travel-distance), or should I implement
   both and compare?
2. Is the **type-compatibility** interpretation of P1 acceptable, or should P1 treat all 197 sites as eligible
   for every department (pure QAP-with-selection)?
3. For the report — **IEEE/academic two-column** style or a cleaner single-column technical-report style?
4. GitHub repo name preference (e.g., `renewable-facility-optimization`, `or-facility-optimization`)?

---

*End of plan. Awaiting review before execution.*
