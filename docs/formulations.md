# Mathematical Formulations

> Reference document for all three optimization problems. Every solver in `solvers/`
> implements one of the models specified here. Notation is kept consistent across problems.
> Instance sizes and statistics below are computed from `Group 1.xlsx` via `data/loader.py`.

**Contents**
- [Conventions](#conventions)
- [P1 — Department Layout (Quadratic Assignment Problem)](#p1--department-layout-quadratic-assignment-problem)
- [P2 — Substation Siting (Capacitated Facility Location)](#p2--substation-siting-capacitated-facility-location)
- [P3 — Storage Layout (Slot Assignment / Cube-per-Order Index)](#p3--storage-layout-slot-assignment--cube-per-order-index)
- [Assumptions to Validate](#assumptions-to-validate)

---

## Conventions

- Sets use calligraphic/uppercase letters ($\mathcal{D}$, $\mathcal{Z}$, $\mathcal{P}$); indices lowercase.
- Decision variables are bold-italic in prose, plain in equations; all binary unless stated.
- "Exact" = MILP/LP solved with PuLP + CBC to proven optimality (or a reported gap).
- "Heuristic" = metaheuristic (GA/SA/PSO) returning a feasible incumbent with no optimality proof.
- Objectives are **minimization** throughout.

---

## P1 — Department Layout (Quadratic Assignment Problem)

Assign 30 organizational departments to physical candidate sites so that departments with
high mutual flow (people, documents, materials per period) are placed close together.

### Instance
- Departments $\mathcal{D}$, $|\mathcal{D}| = 30$.
- Candidate locations $\mathcal{L}$, $|\mathcal{L}| = 197$ (with $(x,y)$ coordinates and a type label; 26 distinct types).
- Flow matrix $f_{ij} \ge 0$ is **asymmetric** ($f_{ij} \neq f_{ji}$), $f_{ii}=0$; 870 of 900 entries are non-zero.

### Parameters
| Symbol | Meaning |
|--------|---------|
| $f_{ij}$ | flow from department $i$ to department $j$ (per period) |
| $d_{k\ell}$ | distance between locations $k$ and $\ell$ — Euclidean from coordinates, $d_{k\ell}=\sqrt{(x_k-x_\ell)^2+(y_k-y_\ell)^2}$ (symmetric) |
| $a_{ik}\in\{0,1\}$ | type-compatibility: $1$ if department $i$ may occupy location $k$ |

### Decision variables
$$x_{ik} = \begin{cases} 1 & \text{if department } i \text{ is placed at location } k \\ 0 & \text{otherwise} \end{cases}$$

### Quadratic model (QAP)
$$\min \sum_{i\in\mathcal{D}}\sum_{j\in\mathcal{D}}\sum_{k\in\mathcal{L}}\sum_{\ell\in\mathcal{L}} f_{ij}\, d_{k\ell}\, x_{ik}\, x_{j\ell}$$

subject to
$$\sum_{k\in\mathcal{L}} x_{ik} = 1 \qquad \forall i\in\mathcal{D} \qquad\text{(each department placed exactly once)}$$
$$\sum_{i\in\mathcal{D}} x_{ik} \le 1 \qquad \forall k\in\mathcal{L} \qquad\text{(each location holds at most one department)}$$
$$x_{ik} \le a_{ik} \qquad \forall i,k \qquad\text{(type compatibility)}$$
$$x_{ik}\in\{0,1\}.$$

Because $|\mathcal{L}|=197 > |\mathcal{D}|=30$, this is a QAP **with location selection**: only 30 of the
197 sites are used.

### Linearization (for the exact solver)
The product $x_{ik}x_{j\ell}$ is replaced by $y_{ikj\ell}\ge 0$ with the standard McCormick constraints:
$$y_{ikj\ell} \le x_{ik},\quad y_{ikj\ell} \le x_{j\ell},\quad y_{ikj\ell} \ge x_{ik}+x_{j\ell}-1,$$
$$\min \sum_{i,j,k,\ell} f_{ij}\,d_{k\ell}\,y_{ikj\ell}.$$

This has $O(|\mathcal{D}|^2|\mathcal{L}|^2) \approx 3.4\times10^{9}$ product terms for the full instance — **intractable**.
**Plan:** solve the exact MILP only on a **reduced instance** (restrict each department to its
type-compatible locations, which collapses most $a_{ik}=0$, and/or shrink $\mathcal{L}$), and use a
**permutation metaheuristic** for the full $30\to197$ problem. A more compact alternative
(Kaufman–Broeckx linearization, $O(|\mathcal{D}||\mathcal{L}|)$ extra variables) is noted as a fallback if the
reduced MILP is still heavy.

### Heuristic encoding
A solution is an injective map $\pi:\mathcal{D}\to\mathcal{L}$ with $a_{i,\pi(i)}=1$. The objective evaluates
in $O(|\mathcal{D}|^2)$:
$$C(\pi)=\sum_{i}\sum_{j} f_{ij}\, d_{\pi(i)\pi(j)}.$$
Neighborhood moves: **swap** two departments' locations, or **relocate** a department to an empty
compatible site.

---

## P2 — Substation Siting (Capacitated Facility Location)

Decide in which zones to build electrical substations and which substation serves each demand
zone, minimizing construction cost plus transmission cost, subject to substation capacity and
forbidden links.

### Instance
- Zones $\mathcal{Z}$, $|\mathcal{Z}| = 137$. A substation may be built in any zone, so candidate sites $=\mathcal{Z}$.
- Distance matrix $d_{zs}$ is **asymmetric**; $5.7\%$ of entries are $+\infty$ (forbidden links). No zone is fully isolated.

### Parameters
| Symbol | Meaning | Raw total |
|--------|---------|-----------|
| $q_z \ge 0$ | expected demand of zone $z$ | $\sum_z q_z = 138{,}255$ |
| $c_s \ge 0$ | cost to construct a substation at $s$ | — |
| $\kappa_s \ge 0$ | capacity of a substation at $s$ | $\sum_s \kappa_s = 254{,}041$ |
| $d_{zs}\in\mathbb{R}_{\ge0}\cup\{\infty\}$ | transmission distance, zone $z$ ↔ substation $s$ ($\infty$ = not allowed). Raw diagonal is $\infty$; we set $d_{ss}=0$ (a substation serves its own zone for free) — see A8. | — |
| $\alpha > 0$ | transmission rate (cost per unit demand·distance); calibrated from the data so construction and transmission are comparable — see A8 | — |

> **⚠ Unit / feasibility note.** The sheet labels demand as "×1000 KWh" and capacity as "×100 KWh".
> Applied literally, total demand ($138{,}255{,}000$ KWh) is $5.4\times$ total capacity
> ($25{,}404{,}100$ KWh), making the instance **infeasible even if every substation is built**.
> We therefore treat the **raw tabulated values as a common unit** (total capacity $254{,}041$ vs
> total demand $138{,}255$, i.e. capacity is $1.84\times$ demand → feasible, ≈54 % utilization if all
> built). See [Assumptions](#assumptions-to-validate) A5.

### Decision variables
$$y_s=\begin{cases}1 & \text{substation built at } s\\ 0 & \text{otherwise}\end{cases} \qquad
x_{zs}=\begin{cases}1 & \text{zone } z \text{ served by substation } s\\ 0 & \text{otherwise}\end{cases}$$

### Model (single-source CFLP)
$$\min \underbrace{\sum_{s\in\mathcal{Z}} c_s\, y_s}_{\text{construction}} \; + \; \alpha\underbrace{\sum_{z\in\mathcal{Z}}\sum_{s\in\mathcal{Z}} d_{zs}\, q_z\, x_{zs}}_{\text{transmission}}$$

with $d_{ss}=0$ (self-service free) and the transmission rate $\alpha$ controlling the construction-vs-transmission trade-off: small $\alpha$ → consolidate onto few substations (save construction, accept transmission); large $\alpha$ → open many substations (minimize transmission). We report the **optimum as a function of $\alpha$** (number of substations and cost split), a classic facility-location sensitivity curve, and pick a calibrated baseline $\alpha_0$ for the headline solution.

subject to
$$\sum_{s} x_{zs}=1 \qquad\forall z \qquad\text{(every zone served exactly once)}$$
$$x_{zs}\le y_s \qquad\forall z,s \qquad\text{(serve only from open substations)}$$
$$\sum_{z} q_z\, x_{zs}\le \kappa_s\, y_s \qquad\forall s \qquad\text{(capacity)}$$
$$x_{zs}=0 \quad\text{whenever } d_{zs}=\infty \qquad\text{(forbidden links)}$$
$$x_{zs},\,y_s\in\{0,1\}.$$

This is the **single-source** CFLP (each zone assigned to exactly one substation): NP-hard, but a
137-zone instance is tractable for CBC within a time limit, typically to optimality or a small gap.
Forbidden links are handled by **omitting** those $x_{zs}$ variables entirely (cleaner and smaller than
big-$M$).

**Variants we will report:**
- *Multi-source relaxation* — allow $x_{zs}\in[0,1]$ (demand splitting). LP-easier, gives a lower bound.
- *Transmission weighting* — objective uses $d_{zs}\cdot q_z$ (cost scales with delivered demand); a pure-distance variant ($d_{zs}$ only) is available for comparison.

### Heuristic encoding
A solution is the open set $S\subseteq\mathcal{Z}$ ($y$). Given $S$, zones are assigned greedily to the
cheapest feasible open substation respecting capacity (with a repair step if capacity is violated).
GA operates on the binary open-vector; SA flips/​swaps open sites.

---

## P3 — Storage Layout (Slot Assignment / Cube-per-Order Index)

Assign products to storage slots to minimize expected material-handling travel. High-throughput,
low-volume products should sit closest to the input/output (I/O) point.

### Instance
- Products $\mathcal{P}$, $|\mathcal{P}| = 20$.
- Storage slots $\mathcal{S}$ (each $1.5\,\text{m}\times1.5\,\text{m}\times1.5\,\text{m}$). Total slots required $\sum_p q_p = 25{,}189$.

### Parameters
| Symbol | Meaning |
|--------|---------|
| $t_p$ | throughput of product $p$ (operations / period); $\sum_p t_p = 118{,}231$ |
| $q_p$ | storage requirement of product $p$ (number of slots) |
| $\delta_s$ | round-trip travel distance from slot $s$ to the I/O point |

### Cube-per-Order Index
$$\text{COI}_p = \frac{q_p}{t_p} \qquad (\text{range in this instance: } 0.036 \ldots 0.686)$$

**Optimality result (Heskett):** total travel is minimized by ranking products by ascending COI and
filling slots from nearest to farthest. Products with low COI (high turnover per unit volume) get the
closest slots.

### Model (transportation / assignment LP)
Let $x_{ps}\in\{0,1\}$ indicate that slot $s$ stores product $p$ (unit-capacity slots). With one trip
per slot scaled by per-slot visit frequency $t_p/q_p$:
$$\min \sum_{p\in\mathcal{P}}\sum_{s\in\mathcal{S}} \frac{t_p}{q_p}\,\delta_s\, x_{ps}$$
subject to
$$\sum_{s} x_{ps}=q_p \qquad\forall p \qquad\text{(each product gets its slots)}$$
$$\sum_{p} x_{ps}\le 1 \qquad\forall s \qquad\text{(each slot holds one product)}$$
$$x_{ps}\in\{0,1\}.$$

The constraint matrix is **totally unimodular** (a transportation problem), so the LP relaxation is
integral — it solves exactly and fast, and its optimum **coincides** with the COI ranking rule. We
report both to demonstrate the equivalence.

### Slot geometry assumption
With no explicit rack layout in the data, $\delta_s$ is generated from a simple single-I/O rack model
(slots ordered by increasing distance from one I/O point). Documented in [Assumptions](#assumptions-to-validate) A6.

---

## Assumptions to Validate

These are modeling judgment calls where the original brief is silent or internally inconsistent.
Each is encoded explicitly in code and revisited if results look wrong.

| # | Assumption | Rationale / impact |
|---|------------|--------------------|
| **A1** | P1 distance = Euclidean from $(x,y)$. The sheet's "fixed-travel-distance" column is offered as a documented variant, not the default. | Euclidean is standard for layout QAP; we can report both. |
| **A2** | P1 uses **location selection** (30 of 197 sites), not a 1-to-1 QAP. | Forced by $|\mathcal{L}|>|\mathcal{D}|$. |
| **A3** | P1 **type compatibility** $a_{ik}$ derived by mapping location types to department semantics (e.g. "Spare Parts Storage" ↔ "Spare Parts Agent"). The full mapping is tabulated in `models/qap.py`. | If too restrictive, fall back to all-eligible ($a_{ik}\equiv1$). |
| **A4** | P2 candidate substation sites = the 137 zones themselves. | No separate candidate-site list is given. |
| **A5** | P2 demand and capacity are treated in a **common unit** (raw tabulated values), overriding the literal "×1000" / "×100" labels, because the literal reading is infeasible (demand $5.4\times$ capacity). | Critical — without this the model has no feasible solution. |
| **A6** | P3 travel uses a **single I/O point** with a simple rack-distance model for $\delta_s$; no dock list is given. | Affects absolute travel numbers, not the COI ranking. |
| **A7** | Units are kept raw internally; conversions to KWh/\$ happen only at reporting time. | Avoids scaling bugs. |
| **A8** | P2 sets the (raw-$\infty$) distance diagonal to $d_{ss}=0$ and introduces a transmission rate $\alpha$, calibrated so mean construction ≈ mean nearest-neighbour transmission. Without $\alpha$, transmission (mean $d\approx10^5$) dwarfs construction ($\approx10^4$) and the model degenerates to "build everywhere". | Makes P2 a genuine, non-degenerate facility-location trade-off; we report an $\alpha$-sweep. |

---

*Cross-references: implementations live in `models/` (instance construction) and `solvers/`
(exact + metaheuristic). Experiment runners in `experiments/` log results to `output/results/`.*
