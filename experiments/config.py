"""
Central experiment configuration
=================================
Single source of truth for data paths, seeds, solver time limits, and
metaheuristic hyperparameters, so that every run is reproducible and the
report regenerates from logged results.
"""

import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "Group 1.xlsx")
RESULTS_DIR = os.path.join(ROOT, "output", "results")
FIGURES_DIR = os.path.join(ROOT, "output", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42                 # master seed
N_SEEDS = 10              # independent metaheuristic restarts for mean +/- std
SEEDS = list(range(SEED, SEED + N_SEEDS))

# ----------------------------------------------------------------------------
# Exact solver (PuLP/CBC)
# ----------------------------------------------------------------------------
EXACT_TIME_LIMIT = 120    # seconds per MILP solve
EXACT_MSG = False         # CBC verbosity

# ----------------------------------------------------------------------------
# Metaheuristic hyperparameters
# ----------------------------------------------------------------------------
GA = dict(
    pop_size=80,
    n_generations=200,
    tournament_k=3,
    elitism=2,
    crossover_rate=0.9,
    mutation_rate=0.15,
)

SA = dict(
    n_iterations=15000,
    t_start=1.0,          # set relative to objective scale at runtime
    t_end=1e-3,
    cooling="exponential",
    restarts=1,
)

PSO = dict(
    n_particles=40,
    n_iterations=200,
    w=0.72,               # inertia
    c1=1.49,              # cognitive
    c2=1.49,              # social
)

# ----------------------------------------------------------------------------
# Problem-specific
# ----------------------------------------------------------------------------
# P2: transmission-rate sweep (multipliers applied to the calibrated alpha_0)
P2_ALPHA_SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
P2_ALPHA_BASELINE = 1.0   # multiplier on alpha_0 used for the headline solution

# P1: reduced instance size for the exact MILP <-> heuristic comparison
P1_EXACT_K = 8            # K departments x K locations square QAP (CBC-tractable)
# QAP objective evaluation is cheap, so the metaheuristics get larger budgets
P1_GA = dict(pop_size=120, n_generations=400, tournament_k=3, elitism=2,
             crossover_rate=0.9, mutation_rate=0.30)
P1_SA = dict(n_iterations=40000, t_end=1e-3, cooling="exponential")
