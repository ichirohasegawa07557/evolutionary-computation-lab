from __future__ import annotations

import numpy as np

from src.fitness import sphere, rastrigin
from src.ga import GAConfig, GeneticAlgorithm
from src.ca import decode_rule, simulate_ca, ca_fitness, target_triangle
from src.evolve_ca import CARuleEvolution, CARuleEvolutionConfig
from src.multiobjective import bi_objective_sphere, fast_non_dominated_sort, NSGAII, NSGAIIConfig
from src.gp import safe_eval, tree_to_string, GeneticProgrammingRegressor, GPConfig
from src.constraints import constraint_violations, penalized_objective
from src.island import run_migrating_island_model


def test_benchmark_minima_are_zero_or_near_zero():
    assert sphere(np.zeros(4)) == 0.0
    assert abs(rastrigin(np.zeros(4))) < 1e-9


def test_ga_improves_sphere():
    cfg = GAConfig(population_size=30, dimensions=4, generations=25, seed=1)
    result = GeneticAlgorithm(sphere, cfg).run()
    assert result.history["best_fitness"].iloc[-1] <= result.history["best_fitness"].iloc[0]
    assert result.best_solution.shape == (4,)


def test_ca_rule_decode_shape():
    bits = decode_rule(90)
    assert bits.shape == (8,)
    assert set(bits.tolist()).issubset({0, 1})


def test_ca_simulation_shape():
    grid = simulate_ca(90, width=31, steps=20)
    assert grid.shape == (20, 31)


def test_ca_fitness_for_target_rule_is_zero():
    target = target_triangle(width=31, steps=20)
    assert ca_fitness(90, target) == 0.0


def test_ca_evolution_runs():
    cfg = CARuleEvolutionConfig(population_size=12, generations=5, width=31, steps=20, seed=4)
    result = CARuleEvolution(cfg).run()
    assert 0 <= result.best_rule <= 255
    assert result.history.shape[0] == 6


def test_non_dominated_sort_finds_front():
    objectives = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [2.0, 2.0]])
    fronts = fast_non_dominated_sort(objectives)
    assert set(fronts[0]) == {0, 1, 2}


def test_nsga2_runs():
    result = NSGAII(config=NSGAIIConfig(population_size=20, generations=5, seed=2)).run()
    assert {"objective_1", "objective_2", "rank"}.issubset(result.population.columns)
    assert len(result.population) == 20
    assert bi_objective_sphere(np.zeros(2))[0] == 0.0


def test_gp_expression_evaluates():
    x = np.array([0.0, 1.0, 2.0])
    tree = ("add", ("mul", "x", "x"), ("const", 1.0))
    y = safe_eval(tree, x)
    assert np.allclose(y, np.array([1.0, 2.0, 5.0]))
    assert "x" in tree_to_string(tree)


def test_gp_runs_short_demo():
    result = GeneticProgrammingRegressor(GPConfig(population_size=12, generations=3, seed=5)).run()
    assert isinstance(result.best_expression, str)
    assert result.history.shape[0] == 4


def test_constraint_penalty_is_positive_for_violation():
    feasible = np.array([0.5, 0.5])
    infeasible = np.array([0.0, 0.0])
    assert constraint_violations(feasible)["sum_at_least_1"] == 0.0
    assert constraint_violations(infeasible)["sum_at_least_1"] > 0.0
    assert penalized_objective(infeasible) > penalized_objective(feasible)


def test_migrating_island_model_runs():
    _, best_fitness, history = run_migrating_island_model(
        sphere,
        islands=2,
        generations=4,
        dimensions=2,
        population_size=8,
        migration_interval=2,
        migration_size=1,
        seed=6,
    )
    assert best_fitness >= 0.0
    assert history["migration_event"].any()
