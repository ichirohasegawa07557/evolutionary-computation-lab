from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.ga import GAConfig, GeneticAlgorithm


def constrained_objective(x: np.ndarray) -> float:
    """Minimize distance from origin under simple linear constraints."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def constraint_violations(x: np.ndarray) -> dict[str, float]:
    """Return positive values for violated constraints.

    Constraints:
    - x0 + x1 >= 1.0
    - x0 <= 1.0
    - x1 <= 1.0
    """
    x = np.asarray(x, dtype=float)
    return {
        "sum_at_least_1": max(0.0, 1.0 - float(x[0] + x[1])),
        "x0_at_most_1": max(0.0, float(x[0] - 1.0)),
        "x1_at_most_1": max(0.0, float(x[1] - 1.0)),
    }


def total_violation(x: np.ndarray) -> float:
    violations = constraint_violations(x)
    return float(sum(v ** 2 for v in violations.values()))


def penalized_objective(x: np.ndarray, penalty_weight: float = 100.0) -> float:
    return constrained_objective(x) + penalty_weight * total_violation(x)


@dataclass
class ConstraintResult:
    best_solution: np.ndarray
    best_objective: float
    total_violation: float
    penalized_fitness: float
    history: pd.DataFrame


def run_constrained_ga(
    generations: int = 80,
    population_size: int = 60,
    penalty_weight: float = 100.0,
    seed: int = 31,
) -> ConstraintResult:
    """Run a GA with quadratic penalty functions for constraint handling."""

    def fitness_fn(x: np.ndarray) -> float:
        return penalized_objective(x, penalty_weight=penalty_weight)

    cfg = GAConfig(
        population_size=population_size,
        dimensions=2,
        generations=generations,
        bounds=(-1.5, 2.0),
        mutation_rate=0.2,
        mutation_sigma=0.15,
        seed=seed,
    )
    result = GeneticAlgorithm(fitness_fn, cfg).run()
    best_solution = result.best_solution
    return ConstraintResult(
        best_solution=best_solution,
        best_objective=constrained_objective(best_solution),
        total_violation=total_violation(best_solution),
        penalized_fitness=result.best_fitness,
        history=result.history,
    )
