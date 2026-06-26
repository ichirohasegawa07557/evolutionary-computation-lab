from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ESConfig:
    dimensions: int = 10
    generations: int = 120
    population_size: int = 60
    elite_size: int = 12
    bounds: tuple[float, float] = (-5.12, 5.12)
    sigma: float = 0.45
    seed: int = 7


@dataclass
class ESResult:
    best_solution: np.ndarray
    best_fitness: float
    history: pd.DataFrame


class EvolutionStrategy:
    """Simple (mu + lambda)-style evolution strategy for minimization."""

    def __init__(self, fitness_fn, config: ESConfig):
        self.fitness_fn = fitness_fn
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def run(self) -> ESResult:
        cfg = self.config
        low, high = cfg.bounds
        population = self.rng.uniform(low, high, size=(cfg.population_size, cfg.dimensions))
        records = []
        best_solution = population[0].copy()
        best_fitness = float("inf")

        sigma = cfg.sigma

        for generation in range(cfg.generations + 1):
            fitness = np.array([self.fitness_fn(ind) for ind in population])
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            elites = population[: cfg.elite_size]

            if fitness[0] < best_fitness:
                best_fitness = float(fitness[0])
                best_solution = population[0].copy()

            records.append({
                "generation": generation,
                "best_fitness": float(fitness[0]),
                "mean_fitness": float(np.mean(fitness)),
                "sigma": float(sigma),
            })

            if generation == cfg.generations:
                break

            children = []
            while len(children) < cfg.population_size:
                parent = elites[int(self.rng.integers(0, len(elites)))]
                child = parent + self.rng.normal(0.0, sigma, size=cfg.dimensions)
                children.append(np.clip(child, low, high))
            population = np.asarray(children)

            # Mild annealing for stability.
            sigma *= 0.985

        return ESResult(best_solution=best_solution, best_fitness=best_fitness, history=pd.DataFrame(records))
