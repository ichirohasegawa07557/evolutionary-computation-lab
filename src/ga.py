from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.operators import tournament_selection, blend_crossover, gaussian_mutation


@dataclass
class GAConfig:
    population_size: int = 80
    dimensions: int = 10
    generations: int = 120
    bounds: tuple[float, float] = (-5.12, 5.12)
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.25
    crossover_rate: float = 0.9
    tournament_size: int = 3
    elitism: int = 2
    seed: int = 42


@dataclass
class GAResult:
    best_solution: np.ndarray
    best_fitness: float
    history: pd.DataFrame


class GeneticAlgorithm:
    """Small from-scratch real-valued genetic algorithm for minimization."""

    def __init__(self, fitness_fn, config: GAConfig):
        self.fitness_fn = fitness_fn
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _initialize_population(self) -> np.ndarray:
        low, high = self.config.bounds
        return self.rng.uniform(low, high, size=(self.config.population_size, self.config.dimensions))

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_fn(ind) for ind in population], dtype=float)

    def run(self) -> GAResult:
        cfg = self.config
        population = self._initialize_population()
        records = []

        best_solution = None
        best_fitness = float("inf")

        for generation in range(cfg.generations + 1):
            fitness = self._evaluate(population)
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            if fitness[0] < best_fitness:
                best_fitness = float(fitness[0])
                best_solution = population[0].copy()

            records.append({
                "generation": generation,
                "best_fitness": float(fitness[0]),
                "mean_fitness": float(np.mean(fitness)),
                "median_fitness": float(np.median(fitness)),
                "diversity": float(np.mean(np.std(population, axis=0))),
            })

            if generation == cfg.generations:
                break

            next_population = [population[i].copy() for i in range(min(cfg.elitism, cfg.population_size))]

            while len(next_population) < cfg.population_size:
                parent_a = tournament_selection(population, fitness, self.rng, cfg.tournament_size)
                parent_b = tournament_selection(population, fitness, self.rng, cfg.tournament_size)

                if self.rng.random() < cfg.crossover_rate:
                    child_a, child_b = blend_crossover(parent_a, parent_b, self.rng)
                else:
                    child_a, child_b = parent_a.copy(), parent_b.copy()

                child_a = gaussian_mutation(child_a, self.rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)
                child_b = gaussian_mutation(child_b, self.rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)

                next_population.append(child_a)
                if len(next_population) < cfg.population_size:
                    next_population.append(child_b)

            population = np.asarray(next_population)

        assert best_solution is not None
        return GAResult(best_solution=best_solution, best_fitness=best_fitness, history=pd.DataFrame(records))
