from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.ca import ca_fitness, target_triangle


@dataclass
class CARuleEvolutionConfig:
    population_size: int = 40
    generations: int = 60
    mutation_rate: float = 0.08
    seed: int = 42
    width: int = 101
    steps: int = 80


@dataclass
class CARuleEvolutionResult:
    best_rule: int
    best_fitness: float
    history: pd.DataFrame


class CARuleEvolution:
    """Evolve elementary cellular automaton rule numbers toward a target pattern."""

    def __init__(self, config: CARuleEvolutionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.target = target_triangle(config.width, config.steps)

    def _mutate_rule(self, rule: int) -> int:
        bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
        mask = self.rng.random(8) < self.config.mutation_rate
        bits[mask] = 1 - bits[mask]
        new_rule = int(sum(int(bits[i]) << i for i in range(8)))
        return new_rule

    def run(self) -> CARuleEvolutionResult:
        cfg = self.config
        population = self.rng.integers(0, 256, size=cfg.population_size)
        records = []
        best_rule = int(population[0])
        best_fitness = float("inf")

        for generation in range(cfg.generations + 1):
            fitness = np.array([ca_fitness(int(rule), self.target) for rule in population])
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            if fitness[0] < best_fitness:
                best_rule = int(population[0])
                best_fitness = float(fitness[0])

            records.append({
                "generation": generation,
                "best_rule": int(population[0]),
                "best_fitness": float(fitness[0]),
                "mean_fitness": float(np.mean(fitness)),
            })

            if generation == cfg.generations:
                break

            elites = population[: max(2, cfg.population_size // 5)]
            next_population = [int(rule) for rule in elites]
            while len(next_population) < cfg.population_size:
                parent = int(elites[int(self.rng.integers(0, len(elites)))])
                next_population.append(self._mutate_rule(parent))
            population = np.array(next_population, dtype=int)

        return CARuleEvolutionResult(best_rule=best_rule, best_fitness=best_fitness, history=pd.DataFrame(records))
