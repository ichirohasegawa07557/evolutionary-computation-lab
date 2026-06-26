from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RandomSearchConfig:
    dimensions: int = 10
    generations: int = 120
    samples_per_generation: int = 60
    bounds: tuple[float, float] = (-5.12, 5.12)
    seed: int = 123


def run_random_search(fitness_fn, config: RandomSearchConfig):
    rng = np.random.default_rng(config.seed)
    low, high = config.bounds
    best_fitness = float("inf")
    best_solution = None
    records = []

    for generation in range(config.generations + 1):
        samples = rng.uniform(low, high, size=(config.samples_per_generation, config.dimensions))
        fitness = np.array([fitness_fn(s) for s in samples])
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_fitness:
            best_fitness = float(fitness[idx])
            best_solution = samples[idx].copy()
        records.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "mean_generation_fitness": float(np.mean(fitness)),
        })

    return best_solution, best_fitness, pd.DataFrame(records)
