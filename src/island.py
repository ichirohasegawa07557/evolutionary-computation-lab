from __future__ import annotations

import numpy as np
import pandas as pd

from src.ga import GAConfig, GeneticAlgorithm
from src.operators import tournament_selection, blend_crossover, gaussian_mutation


def run_island_model(fitness_fn, islands: int = 4, generations_per_island: int = 60, dimensions: int = 10, seed: int = 99):
    """Run independent islands and summarize the best solution found.

    This legacy compact variant treats islands as independent repeated runs.
    Use run_migrating_island_model for actual inter-island migration.
    """
    records = []
    best_fitness = float("inf")
    best_solution = None

    for island in range(islands):
        cfg = GAConfig(
            population_size=60,
            generations=generations_per_island,
            dimensions=dimensions,
            seed=seed + island,
        )
        result = GeneticAlgorithm(fitness_fn, cfg).run()
        hist = result.history.copy()
        hist["island"] = island
        hist["migration_event"] = False
        records.append(hist)
        if result.best_fitness < best_fitness:
            best_fitness = result.best_fitness
            best_solution = result.best_solution.copy()

    return best_solution, best_fitness, pd.concat(records, ignore_index=True)


def _initialize_islands(rng: np.random.Generator, islands: int, population_size: int, dimensions: int, bounds: tuple[float, float]) -> list[np.ndarray]:
    low, high = bounds
    return [rng.uniform(low, high, size=(population_size, dimensions)) for _ in range(islands)]


def _next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    cfg: GAConfig,
) -> np.ndarray:
    order = np.argsort(fitness)
    population = population[order]
    fitness = fitness[order]
    next_population = [population[i].copy() for i in range(min(cfg.elitism, cfg.population_size))]

    while len(next_population) < cfg.population_size:
        parent_a = tournament_selection(population, fitness, rng, cfg.tournament_size)
        parent_b = tournament_selection(population, fitness, rng, cfg.tournament_size)
        if rng.random() < cfg.crossover_rate:
            child_a, child_b = blend_crossover(parent_a, parent_b, rng)
        else:
            child_a, child_b = parent_a.copy(), parent_b.copy()
        child_a = gaussian_mutation(child_a, rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)
        child_b = gaussian_mutation(child_b, rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)
        next_population.append(child_a)
        if len(next_population) < cfg.population_size:
            next_population.append(child_b)
    return np.asarray(next_population)


def _migrate_ring(populations: list[np.ndarray], fitnesses: list[np.ndarray], migration_size: int) -> None:
    """Move elite individuals to the next island and replace its worst members."""
    if migration_size <= 0 or len(populations) < 2:
        return
    migrants = []
    for pop, fit in zip(populations, fitnesses):
        elite_idx = np.argsort(fit)[:migration_size]
        migrants.append(pop[elite_idx].copy())

    for source_island, migrant_block in enumerate(migrants):
        target_island = (source_island + 1) % len(populations)
        target_fit = fitnesses[target_island]
        worst_idx = np.argsort(target_fit)[-migration_size:]
        populations[target_island][worst_idx] = migrant_block


def run_migrating_island_model(
    fitness_fn,
    islands: int = 4,
    generations: int = 80,
    dimensions: int = 10,
    population_size: int = 50,
    migration_interval: int = 10,
    migration_size: int = 2,
    seed: int = 123,
):
    """Run a real ring-migration island model.

    Every migration_interval generations, each island sends its best individuals
    to the next island and replaces the next island's weakest individuals.
    """
    cfg = GAConfig(
        population_size=population_size,
        dimensions=dimensions,
        generations=generations,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    populations = _initialize_islands(rng, islands, population_size, dimensions, cfg.bounds)
    records = []
    best_fitness = float("inf")
    best_solution = None

    for generation in range(generations + 1):
        fitnesses = [np.asarray([fitness_fn(ind) for ind in pop], dtype=float) for pop in populations]
        migration_event = generation > 0 and generation % migration_interval == 0

        for island_id, (pop, fit) in enumerate(zip(populations, fitnesses)):
            best_idx = int(np.argmin(fit))
            if float(fit[best_idx]) < best_fitness:
                best_fitness = float(fit[best_idx])
                best_solution = pop[best_idx].copy()
            records.append({
                "generation": generation,
                "island": island_id,
                "best_fitness": float(np.min(fit)),
                "mean_fitness": float(np.mean(fit)),
                "diversity": float(np.mean(np.std(pop, axis=0))),
                "migration_event": bool(migration_event),
            })

        if generation == generations:
            break

        island_rngs = [np.random.default_rng(seed + generation * 1000 + i) for i in range(islands)]
        populations = [
            _next_generation(populations[i], fitnesses[i], island_rngs[i], cfg)
            for i in range(islands)
        ]

        if (generation + 1) % migration_interval == 0:
            updated_fitnesses = [np.asarray([fitness_fn(ind) for ind in pop], dtype=float) for pop in populations]
            _migrate_ring(populations, updated_fitnesses, migration_size=migration_size)

    assert best_solution is not None
    return best_solution, best_fitness, pd.DataFrame(records)
