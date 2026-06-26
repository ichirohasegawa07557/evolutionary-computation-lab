from __future__ import annotations

import numpy as np


def tournament_selection(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator, tournament_size: int = 3) -> np.ndarray:
    """Select one individual using minimization tournament selection."""
    indices = rng.integers(0, len(population), size=tournament_size)
    best_index = indices[np.argmin(fitness[indices])]
    return population[best_index].copy()


def blend_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha style arithmetic crossover for real-valued vectors."""
    gamma = rng.uniform(-alpha, 1 + alpha, size=parent_a.shape)
    child_a = gamma * parent_a + (1 - gamma) * parent_b
    child_b = gamma * parent_b + (1 - gamma) * parent_a
    return child_a, child_b


def gaussian_mutation(individual: np.ndarray, rng: np.random.Generator, mutation_rate: float, sigma: float, bounds: tuple[float, float]) -> np.ndarray:
    """Apply per-gene Gaussian mutation and clip to bounds."""
    mutant = individual.copy()
    mask = rng.random(size=mutant.shape) < mutation_rate
    mutant[mask] += rng.normal(0.0, sigma, size=np.sum(mask))
    return np.clip(mutant, bounds[0], bounds[1])


def one_point_crossover_bits(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One-point crossover for bit strings."""
    if parent_a.size < 2:
        return parent_a.copy(), parent_b.copy()
    point = int(rng.integers(1, parent_a.size))
    return (
        np.concatenate([parent_a[:point], parent_b[point:]]),
        np.concatenate([parent_b[:point], parent_a[point:]]),
    )


def bit_flip_mutation(bits: np.ndarray, rng: np.random.Generator, mutation_rate: float) -> np.ndarray:
    mutant = bits.copy()
    mask = rng.random(mutant.shape) < mutation_rate
    mutant[mask] = 1 - mutant[mask]
    return mutant
