from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.operators import blend_crossover, gaussian_mutation


def bi_objective_sphere(x: np.ndarray) -> tuple[float, float]:
    """Two-objective benchmark used for the Pareto-front demo.

    Objective 1 is minimized near x=0. Objective 2 is minimized near x=2.
    The trade-off creates a simple Pareto set between those regions.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2)), float(np.sum((x - 2.0) ** 2))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if objective vector a Pareto-dominates b for minimization."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Small from-scratch non-dominated sorting implementation."""
    n = len(objectives)
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    """Compute crowding distance for a Pareto front."""
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distance = {idx: 0.0 for idx in front}
    front_obj = objectives[front]
    num_objectives = front_obj.shape[1]

    for m in range(num_objectives):
        values = front_obj[:, m]
        order = np.argsort(values)
        sorted_indices = [front[i] for i in order]
        distance[sorted_indices[0]] = float("inf")
        distance[sorted_indices[-1]] = float("inf")
        v_min = float(values[order[0]])
        v_max = float(values[order[-1]])
        denom = max(v_max - v_min, 1e-12)
        for j in range(1, len(sorted_indices) - 1):
            prev_v = float(values[order[j - 1]])
            next_v = float(values[order[j + 1]])
            distance[sorted_indices[j]] += (next_v - prev_v) / denom

    return distance


@dataclass
class NSGAIIConfig:
    population_size: int = 80
    dimensions: int = 2
    generations: int = 80
    bounds: tuple[float, float] = (-1.0, 3.0)
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.12
    crossover_rate: float = 0.9
    seed: int = 7


@dataclass
class NSGAIIResult:
    population: pd.DataFrame
    history: pd.DataFrame


class NSGAII:
    """Compact educational NSGA-II style multi-objective optimizer."""

    def __init__(self, objective_fn=bi_objective_sphere, config: NSGAIIConfig | None = None):
        self.objective_fn = objective_fn
        self.config = config or NSGAIIConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _initialize(self) -> np.ndarray:
        low, high = self.config.bounds
        return self.rng.uniform(low, high, size=(self.config.population_size, self.config.dimensions))

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.asarray([self.objective_fn(ind) for ind in population], dtype=float)

    def _rank_and_crowding(self, objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fronts = fast_non_dominated_sort(objectives)
        ranks = np.full(len(objectives), fill_value=999999, dtype=int)
        crowding = np.zeros(len(objectives), dtype=float)
        for rank, front in enumerate(fronts):
            distances = crowding_distance(objectives, front)
            for idx in front:
                ranks[idx] = rank
                crowding[idx] = distances[idx]
        return ranks, crowding

    def _binary_tournament(self, population: np.ndarray, ranks: np.ndarray, crowding: np.ndarray) -> np.ndarray:
        a, b = self.rng.integers(0, len(population), size=2)
        if ranks[a] < ranks[b]:
            return population[a].copy()
        if ranks[b] < ranks[a]:
            return population[b].copy()
        if crowding[a] >= crowding[b]:
            return population[a].copy()
        return population[b].copy()

    def _make_offspring(self, population: np.ndarray, ranks: np.ndarray, crowding: np.ndarray) -> np.ndarray:
        cfg = self.config
        children: list[np.ndarray] = []
        while len(children) < cfg.population_size:
            p1 = self._binary_tournament(population, ranks, crowding)
            p2 = self._binary_tournament(population, ranks, crowding)
            if self.rng.random() < cfg.crossover_rate:
                c1, c2 = blend_crossover(p1, p2, self.rng, alpha=0.25)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = gaussian_mutation(c1, self.rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)
            c2 = gaussian_mutation(c2, self.rng, cfg.mutation_rate, cfg.mutation_sigma, cfg.bounds)
            children.append(c1)
            if len(children) < cfg.population_size:
                children.append(c2)
        return np.asarray(children)

    def _survival_select(self, combined: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        cfg = self.config
        fronts = fast_non_dominated_sort(objectives)
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= cfg.population_size:
                selected.extend(front)
            else:
                distances = crowding_distance(objectives, front)
                remaining = cfg.population_size - len(selected)
                ordered = sorted(front, key=lambda idx: distances[idx], reverse=True)
                selected.extend(ordered[:remaining])
                break
        return combined[selected]

    def run(self) -> NSGAIIResult:
        cfg = self.config
        population = self._initialize()
        history_records = []

        for generation in range(cfg.generations + 1):
            objectives = self._evaluate(population)
            fronts = fast_non_dominated_sort(objectives)
            first_front = fronts[0]
            history_records.append({
                "generation": generation,
                "pareto_front_size": len(first_front),
                "min_objective_1": float(np.min(objectives[:, 0])),
                "min_objective_2": float(np.min(objectives[:, 1])),
                "mean_objective_1": float(np.mean(objectives[:, 0])),
                "mean_objective_2": float(np.mean(objectives[:, 1])),
            })
            if generation == cfg.generations:
                break
            ranks, crowding = self._rank_and_crowding(objectives)
            offspring = self._make_offspring(population, ranks, crowding)
            combined = np.vstack([population, offspring])
            combined_objectives = self._evaluate(combined)
            population = self._survival_select(combined, combined_objectives)

        objectives = self._evaluate(population)
        ranks, crowding = self._rank_and_crowding(objectives)
        records = []
        for i, ind in enumerate(population):
            row = {f"x{j}": float(value) for j, value in enumerate(ind)}
            row.update({
                "objective_1": float(objectives[i, 0]),
                "objective_2": float(objectives[i, 1]),
                "rank": int(ranks[i]),
                "crowding_distance": float(crowding[i]),
            })
            records.append(row)
        return NSGAIIResult(population=pd.DataFrame(records), history=pd.DataFrame(history_records))
