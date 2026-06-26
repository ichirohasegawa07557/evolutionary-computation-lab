from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Tree = Any

BINARY_OPS = ("add", "sub", "mul")
UNARY_OPS = ("sin", "cos")
TERMINALS = ("x",)


def target_function(x: np.ndarray) -> np.ndarray:
    """Symbolic-regression target used by the GP demo."""
    return x ** 2 + x + 1.0


def safe_eval(tree: Tree, x: np.ndarray) -> np.ndarray:
    """Evaluate an expression tree on x values."""
    if tree == "x":
        return x
    if isinstance(tree, (int, float)):
        return np.full_like(x, float(tree), dtype=float)
    op = tree[0]
    if op == "const":
        return np.full_like(x, float(tree[1]), dtype=float)
    if op == "add":
        return safe_eval(tree[1], x) + safe_eval(tree[2], x)
    if op == "sub":
        return safe_eval(tree[1], x) - safe_eval(tree[2], x)
    if op == "mul":
        return safe_eval(tree[1], x) * safe_eval(tree[2], x)
    if op == "sin":
        return np.sin(safe_eval(tree[1], x))
    if op == "cos":
        return np.cos(safe_eval(tree[1], x))
    raise ValueError(f"Unknown GP op: {op}")


def tree_size(tree: Tree) -> int:
    if tree == "x" or isinstance(tree, (int, float)):
        return 1
    if tree[0] == "const":
        return 1
    if tree[0] in UNARY_OPS:
        return 1 + tree_size(tree[1])
    return 1 + tree_size(tree[1]) + tree_size(tree[2])


def tree_to_string(tree: Tree) -> str:
    if tree == "x":
        return "x"
    if isinstance(tree, (int, float)):
        return f"{float(tree):.3f}"
    if tree[0] == "const":
        return f"{float(tree[1]):.3f}"
    if tree[0] == "add":
        return f"({tree_to_string(tree[1])} + {tree_to_string(tree[2])})"
    if tree[0] == "sub":
        return f"({tree_to_string(tree[1])} - {tree_to_string(tree[2])})"
    if tree[0] == "mul":
        return f"({tree_to_string(tree[1])} * {tree_to_string(tree[2])})"
    if tree[0] == "sin":
        return f"sin({tree_to_string(tree[1])})"
    if tree[0] == "cos":
        return f"cos({tree_to_string(tree[1])})"
    raise ValueError(f"Unknown tree: {tree}")


def random_tree(rng: np.random.Generator, max_depth: int) -> Tree:
    if max_depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.65:
            return "x"
        return ("const", float(rng.uniform(-3.0, 3.0)))
    if rng.random() < 0.75:
        op = str(rng.choice(BINARY_OPS))
        return (op, random_tree(rng, max_depth - 1), random_tree(rng, max_depth - 1))
    op = str(rng.choice(UNARY_OPS))
    return (op, random_tree(rng, max_depth - 1))


def get_subtree(tree: Tree, path: tuple[int, ...]) -> Tree:
    node = tree
    for step in path:
        node = node[step]
    return node


def replace_subtree(tree: Tree, path: tuple[int, ...], replacement: Tree) -> Tree:
    if not path:
        return replacement
    if tree == "x" or isinstance(tree, (int, float)) or (isinstance(tree, tuple) and tree[0] == "const"):
        return replacement
    tree_list = list(tree)
    step = path[0]
    tree_list[step] = replace_subtree(tree_list[step], path[1:], replacement)
    return tuple(tree_list)


def subtree_paths(tree: Tree, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if isinstance(tree, tuple):
        if tree[0] in UNARY_OPS:
            paths.extend(subtree_paths(tree[1], prefix + (1,)))
        elif tree[0] in BINARY_OPS:
            paths.extend(subtree_paths(tree[1], prefix + (1,)))
            paths.extend(subtree_paths(tree[2], prefix + (2,)))
    return paths


def mutate_tree(tree: Tree, rng: np.random.Generator, max_depth: int) -> Tree:
    path = subtree_paths(tree)[int(rng.integers(0, len(subtree_paths(tree))))]
    return replace_subtree(tree, path, random_tree(rng, max_depth=max_depth))


def crossover_trees(a: Tree, b: Tree, rng: np.random.Generator) -> tuple[Tree, Tree]:
    path_a = subtree_paths(a)[int(rng.integers(0, len(subtree_paths(a))))]
    path_b = subtree_paths(b)[int(rng.integers(0, len(subtree_paths(b))))]
    sub_a = get_subtree(a, path_a)
    sub_b = get_subtree(b, path_b)
    return replace_subtree(a, path_a, sub_b), replace_subtree(b, path_b, sub_a)


@dataclass
class GPConfig:
    population_size: int = 80
    generations: int = 60
    max_depth: int = 4
    mutation_rate: float = 0.25
    crossover_rate: float = 0.85
    tournament_size: int = 3
    parsimony_weight: float = 0.001
    seed: int = 23


@dataclass
class GPResult:
    best_tree: Tree
    best_expression: str
    best_fitness: float
    history: pd.DataFrame
    fit: pd.DataFrame


class GeneticProgrammingRegressor:
    """Small tree-based genetic programming system for symbolic regression."""

    def __init__(self, config: GPConfig | None = None):
        self.config = config or GPConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.x = np.linspace(-2.0, 2.0, 80)
        self.y = target_function(self.x)

    def _fitness(self, tree: Tree) -> float:
        try:
            y_hat = safe_eval(tree, self.x)
            if not np.all(np.isfinite(y_hat)):
                return float("inf")
            mse = float(np.mean((y_hat - self.y) ** 2))
        except Exception:
            return float("inf")
        return mse + self.config.parsimony_weight * tree_size(tree)

    def _select(self, population: list[Tree], fitness: np.ndarray) -> Tree:
        indices = self.rng.integers(0, len(population), size=self.config.tournament_size)
        best_index = int(indices[np.argmin(fitness[indices])])
        return population[best_index]

    def run(self) -> GPResult:
        cfg = self.config
        population = [random_tree(self.rng, cfg.max_depth) for _ in range(cfg.population_size)]
        best_tree = population[0]
        best_fitness = float("inf")
        records = []

        for generation in range(cfg.generations + 1):
            fitness = np.asarray([self._fitness(tree) for tree in population], dtype=float)
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_fitness:
                best_fitness = float(fitness[best_idx])
                best_tree = population[best_idx]
            records.append({
                "generation": generation,
                "best_fitness": float(np.min(fitness)),
                "mean_fitness": float(np.mean(fitness[np.isfinite(fitness)])),
                "best_size": int(tree_size(population[best_idx])),
            })
            if generation == cfg.generations:
                break

            next_population = [best_tree]
            while len(next_population) < cfg.population_size:
                p1 = self._select(population, fitness)
                p2 = self._select(population, fitness)
                if self.rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover_trees(p1, p2, self.rng)
                else:
                    c1, c2 = p1, p2
                if self.rng.random() < cfg.mutation_rate:
                    c1 = mutate_tree(c1, self.rng, cfg.max_depth)
                if self.rng.random() < cfg.mutation_rate:
                    c2 = mutate_tree(c2, self.rng, cfg.max_depth)
                next_population.append(c1)
                if len(next_population) < cfg.population_size:
                    next_population.append(c2)
            population = next_population

        y_hat = safe_eval(best_tree, self.x)
        fit = pd.DataFrame({"x": self.x, "target": self.y, "prediction": y_hat})
        return GPResult(
            best_tree=best_tree,
            best_expression=tree_to_string(best_tree),
            best_fitness=best_fitness,
            history=pd.DataFrame(records),
            fit=fit,
        )


def save_gp_expression(path: str | Path, result: GPResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_expression": result.best_expression,
        "best_fitness": float(result.best_fitness),
        "tree_size": tree_size(result.best_tree),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
