from __future__ import annotations

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Sphere benchmark. Global minimum is 0 at x=0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin benchmark. Multimodal, global minimum is 0 at x=0."""
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock benchmark. Global minimum is 0 at all ones."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def get_fitness_function(name: str):
    functions = {
        "sphere": sphere,
        "rastrigin": rastrigin,
        "rosenbrock": rosenbrock,
    }
    try:
        return functions[name]
    except KeyError as exc:
        raise ValueError(f"Unknown fitness function: {name}") from exc
