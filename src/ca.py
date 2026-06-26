from __future__ import annotations

import numpy as np


def decode_rule(rule_number: int) -> np.ndarray:
    """Decode elementary cellular automaton rule number into 8 output bits."""
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)


def step_ca(state: np.ndarray, rule_bits: np.ndarray) -> np.ndarray:
    left = np.roll(state, 1)
    center = state
    right = np.roll(state, -1)
    neighborhood = (left << 2) | (center << 1) | right
    return rule_bits[neighborhood]


def simulate_ca(rule_number: int, width: int = 101, steps: int = 80, seed_index: int | None = None) -> np.ndarray:
    rule_bits = decode_rule(rule_number)
    grid = np.zeros((steps, width), dtype=np.uint8)
    if seed_index is None:
        seed_index = width // 2
    grid[0, seed_index] = 1
    for t in range(1, steps):
        grid[t] = step_ca(grid[t - 1], rule_bits)
    return grid


def target_triangle(width: int = 101, steps: int = 80) -> np.ndarray:
    """Create a Sierpinski-like triangle target using rule 90."""
    return simulate_ca(90, width=width, steps=steps)


def ca_fitness(rule_number: int, target: np.ndarray) -> float:
    candidate = simulate_ca(rule_number, width=target.shape[1], steps=target.shape[0])
    return float(np.mean(candidate != target))
