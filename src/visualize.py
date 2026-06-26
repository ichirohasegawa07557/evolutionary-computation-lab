from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.ca import simulate_ca


def save_best_solution(path: str | Path, solution: np.ndarray, fitness: float, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_fitness": float(fitness),
        "best_solution": np.asarray(solution).tolist(),
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_fitness_curve(history: pd.DataFrame, output_path: str | Path, title: str = "Evolutionary Optimization") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["generation"], history["best_fitness"], label="best")
    if "mean_fitness" in history.columns:
        plt.plot(history["generation"], history["mean_fitness"], label="mean", alpha=0.75)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_algorithm_comparison(histories: dict[str, pd.DataFrame], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history["generation"], history["best_fitness"], label=name)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.yscale("log")
    plt.title("Algorithm Comparison")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_ca_grid(grid: np.ndarray, output_path: str | Path, title: str = "Evolved Cellular Automaton") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    plt.xlabel("Cell")
    plt.ylabel("Time step")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_ca_rule_gif(rule_number: int, output_path: str | Path, width: int = 101, steps: int = 80) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = simulate_ca(rule_number, width=width, steps=steps)
    frames = []
    for t in range(8, steps + 1, 4):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(grid[:t], cmap="binary", interpolation="nearest", aspect="auto")
        ax.set_title(f"Rule {rule_number} | step {t}")
        ax.set_xlabel("Cell")
        ax.set_ylabel("Time")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(output_path, frames, duration=0.12)


def plot_pareto_front(population: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    ranks = sorted(population["rank"].unique())
    for rank in ranks:
        subset = population[population["rank"] == rank]
        label = "Pareto front" if rank == 0 else f"rank={rank}"
        alpha = 0.9 if rank == 0 else 0.35
        plt.scatter(subset["objective_1"], subset["objective_2"], label=label, alpha=alpha, s=35)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("NSGA-II Multi-objective Optimization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_gp_fit(fit: pd.DataFrame, output_path: str | Path, title: str = "Genetic Programming Symbolic Regression") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(fit["x"], fit["target"], label="target", linewidth=2)
    plt.plot(fit["x"], fit["prediction"], label="GP prediction", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_constrained_history(history: pd.DataFrame, output_path: str | Path) -> None:
    plot_fitness_curve(history, output_path, title="Constrained GA with Penalty Functions")
