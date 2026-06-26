from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.fitness import get_fitness_function
from src.island import run_migrating_island_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run island model with real ring migration.")
    parser.add_argument("--function", default="rastrigin", choices=["sphere", "rastrigin", "rosenbrock"])
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--migration-interval", type=int, default=10)
    parser.add_argument("--migration-size", type=int, default=2)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    _, best_fitness, history = run_migrating_island_model(
        get_fitness_function(args.function),
        islands=args.islands,
        generations=args.generations,
        dimensions=args.dimensions,
        population_size=args.population_size,
        migration_interval=args.migration_interval,
        migration_size=args.migration_size,
    )

    history_path = results_dir / "migrating_island_model_history.csv"
    plot_path = results_dir / "migrating_island_model_fitness.png"
    history.to_csv(history_path, index=False)

    plt.figure(figsize=(8, 5))
    for island, subset in history.groupby("island"):
        plt.plot(subset["generation"], subset["best_fitness"], label=f"island={island}")
    migration_generations = sorted(history.loc[history["migration_event"], "generation"].unique())
    for generation in migration_generations:
        plt.axvline(generation, color="gray", alpha=0.25, linewidth=1)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.yscale("log")
    plt.title("Migrating Island Model")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(f"Best migrating-island fitness: {best_fitness:.6f}")
    print(f"Saved: {history_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
