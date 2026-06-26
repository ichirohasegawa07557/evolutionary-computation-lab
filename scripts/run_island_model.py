from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fitness import get_fitness_function
from src.island import run_island_model
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact island-model evolutionary search.")
    parser.add_argument("--function", default="rastrigin", choices=["sphere", "rastrigin", "rosenbrock"])
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--generations", type=int, default=60)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    _, best_fitness, history = run_island_model(
        get_fitness_function(args.function),
        islands=args.islands,
        generations_per_island=args.generations,
    )
    history.to_csv(results_dir / "island_model_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    for island, subset in history.groupby("island"):
        plt.plot(subset["generation"], subset["best_fitness"], label=f"island={island}")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.yscale("log")
    plt.title("Island Model Evolution")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "island_model_fitness.png", dpi=180)
    plt.close()

    print(f"Best island-model fitness: {best_fitness:.6f}")
    print("Saved island model results in results/")


if __name__ == "__main__":
    main()
