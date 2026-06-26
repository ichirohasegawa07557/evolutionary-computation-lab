from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fitness import get_fitness_function
from src.ga import GAConfig, GeneticAlgorithm
from src.visualize import plot_fitness_curve, save_best_solution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run from-scratch genetic algorithm demo.")
    parser.add_argument("--function", default="rastrigin", choices=["sphere", "rastrigin", "rosenbrock"])
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    cfg = GAConfig(
        population_size=args.population_size,
        dimensions=args.dimensions,
        generations=args.generations,
        seed=args.seed,
    )
    ga = GeneticAlgorithm(get_fitness_function(args.function), cfg)
    result = ga.run()

    history_path = results_dir / f"ga_{args.function}_history.csv"
    plot_path = results_dir / f"ga_{args.function}_fitness_curve.png"
    solution_path = results_dir / f"ga_{args.function}_best_solution.json"

    result.history.to_csv(history_path, index=False)
    plot_fitness_curve(result.history, plot_path, title=f"Genetic Algorithm on {args.function.title()}")
    save_best_solution(solution_path, result.best_solution, result.best_fitness, {"function": args.function})

    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Saved: {history_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {solution_path}")


if __name__ == "__main__":
    main()
