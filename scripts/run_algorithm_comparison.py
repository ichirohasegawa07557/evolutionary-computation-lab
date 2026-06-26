from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fitness import get_fitness_function
from src.ga import GAConfig, GeneticAlgorithm
from src.es import ESConfig, EvolutionStrategy
from src.random_search import RandomSearchConfig, run_random_search
from src.visualize import plot_algorithm_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GA, evolution strategy, and random search.")
    parser.add_argument("--function", default="rastrigin", choices=["sphere", "rastrigin", "rosenbrock"])
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--dimensions", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    fitness_fn = get_fitness_function(args.function)

    ga = GeneticAlgorithm(fitness_fn, GAConfig(generations=args.generations, dimensions=args.dimensions, seed=1)).run()
    es = EvolutionStrategy(fitness_fn, ESConfig(generations=args.generations, dimensions=args.dimensions, seed=2)).run()
    _, random_best, random_history = run_random_search(
        fitness_fn,
        RandomSearchConfig(generations=args.generations, dimensions=args.dimensions, samples_per_generation=60, seed=3),
    )

    random_history = random_history.rename(columns={"best_fitness": "best_fitness"})

    histories = {
        "Genetic Algorithm": ga.history,
        "Evolution Strategy": es.history,
        "Random Search": random_history,
    }

    comparison_path = results_dir / "algorithm_comparison.png"
    plot_algorithm_comparison(histories, comparison_path)

    summary = []
    for name, hist in histories.items():
        summary.append({"algorithm": name, "final_best_fitness": float(hist["best_fitness"].iloc[-1])})
    import pandas as pd

    summary_df = pd.DataFrame(summary)
    summary_path = results_dir / "algorithm_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print(summary_df)
    print(f"Saved: {comparison_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
