from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.multiobjective import NSGAII, NSGAIIConfig
from src.visualize import plot_pareto_front


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NSGA-II style multi-objective optimization.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    config = NSGAIIConfig(
        population_size=args.population_size,
        dimensions=args.dimensions,
        generations=args.generations,
        seed=args.seed,
    )
    result = NSGAII(config=config).run()

    population_path = results_dir / "multiobjective_pareto_population.csv"
    history_path = results_dir / "multiobjective_history.csv"
    plot_path = results_dir / "multiobjective_pareto_front.png"

    result.population.to_csv(population_path, index=False)
    result.history.to_csv(history_path, index=False)
    plot_pareto_front(result.population, plot_path)

    print(result.population.sort_values(["rank", "objective_1"]).head(12))
    print(f"Saved: {population_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
