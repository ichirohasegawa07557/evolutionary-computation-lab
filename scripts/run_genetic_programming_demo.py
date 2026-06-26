from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gp import GPConfig, GeneticProgrammingRegressor, save_gp_expression
from src.visualize import plot_fitness_curve, plot_gp_fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run genetic programming symbolic regression.")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    config = GPConfig(
        population_size=args.population_size,
        generations=args.generations,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    result = GeneticProgrammingRegressor(config).run()

    history_path = results_dir / "gp_symbolic_regression_history.csv"
    fit_path = results_dir / "gp_symbolic_regression_fit.csv"
    curve_path = results_dir / "gp_symbolic_regression_fitness_curve.png"
    fit_plot_path = results_dir / "gp_symbolic_regression_fit.png"
    expr_path = results_dir / "gp_best_expression.json"

    result.history.to_csv(history_path, index=False)
    result.fit.to_csv(fit_path, index=False)
    plot_fitness_curve(result.history, curve_path, title="GP Symbolic Regression Fitness")
    plot_gp_fit(result.fit, fit_plot_path)
    save_gp_expression(expr_path, result)

    print("Best expression:", result.best_expression)
    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Saved: {history_path}")
    print(f"Saved: {fit_plot_path}")


if __name__ == "__main__":
    main()
