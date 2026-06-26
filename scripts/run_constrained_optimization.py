from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.constraints import run_constrained_ga, constraint_violations
from src.visualize import plot_constrained_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained optimization with penalty functions.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--penalty-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    result = run_constrained_ga(
        generations=args.generations,
        population_size=args.population_size,
        penalty_weight=args.penalty_weight,
        seed=args.seed,
    )

    history_path = results_dir / "constrained_ga_history.csv"
    plot_path = results_dir / "constrained_ga_fitness_curve.png"
    best_path = results_dir / "constrained_ga_best_solution.json"

    result.history.to_csv(history_path, index=False)
    plot_constrained_history(result.history, plot_path)
    payload = {
        "best_solution": result.best_solution.tolist(),
        "objective": result.best_objective,
        "total_violation": result.total_violation,
        "penalized_fitness": result.penalized_fitness,
        "violations": constraint_violations(result.best_solution),
    }
    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(payload)
    print(f"Saved: {history_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
