from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evolve_ca import CARuleEvolution, CARuleEvolutionConfig
from src.ca import simulate_ca, target_triangle
from src.visualize import plot_ca_grid, plot_fitness_curve, make_ca_rule_gif


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve elementary CA rule toward a target pattern.")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    cfg = CARuleEvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
    )
    result = CARuleEvolution(cfg).run()

    result.history.to_csv(results_dir / "ca_rule_evolution_history.csv", index=False)
    plot_fitness_curve(result.history, results_dir / "ca_rule_evolution_fitness_curve.png", title="Evolutionary Search for Cellular Automaton Rule")

    target = target_triangle(cfg.width, cfg.steps)
    best_grid = simulate_ca(result.best_rule, cfg.width, cfg.steps)
    plot_ca_grid(target, results_dir / "ca_target_rule90.png", title="Target Pattern: Rule 90")
    plot_ca_grid(best_grid, results_dir / "ca_evolved_rule.png", title=f"Evolved Rule: {result.best_rule}")
    make_ca_rule_gif(result.best_rule, results_dir / "ca_evolved_rule.gif", width=cfg.width, steps=cfg.steps)

    print(f"Best CA rule: {result.best_rule}")
    print(f"Best pattern mismatch: {result.best_fitness:.6f}")
    print("Saved CA evolution results in results/")


if __name__ == "__main__":
    main()
