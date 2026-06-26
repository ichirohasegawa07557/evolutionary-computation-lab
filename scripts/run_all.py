from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable
    run([py, "scripts/run_ga_demo.py", "--function", "rastrigin", "--generations", "80", "--population-size", "60"])
    run([py, "scripts/run_algorithm_comparison.py", "--function", "rastrigin", "--generations", "80"])
    run([py, "scripts/run_evolve_ca.py", "--generations", "40", "--population-size", "30"])
    run([py, "scripts/run_island_model.py", "--function", "rastrigin", "--islands", "4", "--generations", "40"])
    run([py, "scripts/run_multiobjective_demo.py", "--generations", "50", "--population-size", "60"])
    run([py, "scripts/run_genetic_programming_demo.py", "--generations", "35", "--population-size", "50"])
    run([py, "scripts/run_migrating_island_model.py", "--function", "rastrigin", "--islands", "4", "--generations", "40"])
    run([py, "scripts/run_constrained_optimization.py", "--generations", "50", "--population-size", "50"])


if __name__ == "__main__":
    main()
