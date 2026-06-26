# Evolutionary Computation Lab

This repository contains small, local evolutionary computation experiments written for educational and portfolio purposes.

The project focuses on search, optimization, and artificial-life-style rule discovery. It includes a from-scratch genetic algorithm, an evolution strategy, random-search baselines, multi-objective optimization, genetic programming, migrating island models, constrained optimization with penalty functions, and evolutionary search for cellular automaton rules.

## Overview

Implemented experiments:

- From-scratch Genetic Algorithm for continuous optimization
- Evolution Strategy baseline
- Random Search baseline
- Algorithm comparison plots
- Island-model evolutionary search
- Real ring migration between islands
- Multi-objective optimization with a compact NSGA-II style implementation
- Genetic programming for symbolic expression discovery
- Constraint handling with quadratic penalty functions
- Evolutionary search for elementary cellular automaton rules
- Fitness curves, CSV summaries, JSON best solutions, PNG plots, and GIF animation
- Streamlit viewer for interactive GA experiments
- Pytest-based tests

## Project Structure

```text
evolutionary-computation-lab/
├── README.md
├── requirements.txt
├── requirements-minimal.txt
├── app.py
├── src/
│   ├── fitness.py
│   ├── operators.py
│   ├── ga.py
│   ├── es.py
│   ├── random_search.py
│   ├── island.py
│   ├── multiobjective.py
│   ├── gp.py
│   ├── constraints.py
│   ├── ca.py
│   ├── evolve_ca.py
│   └── visualize.py
├── scripts/
│   ├── run_ga_demo.py
│   ├── run_algorithm_comparison.py
│   ├── run_evolve_ca.py
│   ├── run_island_model.py
│   ├── run_migrating_island_model.py
│   ├── run_multiobjective_demo.py
│   ├── run_genetic_programming_demo.py
│   ├── run_constrained_optimization.py
│   └── run_all.py
├── results/
├── tests/
└── notebooks/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For a lighter install without Streamlit:

```bash
pip install -r requirements-minimal.txt
```

## Run Tests

```bash
python -m pytest -q
```

Expected result:

```text
12 passed
```

## Run All Experiments

```bash
python scripts/run_all.py
```

This generates output files in `results/`.

## Individual Commands

### Genetic Algorithm

```bash
python scripts/run_ga_demo.py --function rastrigin --generations 120 --population-size 80 --dimensions 10
```

Outputs:

```text
results/ga_rastrigin_history.csv
results/ga_rastrigin_fitness_curve.png
results/ga_rastrigin_best_solution.json
```

### Algorithm Comparison

```bash
python scripts/run_algorithm_comparison.py --function rastrigin --generations 120
```

Outputs:

```text
results/algorithm_comparison.csv
results/algorithm_comparison.png
```

### Multi-objective Optimization

```bash
python scripts/run_multiobjective_demo.py --generations 80 --population-size 80
```

Outputs:

```text
results/multiobjective_pareto_population.csv
results/multiobjective_history.csv
results/multiobjective_pareto_front.png
```

### Genetic Programming for Symbolic Expressions

```bash
python scripts/run_genetic_programming_demo.py --generations 60 --population-size 80
```

Outputs:

```text
results/gp_symbolic_regression_history.csv
results/gp_symbolic_regression_fit.csv
results/gp_symbolic_regression_fitness_curve.png
results/gp_symbolic_regression_fit.png
results/gp_best_expression.json
```

### Migrating Island Model

```bash
python scripts/run_migrating_island_model.py --function rastrigin --islands 4 --generations 80 --migration-interval 10 --migration-size 2
```

Outputs:

```text
results/migrating_island_model_history.csv
results/migrating_island_model_fitness.png
```

### Constrained Optimization with Penalty Functions

```bash
python scripts/run_constrained_optimization.py --generations 80 --population-size 60 --penalty-weight 100
```

Outputs:

```text
results/constrained_ga_history.csv
results/constrained_ga_fitness_curve.png
results/constrained_ga_best_solution.json
```

### Evolve Cellular Automaton Rules

```bash
python scripts/run_evolve_ca.py --generations 60 --population-size 40
```

Outputs:

```text
results/ca_rule_evolution_history.csv
results/ca_rule_evolution_fitness_curve.png
results/ca_target_rule90.png
results/ca_evolved_rule.png
results/ca_evolved_rule.gif
```

### Legacy Independent Island Model

```bash
python scripts/run_island_model.py --function rastrigin --islands 4 --generations 60
```

Outputs:

```text
results/island_model_history.csv
results/island_model_fitness.png
```

### Streamlit Viewer

```bash
streamlit run app.py
```

## Results

### Genetic Algorithm on Rastrigin

![GA Fitness Curve](results/ga_rastrigin_fitness_curve.png)

### Algorithm Comparison

![Algorithm Comparison](results/algorithm_comparison.png)

### Multi-objective Optimization

![Multi-objective Pareto Front](results/multiobjective_pareto_front.png)

### Genetic Programming Symbolic Regression

![GP Symbolic Regression Fit](results/gp_symbolic_regression_fit.png)

### Migrating Island Model

![Migrating Island Model](results/migrating_island_model_fitness.png)

### Constrained Optimization

![Constrained GA](results/constrained_ga_fitness_curve.png)

### Evolving Cellular Automaton Rules

Target pattern:

![CA Target](results/ca_target_rule90.png)

Evolved rule:

![Evolved CA Rule](results/ca_evolved_rule.png)

Evolved CA animation:

![Evolved CA Rule GIF](results/ca_evolved_rule.gif)

### Legacy Independent Island Model

![Island Model](results/island_model_fitness.png)

## Method

The main optimizer is a from-scratch real-valued genetic algorithm:

1. Initialize a population of candidate vectors
2. Evaluate fitness using benchmark functions
3. Select parents using tournament selection
4. Recombine parents using blend crossover
5. Mutate children with Gaussian noise
6. Preserve a small elite set
7. Track best fitness, mean fitness, and diversity over generations

The multi-objective optimizer uses a compact NSGA-II style loop with non-dominated sorting and crowding distance. It demonstrates how a population can approximate a Pareto front instead of optimizing a single scalar objective.

The genetic programming experiment evolves expression trees for symbolic regression. Candidate programs are composed of arithmetic operators, constants, and the variable `x`. Fitness is based on prediction error with a small parsimony penalty.

The migrating island model maintains several separate populations and periodically sends elite individuals from each island to the next island in a ring topology. This is different from independent repeated runs because information actually moves between islands during the run.

The constrained optimization experiment uses quadratic penalty functions. It minimizes a base objective while adding a penalty term when constraints are violated.

The cellular automaton experiment evolves elementary cellular automaton rule numbers. Each rule is evaluated by comparing its generated space-time pattern with a target pattern. This connects evolutionary computation with artificial life and local-rule discovery.

## Implemented Extensions

This expanded version includes:

- Multi-objective optimization
- Genetic programming for symbolic expressions
- Real migration between islands
- Constraint handling and penalty functions
- Evolution Strategy baseline
- Random Search baseline
- Algorithm comparison plots
- Island-model evolutionary search
- Evolutionary search for cellular automaton rules
- GIF animation of evolved CA rule growth
- Streamlit interactive viewer
- CSV, JSON, PNG, and GIF outputs for GitHub display
- Pytest tests for core components

## Limitations

This is an educational implementation, not a production-grade optimization library.

Current limitations:

- The GA uses simple tournament selection and blend crossover
- The NSGA-II implementation is compact and intended for small local demonstrations
- The genetic programming system uses a small operator set and no advanced simplification
- The constrained optimization demo uses quadratic penalties rather than a full constrained optimizer
- The CA rule search uses elementary one-dimensional cellular automata only
- Hyperparameters are manually chosen for small local experiments
- Results are stochastic and may vary by seed

## Future Work

Further possible extensions:

- Add more benchmark functions
- Add experiment configuration files
- Add GitHub Actions CI
- Add larger benchmark runs with repeated seeds
- Add symbolic simplification for genetic programming outputs
- Add additional migration topologies such as star and fully connected islands

## Why This Project?

Evolutionary computation is a useful bridge between optimization, artificial life, and machine learning. This repository demonstrates how population-based search can discover good solutions, optimize benchmark functions, approximate Pareto fronts, evolve symbolic expressions, handle constrained problems, and search for local update rules that generate structured behavior.
