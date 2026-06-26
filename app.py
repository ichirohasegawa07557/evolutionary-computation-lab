from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

from src.fitness import get_fitness_function
from src.ga import GAConfig, GeneticAlgorithm


st.set_page_config(page_title="Evolutionary Computation Lab", layout="wide")
st.title("Evolutionary Computation Lab")
st.write("Interactive local viewer for a small from-scratch genetic algorithm.")

with st.sidebar:
    function_name = st.selectbox("Fitness function", ["rastrigin", "sphere", "rosenbrock"])
    dimensions = st.slider("Dimensions", 2, 30, 10)
    generations = st.slider("Generations", 20, 200, 80)
    population_size = st.slider("Population size", 20, 200, 60)
    mutation_rate = st.slider("Mutation rate", 0.01, 0.5, 0.15)
    seed = st.number_input("Seed", value=42, step=1)

if st.button("Run genetic algorithm"):
    cfg = GAConfig(
        population_size=population_size,
        dimensions=dimensions,
        generations=generations,
        mutation_rate=mutation_rate,
        seed=int(seed),
    )
    result = GeneticAlgorithm(get_fitness_function(function_name), cfg).run()
    st.metric("Best fitness", f"{result.best_fitness:.6f}")
    st.line_chart(result.history.set_index("generation")[["best_fitness", "mean_fitness"]])
    st.write("Best solution")
    st.json(result.best_solution.tolist())
else:
    st.info("Choose parameters and run the genetic algorithm.")
