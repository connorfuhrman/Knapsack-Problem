#!/bin/bash


julia knapsack_solver.jl --genetic \
      --input ./items/items_n50.csv \
      --max_capacity 1500 \
      --initial_population_size 2500 \
      --selected_population_size 2500 \
      --tournament_size 100 \
      --p_crossover 0.75 \
      --p_mutate 0.05
