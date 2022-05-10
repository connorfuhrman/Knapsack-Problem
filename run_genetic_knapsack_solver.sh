#!/bin/bash


julia knapsack_solver.jl --genetic \
      --input ./items/items_n350.csv \
      --max_capacity 500 \
      --initial_population_size 3500 \
      --selected_population_size 3500 \
      --p_crossover 0.75 \
      --p_mutate 0.025
