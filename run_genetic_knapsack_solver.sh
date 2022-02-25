#!/bin/bash


julia knapsack_solver.jl --genetic \
      --input ./items/items_n350.csv \
      --max_capacity 1500 \
      --initial_population_size 35000 \
      --selected_population_size 35000 \
      --p_crossover 0.75 \
      --p_mutate 0.025
