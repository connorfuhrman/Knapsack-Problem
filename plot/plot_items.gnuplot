#!/usr/bin/gnuplot

reset


set terminal pngcairo

set output "knapsack_items.png"

set xlabel "Item Value"
set ylabel "Item Weight"
set title "Knapsack Items"

plot '<cat' with points pointtype 3 title ""
