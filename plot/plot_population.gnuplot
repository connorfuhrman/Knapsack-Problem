#!/usr/bin/gnuplot


#
# Plot the population for the 0-1 knapsack problem
#
# Adapted from http://www.gnuplotting.org/code/heat_map_interpolation1.gnu

reset



# Configure the png output
set terminal pngcairo size 854,480 enhanced font 'Verdana,10'
set output filename

set border linewidth 0
unset key
unset colorbox
unset tics
set lmargin screen 0.1
set rmargin screen 0.9
set tmargin screen 0.9
set bmargin screen 0.1
set palette grey

set pm3d map
splot '<cat' matrix