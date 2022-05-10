#!/usr/bin/gnuplot

reset

stats filename using 1:4 name 'fit'

set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'

set output "fitness.png"


#set multiplot layout 2,1

set key bottom
set xlabel "Generation"
set ylabel "Fitness value"
set title "Avg, Min, and Max Fitness over Generations"
set xrange [fit_min_x:fit_max_x]
set ytics add (fit_max_y)
plot filename using 1:2 with lines title "Average Fitness", \
     filename using 1:3 with lines title "Minimum Nonzero Fitness", \
     filename using 1:4 with lines title "Maximum Fitness"


set output "zerov_fitness.png"

set multiplot layout 2,1
set title "Percentage of 0-valued fitness"


set key top
set ylabel "Percentage"
set ytics

set xrange[fit_min_x:25]
plot filename using 1:5 with lines title ""

unset title
unset yrange
set xrange [fit_min_x:fit_max_x]
plot filename using 1:5 with lines title ""

