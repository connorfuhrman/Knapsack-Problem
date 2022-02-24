""" 
The knapsack problem: 
   
Given a set of items, each with a weight and a value, determine the number of each item to 
include in a collection so that the total weight is less than or equal to a given limit and 
the total value is as large as possible.
"""

push!(LOAD_PATH, pwd()) # Add the current path so we can load other files
using GeneticTypes, GeneticFuncs

using Printf
using Random
using Distributions
using DelimitedFiles
using ArgParse


# Struct represents the items available to be placed inside the knapsack
# where the value and weight are both floating-point values
struct Item
    value::Float64
    weight::Float64

    Item(v, w) = new(v, w)
end

# Add another method to Base.show to print the Item struct
Base.show(io::IO, ::MIME"text/plain", i::Item) = @printf(io, "(weight: %f, value: %f)\n", i.weight, i.value)


# Method to read items from a config file in the format 'value, weight'.
# Returns an array of Items
function read_items_file(fpath)
    data = readdlm(fpath, ',', Float64, '\n')
    items = []
    for i = 1:size(data)[1]
        push!(items, Item(data[i,1], data[i,2]))
    end
    return items
end


function weight_calc(gene, items)
    w = 0.0
    for i in 1:length(gene)
        w += gene[i] * items[i].weight
    end
    return w
end


function fitness(gene, items, max_weight)
    fit = 0.0
    weight = 0.0
    for i in 1:length(items)
        fit += gene[i] * items[i].value
        weight += gene[i] * items[i].weight
    end
#    for (g,i) in zip(gene, items)
#        fit += g*i.value
#        weight += g*i.weight
#    end

    # Return the fitness value or 0 if we're over weight

    return fit * (weight <= max_weight)
end

function make_init_pop(N::Int64, num_items::Int64)
    pop = [Gene{Bool}(N) for i in 1:num_items]

    # Push zeros, ones, and 1 of each for each item to the pool
    push!(pop, Gene{Bool}(BitVector(zeros(num_items))))
    push!(pop, Gene{Bool}(BitVector(ones(num_items))))

    for i in 1:num_items
        to_add = BitVector(zeros(num_items))
        to_add[i] = true
        push!(pop, Gene{Bool}(to_add))
    end

    return pop
end


# Sort the population acording to the fitness
function sort_population(pop, _fitness)
    sorted = similar(pop)
    fitness = copy(_fitness)
    row = 1
    while length(fitness) > 0
        i = argmax(fitness)
        deleteat!(fitness, i)
        sorted[row,:] = pop[i,:]
        row += 1
    end
    return sorted
end

""" Function to write the population to a text file 

    Accepts arguments (1) the population and (2) the filename to write """
function write_population(pop, filename)
    open(filename, "w") do f
        writedlm(f, 1*pop)
    end
end

function debug_items(items)
    weights = [it.weight for it in items]
    values = [it.value for it in items]

    @printf "Items summary for %d items:\n" length(items)
    @printf "\tWeight summary:\n"
    @printf "\t\tMax: %f\n" maximum(weights)
    @printf "\t\tMin: %f\n" minimum(weights)
    @printf "\t\tMean: %f\n" mean(weights)
    @printf "\tValues summary:\n"
    @printf "\t\tMax: %f\n" maximum(values)
    @printf "\t\tMin: %f\n" minimum(values)
    @printf "\t\tMean: %f\n" mean(values)
end

function debug_population(pop, fitness, items)
    
end

function tournament_select(pop, fit, n_selected, n_competing)
    selected = similar(pop)
    for i in 1:n_selected
        idx = rand(1:length(pop), n_competing)
        max_fit_idx = idx[1]
        for i in 2:length(idx)
            if fit[idx[i]] > fit[max_fit_idx]
                max_fit_idx = idx[i]
            end
        end
        # Add the winning gene to the new pool
        push!(selected, pop[max_fit_idx])
        # Remove the gene from the old pool
        deleteat!(pop, max_fit_idx)
    end

    return selected
end

function do_crossover(p1, p2, pc)
    coin = Bernoulli(pc)
    if rand(coin)
        c1 = copy(p1)
        c2 = copy(p2)
        split_idx = rand(eachindex(c1))
        for i = 1:split_idx
            c1[i] = p2[i]
            c2[i] = p1[i]
        end
        return c1, c2
    else
        return p1, p2
    end
end

function crossover(pop, p_cross)
    new_pop = []
    while  length(pop) >= 2
        # Select at random 2 genes (without replacement), perform crossover with some
        # probability, and add the resulting two genes back to the pool
        parent1 = splice!(pop, rand(eachindex(pop)))
        parent2 = splice!(pop, rand(eachindex(pop)))
        child1, child2 = do_crossover(parent1, parent2, p_cross)
        push!(new_pop, child1)
        push!(new_pop, child2)
    end

    # Place the "date mike" genes who did *not* get lucky back into the gene pool to try again
    # next time
    for g in pop push!(new_pop, g) end
    
    return new_pop
end

function mutation(pop, p_mutate)
    coin = Bernoulli(p_mutate)
    for ip = 1:length(pop)
        for ig = 1:length(pop[1])
            if rand(coin)
                pop[ip][ig] = !pop[ip][ig]
            end
        end
    end
    return pop
end

function optimize(n_init_pop, items, max_capacity, tourn_size, select_pop_size, p_cross, p_mutate, exit_tol = 150, save_population_frames = false)
    # Create an initial population 
    pop = make_init_pop(n_init_pop, length(items))
#    for g in pop
#        @show g
#    end
    # Loop until a satisfactory solution is found
    max_fitness = 0.0
    opt_weight = 0.0
    n_loops_without_increase = 0
    optimal = nothing
    generation = 0

    debug_items(items)
    
    while true
        # Display some information about the population
        #debug_population(pop, items)
        
        # Calculate the fitness for the population
        fit = [fitness(g, items, max_capacity) for g in pop]
        i = argmax(fit)
        if fit[i] > max_fitness
            @show max_fitness = fit[i]
            optimal = pop[i]
            @show opt_weight = weight_calc(optimal, items)
            @assert(opt_weight <= max_capacity)
            n_loops_without_increase = 0
        else
            n_loops_without_increase += 1

            # Exit condition
            if n_loops_without_increase > exit_tol break end
        end
        if save_population_frames
            # Sort the population according to fitness 
            sorted_pop = sort_population(pop, fit)
            # Write this generation to a file and plot via the GNUplot script
            filename = ".knapsack_data/data.txt"
            write_population(sorted_pop, filename)
            img_filename = @sprintf ".knapsack_data/generation_%05i.png" generation
            run(pipeline(`cat $filename`, `gnuplot -e "filename='$img_filename'" plot/plot_population.gnuplot`))
        end
        generation += 1
        if generation % 10 == 0 println("Generation: $generation") end
        # Now based on that fitness select the genes which will move on
        pop = tournament_select(pop, fit, select_pop_size, tourn_size)
        # Perform crossover
        pop = crossover(pop, p_cross)
        # Perform mutations
        pop = mutation(pop, p_mutate)
    end

    if optimal == nothing
        error("Was never able to find a fitness > 0")
    end

    if opt_weight > max_capacity
        @show opt_weight
        error("Weight exceeds capacity")
    end
    
    return max_fitness, opt_weight
end


# Brute force solution taken from Python implimentation at
# https://www.educative.io/blog/0-1-knapsack-problem-dynamic-solution
function brute_force_recursive(items, remaining_capacity, total_weight, index)
    # Recursive exit condition
    if remaining_capacity <= 0.0 || index > length(items)
        return 0.0, total_weight
    end

    # Chose the element at the current index. If the weight exceeds the capaity then
    # we can't carry this item so don't consider
    val1 = 0.0
    weight_with = total_weight
    if items[index].weight <= remaining_capacity
        val1, weight_with = brute_force_recursive(items, remaining_capacity-items[index].weight, total_weight + items[index].weight, index+1)
        val1 += items[index].value
    end

    # Exclude the current element and consider everything else
    val2, weight_without = brute_force_recursive(items, remaining_capacity, total_weight, index+1)
    if val1 == val2
        return val1, min(weight_with, weight_without)
    elseif val1 > val2
        return val1, weight_with
    else
        return val2, weight_without
    end
end

function brute_force(items, max_weight)
    return brute_force_recursive(items, max_weight, 0.0, 1)
end


# Taken from https://rosettacode.org/wiki/Knapsack_problem/0-1#Python
function items_value(items, max_weight)
    val = 0.0
    weight = 0
    for i in items
        val += i.value
        weight += i.weight
        if weight > max_weight return 0 end
    end
    return val
end

function dynamic_recursive(items, max_weight, cache)
    if length(items) == 0
        return []
    end

    # If we've not cached this value run the calculation
    if !haskey(cache, (items, max_weight))
        head = items[1]
        tail = items[2:end]

        include = vcat(head, dynamic_recursive(tail, max_weight - head.weight, cache))
        dont_include = dynamic_recursive(tail, max_weight, cache)

        if items_value(include, max_weight) > items_value(dont_include, max_weight)
            ans = include
        else
            ans = dont_include
        end

        cache[(items, max_weight)] = ans
    end

    return cache[(items, max_weight)]
    
end

function dynamic(items, max_weight)
    cache = Dict()
    selected = dynamic_recursive(items, max_weight, cache)
    v, w = 0.0, 0
    for s in selected
        v += s.value
        w += s.weight
    end
    return v, w
end


function main(args)
    # Are we running the genetic optimier or the brute force method?
    parser = ArgParseSettings(description = "Knapsack Problem solved either via the genetic optimizer or brute force")

    @add_arg_table! parser begin
        "--genetic"
        action = :store_true
        help = "Run the genetic optimizer for the problem" 
        "--brute_force"
        action = :store_true
        help = "Run the brute force optimizer"
        "--dynamic"
        action = :store_true
        help = "Run the dynamic programming brute-force solver"
        "--input"
        required = true
        help = "Input file"
        "--max_capacity"
        default = 15.0
        arg_type = Float64
        help = "The maximum capacity of the knapsack"
        "--initial_population_size"
        default = 1000
        arg_type = Int64
        help = "The size of the initial population"
        "--selected_population_size"
        default = 1000
        arg_type = Int64
        help = "The size of the population selected"
        "--tournament_size"
        default = 25
        arg_type = Int64
        help = "The number of genes competing in the tournament"
        "--p_crossover"
        default = 0.5
        arg_type = Float64
        help = "The probability of crossover happening"
        "--p_mutate"
        default = 0.05
        arg_type = Float64
        help = "The probability of mutation occuring"
    end

    args = parse_args(parser)

    if args["genetic"] && args["brute_force"] 
        error("Cannot run both the genetic and brute force at once. Pick one")
    end

    # Make the list of items we can put in the knapsack with the (value, weight) value
    possible_items = read_items_file(args["input"])

    if args["genetic"]
        # Run the genetic solver
        @timev opt, weight = optimize(args["initial_population_size"],
                                      possible_items, args["max_capacity"],
                                      args["tournament_size"],
                                      args["selected_population_size"],
                                      args["p_crossover"],
                                      args["p_mutate"])
    elseif args["dynamic"] && args["brute_force"]
        @timev opt, weight = dynamic(possible_items, args["max_capacity"])
    elseif args["brute_force"]
        # Run the brute force recursive
        @timev opt, weight = brute_force(possible_items, args["max_capacity"])
    else
        error("Neither genetic or the brute force method was selected. Pick one")
    end

    println("Optimal calculated to be $opt with weight $weight")
end


main(ARGS)
