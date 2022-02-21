""" 
The knapsack problem: 
   
Given a set of items, each with a weight and a value, determine the number of each item to 
include in a collection so that the total weight is less than or equal to a given limit and 
the total value is as large as possible.
"""

using Printf
using Random
using Distributions
using DelimitedFiles
using ArgParse

# Struct represents the items available to be placed inside the knapsack
struct Item
    value
    weight
end

# Add another method to Base.show to print the Item struct
Base.show(io::IO, ::MIME"text/plain", i::Item) = @printf(io, "Item with weight %f and value %f\n", i.weight, i.value)

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


function fitness(gene, items, max_weight)
    fit = 0.0
    weight = 0.0
    for (g,i) in zip(gene, items)
        fit += g*i.value
        weight += g*i.weight
    end

    # Return the fitness value or 0 if we're over weight
    return fit * (weight <= max_weight)
end

function make_init_pop(n, num_items)
    pop = [] # TOOD make this a fixed size (we know here)
    for i in 1:n
        push!(pop, bitrand(num_items))
    end

    return pop
end

function tournament_select(pop, fit, n_selected, n_competing)
    # Select n_selected number of genes from the pool pop (with replacement)
    # where n_competing randomly sampled genes compete at once
    selected = []
    for i in 1:n_selected
        idx = rand(1:length(pop), n_competing)
        max_fit_idx = idx[1]
        for i in 2:length(idx)
            if fit[idx[i]] > fit[max_fit_idx]
                max_fit_idx = idx[i]
            end
        end
        push!(selected, pop[max_fit_idx])
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

function optimize(n_init_pop, items, max_weight, tourn_size, select_pop_size, p_cross, p_mutate, exit_tol = 150)
    # Create an initial population 
   pop = make_init_pop(n_init_pop, length(items))
    # Loop until a satisfactory solution is found
    max_fitness = 0.0
    max_weight = 0.0
    n_loops_without_increase = 0
    optimal = nothing
    while true
        # Calculate the fitness for the population
        fit = [fitness(g, items, max_weight) for g in pop]
        i = argmax(fit)
        if fit[i] > max_fitness
            @show max_fitness = fit[i]
            @show optimal = pop[i]
        else
            n_loops_without_increase = n_loops_without_increase + 1
            if n_loops_without_increase > exit_tol break end
        end
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

    opt_weight = 0.0
    for i = 1:length(items)
        opt_weight += items[i].weight * optimal[i]
    end

    if opt_weight > max_weight
        error("Weight exceeds capacity")
    end
    
    return fitness(optimal, items, max_weight), opt_weight
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

    init_pop_size = args["initial_population_size"]
    pop_size = args["selected_population_size"]
    tourn_size = args["tournament_size"]
    p_cross = args["p_crossover"]
    p_mut = args["p_mutate"]

    if args["genetic"]
        # Run the genetic solver
        @timev opt, weight = optimize(init_pop_size, possible_items, args["max_capacity"], tourn_size, pop_size, p_cross, p_mut)
    elseif args["brute_force"]
        # Run the brute force recursive
        @timev opt, weight = brute_force(possible_items, args["max_capacity"])
    else
        error("Neither genetic or the brute force method was selected. Pick one")
    end

    println("Optimal calculated to be $opt with weight $weight")
end


main(ARGS)
