push!(LOAD_PATH, pwd()) # Add the current path so we can load other files

using GeneticTypes, GeneticFuncs


using Printf
using Random
using StatsBase

using StatProfilerHTML

# Function to create a random chromosome with some number of 1's
function make_random_chromosome(num_items::Int64, num_true::Int64)::Vector{Bool}
    c = [false for _ in 1:num_items]
    idxs = sample(1:num_items, num_true, replace = false)
    c[idxs] .= true

    return c
end

function make_genepool(N::Int64, num_items::Int64)::Vector{Gene{Bool}}
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


function main1(args)
    @show bg1 = Gene{Bool}(5, bitrand)
    @show bg2 = Gene{Bool}(5, bitrand)
    mutation!(bg1, 1.0)
    @show bg1

    g1 = Gene{Int64}([1,2,3,4,5])
    g2 = Gene{Int64}([6,7,8,9,10])
    
    @show g1
    @show g2

    one_point_crossover!(g1, g2, 1.0)

    @show g1
    @show g2

    g3 = Gene{Int64}([1,2,3,4,5])
    @show g3
    addme(x) = x+1
    mutation!(g3, 1.0, addme)
    @show g3
    
end

function main2(args)
    pop = make_genepool(10, 5)
end

function main3(args)
    G = Gene{Int64}
    @show genes = [G(collect(1:4)), G(collect(5:8)), G(zeros(Int, 4)), G(zeros(Int, 4))]

    @show parents = genes[1:2]
    @show children = genes[3:4]

    println("Doing crossover with p = 1.0")
    @show methods(one_point_crossover!)
    one_point_crossover!(parents, children, 1.0)
    @show parents
    @show children
   
end

#main1(ARGS)
#@timev main2(ARGS)
main3(ARGS)
