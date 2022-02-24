push!(LOAD_PATH, pwd()) # Add the current path so we can load other files

using GeneticTypes, GeneticFuncs


using Printf
using Random

addme(x) = x+1

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
    mutation!(g3, 1.0, addme)
    @show g3
    
end

function main2(args)
    @show pop = make_genepool(10, 5)
end

#main1(ARGS)

main2(ARGS)
