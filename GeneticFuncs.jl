module GeneticFuncs

push!(LOAD_PATH, pwd()) # Add the current path so we can load other files
using GeneticTypes

using Printf
using Random

using Distributions


export one_point_crossover!, mutation!




# Macro to swap the content of two variables given a tmp location 
macro swap(x,y)
   quote
      local tmp = $(esc(x))
      $(esc(x)) = $(esc(y))
      $(esc(y)) = tmp
    end
end

# Function to crossover two genes of the same size. Crossover occurs with some random
# probability provided in the third argument.
#
# Crossover occurs in-place
function one_point_crossover!(g1::Gene{<:Any}, g2::Gene{<:Any}, prob::Float64,
                             minpt::Int64 = 2, maxpt::Int64 = -1)
    if g1.n_chromosomes != g2.n_chromosomes
        error("The number of chromosomes in the genes must be the same to perform crossover")
    end

    coin = Bernoulli(prob)
    if rand(coin)
        if maxpt == -1 maxpt = g1.n_chromosomes-1 end
        # Perform 1-point crossover via https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover
        for i in 1:rand(minpt:maxpt)
            @swap(g1.chromosomes[i], g2.chromosomes[i])
        end
    end
end


# Function to perform crossover from two genes of the same size. Crossover occurs with
# some random probability
#
# Parents in the first two arguments are crossedover into the childern in the second two arguments
function one_point_crossover!(p1::Gene{<:Any}, p2::Gene{<:Any},
                              c1::Gene{<:Any}, c2::Gene{<:Any},
                              prob::Float64, minpt::Int64 = 2, maxpt::Int64 = -1)
    if p1.n_chromosomes != p2.n_chromosomes != c1.n_chromosomes != c2.n_chromosomes
        error("The number of chromosomes in the genes must be the same to perform crossover")
    end

    coin = Bernoulli(prob)
    if rand(coin)
        if maxpt == -1 maxpt = p1.n_chromosomes-1 end
        # Perform 1-point crossover via https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover
        pt = rand(minpt:maxpt)
        for i in 1:pt
            # Up until the crossover point copy pi to ci and pj to cj
            c1.chromosomes[i] = p1.chromosomes[i]
            c2.chromosomes[i] = p2.chromosomes[i]
        end
        for i in pt+1:p1.n_chromosomes
            # After the crossover point copy pi to cj and pj to ci
            c1.chromosomes[i] = p2.chromosomes[i]
            c2.chromosomes[i] = p1.chromosomes[i]
        end
    end
end

# Helper which takes size-2 arrays for children and parents
function one_point_crossover!(parents::Vector{Gene{T}}, children::Vector{Gene{T}},
                               prob::Float64, minpt::Int64 = 2, maxpt::Int64 = -1) where T <:Any
    if length(parents) != length(children) != 2
        error("One point crossover currently only supports 2 parents and 2 children")
    end

    one_point_crossover!(parents[1], parents[2], children[1], children[2], prob, minpt, maxpt)
end



# General mutation function which accepts in the third argument a mutator function
# which takes an element of the gene's chromosome as the argument.
# Mutation occurs with a given probability via a Bernoulli distribution
#
# Mutation occurs in-place
function mutation!(g::Gene{<:Any}, prob::Float64, mutator::Function)
    coin = Bernoulli(prob)
    for i in eachindex(g.chromosomes)
        if rand(coin) g.chromosomes[i] = mutator(g.chromosomes[i]) end
    end
end

# Specialize mutation for the binary gene for not operation
mutation!(g::Gene{Bool}, prob::Float64) = mutation!(g, prob, !)

end
