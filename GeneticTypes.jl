module GeneticTypes

export Gene, make_binary_gene

using StatsBase


# Type for a "gene" which holds an array of T-type chromosomes
mutable struct Gene{T}
    n_chromosomes::Int64 # How many chromosomes in the gene?
    chromosomes::Array{T} # The representation of the gene

    # Constructor method to create an N-sized vector of T-type variables
    # which are uninitialized
    Gene{T}(sz::Int64) where T<:Any = new(sz, Array{T}(undef,sz))

    # Constructor method to create an N-sized vector of T-type variables
    # which are randomly initialized via the randomer function which takes
    # a size as the argument
    Gene{T}(sz::Int64, randomer::Function) where T<:Any = new(sz, Array{T}(randomer(sz)))

    # Constructor given an array already. The array is copied
    Gene{T}(a::Array{T}) where T<:Any = new(length(a), a)
    # Specialized for the Bool type
    Gene{Bool}(a::BitVector) = new(length(a), a)
end


function make_binary_gene(sz::Int64, nTrue::Int64)::Gene{Bool}
    g = Gene{Bool}(BitVector(zeros(sz)))
    g.chromosomes[sample(collect(1:sz), nTrue, replace=false)] .= true

    return g
end


end
