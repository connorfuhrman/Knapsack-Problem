module GeneticTypes

export Item, Gene, Base

using Printf


# Type for a "gene" which holds an array of T-type chromosomes
mutable struct Gene{T}
    n_chromosomes::Int64 # How many chromosomes in the gene?
    chromosomes::Array{T} # The representation of the gene

    # Constructor method to create an N-sized vector of T-type variables
    # which are uninitialized
    Gene{T}(sz::Int64) where T = new(sz, Array{T}(undef,sz))

    # Constructor method to create an N-sized vector of T-type variables
    # which are randomly initialized via the randomer function which takes
    # a size as the argument
    Gene{T}(sz::Int64, randomer::Function) where T = new(sz, Array{T}(randomer(sz)))
end


end
