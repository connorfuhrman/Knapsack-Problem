push!(LOAD_PATH, pwd()) # Add the current path so we can load other files

using GeneticTypes


using Printf
using Random


function main(args)
    g = Gene{Bool}(5)
    @show g

    g2 = Gene{Bool}(5, bitrand)
    @show g2
                    
    
end

main(ARGS)
