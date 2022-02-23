# Helper script to generate a dummy knapsack problem

using DelimitedFiles

function make_knapsack_problem(filename, num_items, max_value, max_weight, min_value = 5, min_weight = 1)
    values = rand(min_value:max_value, num_items)
    weights = rand(min_weight:max_weight, num_items)

    open(filename, "w") do io
        writedlm(io, zip(values, weights), ',')
    end
end


#make_knapsack_problem("items/items_01.csv", 50, 100, 50)
#make_knapsack_problem("items/items_n150.csv", 150, 1000, 100)
make_knapsack_problem("items/item_n5000.csv", 5000, 100, 50)
