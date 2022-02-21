#include <vector>
#include <iostream>

tempalte<typename T1, typename T2>
struct Item {
  T1 value;
  T2 weight;
};


typedef Item<double, double> item_t;
typedef item_vec std::vector<item_t>;

// Function to read the config file and return the items we're to
// consider in the optimization
auto read_item_file([[maybe_unused]] const std::string fpath) {
  item_vec items;
  
  items.emplace_back(15, 10);
  items.emplace_back(5, 2);
  items.emplace_back(1, 7);

  return items;
}

// Function to create an initial population

// Function to run the genetic optimizer

// Functions to brute-force the solution (recursively)
auto brute_force_solve_rec(const auto& items, const auto cap_left, const auto index) {
  // Exit condition
  if (cap_left <= 0 or index >= items.size()) {
    return 0;
  }

  // Chose the element at the current index and if the weight exceeds the capacity we
  // cant' carry so we don't care
  const auto& thisItem = items.at(index);
  auto val1 = 0;
  
  
}

auto brute_force_solve(const auto& items, const auto weight_cap) {
  return brute_force_solve_rec(items, weight_cap, 0);
}


int main(int ac, char** av) {
  
  

  return EXIT_SUCCESS;
}
