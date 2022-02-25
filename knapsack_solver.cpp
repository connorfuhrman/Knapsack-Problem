// Cpp implimentation of the Genetic Optimizer for the Knapsack Problem
// Purpose is to compare efficiency against the Julia implimentation 


#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <cassert>
#include <memory_resource>
#include <memory>
#include <cstdint>
#include <algorithm>

template<typename T1, typename T2>
struct Item {
  T1 value;
  T2 weight;
};


typedef Item<double, double> item_t;
typedef std::vector<item_t> item_vec;

template<typename ChromoT>
struct Gene {
  typedef std::vector<ChromoT> chromo_t;

  // Create a random chromosome given some sampler
  Gene(std::size_t sz, auto& dist) : chromosomes(sz) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    for (std::size_t i = 0; i != sz; ++i) chromosomes[i] = static_cast<ChromoT>(dist(gen));
  }
  Gene(const auto& other) : chromosomes(other) {};
  
  chromo_t chromosomes;
};

typedef Gene<bool> gene_t;

// Function to print a Gene<T>-type object. T must be defined in
// std::ostream& operator<< (std::ostream&, const T&), e.g., POD.
template<typename T>
std::ostream& operator<< (std::ostream& str, const Gene<T>& g) {
  str << "[";
  for (std::size_t i = 0; i != g.chromosomes.size()-1; ++i) {
    str << g.chromosomes.at(i) << " ";
  }
  str << g.chromosomes.at(g.chromosomes.size()-1) << "]";
  return str;
}

// Function to print a Item<T1, T2>-type object. Both T1 and T2 must
// have defined operator<< for std::ostream
template<typename T1, typename T2>
std::ostream& operator<< (std::ostream& str, const Item<T1, T2>& i) {
  str << "{" << i.value << ", " << i.weight << "}";
  return str;
}

typedef std::vector<gene_t> GenePool;

// Function to read the config file and return the items we're to
// consider in the optimization
template <typename T1, typename T2>
auto read_item_file(const std::string& fpath) {
  item_vec items;

  std::ifstream f(fpath);
  T1 value;
  T2 weight;
  char comma;

  while(f >> value >> comma >> weight) items.emplace_back(value, weight);

  return items;
}

// Function to create an initial population
auto make_genepool(const std::size_t pop_size, const std::size_t num_items) {
  GenePool pool;
  std::uniform_int_distribution<> dist(0,1);
  for (std::size_t i = 0; i != pop_size; ++i) {
    pool.emplace_back(num_items, dist); // TODO init like in Julia
  }

  return pool;
}

// Function to calculate the fitness of a gene
template<typename T>
auto calc_fitness(const Gene<T>& g, const auto& items, double capacity) {
  double fit = 0;
  double weight = 0;
  
  assert(std::ranges::size(g.chromosomes) == std::ranges::size(items));

  auto chromo = std::ranges::begin(g.chromosomes);
  auto item   = std::ranges::begin(items);

  while (chromo != std::ranges::end(g.chromosomes)) {
    fit += *chromo * (item->value);
    weight += *chromo * (item -> weight);

    ++chromo; ++item;
  }
  
  return fit * (weight < capacity);
}

// Function to calculate all fitness and place into the container of fitnesses
inline auto calc_fitnesses(auto& fitness, const auto& genes,
			   const auto& items, const double capacity)
{
  assert(std::ranges::size(fitness) == std::ranges::size(genes));
  auto f = std::ranges::begin(fitness);
  auto g = std::ranges::begin(genes);

  while (f != std::ranges::end(fitness)) {
    *f = calc_fitness(*g, items, capacity);
    ++f; ++g;
  }
  return;
}

// Function to run the genetic optimizer
auto optimize(std::size_t pop_size, auto max_capacity, std::string items_fpath) {
  namespace r = std::ranges;
  
  // Read the items from configuration file
  auto items = read_item_file<double, double>(items_fpath);

  // Create the initial gene poo.
  auto gene_pool = make_genepool(pop_size, items.size());

  // Allocate for the fitness values but store data on the stack in a pmr resource
  // monotonic buffer is used because the fitness is allocated once and not deallocated
  // until optimizer ends
  const auto fitness_buff_sz = pop_size + 10;
  char fitness_buff[fitness_buff_sz];
  std::pmr::monotonic_buffer_resource fitness_buff_rsrc(fitness_buff, fitness_buff_sz);
  std::pmr::vector<double> fitness(pop_size, &fitness_buff_rsrc);

  unsigned int generation = 0, gen_without_improvement = 0;
  auto max_fitness = r::end(fitness);
  auto optimal_gene = r::end(gene_pool);

  for(;;) {
    std::cout << "Generation " << generation << std::endl;
    
    // Calculate the fitness for this generation
    calc_fitnesses(fitness, gene_pool, items, max_capacity);
    // and find the gene with the maximum fitness
    auto _max_fitness = r::max_element(fitness);
    if (_max_fitness != max_fitness) {
      max_fitness = _max_fitness;
      const auto d = r::distance(r::begin(fitness), max_fitness);
      optimal_gene = r::begin(gene_pool) + d;

      std::cout << "New optimal gene found in generation " << generation << std::endl;
      std::cout << "\tOptimal Gene: " << *optimal_gene << std::endl;
      std::cout << "\tFitness: " << *max_fitness << std::endl;

      gen_without_improvement = 0;
    }
    else {
      if (++gen_without_improvement == 10) return;
    }

    // Perform selection to select parents

    // Perform crossover to create children from selected parents

    // Perform mutation on the new children genes


    ++generation;
  }
  
  
  return;
}


int main([[maybe_unused]] int ac, [[maybe_unused]] char** av) {
  // TOOD command line args
  
  optimize(15, 150, "./items/items_n50.csv");

  return EXIT_SUCCESS;
}
