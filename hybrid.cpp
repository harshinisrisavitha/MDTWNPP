#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <random> 
#include <ctime>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <list>

using namespace std;

// ----------- CONFIGURABLE PARAMETERS ------------
const string filename = "/home/harshini/python_tutorials/cs23b2055/dataset/mdtwnpp_50_20c.txt";
int POPULATION_SIZE_INPUT;
int CHROMOSOME_LENGTH_INPUT;
const int GENERATIONS_INPUT = 4000;
int NUM_INPUT_VECTORS_INPUT;
int DIMENSION_INPUT;

const string SELECTION_METHOD_INPUT = "tournament";
const string CROSSOVER_METHOD_INPUT = "uniform";
float MUTATION_RATE_INPUT = 0.015f;
const bool ELITISM_ENABLED_INPUT = true;
const int TOURNAMENT_SIZE = 3;
const float CONVERGENCE_THRESHOLD = 50.0f;
const int STAGNATION_LIMIT = 400;
// --------------------------

struct Individual {
    vector<int> chromosome;
    float fitness;
    bool fitness_evaluated;
    
    Individual() : fitness_evaluated(false), fitness(1e9f) {}
    Individual(const vector<int>& chrom) : chromosome(chrom), fitness_evaluated(false), fitness(1e9f) {}
};

struct GAResult {
    vector<int> best_solution;
    float best_fitness;
    vector<float> fitness_history;
    int generations_run;
    float convergence_time;
};

struct TabuMove {
    int position;
    int generation_added;
    
    TabuMove(int pos, int gen) : position(pos), generation_added(gen) {}
};

class TabuList {
private:
    std::queue<TabuMove> tabu_moves;
    std::unordered_set<int> tabu_positions;
    int max_size;
    int current_generation;
    
public:
    TabuList(int size = 10) : max_size(size), current_generation(0) {}
    
    void add_move(int position) {
        tabu_moves.push(TabuMove(position, current_generation));
        tabu_positions.insert(position);
        
        // Remove old moves
        while (tabu_moves.size() > max_size) {
            TabuMove old_move = tabu_moves.front();
            tabu_moves.pop();
            tabu_positions.erase(old_move.position);
        }
    }
    
    bool is_tabu(int position) const {
        return tabu_positions.find(position) != tabu_positions.end();
    }
    
    void next_generation() {
        current_generation++;
    }
    
    void clear() {
        while (!tabu_moves.empty()) tabu_moves.pop();
        tabu_positions.clear();
        current_generation = 0;
    }
};



random_device rd;
mt19937 rng(rd());
unordered_map<string, float> fitness_cache;


string chromosome_to_string(const vector<int>& chromosome) {
    string result;
    result.reserve(chromosome.size());
    for (int bit : chromosome) {
        result += (bit + '0');
    }
    return result;
}

bool read_vectors_from_file(const string& filename, int& num_of_vectors, int& dimension, vector<vector<float>>& vectors) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    infile >> num_of_vectors >> dimension;
    if (num_of_vectors <= 0 || dimension <= 0) {
        cerr << "Invalid vector count or dimension in file." << endl;
        return false;
    }

    vectors.assign(num_of_vectors, vector<float>(dimension));

    for (int i = 0; i < num_of_vectors; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (!(infile >> vectors[i][j])) {
                cerr << "Error reading vector element at [" << i << "][" << j << "]" << endl;
                return false;
            }
        }
    }

    infile.close();
    return true;
}

// CORRECTED: Back to using SUMS (not averages) but without initial bias
float compute_fitness_optimized(Individual& individual, const vector<vector<float>>& vectors, int dimension) {
    if (individual.fitness_evaluated) {
        return individual.fitness;
    }
    
    string chrom_str = chromosome_to_string(individual.chromosome);
    auto cache_it = fitness_cache.find(chrom_str);
    if (cache_it != fitness_cache.end()) {
        individual.fitness = cache_it->second;
        individual.fitness_evaluated = true;
        return individual.fitness;
    }
    
    // NO INITIAL VALUES - start from pure zero
    vector<float> sum_0(dimension, 0.0f);
    vector<float> sum_1(dimension, 0.0f);
    int count_0 = 0, count_1 = 0;
    
    // Calculate sums for each partition
    for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
        if (individual.chromosome[i] == 0) {
            count_0++;
            for (int j = 0; j < dimension; ++j) {
                sum_0[j] += vectors[i][j];
            }
        } else {
            count_1++;
            for (int j = 0; j < dimension; ++j) {
                sum_1[j] += vectors[i][j];
            }
        }
    }
    
    // Heavy penalty for empty partitions
    if (count_0 == 0 || count_1 == 0) {
        individual.fitness = 1e8f;
        individual.fitness_evaluated = true;
        fitness_cache[chrom_str] = individual.fitness;
        return individual.fitness;
    }
    
    // Calculate absolute differences between SUMS (as in original)
    vector<float> abs_diff(dimension);
    for (int i = 0; i < dimension; ++i) {
        abs_diff[i] = fabs(sum_0[i] - sum_1[i]);
    }
    
    // Return the maximum absolute difference
    float max_diff = *max_element(abs_diff.begin(), abs_diff.end());
    
    individual.fitness = max_diff;
    individual.fitness_evaluated = true;
    fitness_cache[chrom_str] = individual.fitness;
    
    return individual.fitness;
}

// Greedy heuristic for initial solution
Individual generate_greedy_solution(const vector<vector<float>>& vectors, int dimension) {
    vector<int> chromosome(vectors.size(), 0);
    vector<float> sum_0(dimension, 0.0f);
    vector<float> sum_1(dimension, 0.0f);
    int count_0 = 0, count_1 = 0;
    
    // Start with first vector in partition 0
    chromosome[0] = 0;
    count_0 = 1;
    for (int j = 0; j < dimension; ++j) {
        sum_0[j] = vectors[0][j];
    }
    
    // For each remaining vector, assign to partition that minimizes max difference
    for (int i = 1; i < vectors.size(); ++i) {
        // Try adding to partition 0
        vector<float> temp_sum_0 = sum_0;
        for (int j = 0; j < dimension; ++j) {
            temp_sum_0[j] += vectors[i][j];
        }
        
        float max_diff_0 = 0.0f;
        for (int j = 0; j < dimension; ++j) {
            max_diff_0 = max(max_diff_0, fabs(temp_sum_0[j] - sum_1[j]));
        }
        
        // Try adding to partition 1
        vector<float> temp_sum_1 = sum_1;
        for (int j = 0; j < dimension; ++j) {
            temp_sum_1[j] += vectors[i][j];
        }
        
        float max_diff_1 = 0.0f;
        for (int j = 0; j < dimension; ++j) {
            max_diff_1 = max(max_diff_1, fabs(sum_0[j] - temp_sum_1[j]));
        }
        
        // Assign to partition that gives smaller max difference
        if (max_diff_0 <= max_diff_1) {
            chromosome[i] = 0;
            sum_0 = temp_sum_0;
            count_0++;
        } else {
            chromosome[i] = 1;
            sum_1 = temp_sum_1;
            count_1++;
        }
    }
    
    return Individual(chromosome);
}

// Improved initial population with multiple strategies
vector<Individual> generate_initial_population_improved(int pop_size, int chrom_length, const vector<vector<float>>& vectors, int dimension) {
    vector<Individual> population;
    population.reserve(pop_size);
    
    uniform_int_distribution<int> bit_dist(0, 1);
    uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    // Add greedy solution
    if (pop_size > 0) {
        population.push_back(generate_greedy_solution(vectors, dimension));
    }
    
    // Add some balanced solutions
    for (int i = 1; i < pop_size / 3; ++i) {
        vector<int> chromosome(chrom_length);
        int target_ones = chrom_length / 2 + (rng() % 21) - 10; // ±10 variation
        target_ones = max(1, min(chrom_length - 1, target_ones));
        
        fill(chromosome.begin(), chromosome.begin() + target_ones, 1);
        fill(chromosome.begin() + target_ones, chromosome.end(), 0);
        shuffle(chromosome.begin(), chromosome.end(), rng);
        
        population.emplace_back(chromosome);
    }
    
    // Add random solutions with bias toward balanced
    for (int i = pop_size / 3; i < 2 * pop_size / 3; ++i) {
        vector<int> chromosome(chrom_length);
        float bias = 0.3f + prob_dist(rng) * 0.4f; // 0.3 to 0.7 bias
        
        for (int j = 0; j < chrom_length; ++j) {
            chromosome[j] = (prob_dist(rng) < bias) ? 1 : 0;
        }
        
        // Ensure both partitions have elements
        if (all_of(chromosome.begin(), chromosome.end(), [](int x) { return x == 0; })) {
            chromosome[rng() % chrom_length] = 1;
        }
        if (all_of(chromosome.begin(), chromosome.end(), [](int x) { return x == 1; })) {
            chromosome[rng() % chrom_length] = 0;
        }
        
        population.emplace_back(chromosome);
    }
    
    // Fill rest with random
    for (int i = 2 * pop_size / 3; i < pop_size; ++i) {
        vector<int> chromosome(chrom_length);
        for (int j = 0; j < chrom_length; ++j) {
            chromosome[j] = bit_dist(rng);
        }
        
        // Ensure both partitions have elements
        if (all_of(chromosome.begin(), chromosome.end(), [](int x) { return x == 0; })) {
            chromosome[rng() % chrom_length] = 1;
        }
        if (all_of(chromosome.begin(), chromosome.end(), [](int x) { return x == 1; })) {
            chromosome[rng() % chrom_length] = 0;
        }
        
        population.emplace_back(chromosome);
    }
    
    return population;
}

Individual tournament_selection(vector<Individual>& population, const vector<vector<float>>& vectors, int k = TOURNAMENT_SIZE) {
    uniform_int_distribution<int> dist(0, population.size() - 1);
    
    Individual best = population[dist(rng)];
    float best_fitness = compute_fitness_optimized(best, vectors, DIMENSION_INPUT);
    
    for (int i = 1; i < k; ++i) {
        Individual current = population[dist(rng)];
        float current_fitness = compute_fitness_optimized(current, vectors, DIMENSION_INPUT);
        
        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            best = current;
        }
    }
    
    return best;
}

pair<Individual, Individual> uniform_crossover(const Individual& parent1, const Individual& parent2) {
    uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    vector<int> child1, child2;
    child1.reserve(parent1.chromosome.size());
    child2.reserve(parent1.chromosome.size());
    
    for (size_t i = 0; i < parent1.chromosome.size(); ++i) {
        if (prob_dist(rng) < 0.5f) {
            child1.push_back(parent1.chromosome[i]);
            child2.push_back(parent2.chromosome[i]);
        } else {
            child1.push_back(parent2.chromosome[i]);
            child2.push_back(parent1.chromosome[i]);
        }
    }
    
    return {Individual(child1), Individual(child2)};
}

Individual mutate(const Individual& individual, float mutation_rate) {
    uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    Individual mutated = individual;
    mutated.fitness_evaluated = false;
    
    for (size_t i = 0; i < mutated.chromosome.size(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            mutated.chromosome[i] = 1 - mutated.chromosome[i];
        }
    }
    
    // Ensure both partitions have at least one element
    int count_0 = count(mutated.chromosome.begin(), mutated.chromosome.end(), 0);
    int count_1 = mutated.chromosome.size() - count_0;
    
    if (count_0 == 0) {
        uniform_int_distribution<int> pos_dist(0, mutated.chromosome.size() - 1);
        mutated.chromosome[pos_dist(rng)] = 0;
    }
    if (count_1 == 0) {
        uniform_int_distribution<int> pos_dist(0, mutated.chromosome.size() - 1);
        mutated.chromosome[pos_dist(rng)] = 1;
    }
    
    return mutated;
}

// Aggressive local search with multiple neighborhoods
Individual local_search_aggressive(Individual individual, const vector<vector<float>>& vectors, int max_iterations = 100) {
    float current_fitness = compute_fitness_optimized(individual, vectors, DIMENSION_INPUT);
    bool improved = true;
    int iterations = 0;
    
    while (improved && iterations < max_iterations) {
        improved = false;
        
        // Try single bit flips
        for (size_t i = 0; i < individual.chromosome.size(); ++i) {
            Individual neighbor = individual;
            neighbor.chromosome[i] = 1 - neighbor.chromosome[i];
            neighbor.fitness_evaluated = false;
            
            // Check if partitions are still valid
            int count_0 = count(neighbor.chromosome.begin(), neighbor.chromosome.end(), 0);
            int count_1 = neighbor.chromosome.size() - count_0;
            
            if (count_0 > 0 && count_1 > 0) {
                float neighbor_fitness = compute_fitness_optimized(neighbor, vectors, DIMENSION_INPUT);
                
                if (neighbor_fitness < current_fitness) {
                    individual = neighbor;
                    current_fitness = neighbor_fitness;
                    improved = true;
                    break;
                }
            }
        }
        
        // If no single flip helped, try swaps
        if (!improved && iterations < max_iterations / 2) {
            for (size_t i = 0; i < individual.chromosome.size() - 1; ++i) {
                for (size_t j = i + 1; j < individual.chromosome.size(); ++j) {
                    if (individual.chromosome[i] != individual.chromosome[j]) {
                        Individual neighbor = individual;
                        swap(neighbor.chromosome[i], neighbor.chromosome[j]);
                        neighbor.fitness_evaluated = false;
                        
                        float neighbor_fitness = compute_fitness_optimized(neighbor, vectors, DIMENSION_INPUT);
                        
                        if (neighbor_fitness < current_fitness) {
                            individual = neighbor;
                            current_fitness = neighbor_fitness;
                            improved = true;
                            goto next_iteration;
                        }
                    }
                }
            }
        }
        
        next_iteration:
        iterations++;
    }
    
    return individual;
}

// tabu search
Individual tabu_search(const vector<vector<float>>& vectors, 
                      Individual initial_solution = Individual(),
                      int max_iterations = 1000,
                      int tabu_tenure = 15) {
    
    // Generate initial solution if not provided
    Individual current_solution;
    if (initial_solution.chromosome.empty()) {
        current_solution = generate_greedy_solution(vectors, DIMENSION_INPUT);
    } else {
        current_solution = initial_solution;
    }
    
    Individual best_solution = current_solution;
    float best_fitness = compute_fitness_optimized(best_solution, vectors, DIMENSION_INPUT);
    float current_fitness = best_fitness;
    
    TabuList tabu_list(tabu_tenure);
    int no_improvement_count = 0;
    const int max_no_improvement = 200;
    
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Individual best_neighbor;
        float best_neighbor_fitness = 1e9f;
        int best_move = -1;
        bool found_valid_move = false;
        
        // Explore neighborhood (bit flip moves)
        for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
            // Skip if move is tabu (unless it leads to best solution - aspiration criteria)
            if (tabu_list.is_tabu(i)) {
                continue;
            }
            
            Individual neighbor = current_solution;
            neighbor.chromosome[i] = 1 - neighbor.chromosome[i];
            neighbor.fitness_evaluated = false;
            
            // Check if solution is valid (both partitions non-empty)
            int count_0 = count(neighbor.chromosome.begin(), neighbor.chromosome.end(), 0);
            int count_1 = neighbor.chromosome.size() - count_0;
            
            if (count_0 == 0 || count_1 == 0) {
                continue; // Invalid solution
            }

            float neighbor_fitness = compute_fitness_optimized(neighbor, vectors, DIMENSION_INPUT);
            // Accept if better than current best neighbor OR if it's best solution found so far (aspiration)
            if (!found_valid_move || neighbor_fitness < best_neighbor_fitness || neighbor_fitness < best_fitness) {
                best_neighbor = neighbor;
                best_neighbor_fitness = neighbor_fitness;
                best_move = i;
                found_valid_move = true;
            }
        }
        // Move to best neighbor
        current_solution = best_neighbor;
        current_fitness = best_neighbor_fitness;
        tabu_list.add_move(best_move);
        tabu_list.next_generation();
        
        // Update best solution
        if (current_fitness < best_fitness) {
            best_solution = current_solution;
            best_fitness = current_fitness;
            no_improvement_count = 0;
            
        } else {
            no_improvement_count++;
        }
        
        // Diversification - restart if no improvement for too long
        if (no_improvement_count > max_no_improvement) {
            current_solution = generate_greedy_solution(vectors, DIMENSION_INPUT);
            current_fitness = compute_fitness_optimized(current_solution, vectors, DIMENSION_INPUT);
            tabu_list.clear();
            no_improvement_count = 0;
        }
        
    }
    return best_solution;
}

// variable neighbourhood_search

Individual flip_k_bits(const Individual& solution, int k) {
    Individual neighbor = solution;
    neighbor.fitness_evaluated = false;
    
    uniform_int_distribution<int> pos_dist(0, solution.chromosome.size() - 1);
    unordered_set<int> flipped_positions;
    
    while (flipped_positions.size() < k) {
        int pos = pos_dist(rng);
        if (flipped_positions.find(pos) == flipped_positions.end()) {
            neighbor.chromosome[pos] = 1 - neighbor.chromosome[pos];
            flipped_positions.insert(pos);
        }
    }
    // Ensure valid solution
    int count_0 = count(neighbor.chromosome.begin(), neighbor.chromosome.end(), 0);
    int count_1 = neighbor.chromosome.size() - count_0;
    
    if (count_0 == 0) {
        neighbor.chromosome[pos_dist(rng)] = 0;
    }
    if (count_1 == 0) {
        neighbor.chromosome[pos_dist(rng)] = 1;
    }
    
    return neighbor;
}


Individual swap_k_pairs(const Individual& solution, int k) {
    Individual neighbor = solution;
    neighbor.fitness_evaluated = false;
    
    uniform_int_distribution<int> pos_dist(0, solution.chromosome.size() - 1);
    
    for (int i = 0; i < k; ++i) {
        int pos1 = pos_dist(rng);
        int pos2 = pos_dist(rng);
        
        if (pos1 != pos2) {
            swap(neighbor.chromosome[pos1], neighbor.chromosome[pos2]);
        }
    }
    
    return neighbor;
}


Individual variable_neighborhood_search(const vector<vector<float>>& vectors,
                                      Individual initial_solution = Individual(),
                                      int max_iterations = 50) {
    
    // Generate initial solution if not provided
    Individual current_solution;
    if (initial_solution.chromosome.empty()) {
        current_solution = generate_greedy_solution(vectors, DIMENSION_INPUT);
    } else {
        current_solution = initial_solution;
    }
    
    Individual best_solution = current_solution;
    float best_fitness = compute_fitness_optimized(best_solution, vectors, DIMENSION_INPUT);
    
    const int max_neighborhoods = 4;
    const vector<int> k_values = {1, 3, 5, 8}; // Different neighborhood sizes
    
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Individual current = best_solution;
        for (int k = 0; k < max_neighborhoods; ++k) {
            // Shaking: generate random solution in k-th neighborhood
            Individual shaken_solution;
            if (k < 2) {
                shaken_solution = flip_k_bits(current, k_values[k]);
            } else {
                shaken_solution = swap_k_pairs(current, k_values[k] / 2);
            }
            
            // Local search from shaken solution
            Individual improved_solution = local_search_aggressive(shaken_solution, vectors, 30);
            float improved_fitness = compute_fitness_optimized(improved_solution, vectors, DIMENSION_INPUT);
            
            // Move or not
            if (improved_fitness < best_fitness) {
                best_solution = improved_solution;
                best_fitness = improved_fitness;
                k = 0; // Restart from first neighborhood
                
                break;
            }
        }
    }
    return best_solution;
}

Individual destroy_random(const Individual& solution, float destroy_rate = 0.3f) {
    Individual destroyed = solution;
    destroyed.fitness_evaluated = false;
    
    int destroy_count = max(1, static_cast<int>(solution.chromosome.size() * destroy_rate));
    uniform_int_distribution<int> pos_dist(0, solution.chromosome.size() - 1);
    uniform_int_distribution<int> bit_dist(0, 1);
    
    unordered_set<int> destroyed_positions;
    while (destroyed_positions.size() < destroy_count) {
        destroyed_positions.insert(pos_dist(rng));
    }
    
    for (int pos : destroyed_positions) {
        destroyed.chromosome[pos] = bit_dist(rng);
    }
    
    // Ensure valid solution
    int count_0 = count(destroyed.chromosome.begin(), destroyed.chromosome.end(), 0);
    int count_1 = destroyed.chromosome.size() - count_0;
    
    if (count_0 == 0) {
        destroyed.chromosome[pos_dist(rng)] = 0;
    }
    if (count_1 == 0) {
        destroyed.chromosome[pos_dist(rng)] = 1;
    }
    
    return destroyed;
}

Individual destroy_worst_segments(const Individual& solution, const vector<vector<float>>& vectors, float destroy_rate = 0.3f) {
    Individual destroyed = solution;
    destroyed.fitness_evaluated = false;
    
    int destroy_count = max(1, static_cast<int>(solution.chromosome.size() * destroy_rate));
    
    // Calculate contribution of each vector to current imbalance
    vector<float> sum_0(DIMENSION_INPUT, 0.0f), sum_1(DIMENSION_INPUT, 0.0f);
    for (size_t i = 0; i < solution.chromosome.size(); ++i) {
        if (solution.chromosome[i] == 0) {
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                sum_0[j] += vectors[i][j];
            }
        } else {
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                sum_1[j] += vectors[i][j];
            }
        }
    }
    // Calculate difference vector
    vector<float> diff(DIMENSION_INPUT);
    for (int j = 0; j < DIMENSION_INPUT; ++j) {
        diff[j] = sum_0[j] - sum_1[j];
    }
    
    // Score each vector by how much it contributes to imbalance
    vector<pair<float, int>> vector_scores;
    for (size_t i = 0; i < solution.chromosome.size(); ++i) {
        float score = 0.0f;
        for (int j = 0; j < DIMENSION_INPUT; ++j) {
            if (solution.chromosome[i] == 0) {
                score += vectors[i][j] * diff[j]; // Positive if adding to imbalance
            } else {
                score -= vectors[i][j] * diff[j]; // Negative if reducing imbalance
            }
        }
        vector_scores.push_back({score, i});
    }
    // Sort by score (highest first - these contribute most to imbalance)
    sort(vector_scores.begin(), vector_scores.end(), greater<pair<float, int>>());
    
    // Destroy worst contributors
    uniform_int_distribution<int> bit_dist(0, 1);
    for (int i = 0; i < destroy_count; ++i) {
        int pos = vector_scores[i].second;
        destroyed.chromosome[pos] = bit_dist(rng);
    }
    
    return destroyed;
}

Individual repair_greedy(const Individual& destroyed_solution, const vector<vector<float>>& vectors) {
    Individual repaired = destroyed_solution;
    repaired.fitness_evaluated = false;
    
    // Calculate current sums for both partitions
    vector<float> sum_0(DIMENSION_INPUT, 0.0f);
    vector<float> sum_1(DIMENSION_INPUT, 0.0f);
    int count_0 = 0, count_1 = 0;
    
    for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
        if (repaired.chromosome[i] == 0) {
            count_0++;
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                sum_0[j] += vectors[i][j];
            }
        } else {
            count_1++;
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                sum_1[j] += vectors[i][j];
            }
        }
    }
    
    // Ensure both partitions have at least one element
    if (count_0 == 0) {
        uniform_int_distribution<int> pos_dist(0, CHROMOSOME_LENGTH_INPUT - 1);
        int pos = pos_dist(rng);
        repaired.chromosome[pos] = 0;
        count_0 = 1;
        count_1--;
        // Recalculate sums
        fill(sum_0.begin(), sum_0.end(), 0.0f);
        fill(sum_1.begin(), sum_1.end(), 0.0f);
        for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
            if (repaired.chromosome[i] == 0) {
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    sum_0[j] += vectors[i][j];
                }
            } else {
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    sum_1[j] += vectors[i][j];
                }
            }
        }
    }
    
    if (count_1 == 0) {
        uniform_int_distribution<int> pos_dist(0, CHROMOSOME_LENGTH_INPUT - 1);
        int pos = pos_dist(rng);
        repaired.chromosome[pos] = 1;
        count_1 = 1;
        count_0--;
        // Recalculate sums
        fill(sum_0.begin(), sum_0.end(), 0.0f);
        fill(sum_1.begin(), sum_1.end(), 0.0f);
        for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
            if (repaired.chromosome[i] == 0) {
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    sum_0[j] += vectors[i][j];
                }
            } else {
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    sum_1[j] += vectors[i][j];
                }
            }
        }
    }
    
    // Greedy improvement: try to reassign vectors to reduce maximum difference
    bool improved = true;
    int max_iterations = 50;
    int iteration = 0;
    
    while (improved && iteration < max_iterations) {
        improved = false;
        iteration++;
        
        // Try moving each vector to the other partition
        for (int i = 0; i < CHROMOSOME_LENGTH_INPUT; ++i) {
            // Skip if this would empty a partition
            if ((repaired.chromosome[i] == 0 && count_0 == 1) || 
                (repaired.chromosome[i] == 1 && count_1 == 1)) {
                continue;
            }
            
            // Calculate current maximum difference
            float current_max_diff = 0.0f;
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                current_max_diff = max(current_max_diff, fabs(sum_0[j] - sum_1[j]));
            }
            
            // Try flipping this vector
            vector<float> new_sum_0 = sum_0;
            vector<float> new_sum_1 = sum_1;
            
            if (repaired.chromosome[i] == 0) {
                // Move from partition 0 to partition 1
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    new_sum_0[j] -= vectors[i][j];
                    new_sum_1[j] += vectors[i][j];
                }
            } else {
                // Move from partition 1 to partition 0
                for (int j = 0; j < DIMENSION_INPUT; ++j) {
                    new_sum_0[j] += vectors[i][j];
                    new_sum_1[j] -= vectors[i][j];
                }
            }
            
            // Calculate new maximum difference
            float new_max_diff = 0.0f;
            for (int j = 0; j < DIMENSION_INPUT; ++j) {
                new_max_diff = max(new_max_diff, fabs(new_sum_0[j] - new_sum_1[j]));
            }
            
            // If improvement, make the move
            if (new_max_diff < current_max_diff - 1e-6f) {
                repaired.chromosome[i] = 1 - repaired.chromosome[i];
                sum_0 = new_sum_0;
                sum_1 = new_sum_1;
                
                if (repaired.chromosome[i] == 0) {
                    count_0++;
                    count_1--;
                } else {
                    count_0--;
                    count_1++;
                }
                
                improved = true;
                break; // Try again from the beginning
            }
        }
    }
    
    return repaired;
}

// large neighbourhood search
Individual large_neighborhood_search(const vector<vector<float>>& vectors,
                                   Individual initial_solution = Individual(),
                                   int max_iterations = 100) {
    
    // Generate initial solution if not provided
    Individual current_solution;
    if (initial_solution.chromosome.empty()) {
        current_solution = generate_greedy_solution(vectors, DIMENSION_INPUT);
    } else {
        current_solution = initial_solution;
    }
    
    Individual best_solution = current_solution;
    float best_fitness = compute_fitness_optimized(best_solution, vectors, DIMENSION_INPUT);
    float current_fitness = best_fitness;
    
    const vector<float> destroy_rates = {0.2f, 0.3f, 0.4f, 0.5f};
    int no_improvement_count = 0;
    const int max_no_improvement = 100;
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Choose destroy rate
        float destroy_rate = destroy_rates[iteration % destroy_rates.size()];
        
        // Choose destroy method (alternate between random and worst segments)
        Individual destroyed;
        if (iteration % 2 == 0) {
            destroyed = destroy_random(current_solution, destroy_rate);
        } else {
            destroyed = destroy_worst_segments(current_solution, vectors, destroy_rate);
        }
        
        // Repair
        Individual repaired = repair_greedy(destroyed, vectors);
        float repaired_fitness = compute_fitness_optimized(repaired, vectors, DIMENSION_INPUT);
        
        // Acceptance criteria (simulated annealing-like)
        bool accept = false;
        if (repaired_fitness < current_fitness) {
            accept = true;
        } else {
            float temperature = 100.0f * exp(-iteration / 200.0f);
            float probability = exp(-(repaired_fitness - current_fitness) / temperature);
            uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
            accept = (prob_dist(rng) < probability);
        }
        
        if (accept) {
            current_solution = repaired;
            current_fitness = repaired_fitness;
            
            if (current_fitness < best_fitness) {
                best_solution = current_solution;
                best_fitness = current_fitness;
                no_improvement_count = 0;
                
            }
        } else {
            no_improvement_count++;
        }
        
        // Restart if stuck
        if (no_improvement_count > max_no_improvement) {
            current_solution = generate_greedy_solution(vectors, DIMENSION_INPUT);
            current_fitness = compute_fitness_optimized(current_solution, vectors, DIMENSION_INPUT);
            no_improvement_count = 0;
        }
    }
    return best_solution;
}



GAResult genetic_algorithm_optimized(const vector<vector<float>>& vectors,
                                   int pop_size = POPULATION_SIZE_INPUT,
                                   int chrom_length = CHROMOSOME_LENGTH_INPUT,
                                   int generations = GENERATIONS_INPUT) {
    
    auto start_time = chrono::high_resolution_clock::now();
    
    vector<Individual> population = generate_initial_population_improved(pop_size, chrom_length, vectors, DIMENSION_INPUT);
    
    vector<float> fitness_history;
    fitness_history.reserve(generations);
    
    float best_fitness = 1e6f;
    Individual best_individual;
    int stagnation_counter = 0;
    float current_mutation_rate = MUTATION_RATE_INPUT;
    // Define the metaheuristic methods "vns" , 
    vector<string> methods = { "tabu","lns"};
    
    for (int gen = 0; gen < generations; ++gen) {
        // Evaluate fitness for all individuals
        for (auto& individual : population) {
            compute_fitness_optimized(individual, vectors, DIMENSION_INPUT);
        }
        
        
        // Sort population by fitness
        sort(population.begin(), population.end(), 
             [&](Individual& a, Individual& b) {
                 return compute_fitness_optimized(a, vectors, DIMENSION_INPUT) < 
                        compute_fitness_optimized(b, vectors, DIMENSION_INPUT);
             });
        
        // Track best fitness
        float gen_best_fitness = compute_fitness_optimized(population[0], vectors, DIMENSION_INPUT);
        
        if (gen_best_fitness < best_fitness - 0.5f) {  // Improvement threshold
            best_fitness = gen_best_fitness;
            best_individual = population[0];
            stagnation_counter = 0;
        } else {
            stagnation_counter++;
        }
        
        fitness_history.push_back(best_fitness);
        
        // Early stopping
        if (best_fitness < CONVERGENCE_THRESHOLD) {
            cout << "Converged at generation " << gen << " with fitness " << best_fitness << "\n";
            break;
        }
        
        // Adaptive mutation rate
        if (stagnation_counter > 50) {
            current_mutation_rate = min(0.08f, MUTATION_RATE_INPUT * 3.0f);
        } else if (stagnation_counter > 25) {
            current_mutation_rate = min(0.04f, MUTATION_RATE_INPUT * 2.0f);
        } else {
            current_mutation_rate = MUTATION_RATE_INPUT;
        }


        // Apply different metaheuristic approaches every 50 generations
        if (gen % 50 == 0 && gen > 0) {
            cout << "Applying metaheuristics at generation " << gen << "..." << endl;
            
            // Apply to best 3-5 individuals
            int individuals_to_improve = min(5, max(3, (int)population.size() / 10));
            
            for (int i = 0; i < individuals_to_improve; ++i) {
                // Randomly select a method
                uniform_int_distribution<int> method_dist(0, methods.size() - 1);
                string method = methods[method_dist(rng)];
                
                Individual improved;
                
                if (method == "tabu") {
                    cout << "  Applying Tabu Search to individual " << i << endl;
                    improved = tabu_search(vectors, population[i], 50);  // Reduced iterations for integration
                } else if (method == "vns") {
                    cout << "  Applying VNS to individual " << i << endl;
                    improved = variable_neighborhood_search(vectors, population[i], 100);  // Reduced iterations
                } else { // lns
                    cout << "  Applying LNS to individual " << i << endl;
                    improved = large_neighborhood_search(vectors, population[i], 75);  // Reduced iterations
                }
                
                float improved_fitness = compute_fitness_optimized(improved, vectors, DIMENSION_INPUT);
                float original_fitness = compute_fitness_optimized(population[i], vectors, DIMENSION_INPUT);
                
                if (improved_fitness < original_fitness) {
                    cout << "    Improvement found: " << original_fitness << " -> " << improved_fitness << endl;
                    population[i] = improved;
                    
                    // Update global best if necessary
                    if (improved_fitness < best_fitness) {
                        best_fitness = improved_fitness;
                        best_individual = improved;
                        stagnation_counter = 0;
                        cout << "    New global best: " << best_fitness << endl;
                    }
                } else {
                    cout << "    No improvement: " << original_fitness << " vs " << improved_fitness << endl;
                }
            }
            
            // Re-sort population after improvements
            sort(population.begin(), population.end(), 
                 [&](Individual& a, Individual& b) {
                     return compute_fitness_optimized(a, vectors, DIMENSION_INPUT) < 
                            compute_fitness_optimized(b, vectors, DIMENSION_INPUT);
                 });
        }
        
        // Apply original local search occasionally (less frequently now)
        if (gen % 100 == 0 && gen > 0) {
            for (int i = 0; i < min(2, (int)population.size()); ++i) {
                Individual improved = local_search_aggressive(population[i], vectors, 30);
                float improved_fitness = compute_fitness_optimized(improved, vectors, DIMENSION_INPUT);
                
                if (improved_fitness < compute_fitness_optimized(population[i], vectors, DIMENSION_INPUT)) {
                    population[i] = improved;
                    if (i == 0 && improved_fitness < best_fitness) {
                        best_fitness = improved_fitness;
                        best_individual = improved;
                        stagnation_counter = 0;
                    }
                }
            }
        }
        
        // Restart if stagnation is too long
        if (stagnation_counter > STAGNATION_LIMIT) {
            cout << "Restarting due to stagnation at generation " << gen << "\n";
            
            // Keep best 15% of population
            int keep_count = max(1, pop_size / 7);
            vector<Individual> new_population = generate_initial_population_improved(pop_size - keep_count, chrom_length, vectors, DIMENSION_INPUT);
            
            for (int i = 0; i < keep_count; ++i) {
                new_population.push_back(population[i]);
            }
            
            population = new_population;
            stagnation_counter = 0;
            current_mutation_rate = MUTATION_RATE_INPUT * 2.0f;
        }
        
        // Generate new population
        vector<Individual> new_population;
        new_population.reserve(pop_size);
        
        // Elitism: keep best individuals
        int elite_count = max(2, pop_size / 8);
        for (int i = 0; i < elite_count; ++i) {
            new_population.push_back(population[i]);
        }
        
        // Generate offspring
        while (new_population.size() < pop_size) {
            Individual parent1 = tournament_selection(population, vectors);
            Individual parent2 = tournament_selection(population, vectors);
            
            auto children = uniform_crossover(parent1, parent2);
            
            children.first = mutate(children.first, current_mutation_rate);
            children.second = mutate(children.second, current_mutation_rate);
            
            new_population.push_back(children.first);
            if (new_population.size() < pop_size) {
                new_population.push_back(children.second);
            }
        }
        
        population = new_population;
        
        // Progress reporting
        if (gen % 100 == 0 || gen < 10) {
            cout << "Generation " << gen << ": Best Fitness = " << best_fitness 
                 << ", Mutation Rate = " << current_mutation_rate
                 << ", Stagnation = " << stagnation_counter << "\n";
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    return GAResult{
        best_individual.chromosome,
        best_fitness,
        fitness_history,
        static_cast<int>(fitness_history.size()),
        static_cast<float>(duration.count()) / 1000.0f
    };
}

int main() {
    cout << "Starting Multi-Dimensional Two-Way Number Partitioning GA...\n";
    
    int num_vectors, dimension;
    vector<vector<float>> vectors;

    if (read_vectors_from_file(filename, num_vectors, dimension, vectors)) {
        cout << "Loaded " << num_vectors << " vectors, each of dimension " << dimension << endl;
    } else {
        cout << "Failed to load vectors from file." << endl;
        return 1;
    }
        
    POPULATION_SIZE_INPUT = max(150, min(600, num_vectors));
    CHROMOSOME_LENGTH_INPUT = num_vectors;
    NUM_INPUT_VECTORS_INPUT = num_vectors;
    DIMENSION_INPUT = dimension;
    MUTATION_RATE_INPUT = max(0.008f, min(0.025f, 2.0f / sqrt(static_cast<float>(num_vectors))));
    
    cout << "\n---- GA Configuration ----" << endl;
    cout << "Population Size     : " << POPULATION_SIZE_INPUT << endl;
    cout << "Chromosome Length   : " << CHROMOSOME_LENGTH_INPUT << endl;
    cout << "Generations         : " << GENERATIONS_INPUT << endl;
    cout << "Mutation Rate       : " << MUTATION_RATE_INPUT << endl;
    cout << "-----------------------------\n" << endl;
    
    // Run multiple times and take the best result
    GAResult best_result;
    best_result.best_fitness = 1e9f;
    
    int runs = 10;  // Run 5 times and take the best
    for (int run = 0; run < runs; ++run) {
        cout << "\n=== RUN " << (run + 1) << " ===" << endl;
        
        // Clear cache between runs for fresh start
        fitness_cache.clear();
        
        GAResult result = genetic_algorithm_optimized(vectors);
        
        if (result.best_fitness < best_result.best_fitness) {
            best_result = result;
        }
        
        cout << "Run " << (run + 1) << " Best Fitness: " << result.best_fitness << endl;
    }
    
    // Output the best result
    cout << "\n==== FINAL BEST RESULTS ====" << endl;
    cout << "Best Fitness        : " << best_result.best_fitness << endl;
    cout << "Generations Run     : " << best_result.generations_run << endl;
    cout << "Convergence Time    : " << best_result.convergence_time << " seconds" << endl;
    
    cout << "\nBest Solution: ";
    for (size_t i = 0; i < min(size_t(50), best_result.best_solution.size()); ++i) {
        cout << best_result.best_solution[i];
    }
    if (best_result.best_solution.size() > 50) cout << "...";
    cout << endl;
    
    // Count partition sizes
    int count_0 = 0, count_1 = 0;
    for (int gene : best_result.best_solution) {
        if (gene == 0) count_0++;
        else count_1++;
    }
    cout << "Partition sizes: " << count_0 << " (0s) and " << count_1 << " (1s)" << endl;
    
    // Verify the solution by computing actual fitness
    vector<float> sum_0(dimension, 0.0f), sum_1(dimension, 0.0f);
    for (int i = 0; i < num_vectors; ++i) {
        if (best_result.best_solution[i] == 0) {
            for (int j = 0; j < dimension; ++j) {
                sum_0[j] += vectors[i][j];
            }
        } else {
            for (int j = 0; j < dimension; ++j) {
                sum_1[j] += vectors[i][j];
            }
        }
    }
    
    cout << "\nPartition sums verification (first 5 dimensions):" << endl;
    cout << "Partition 0 sums: ";
    for (int j = 0; j < min(5, dimension); ++j) {
        cout << sum_0[j] << " ";
    }
    cout << endl;
    
    cout << "Partition 1 sums: ";
    for (int j = 0; j < min(5, dimension); ++j) {
        cout << sum_1[j] << " ";
    }
    cout << endl;
    
    cout << "Absolute differences: ";
    float max_diff_verify = 0.0f;
    for (int j = 0; j < min(5, dimension); ++j) {
        float diff = fabs(sum_0[j] - sum_1[j]);
        cout << diff << " ";
        max_diff_verify = max(max_diff_verify, diff);
    }
    cout << endl;
    cout << "Maximum difference: " << max_diff_verify << endl;
    
    return 0;
}


// Best Fitness = 48553-> 500_20d(38k)
// Best Fitness        : 79583.5 50_20c(50k)
















