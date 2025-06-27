#include <iostream>                             //DYNAMIC_PARAMETER FUNCTIONS
#include <fstream>
#include <vector>                               //BETTER_FITNESS(NOT SO GOOD AVG_VALUES)
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

const string filename = "/home/harshini/python_tutorials/cs23b2055/dataset/mdtwnpp_50_10a.txt";

const int DIMENSION = 10;       //vector size
int CHROMO_LENGTH;              //no of vectors
const int GENERATIONS = 50000;    //max iterations for bba
const int RUN = 15;
const int NUM_BATS = 100;

const float A = 1.0f;           //initial loudness
const float r = 0.5f;           //initial pulse rate
const float ALPHA = 0.5f; 
const float FREQ_MIN = 0.0f;
const float FREQ_MAX = 1.0f;

// Global stagnation counter for dynamic parameter adjustment
int global_stagnation_count = 0;
const int STAGNATION_THRESHOLD = 50;  // Threshold for parameter adjustment

random_device rd;
mt19937 rng(rd());

//-----------------------------------STRUCTURES----------------------------------------

struct Bat {
    vector<int> bat;        //binary chromosome
    vector<float> velocity;     
    float fitness = 1e8f;
    bool evaluated = false;
    float loudness = A;
    float pulse_rate = r;
};

struct BBAResult {
    vector<int> best_solution;
    float best_fitness;
    vector<float> fitness_history;
    int generations_run;
    float convergence_time;
};

//-----------------------------------DYNAMIC PARAMETER FUNCTIONS----------------------

// Dynamic loudness based on global stagnation count
float dynamic_loudness(float base_loudness, int stagnation_count) {
    if (stagnation_count == 0) {
        return base_loudness;
    }
    // Increase loudness (exploration) during stagnation
    // Formula: A * (1 + stagnation_factor * log(1 + stagnation_count))
    float stagnation_factor = 0.1f;  // Controls how much stagnation affects loudness
    float multiplier = 1.0f + stagnation_factor * log(1.0f + stagnation_count);
    
    // Cap the maximum loudness to prevent excessive exploration
    return min(base_loudness * multiplier, 2.0f * A);
}

// Dynamic pulse rate based on global stagnation count
float dynamic_pulse_rate(float base_pulse_rate, int stagnation_count) {
    if (stagnation_count == 0) {
        return base_pulse_rate;
    }
    
    // Decrease pulse rate (reduce exploitation) during stagnation to encourage exploration
    // Formula: r * exp(-decay_factor * stagnation_count)
    float decay_factor = 0.01f;  // Controls how much stagnation affects pulse rate
    float multiplier = exp(-decay_factor * stagnation_count);
    
    // Ensure pulse rate doesn't go too low
    return max(base_pulse_rate * multiplier, 0.1f);
}


// Update all bats' parameters based on global stagnation
void update_population_parameters(vector<Bat>& population, int stagnation_count) {
    for (auto& bat : population) {
        bat.loudness = dynamic_loudness(A, stagnation_count);
        bat.pulse_rate = dynamic_pulse_rate(r, stagnation_count);
    }
}

// // Adaptive parameters based on generation and stagnation
// float adaptive_loudness_with_stagnation(int generation, int max_gen, int stagnation_count) {
//     // Base decay over generations
//     float base_loudness = A * exp(-0.5f * generation / max_gen);
//     // Apply stagnation-based adjustment
//     return dynamic_loudness(base_loudness, stagnation_count);
// }
// float adaptive_pulse_rate_with_stagnation(int generation, int max_gen, int stagnation_count) {
//     // Base increase over generations
//     float base_pulse_rate = r * (1.0f - exp(-2.0f * generation / max_gen));
//     // Apply stagnation-based adjustment
//     return dynamic_pulse_rate(base_pulse_rate, stagnation_count);
// }

//-----------------------------------FILE HANDLING--------------------------------------

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

void repair_chromosome(vector<int>& chromosome) {
    int count_A = 0, count_B = 0;
    for (int bit : chromosome) {
        if (bit == 0) count_A++;
        else count_B++;
    }
    
    if (count_A == 0) {
        // All bits are 1, flip one random bit to 0
        int idx = uniform_int_distribution<int>(0, chromosome.size() - 1)(rng);
        chromosome[idx] = 0;
    } else if (count_B == 0) {
        // All bits are 0, flip one random bit to 1
        int idx = uniform_int_distribution<int>(0, chromosome.size() - 1)(rng);
        chromosome[idx] = 1;
    }
}

//----------------------------------- FITNESS METHODS----------------------------------

float compute_fitness_optimized(const vector<int>& chromosome, const vector<vector<float>>& vectors, int dimension) {
    int countA = 0, countB = 0;
    vector<float> sumA(dimension, 0.0f);
    vector<float> sumB(dimension, 0.0f);
    
    for (int i = 0; i < chromosome.size(); i++) {
        if (chromosome[i] == 0) {
            for (int j = 0; j < dimension; j++) {
                sumA[j] += vectors[i][j];
            }
            countA++;
        } else {
            for (int j = 0; j < dimension; j++) {
                sumB[j] += vectors[i][j];
            }
            countB++;
        }
    }
    
    if (countA == 0 || countB == 0) {
        return 1e8f;
    }
    
    float max_diff = 0.0f;
    for (int j = 0; j < dimension; j++) {
        float dim_diff = fabs(sumA[j] - sumB[j]);
        if (dim_diff > max_diff)
            max_diff = dim_diff;
    }
    
    return max_diff;
}

float fitness_fn(Bat& bat, const vector<vector<float>>& vectors, int dimension) {
    if (bat.evaluated) {
        return bat.fitness;
    }
    
    bat.fitness = compute_fitness_optimized(bat.bat, vectors, dimension);
    bat.evaluated = true;
    return bat.fitness;
}

// -------------------------------------SEARCH METHODS-----------------------------------
// Bat local_search(Bat individual, const vector<vector<float>>& vectors, int max_iterations = 50) {
//     float current_fitness = compute_fitness_optimized(individual.bat, vectors, DIMENSION);
//     bool improved = true;
//     int iterations = 0;
//     while (improved && iterations < max_iterations) {
//         improved = false;
//         // Try single bit flips
//         for (size_t i = 0; i < individual.bat.size(); ++i) {
//             Bat neighbor = individual;
//             neighbor.bat[i] = 1 - neighbor.bat[i];
//             neighbor.evaluated = false;
//             // Check if partitions are still valid
//             int count_0 = count(neighbor.bat.begin(), neighbor.bat.end(), 0);
//             int count_1 = neighbor.bat.size() - count_0;
//             if (count_0 > 0 && count_1 > 0) {
//                 float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
//                 if (neighbor_fitness < current_fitness) {
//                     individual = neighbor;
//                     individual.fitness = neighbor_fitness;
//                     individual.evaluated = true;
//                     current_fitness = neighbor_fitness;
//                     improved = true;
//                     break;
//                 }
//             }
//         }
//         // If no single flip helped, try swaps 
//         if (!improved && iterations < max_iterations / 2) {
//             for (size_t i = 0; i < individual.bat.size() - 1; ++i) {
//                 for (size_t j = i + 1; j < individual.bat.size(); ++j) {
//                     if (individual.bat[i] != individual.bat[j]) {
//                         Bat neighbor = individual;
//                         swap(neighbor.bat[i], neighbor.bat[j]);
//                         neighbor.evaluated = false;
//                         float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
//                         if (neighbor_fitness < current_fitness) {
//                             individual = neighbor;
//                             individual.fitness = neighbor_fitness;
//                             individual.evaluated = true;
//                             current_fitness = neighbor_fitness;
//                             improved = true;
//                             goto next_iteration;
//                         }
//                     }
//                 }
//             }
//         }
//         next_iteration:
//         iterations++;
//     }
//     return individual;
// }

// Function 1: Enhanced local search with k-flips
Bat local_search_with_kflips(Bat individual, const vector<vector<float>>& vectors, int max_iterations = 50, int max_k = 3) {
    float current_fitness = compute_fitness_optimized(individual.bat, vectors, DIMENSION);
    bool improved = true;
    int iterations = 0;
    
    while (improved && iterations < max_iterations) {
        improved = false;
        
        // Try single bit flips
        for (size_t i = 0; i < individual.bat.size(); ++i) {
            Bat neighbor = individual;
            neighbor.bat[i] = 1 - neighbor.bat[i];
            neighbor.evaluated = false;
            
            // Check if partitions are still valid
            int count_0 = count(neighbor.bat.begin(), neighbor.bat.end(), 0);
            int count_1 = neighbor.bat.size() - count_0;
            if (count_0 > 0 && count_1 > 0) {
                float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
                if (neighbor_fitness < current_fitness) {
                    individual = neighbor;
                    individual.fitness = neighbor_fitness;
                    individual.evaluated = true;
                    current_fitness = neighbor_fitness;
                    improved = true;
                    break;
                }
            }
        }
        
        // If no single flip helped, try swaps 
        if (!improved && iterations < max_iterations / 2) {
            for (size_t i = 0; i < individual.bat.size() - 1; ++i) {
                for (size_t j = i + 1; j < individual.bat.size(); ++j) {
                    if (individual.bat[i] != individual.bat[j]) {
                        Bat neighbor = individual;
                        swap(neighbor.bat[i], neighbor.bat[j]);
                        neighbor.evaluated = false;
                        float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
                        if (neighbor_fitness < current_fitness) {
                            individual = neighbor;
                            individual.fitness = neighbor_fitness;
                            individual.evaluated = true;
                            current_fitness = neighbor_fitness;
                            improved = true;
                            goto next_iteration;
                        }
                    }
                }
            }
        }
        
        // Try k-flips (2 to max_k bits) if still no improvement and in early iterations
        if (!improved && iterations < max_iterations / 3) {
            for (int k = 2; k <= max_k && k <= (int)individual.bat.size(); ++k) {
                vector<size_t> indices(individual.bat.size());
                iota(indices.begin(), indices.end(), 0);
                
                // Try a limited number of random k-flip combinations to avoid exponential complexity
                int max_tries = min(100, (int)individual.bat.size() * k);
                for (int try_count = 0; try_count < max_tries; ++try_count) {
                    // Generate random k indices
                    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
                    
                    Bat neighbor = individual;
                    for (int flip = 0; flip < k; ++flip) {
                        neighbor.bat[indices[flip]] = 1 - neighbor.bat[indices[flip]];
                    }
                    neighbor.evaluated = false;
                    
                    // Check if partitions are still valid
                    int count_0 = count(neighbor.bat.begin(), neighbor.bat.end(), 0);
                    int count_1 = neighbor.bat.size() - count_0;
                    if (count_0 > 0 && count_1 > 0) {
                        float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
                        if (neighbor_fitness < current_fitness) {
                            individual = neighbor;
                            individual.fitness = neighbor_fitness;
                            individual.evaluated = true;
                            current_fitness = neighbor_fitness;
                            improved = true;
                            goto next_iteration;
                        }
                    }
                }
                if (improved) break;
            }
        }
        
        next_iteration:
        iterations++;
    }
    return individual;
}

// Function 2: Dimension-based local search
Bat local_search_dimension_based(Bat individual, const vector<vector<float>>& vectors, int max_iterations = 50) {
    float current_fitness = compute_fitness_optimized(individual.bat, vectors, DIMENSION);
    bool improved = true;
    int iterations = 0;
    while (improved && iterations < max_iterations) {
        improved = false;
        // Calculate current partition centroids and analyze dimensions
        vector<float> centroid_0(DIMENSION, 0.0f);
        vector<float> centroid_1(DIMENSION, 0.0f);
        int count_0 = 0, count_1 = 0;
        for (size_t i = 0; i < individual.bat.size(); ++i) {
            if (individual.bat[i] == 0) {
                for (int d = 0; d < DIMENSION; ++d) {
                    centroid_0[d] += vectors[i][d];
                }
                count_0++;
            } else {
                for (int d = 0; d < DIMENSION; ++d) {
                    centroid_1[d] += vectors[i][d];
                }
                count_1++;
            }
        }
        if (count_0 > 0) {
            for (int d = 0; d < DIMENSION; ++d) {
                centroid_0[d] /= count_0;
            }
        }
        if (count_1 > 0) {
            for (int d = 0; d < DIMENSION; ++d) {
                centroid_1[d] /= count_1;
            }
        }
        // Find points that might benefit from moving based on dimensional analysis
        vector<pair<size_t, float>> candidates;
        for (size_t i = 0; i < individual.bat.size(); ++i) {
            float dist_to_current = 0.0f;
            float dist_to_other = 0.0f;
            // Calculate distance to current partition centroid and other partition centroid
            if (individual.bat[i] == 0) {
                for (int d = 0; d < DIMENSION; ++d) {
                    float diff_current = vectors[i][d] - centroid_0[d];
                    float diff_other = vectors[i][d] - centroid_1[d];
                    dist_to_current += diff_current * diff_current;
                    dist_to_other += diff_other * diff_other;
                }
            } else {
                for (int d = 0; d < DIMENSION; ++d) {
                    float diff_current = vectors[i][d] - centroid_1[d];
                    float diff_other = vectors[i][d] - centroid_0[d];
                    dist_to_current += diff_current * diff_current;
                    dist_to_other += diff_other * diff_other;
                }
            }
            // If point is closer to the other partition's centroid, it's a candidate for moving
            if (dist_to_other < dist_to_current) {
                candidates.push_back({i, dist_to_current - dist_to_other});
            }
        }
        // Sort candidates by potential improvement (highest difference first)
        sort(candidates.begin(), candidates.end(), 
             [](const pair<size_t, float>& a, const pair<size_t, float>& b) {
                 return a.second > b.second;
             });
        // Try moving the most promising candidates
        for (const auto& candidate : candidates) {
            size_t i = candidate.first;
            Bat neighbor = individual;
            neighbor.bat[i] = 1 - neighbor.bat[i];
            neighbor.evaluated = false;
            // Check if partitions are still valid
            int new_count_0 = count(neighbor.bat.begin(), neighbor.bat.end(), 0);
            int new_count_1 = neighbor.bat.size() - new_count_0;
            if (new_count_0 > 0 && new_count_1 > 0) {
                float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
                if (neighbor_fitness < current_fitness) {
                    individual = neighbor;
                    individual.fitness = neighbor_fitness;
                    individual.evaluated = true;
                    current_fitness = neighbor_fitness;
                    improved = true;
                    break;
                }
            }
        }
        // If no single move helped, try dimension-weighted swaps
        if (!improved && iterations < max_iterations / 2) {
            // Calculate dimension weights based on variance
            vector<float> dim_weights(DIMENSION, 0.0f);
            for (int d = 0; d < DIMENSION; ++d) {
                float mean_d = 0.0f;
                for (size_t i = 0; i < vectors.size(); ++i) {
                    mean_d += vectors[i][d];
                }
                mean_d /= vectors.size();
                float variance = 0.0f;
                for (size_t i = 0; i < vectors.size(); ++i) {
                    float diff = vectors[i][d] - mean_d;
                    variance += diff * diff;
                }
                dim_weights[d] = variance / vectors.size();
            }
            // Try swaps prioritizing high-variance dimensions
            vector<pair<size_t, float>> swap_candidates_0, swap_candidates_1;
            for (size_t i = 0; i < individual.bat.size(); ++i) {
                float weighted_score = 0.0f;
                for (int d = 0; d < DIMENSION; ++d) {
                    weighted_score += vectors[i][d] * dim_weights[d];
                }
                if (individual.bat[i] == 0) {
                    swap_candidates_0.push_back({i, weighted_score});
                } else {
                    swap_candidates_1.push_back({i, weighted_score});
                }
            }
            // Sort by weighted score
            sort(swap_candidates_0.begin(), swap_candidates_0.end(),
                 [](const pair<size_t, float>& a, const pair<size_t, float>& b) {
                     return a.second > b.second;
                 });
            sort(swap_candidates_1.begin(), swap_candidates_1.end(),
                 [](const pair<size_t, float>& a, const pair<size_t, float>& b) {
                     return a.second < b.second;
                 });
            // Try swapping high-scoring 0s with low-scoring 1s
            int max_swaps = min(10, min((int)swap_candidates_0.size(), (int)swap_candidates_1.size()));
            for (int s = 0; s < max_swaps; ++s) {
                Bat neighbor = individual;
                swap(neighbor.bat[swap_candidates_0[s].first], neighbor.bat[swap_candidates_1[s].first]);
                neighbor.evaluated = false;
                float neighbor_fitness = compute_fitness_optimized(neighbor.bat, vectors, DIMENSION);
                if (neighbor_fitness < current_fitness) {
                    individual = neighbor;
                    individual.fitness = neighbor_fitness;
                    individual.evaluated = true;
                    current_fitness = neighbor_fitness;
                    improved = true;
                    break;
                }
            }
        }
        iterations++;
    }
    return individual;
}


// Destroy operator - removes a portion of the solution
void destroy_solution(vector<int>& solution, float destroy_ratio) {
    int destroy_count = (int)(solution.size() * destroy_ratio);
    if (destroy_count == 0) destroy_count = 1;
    
    // Choose random positions to destroy
    unordered_set<int> positions_to_destroy;
    while (positions_to_destroy.size() < destroy_count) {
        int pos = uniform_int_distribution<int>(0, solution.size() - 1)(rng);
        positions_to_destroy.insert(pos);
    }
    
    // Mark destroyed positions (we'll use -1 to indicate destroyed)
    for (int pos : positions_to_destroy) {
        solution[pos] = -1;
    }
}

// Repair operator - reconstructs the destroyed solution using greedy approach
void repair_solution(vector<int>& solution, const vector<vector<float>>& vectors, int dimension) {
    vector<int> destroyed_positions;
    
    // Find destroyed positions
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i] == -1) {
            destroyed_positions.push_back(i);
        }
    }
    
    if (destroyed_positions.empty()) return;
    
    // Current sums for existing assignments
    vector<float> sumA(dimension, 0.0f), sumB(dimension, 0.0f);
    int countA = 0, countB = 0;
    
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i] == 0) {
            for (int j = 0; j < dimension; j++) {
                sumA[j] += vectors[i][j];
            }
            countA++;
        } else if (solution[i] == 1) {
            for (int j = 0; j < dimension; j++) {
                sumB[j] += vectors[i][j];
            }
            countB++;
        }
    }
    
    // Greedily assign destroyed positions
    for (int pos : destroyed_positions) {
        // Try assigning to group A
        vector<float> temp_sumA = sumA;
        for (int j = 0; j < dimension; j++) {
            temp_sumA[j] += vectors[pos][j];
        }
        
        // Try assigning to group B
        vector<float> temp_sumB = sumB;
        for (int j = 0; j < dimension; j++) {
            temp_sumB[j] += vectors[pos][j];
        }
        
        // Calculate max differences for both assignments
        float max_diff_A = 0.0f, max_diff_B = 0.0f;
        for (int j = 0; j < dimension; j++) {
            max_diff_A = max(max_diff_A, fabs(temp_sumA[j] - sumB[j]));
            max_diff_B = max(max_diff_B, fabs(sumA[j] - temp_sumB[j]));
        }
        
        // Choose assignment that gives better (lower) max difference
        if (max_diff_A <= max_diff_B) {
            solution[pos] = 0;
            sumA = temp_sumA;
            countA++;
        } else {
            solution[pos] = 1;
            sumB = temp_sumB;
            countB++;
        }
    }
    
    // Ensure both partitions are non-empty
    repair_chromosome(solution);
}

// LNS operator - combines destroy and repair
Bat large_neighborhood_search(Bat individual, const vector<vector<float>>& vectors, float destroy_ratio = 0.3f, int max_iterations = 10) {
    Bat best_solution = individual;
    float best_fitness = compute_fitness_optimized(individual.bat, vectors, DIMENSION);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        Bat current_solution = individual;
        
        // Destroy phase
        destroy_solution(current_solution.bat, destroy_ratio);
        
        // Repair phase
        repair_solution(current_solution.bat, vectors, DIMENSION);
        current_solution.evaluated = false;
        
        // Evaluate repaired solution
        float current_fitness = compute_fitness_optimized(current_solution.bat, vectors, DIMENSION);
        
        if (current_fitness < best_fitness) {
            best_solution = current_solution;
            best_solution.fitness = current_fitness;
            best_solution.evaluated = true;
            best_fitness = current_fitness;
            individual = best_solution; // Update base for next iteration
        }
        
        // Adaptive destroy ratio based on improvement
        if (current_fitness < best_fitness * 1.1f) { // If reasonably good
            destroy_ratio = min(0.5f, destroy_ratio * 1.1f); // Increase exploration
        } else {
            destroy_ratio = max(0.1f, destroy_ratio * 0.9f); // Decrease exploration
        }
    }
    
    return best_solution;
}



//------------------------------------POPULATION_GENERATION METHODS---------------------
Bat generate_greedy(const vector<vector<float>>& vectors, int dimension) {
    int chrom_length = vectors.size();
    vector<int> chromosome(chrom_length, 0);
    vector<float> sumA(dimension, 0.0f), sumB(dimension, 0.0f);
    int count_A = 0, count_B = 0;

    // Start with first vector in group A
    chromosome[0] = 0;
    count_A = 1;
    for (int j = 0; j < dimension; j++) {
        sumA[j] = vectors[0][j];
    }
    
    for (int i = 1; i < chrom_length; ++i) {
        vector<float> temp_sumA = sumA;
        vector<float> temp_sumB = sumB;

        //adding vector i to group A
        for (int j = 0; j < dimension; ++j) {
            temp_sumA[j] += vectors[i][j];
        }
        
        //adding vector i to group B
        for (int j = 0; j < dimension; ++j) {
            temp_sumB[j] += vectors[i][j];
        }

        float max_diff_A = 0.0f;  // If added to A
        float max_diff_B = 0.0f;  // If added to B

        for (int j = 0; j < dimension; ++j) {
            max_diff_A = max(max_diff_A, fabs(temp_sumA[j] - sumB[j]));
            max_diff_B = max(max_diff_B, fabs(sumA[j] - temp_sumB[j]));
        }

        if (max_diff_A <= max_diff_B) {
            chromosome[i] = 0;
            for (int j = 0; j < dimension; ++j)
                sumA[j] += vectors[i][j];
            count_A++;
        } else {
            chromosome[i] = 1;
            for (int j = 0; j < dimension; ++j)
                sumB[j] += vectors[i][j];
            count_B++;
        }
    }

    // Ensure both partitions are non-empty
    repair_chromosome(chromosome);

    Bat b;
    b.bat = chromosome;
    b.velocity.assign(chrom_length, 0.0f);
    return b;
}

vector<Bat> generate_population(int pop_size, int chrom_length, const vector<vector<float>>& vectors, int dimension) {
    vector<Bat> population;
    population.reserve(pop_size);
    
    uniform_int_distribution<int> bit_dist(0, 1);
    uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    // Adjust population composition to fit within pop_size
    int greedy_count = pop_size * 3 / 10;  
    int balanced_count = pop_size * 3 /10;
    int semi_balanced_count = pop_size * 2/10;
    int random_count = pop_size - (greedy_count + balanced_count + semi_balanced_count);

    // Generate greedy solution
    for (int i = 0; i < greedy_count; ++i) {
        Bat greedy = generate_greedy(vectors, dimension);
        population.push_back(greedy);
    }
    
    // Balanced solutions
    for (int i = 0; i < balanced_count; ++i) {
        vector<int> chromosome(chrom_length);
        int target_ones = chrom_length / 2 + (rng() % 21) - 10; // ±10 variation
        target_ones = max(1, min(chrom_length - 1, target_ones));

        fill(chromosome.begin(), chromosome.begin() + target_ones, 1);
        fill(chromosome.begin() + target_ones, chromosome.end(), 0);
        shuffle(chromosome.begin(), chromosome.end(), rng);

        Bat b;
        b.bat = chromosome;
        b.velocity.assign(chrom_length, 0.0f);
        population.push_back(b);
    }
    
    // Semi-balanced solutions
    for (int i = 0; i < semi_balanced_count; ++i) {
        vector<int> chromosome(chrom_length);
        float bias = 0.3f + prob_dist(rng) * 0.4f; // 0.3 to 0.7 bias

        for (int j = 0; j < chrom_length; ++j) {
            chromosome[j] = (prob_dist(rng) < bias) ? 1 : 0;
        }

        repair_chromosome(chromosome);

        Bat b;
        b.bat = chromosome;
        b.velocity.assign(chrom_length, 0.0f);
        population.push_back(b);
    }
    
    // Random solutions
    for (int i = 0; i < random_count; ++i) {
        vector<int> chromosome(chrom_length);
        for (int j = 0; j < chrom_length; ++j) {
            chromosome[j] = bit_dist(rng);
        }

        repair_chromosome(chromosome);

        Bat b;
        b.bat = chromosome;
        b.velocity.assign(chrom_length, 0.0f);
        population.push_back(b);
    }

    // cout << "Population generated: " << population.size() << " individuals" << endl;
    return population;
}

//-------------------------------------TRANSFER FUNCTION-------------------------------
float sigmoid(float v) {
    return fabs(v / sqrt(1 + v * v));
}

float time_varying_transfer(float v, int current_gen, int max_gen, float alpha = ALPHA) {
    float decay = 1.0f + alpha * (1.0f - static_cast<float>(current_gen) / max_gen);
    float v_shaped = fabs(v / sqrt(1.0f + v * v));
    float result = decay * v_shaped;
    return std::min(result, 1.0f);  // Keep in [0,1]
}

//-------------------------------------BBA ALGORITHM-----------------------------------
BBAResult binary_bat_run(const vector<vector<float>>& vectors, int pop_size = NUM_BATS, int chrom_length = CHROMO_LENGTH, int generations = GENERATIONS) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Reset global stagnation count for each run
    global_stagnation_count = 0;
    
    vector<Bat> population = generate_population(pop_size, chrom_length, vectors, DIMENSION);

    vector<float> fitness_history;
    fitness_history.reserve(generations);
    
    float best_fitness = 1e6f;
    float previous_best = 1e6f;
    Bat best_individual;

    // Evaluate initial population
    for (int i = 0; i < pop_size; ++i) {
        population[i].fitness = fitness_fn(population[i], vectors, DIMENSION);
        if (population[i].fitness < best_fitness) {
            best_fitness = population[i].fitness;
            best_individual = population[i];
        }
    }
    
    for (int t = 0; t < generations; t++) {
        // Update global stagnation count
        if (t > 0 && abs(best_fitness - previous_best) < 1e-6) {
            global_stagnation_count++;
        } else {
            global_stagnation_count = 0;
        }
        previous_best = best_fitness;
        
        // Update population parameters based on global stagnation
        update_population_parameters(population, global_stagnation_count);
        
        //-------------------------------population------------------------------------------
        for (int i = 0; i < pop_size; i++) {
            // Update frequency
            float freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            
            // Update velocity
            for (int j = 0; j < chrom_length; ++j) {
                int bit_diff = population[i].bat[j] ^ best_individual.bat[j];
                population[i].velocity[j] += bit_diff * freq;
            }
            
            // Update position using transfer function (sigmoid)
            for (int j = 0; j < chrom_length; ++j) {
                // population[i].bat[j] = (uniform_real_distribution<float>(0.0f, 1.0f)(rng) < sigmoid(population[i].velocity[j])) ? 1 : 0;
                 if (uniform_real_distribution<float>(0.0f, 1.0f)(rng) < time_varying_transfer(population[i].velocity[j],t,generations)) {
                population[i].bat[j] = 1 - population[i].bat[j]; // Flip the bit
            }
            }
            
            repair_chromosome(population[i].bat);
            population[i].evaluated = false;
            
            // Local search with probability based on dynamic loudness
            if (uniform_real_distribution<float>(0.0f, 1.0f)(rng) > population[i].pulse_rate) {
                // Generate new solution around best solution
                // Mutation rate increases with stagnation
                // float mutation_rate = 0.1f + 0.05f * min(global_stagnation_count / 10.0f, 1.0f);

                if (uniform_real_distribution<float>(0.0f, 1.0f)(rng) > population[i].loudness) {
                population[i] = local_search_with_kflips(population[i], vectors, 20); // light local search
                population[i].evaluated = false;
                }

                repair_chromosome(population[i].bat);
                population[i].evaluated = false;
            }

            // local search during stagnation
            float local_search_prob = 0.3f + 0.2f * min(global_stagnation_count / 20.0f, 1.0f);
            if (t % 10 == 0 && uniform_real_distribution<float>(0.0f, 1.0f)(rng) < local_search_prob) {
                population[i] = local_search_with_kflips(population[i], vectors, 30);
            }


            // Evaluate new solution
            population[i].fitness = fitness_fn(population[i], vectors, DIMENSION);
            
            // Accept new solution if better and random condition met
            if (population[i].fitness < best_fitness && uniform_real_distribution<float>(0.0f, 1.0f)(rng) < population[i].loudness) {
                
                best_fitness = population[i].fitness;
                best_individual = population[i];
                global_stagnation_count = 0;  
                // Update individual loudness and pulse rate (less aggressive since we have global control)
                population[i].loudness = max(0.05f, population[i].loudness * 0.95f);
                population[i].pulse_rate = min(0.95f, population[i].pulse_rate * 1.05f);
            }
        }
        //-----------------------------------------------------------------------------------------------------
        fitness_history.push_back(best_fitness);
        
        // Enhanced stagnation handling with dynamic thresholds
        int dynamic_threshold = STAGNATION_THRESHOLD + (t / 1000) * 10; // Increase threshold over time
        
        if (global_stagnation_count > dynamic_threshold) {
            
            // Keep the best individual and regenerate the rest
            sort(population.begin(), population.end(), [](const Bat& a, const Bat& b) {
            return a.fitness < b.fitness; // Ascending fitness (lower is better)
            });

            int num_elites = max(1, static_cast<int>(pop_size * 0.05));
            vector<Bat> elites(population.begin(), population.begin() + num_elites);
            population = generate_population(pop_size, chrom_length, vectors, DIMENSION);
            
            // Replace multiple individuals with elite and its variations
            for (int i = 0; i < num_elites; ++i) {
            population[i] = elites[i];
            }

            population[0] = local_search_dimension_based(population[0], vectors, 10); 
            population[0].evaluated = false;
            
            // Apply aggressive local search to elite and random individuals
            for (int ls_i = 0; ls_i < min(5, pop_size); ++ls_i) {
                population[ls_i] = local_search_with_kflips(population[ls_i], vectors, 50);
            }
            
            // Reset stagnation counter
            global_stagnation_count = 0;
            
            // Re-evaluate best fitness from new population
            for (int i = 0; i < pop_size; ++i) {
                if (!population[i].evaluated) {
                    population[i].fitness = fitness_fn(population[i], vectors, DIMENSION);
                }
                if (population[i].fitness < best_fitness) {
                    best_fitness = population[i].fitness;
                    best_individual = population[i];
                }
            }
        }

        if (t % 1000 == 0) {
            cout << "Generation " << t << ", Best fitness: " << best_fitness 
                 << ", Stagnation: " << global_stagnation_count << endl;
        }
    }
    
    // Final aggressive local search on best solution
    cout << "Applying final aggressive local search on best solution..." << endl;
    best_individual = local_search_with_kflips(best_individual, vectors, 100);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    BBAResult result;
    result.best_solution = best_individual.bat;
    result.best_fitness = best_individual.fitness;
    result.fitness_history = fitness_history;
    result.convergence_time = duration.count();
    result.generations_run = min((int)fitness_history.size(), generations);
    
    return result;
}

int main() {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Load dataset vectors
    vector<vector<float>> vectors;
    int num_vectors, dimension;
    
    if (!read_vectors_from_file(filename, num_vectors, dimension, vectors)) {
        cerr << "Failed to load dataset. Exiting." << endl;
        return 1;
    }
    cout << "Loaded " << num_vectors << " vectors, each of dimension " << dimension << endl;
    cout<<"filename"<<filename<<endl;
    CHROMO_LENGTH = num_vectors;  // Set chromosome length to number of vectors
    
    cout << "Enhanced Binary Bat Algorithm with Dynamic Loudness and Pulse Rate\n";
    cout << "===================================================================\n";
    cout << "Dataset size: " << vectors.size() << " samples\n";
    cout << "Dimensions: " << (vectors.empty() ? 0 : vectors[0].size()) << "\n";
    cout << "Population size: " << NUM_BATS << "\n";
    cout << "Chromosome length: " << CHROMO_LENGTH << "\n";
    cout << "Generations: " << GENERATIONS << "\n";
    cout << "Dynamic parameter adjustment based on global stagnation count\n\n";
    
    vector<BBAResult> results;
    results.reserve(RUN);
    
    // Run multiple independent trials
    for (int run = 1; run <= RUN; ++run) {
        cout << "Run " << run << "/" << RUN << "\n";
        cout << "--------\n";
        
        auto start_time = chrono::high_resolution_clock::now();
        
        // Execute enhanced binary bat algorithm with dynamic parameters
        BBAResult result = binary_bat_run(vectors, NUM_BATS, CHROMO_LENGTH, GENERATIONS);
        cout << "=========================================================\n";
        cout << "Algorithm execution completed" << endl;
        results.push_back(result);
        
        auto end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        // Display results for this run
        cout<<"filename"<<filename<<endl;
        cout << "Best fitness: " << result.best_fitness << "\n";
        cout << "Generations completed: " << result.generations_run << "\n";
        cout << "Execution time: " << total_duration.count() << " ms\n";
        
        // Count selected features
        int selected_features = 0;
        for (int i = 0; i < result.best_solution.size(); ++i) {
            if (result.best_solution[i] == 1) {
                selected_features++;
            }
        }
        cout << "Selected features: " << selected_features << "/" << result.best_solution.size() << "\n";
        
        // Show feature selection pattern (first 20 features for brevity)
        cout << "Feature pattern (first 20): ";
        for (int i = 0; i < min(20, (int)result.best_solution.size()); ++i) {
            cout << result.best_solution[i];
        }
        if (result.best_solution.size() > 20) {
            cout << "...";
        }
        cout << "\n\n";
    }
    
    // Analyze results across all runs
    cout << "Summary Across All Runs\n";
    cout << "=======================\n";
    
    // Find best, worst, and average fitness
    float best_fitness = results[0].best_fitness;
    float worst_fitness = results[0].best_fitness;
    float total_fitness = 0.0f;
    int best_run = 0;
    
    for (int i = 0; i < results.size(); ++i) {
        total_fitness += results[i].best_fitness;
        
        if (results[i].best_fitness < best_fitness) {
            best_fitness = results[i].best_fitness;
            best_run = i;
        }
        
        if (results[i].best_fitness > worst_fitness) {
            worst_fitness = results[i].best_fitness;
        }
    }
    
    float average_fitness = total_fitness / results.size();
    
    cout << "Best fitness: " << best_fitness << " (Run " << (best_run + 1) << ")\n";
    cout << "Worst fitness: " << worst_fitness << "\n";
    cout << "Average fitness: " << average_fitness << "\n";
    
    
    return 0;
}

