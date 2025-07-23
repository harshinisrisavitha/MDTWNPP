# MDTWNPP-Multi Dimensional Two Way Number Partitioning Problem 

**MDTWNPP** (Multi-Dimensional Two-Way Number Partitioning Problem)involves dividing a set of vectors into two subsets such that the **difference between the maximum dimension-sums** of each subset is minimized.
>In other words, for each subset, we compute the sum of each dimension across all vectors, and then take the **maximum** among those sums. The objective is to make the difference between these two maxima (from each subset) as small as possible. 


# Metaheuristic Algorithms

**Genetic Algorithm (GA)** :
>-   Inspired by the process of natural selection and genetics.
>-   Uses a population of candidate solutions (chromosomes), evolving over generations via:
>-   **Selection**
> -   **Crossover (recombination)**
> -   **Mutation**
> -   Well-suited for discrete and binary optimization problems.

**Binary Particle Swarm Optimization (BPSO)** 
>-   Based on the social behavior of birds flocking or fish schooling.
>-   Each "particle" represents a solution and moves in the binary search space influenced by: 
>-   Its own best position (personal best)
> -   The global best position in the swarm
> -   Velocity is used to determine the probability of a bit flip (via sigmoid function).

**Binary Bat Algorithm (BBA)**
>-   Inspired by the echolocation behavior of bats.-   Each bat adjusts its position based on: 
> -   A **velocity vector** 
>  -   A **frequency parameter**  
>   -   **Loudness** (controls acceptance of new solutions)
>   -   **Pulse rate** (controls exploration)
>   -   Binarized version adapts the algorithm for bitstring solutions like in MDTWNPP.

## Hardware Optimization(FPGA)
