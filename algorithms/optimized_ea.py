import random
import networkx as nx
from collections import defaultdict
from deap import base, creator, tools, algorithms
import copy
import numpy as np
import math # Import math for infinity

# Define a large constant for penalty (effectively infinity for practical path lengths)
CONFLICT_PENALTY_BASE = 1e9


class PathBasedEA_DEAP():

    def __init__(self, graph, robots, population_size=50, generations=50, p_cross=0.8, p_mut=0.2):
        """
        Initializes the Path-Based Evolutionary Algorithm.

        Args:
            graph (GraphWrapper): Wrapper containing the NetworkX graph.
            robots (list[Robot]): List of Robot objects with IDs, start, and targets.
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run the algorithm.
            p_cross (float): Probability of crossover.
            p_mut (float): Probability of mutation.
        """
        self.graph = graph
        self.robots = robots
        self.robot_map = {r.robot_id: r for r in robots} 
        self.population_size = population_size
        self.generations = generations
        self.p_crossover = p_cross
        self.p_mutation = p_mut


        # --- DEAP Setup ---
        # Create FitnessMin: objective is to minimize the value(s)
        # weights=(-1.0,) means minimize the first (and only) objective value
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Individual is a dictionary mapping robot_id -> path (list of nodes)
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        
        self.toolbox = base.Toolbox()
        # Register functions for creating individuals and populations
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_fitness) # Renamed for clarity
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutation)
        self.toolbox.register("select", tools.selRoulette)

    def generate_individual(self):
        """
        Generates a single individual (potential solution).
        Each robot gets a path generated using a heuristic on a randomly weighted graph.
        The individual is then repaired to ensure basic validity (connectivity, targets).
        """
        individual = {}
        # Use a temporary graph with randomized weights to encourage diverse starting paths
        temp_graph = self.randomize_graph_weights()
        for robot in self.robots:
            # Use the heuristic to find an initial path for this robot
            path = self.heuristic_path(robot, temp_graph)
            individual[robot.robot_id] = path
        # Repair ensures connectivity and target inclusion, conflict-free is argument
        repaired_individual = self.repair_individual(individual, conflict_repair=True)
        return creator.Individual(repaired_individual)

    def randomize_graph_weights(self):
        """Creates a copy of the graph with randomized edge weights."""
        temp_graph = self.graph.G.copy()
        # Assign random weights to encourage path diversity in initial population
        for u, v in temp_graph.edges():
            temp_graph[u][v]['weight'] = random.uniform(0.5, 2.0) # Example range
        return temp_graph

    def heuristic_path(self, robot, graph):
        """
        Generates a plausible path for a single robot using a nearest-target heuristic.
        Note: This is a simple heuristic, more complex ones (like TSP solvers) could be used.
        """
        current_pos = robot.start
        path = [current_pos]
        # Copy targets to avoid modifying the original robot object's list
        targets_to_visit = robot.targets[:]
        # Optional: Shuffle to add randomness beyond just nearest selection
        # random.shuffle(targets_to_visit)

        while targets_to_visit:
            # Find the target nearest to the current position based on shortest path length
            try:
                next_target = min(targets_to_visit,
                                  key=lambda t: nx.shortest_path_length(graph, current_pos, t, weight='weight'))
                # Find the shortest path segment to that nearest target
                segment = nx.shortest_path(graph, current_pos, next_target, weight='weight')
                # Append the segment (excluding the start node, which is already in path)
                path.extend(segment[1:])
                # Update current position and remove the visited target
                current_pos = next_target
                targets_to_visit.remove(next_target)
            except nx.NetworkXNoPath:
                # Handle cases where a target might be unreachable (e.g., disconnected graph)
                print(f"Warning: Cannot find path for Robot {robot.robot_id} from {current_pos} to targets {targets_to_visit}. Skipping remaining targets.")
                break # Stop trying to path for this robot if one target is unreachable

        return path

    def evaluate_fitness(self, individual):
        # 1. Detect Conflicts
        conflicts = self._detect_conflicts(individual)
        conflict_count = len(conflicts)


        # 3. Calculate the primary objective value (e.g., cost)
        total_distance = sum(len(p) - 1 for p in individual.values() if p)

        if conflict_count > 0:
            fitness_value = total_distance + conflict_count * CONFLICT_PENALTY_BASE # Penalize for conflicts
        else:
            fitness_value = total_distance

            # Alternative cost: Makespan (length of the longest path)
            # makespan = max(len(p) for p in individual.values()) if individual else 0
            # total_distance = makespan

        return (fitness_value,)

    def _detect_conflicts(self, individual):
        """
        Helper function to detect vertex and edge conflicts in a set of paths.

        Args:
            individual (dict): The individual (robot_id -> path) to check.

        Returns:
            list: A list of conflict descriptions (e.g., tuples describing the conflict).
                  Returns an empty list if no conflicts are found.
        """
        conflicts = []
        max_steps = 0
        for path in individual.values():
            if path:
                max_steps = max(max_steps, len(path))

        if max_steps == 0: # Handle empty individual case
             return []

        # Store positions and movements at each timestep
        # vertex_occupation[t][node] = robot_id that occupies 'node' at time 't'
        vertex_occupation = defaultdict(dict)
        # edge_traversal[t][(u, v)] = robot_id that traverses edge u->v ending at time 't'
        edge_traversal = defaultdict(dict)

        for t in range(max_steps):
            current_vertex_occupations = {} # Track occupations for this specific timestep 't'

            # --- Check Vertex Conflicts (including goal blocking) ---
            for robot_id, path in individual.items():
                if t < len(path):
                    current_node = path[t]
                    # Check if another robot is already planned to be at this node at this time
                    if current_node in current_vertex_occupations:
                        conflicts.append(('vertex', t, current_node, robot_id, current_vertex_occupations[current_node]))
                    else:
                         current_vertex_occupations[current_node] = robot_id
                else:
                     # Robot has finished its path, stays at its final node
                     final_node = path[-1] if path else None
                     if final_node is not None:
                         if final_node in current_vertex_occupations:
                             conflicts.append(('vertex_wait', t, final_node, robot_id, current_vertex_occupations[final_node]))
                         else:
                             current_vertex_occupations[final_node] = robot_id # Occupies final spot indefinitely


            vertex_occupation[t] = current_vertex_occupations # Store for next step's edge check

            # --- Check Edge Conflicts (Swapping) ---
            if t > 0:
                current_edge_traversals = {}
                for robot_id, path in individual.items():
                    if t < len(path):
                        current_node = path[t]
                        prev_node = path[t-1]
                        edge = (prev_node, current_node)
                        swap_edge = (current_node, prev_node) # The potential conflicting move

                        # Check if another robot is making the opposite move ending at the same time
                        if swap_edge in current_edge_traversals:
                             conflicts.append(('edge', t, edge, robot_id, current_edge_traversals[swap_edge]))
                        current_edge_traversals[edge] = robot_id

                edge_traversal[t] = current_edge_traversals

        return conflicts


    def crossover(self, ind1, ind2):
        """
        Performs crossover between two individuals.
        Uses a path-based crossover: picks a common node and swaps path segments.
        Repairs the resulting children.
        """
        child1_data, child2_data = {}, {}
        # Ensure ind1 and ind2 are dictionaries if they are DEAP individuals
        ind1_data = dict(ind1)
        ind2_data = dict(ind2)

        for robot_id in self.robot_map.keys():
            path1 = ind1_data.get(robot_id, [])
            path2 = ind2_data.get(robot_id, [])

            # Simple case: if either path is empty, just copy
            if not path1 or not path2:
                child1_data[robot_id] = path1[:]
                child2_data[robot_id] = path2[:]
                continue

            # Find common nodes, excluding the final destination (less likely to be good crossover points)
            common_nodes = list(set(path1[:-1]) & set(path2[:-1]))

            if common_nodes:
                # Choose a random common node as the crossover point
                crossover_vertex = random.choice(common_nodes)
                # Find the first index of the crossover vertex in each path
                idx1 = path1.index(crossover_vertex)
                idx2 = path2.index(crossover_vertex)

                # Swap segments
                child1_path = path1[:idx1+1] + path2[idx2+1:]
                child2_path = path2[:idx2+1] + path1[idx1+1:]
            else:
                # No common nodes (except maybe endpoint), just copy parents
                child1_path = path1[:]
                child2_path = path2[:]

            child1_data[robot_id] = child1_path
            child2_data[robot_id] = child2_path

        # Create new individuals from the generated path data
        child1 = creator.Individual(child1_data)
        child2 = creator.Individual(child2_data)

        # Repair children to ensure connectivity and target inclusion
        # Note: Repair does NOT guarantee conflict-free paths here.
        child1 = creator.Individual(self.repair_individual(child1))
        child2 = creator.Individual(self.repair_individual(child2))

        return child1, child2

    def mutation(self, individual):
        """
        Performs mutation on an individual.
        Randomly chooses a robot and applies one type of mutation:
        - 'rewire': Replaces a path segment with a new shortest path.
        - 'insert_vertex': Inserts a random node into the path.
        - 'delete_vertex': Deletes a random node (not start/end) from the path.
        Repairs the resulting mutant.
        """
        # Ensure we are working with the dictionary data
        mutant_data = dict(individual)

        # Choose a random robot to mutate
        robot_id_to_mutate = random.choice(list(self.robot_map.keys()))
        path = mutant_data.get(robot_id_to_mutate, [])

        if len(path) < 2: # Cannot mutate empty or single-node paths effectively
             return creator.Individual(self.repair_individual(mutant_data)), # Return repaired original

        # Choose a mutation type
        mutation_type = random.choice(['rewire', 'insert_vertex', 'delete_vertex'])

        try:
            if mutation_type == 'rewire' and len(path) >= 3:
                # Select two distinct indices, excluding start (0) and end (-1)
                idx1, idx2 = sorted(random.sample(range(1, len(path) - 1), 2))
                start_node = path[idx1]
                end_node = path[idx2]
                # Find a new shortest path between these nodes (using original graph weights)
                new_segment = self.graph.shortest_path(start_node, end_node)
                # Replace the old segment with the new one
                mutated_path = path[:idx1] + new_segment + path[idx2+1:]
                mutant_data[robot_id_to_mutate] = mutated_path

            elif mutation_type == 'insert_vertex':
                # Choose a random node from the graph to insert
                # Avoid inserting start/end nodes of the current path to prevent trivial loops
                possible_nodes = list(self.graph.G.nodes - {path[0], path[-1]})
                if not possible_nodes: # Graph might be too small
                   return creator.Individual(self.repair_individual(mutant_data)),

                insert_node = random.choice(possible_nodes)
                # Choose a random index to insert at (not at the very beginning or end)
                insert_idx = random.randint(1, len(path) - 1)
                # Create the new path
                mutated_path = path[:insert_idx] + [insert_node] + path[insert_idx:]
                mutant_data[robot_id_to_mutate] = mutated_path


            elif mutation_type == 'delete_vertex' and len(path) >= 3:
                 # Choose a random index to delete (not start or end)
                 delete_idx = random.randint(1, len(path) - 2)
                 # Create the new path
                 mutated_path = path[:delete_idx] + path[delete_idx+1:]
                 mutant_data[robot_id_to_mutate] = mutated_path

            else: # If conditions for rewire/delete aren't met, or invalid type somehow
                 mutated_path = path # No change
                 mutant_data[robot_id_to_mutate] = mutated_path


        except nx.NetworkXNoPath:
            # If rewire fails to find a path, keep the original path segment
             mutant_data[robot_id_to_mutate] = path # No change on error
             print(f"Warning: Mutation 'rewire' failed for robot {robot_id_to_mutate}, no path found.")
        except Exception as e:
             print(f"Warning: Error during mutation for robot {robot_id_to_mutate}: {e}")
             mutant_data[robot_id_to_mutate] = path # Revert on unexpected error


        # Repair the mutated individual (for connectivity/targets)
        # It returns a new dictionary, which we wrap in the Individual creator
        repaired_mutant = self.repair_individual(mutant_data)
        return creator.Individual(repaired_mutant), # Return as tuple as required by DEAP

    def repair_individual(self, individual: dict, conflict_repair = False) -> dict:
        initial_individual = individual.copy()
        cache = {}

        def safe_shortest(u, v):
            if (u, v) not in cache:
                try:
                    cache[(u, v)] = nx.shortest_path(self.graph.G, u, v)
                except nx.NetworkXNoPath:
                    cache[(u, v)] = [u]  # Stay in place if unreachable
            return cache[(u, v)]

        # Step 1: Basic path repair (connectivity + targets)
        repaired = {}
        for robot in self.robots:
            raw = individual[robot.robot_id]
            if not raw:
                raw = [robot.start]
            path = [raw[0]]
            for v in raw[1:]:
                if v == path[-1]:
                    path.append(v)
                elif not self.graph.G.has_edge(path[-1], v):
                    path += safe_shortest(path[-1], v)[1:]
                else:
                    path.append(v)

            for t in robot.targets:
                if t not in path:
                    path += safe_shortest(path[-1], t)[1:]

            repaired[robot.robot_id] = path
        if(conflict_repair):
            # Step 2: Conflict-aware timing adjustment
            max_steps = max(len(p) for p in repaired.values())
            schedule = defaultdict(dict)  # schedule[t][node] = robot_id
            delay_map = {r.robot_id: 0 for r in self.robots}

            # Use a simple prioritization by robot_id order
            for robot in sorted(self.robots, key=lambda r: r.robot_id):
                rid = robot.robot_id
                path = repaired[rid][:]

                while True:
                    conflict_found = False
                    for t, node in enumerate(path):
                        real_time = t + delay_map[rid]
                        # Vertex conflict
                        if node in schedule[real_time]:
                            conflict_found = True
                            break
                        # Edge conflict
                        if t > 0:
                            prev = path[t - 1]
                            for other_robot, other_path in repaired.items():
                                if other_robot == rid:
                                    continue
                                if real_time - 1 < len(other_path) and real_time < len(other_path):
                                    if other_path[real_time - 1] == node and other_path[real_time] == prev:
                                        conflict_found = True
                                        break
                        if conflict_found:
                            break

                    if conflict_found:
                        delay_map[rid] += 1
                        path = [path[0]] * delay_map[rid] + repaired[rid]  # Add wait at start
                    else:
                        break

                # Update schedule
                for t, node in enumerate(path):
                    schedule[t][node] = rid

                repaired[rid] = path
        return repaired



    def run(self):
        """
        Executes the evolutionary algorithm.
        """
        # Initialize population
        pop = self.toolbox.population(n=self.population_size)

        # Hall of Fame stores the best individual found so far
        # Since we minimize, it will store the individual with the lowest fitness value
        hof = tools.HallOfFame(1)

        # Statistics to track during evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values[0]) # Get the single fitness value
        stats.register("avg", np.mean)
        stats.register("min", np.min) # Minimum fitness value (best individual's cost)
        stats.register("max", np.max) # Maximum fitness value (worst individual's cost/penalty)

        # --- Run the EA ---
        # eaMuPlusLambda: (μ + λ) evolutionary algorithm
        pop, logbook = algorithms.eaMuPlusLambda(
            population=pop,
            toolbox=self.toolbox,
            mu=self.population_size,          # Number of individuals to select for the next generation
            lambda_=self.population_size,     # Number of children to produce at each generation
            cxpb=self.p_crossover,           # Probability of mating two individuals
            mutpb=self.p_mutation,           # Probability of mutating an individual
            ngen=self.generations,           # Number of generations
            stats=stats,                     # Statistics object
            halloffame=hof,                  # Hall of Fame object
            verbose=True                     # Print progress
        )

        # --- Store History ---
        gen = logbook.select("gen")
        self.mean_fitness_history = logbook.select("avg")
        self.min_fitness_history = logbook.select("min") # Best fitness per gen
        self.max_fitness_history = logbook.select("max") # Worst fitness per gen

        # Calculate average conflict count per generation (approximate)
        # Note: This requires evaluating fitness again or modifying the stats/logging
        # For simplicity, let's calculate it post-hoc for the final population
        # Or ideally, integrate conflict counting into the stats during the run (more advanced DEAP use)
        final_pop_conflicts = [self._detect_conflicts(ind) for ind in pop]
        avg_final_conflicts = np.mean([len(c) for c in final_pop_conflicts])
        print(f"Average conflicts in final population: {avg_final_conflicts:.2f}")
        # A more robust history would require modifying the EA loop or stats object.

        # The best individual found is in the Hall of Fame
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]

        print("-" * 30)
        print(f"Best Individual Fitness (Cost): {best_fitness}")
        if best_fitness >= CONFLICT_PENALTY_BASE:
             print("Warning: Best solution found still contains conflicts!")
             num_conflicts = self._detect_conflicts(best_individual)
             print(f"Number of conflicts in best solution: {len(num_conflicts)}")
             # print("Conflicts:", num_conflicts) # Optional: print details
        else:
             print("Best solution appears conflict-free.")
             # print("Best Individual Paths:", best_individual) # Optional: print paths

        return best_individual, logbook
