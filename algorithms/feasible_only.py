import random
import networkx as nx
from collections import defaultdict
from deap import base, creator, tools, algorithms
import copy
import numpy as np
import math # Import math for infinity


class PathBasedEA_DEAP_FeasibleOnly():

    def __init__(self, graph, robots, population_size=50, generations=50, p_cross=0.8, p_mut=0.2):
        """
        Initializes the Path-Based Evolutionary Algorithm.
        Operates on feasible (conflict-free) solutions by incorporating
        conflict resolution into the repair step.

        Args:
            graph (GraphWrapper): Wrapper containing the NetworkX graph.
            robots (list[Robot]): List of Robot objects with IDs, start, and targets.
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run the algorithm.
            p_cross (float): Probability of crossover.
            p_mut (float): Probability of mutation.
        """
        self.graph = graph # Assumes graph.G is the NetworkX graph
        self.robots = robots
        self.robot_map = {r.robot_id: r for r in robots}
        self.population_size = population_size
        self.generations = generations
        self.p_crossover = p_cross
        self.p_mutation = p_mut
        self.generation_best = []
        self.all_pairs_shortest_paths = dict(nx.all_pairs_shortest_path(self.graph.G))

        # --- DEAP Setup ---
        # Fitness is now just minimizing total path length (or makespan)
        # Weights set to minimize the first (and only) objective value
        #creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimise sum of path lengths
        # OR, uncomment below to minimize makespan (longest path)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimise makespan

        # Individual is a dictionary mapping robot_id -> path (list of nodes)
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        # Register functions for creating individuals and populations
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_fitness) # Simplified fitness
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection is often good

    def generate_individual(self):
        """
        Generates a single individual (potential solution).
        Each robot gets a path generated using a heuristic on a randomly weighted graph.
        The individual is then repaired to ensure basic validity AND resolve conflicts.
        """

        methods = [self.heuristic_path, self.random_path]
        individual_data = {}
        temp_graph = self.randomize_graph_weights()
        for robot in self.robots:
            method = random.choice(methods)
            raw_path = method(robot, temp_graph)
            individual_data[robot.robot_id] = raw_path

        # Repair ensures connectivity, target inclusion, AND resolves conflicts
        repaired_individual_data = self.repair_individual(individual_data)
        print("Individual generated and repaired")
        return creator.Individual(repaired_individual_data)

    # Removed _calculate_penalty_base as it's no longer needed

    def randomize_graph_weights(self):
        """Creates a copy of the graph with randomized edge weights."""
        temp_graph = self.graph
        for u, v in temp_graph.edges():
            temp_graph[u][v]['weight'] = random.uniform(0.5, 2.0) # Example range
        return temp_graph

    def random_path(self, robot, graph):
        targets = robot.targets[:]
        random.shuffle(targets)
        path = [robot.start]
        for target in targets:
            segment = nx.shortest_path(graph, path[-1], target, weight='weight')
            path.extend(segment[1:])
        #print(f"Random path generated for robot {robot.robot_id}: {path}")
        return path

    def heuristic_path(self, robot, graph):
        """
        Generates a plausible path for a single robot using a nearest-target heuristic.
        """
        current_pos = robot.start
        path = [current_pos]
        targets_to_visit = robot.targets[:]
        #random.shuffle(targets_to_visit) # Optional: Add randomness

        G_heuristic = graph # Use the potentially randomized graph for initial path generation

        while targets_to_visit:
            try:
                # Find the target nearest to the current position
                next_target = min(targets_to_visit,
                                  key=lambda t: nx.shortest_path_length(G_heuristic, current_pos, t, weight='weight'))
                # Find the shortest path segment using the *original* graph weights for realism?
                # Or stick with potentially randomized weights used for heuristic? Let's use G_heuristic for consistency here.
                segment = nx.shortest_path(G_heuristic, current_pos, next_target, weight='weight')
                path.extend(segment[1:]) # Append segment excluding the start node
                current_pos = next_target
                targets_to_visit.remove(next_target)
            except nx.NetworkXNoPath:
                print(f"Warning: Heuristic path failed for Robot {robot.robot_id} from {current_pos} to targets {targets_to_visit}. Stopping path generation.")
                # Decide how to handle: stop, try alternative target, etc. Here we stop.
                break
            except ValueError: # Happens if targets_to_visit is empty
                 break

        return path

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of a conflict-free individual.
        Fitness is the sum of the lengths of all paths (lower is better).
        Alternatively, makespan (length of the longest path) can be used.
        """
        # Ensure individual is a dict
        individual_data = dict(individual)

        # Calculate sum of path lengths (makespan - 1 step moves)
        total_distance = sum(len(p) - 1 for p in individual_data.values() if p)

        # OR: Calculate makespan (time the last robot finishes)
        makespan = 0
        for path in individual_data.values():
            if path:
                makespan = max(makespan, len(path) - 1)

        # Return the desired metric. Make sure this matches the weights in __init__
        # return (total_distance,)
        return (makespan,) # Using makespan here, matches weights=(-1.0,)

    def _get_location(self, path, time):
        """Helper to get robot location at a given time, stays at end if path finished"""
        if not path:
            return None # Or handle appropriately, maybe raise error?
        if time < 0:
            return path[0] # Or None? Assume time starts at 0
        if time < len(path):
            return path[time]
        else:
            return path[-1] # Stays at the last node

    def repair_individual(self, individual_data: dict, max_path_length=150, pad_steps=5) -> dict:
        """
        Optimized repair: connectivity, target coverage, and fast conflict resolution.
        """
        repaired_paths = {}
        G = self.graph.G
        shortest_paths = self.all_pairs_shortest_paths  # Precomputed in __init__

        def get_path(u, v):
            return shortest_paths.get(u, {}).get(v, [u])

        # Step 1: Skip if already conflict-free
        if not self._detect_conflicts(individual_data):
            return individual_data

        # Step 2: Connectivity and target inclusion repair
        for robot in self.robots:
            rid = robot.robot_id
            path = individual_data.get(rid, [])
            if not path or path[0] != robot.start:
                path = [robot.start] + path

            # Repair invalid segments
            fixed = [path[0]]
            for i in range(1, len(path)):
                u, v = path[i-1], path[i]
                if G.has_edge(u, v) or u == v:
                    fixed.append(v)
                else:
                    segment = get_path(u, v)
                    fixed.extend(segment[1:] if len(segment) > 1 else [u])

            # Ensure all targets are visited
            visited = set(fixed)
            remaining_targets = [t for t in robot.targets if t not in visited]
            current = fixed[-1]
            for target in remaining_targets:
                seg = get_path(current, target)
                fixed.extend(seg[1:] if len(seg) > 1 else [current])
                current = target

            repaired_paths[rid] = fixed if fixed else [robot.start]

        # Step 3: Fast conflict resolution (priority-based, fixed delays)
        paths = copy.deepcopy(repaired_paths)
        for rid in paths:
            paths[rid].extend([paths[rid][-1]] * pad_steps)  # Pad fixed wait

        robot_ids = sorted(paths.keys())
        max_len = max(len(p) for p in paths.values())
        reserved_nodes = [dict() for _ in range(max_len)]
        reserved_edges = [dict() for _ in range(max_len)]

        for t in range(max_len - 1):
            for rid in robot_ids:
                p = paths[rid]
                if t >= len(p) - 1:
                    continue

                u = p[t]
                v = p[t + 1]

                # Check vertex conflict
                if reserved_nodes[t + 1].get(v) not in (None, rid):
                    p[t + 1] = u  # Wait
                    v = u  # Update target node

                # Check edge swap
                if reserved_edges[t + 1].get((v, u)) not in (None, rid):
                    p[t + 1] = u  # Wait
                    v = u

                reserved_nodes[t + 1][v] = rid
                if u != v:
                    reserved_edges[t + 1][(u, v)] = rid

        # Step 4: Clamp overly long paths
        for rid in paths:
            if len(paths[rid]) > max_path_length:
                paths[rid] = paths[rid][:max_path_length]

        return paths



    def crossover(self, ind1, ind2):
        """
        Performs path-based crossover and repairs the children to be conflict-free.
        """
        child1_data, child2_data = {}, {}
        ind1_data = dict(ind1)
        ind2_data = dict(ind2)

        for robot_id in self.robot_map.keys():
            path1 = ind1_data.get(robot_id, [])
            path2 = ind2_data.get(robot_id, [])

            if not path1 or not path2 or len(path1) < 2 or len(path2) < 2:
                child1_data[robot_id] = path1[:]
                child2_data[robot_id] = path2[:]
                continue

            # Find common nodes excluding start/end
            common_nodes = list(set(path1[1:-1]) & set(path2[1:-1]))

            if common_nodes:
                crossover_vertex = random.choice(common_nodes)
                try:
                    idx1 = path1.index(crossover_vertex)
                    idx2 = path2.index(crossover_vertex)

                    child1_path = path1[:idx1+1] + path2[idx2+1:]
                    child2_path = path2[:idx2+1] + path1[idx1+1:]
                except ValueError: # Should not happen if common_nodes is correct
                    child1_path = path1[:]
                    child2_path = path2[:]
            else:
                child1_path = path1[:]
                child2_path = path2[:]

            child1_data[robot_id] = child1_path
            child2_data[robot_id] = child2_path

        # Repair children (includes conflict resolution)
        repaired_child1_data = self.repair_individual(child1_data)
        repaired_child2_data = self.repair_individual(child2_data)

        child1 = creator.Individual(repaired_child1_data)
        child2 = creator.Individual(repaired_child2_data)

        return child1, child2

    def mutation(self, individual):
        """
        Performs mutation on an individual and repairs the mutant to be conflict-free.
        """
        mutant_data = dict(individual)
        robot_id_to_mutate = random.choice(list(self.robot_map.keys()))
        path = mutant_data.get(robot_id_to_mutate, [])

        if len(path) < 2:
             # Repair even if no mutation occurs, ensures consistency
             repaired_mutant_data = self.repair_individual(mutant_data)
             return creator.Individual(repaired_mutant_data),

        mutation_type = random.choice(['rewire', 'insert_vertex', 'delete_vertex', 'wait'])

        mutated_path = path[:] # Work on a copy

        try:
            if mutation_type == 'rewire' and len(path) >= 3:
                idx1, idx2 = sorted(random.sample(range(0, len(path)), 2)) # Allow start/end rewire
                start_node = path[idx1]
                end_node = path[idx2]
                # Find a new shortest path (use original graph, unweighted?)
                # Using unweighted for mutation rewire simplicity
                new_segment = nx.shortest_path(self.graph.G, start_node, end_node, weight=None)
                mutated_path = path[:idx1] + new_segment + path[idx2+1:]

            elif mutation_type == 'insert_vertex' and len(path) >= 1:
                possible_nodes = list(self.graph.G.nodes) # Allow any node
                if possible_nodes:
                    insert_node = random.choice(possible_nodes)
                    insert_idx = random.randint(0, len(path)) # Allow insertion at start/end
                    # Find path to inserted node and from inserted node
                    path_to_insert = []
                    path_from_insert = []
                    prev_node = path[insert_idx-1] if insert_idx > 0 else self.robot_map[robot_id_to_mutate].start # Handle insertion at index 0
                    next_node = path[insert_idx] if insert_idx < len(path) else None # Handle insertion at end

                    if prev_node != insert_node:
                       path_to_insert = nx.shortest_path(self.graph.G, prev_node, insert_node, weight=None)[1:] # Exclude prev_node

                    if next_node is not None and insert_node != next_node:
                        path_from_insert = nx.shortest_path(self.graph.G, insert_node, next_node, weight=None)[1:] # Exclude insert_node

                    # Construct the new path
                    mutated_path = path[:insert_idx] + path_to_insert + [insert_node] + path_from_insert + path[insert_idx:]
                    # Clean up potential duplicates if paths were trivial
                    cleaned_path = []
                    if mutated_path:
                        cleaned_path.append(mutated_path[0])
                        for i in range(1, len(mutated_path)):
                            if mutated_path[i] != mutated_path[i-1]:
                                cleaned_path.append(mutated_path[i])
                    mutated_path = cleaned_path

            elif mutation_type == 'delete_vertex' and len(path) >= 3:
                # Choose a random index to delete (not start or absolute end if it's a target)
                start_node = self.robot_map[robot_id_to_mutate].start
                targets = self.robot_map[robot_id_to_mutate].targets
                can_delete_indices = [i for i in range(1, len(path)-1) if path[i] != start_node and path[i] not in targets]
                if not can_delete_indices: # Cannot delete anything safely
                     can_delete_indices = [i for i in range(1, len(path)-1)] # Fallback: allow deleting targets too if needed

                if can_delete_indices:
                    delete_idx = random.choice(can_delete_indices)
                    prev_node = path[delete_idx - 1]
                    next_node = path[delete_idx + 1]
                    # Reconnect path
                    reconnect_segment = nx.shortest_path(self.graph.G, prev_node, next_node, weight=None)
                    mutated_path = path[:delete_idx] + reconnect_segment[1:] + path[delete_idx+1:]
                else: # No node to delete
                    pass # No change

            elif mutation_type == 'wait' and len(path) >= 1:
                 # Insert a wait step at a random position
                 wait_idx = random.randint(0, len(path)-1) # Index of the node to repeat
                 mutated_path.insert(wait_idx + 1, path[wait_idx])


        except nx.NetworkXNoPath:
             # If mutation fails to find a path, keep the original (will be repaired)
             mutated_path = path # Revert to original path before mutation
             print(f"Warning: Mutation '{mutation_type}' failed for robot {robot_id_to_mutate}, no path found.")
        except Exception as e:
             print(f"Warning: Error during mutation '{mutation_type}' for robot {robot_id_to_mutate}: {e}")
             mutated_path = path # Revert on unexpected error

        mutant_data[robot_id_to_mutate] = mutated_path

        # Repair the mutated individual (includes conflict resolution)
        repaired_mutant_data = self.repair_individual(mutant_data)
        # DEAP expects mutation to return a tuple containing the modified individual
        return creator.Individual(repaired_mutant_data),


    def run(self):
        """
        Executes the evolutionary algorithm using feasible solutions.
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        # Statistics track the fitness (e.g., makespan or total distance)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # --- Run the EA ---
        # eaSimple or eaMuPlusLambda can be used. eaMuPlusLambda is often good.
        pop, logbook = algorithms.eaMuPlusLambda(
            population=pop,
            toolbox=self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size, # Generate as many children as parents
            cxpb=self.p_crossover,
            mutpb=self.p_mutation,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # --- Results ---
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0] # This is now the actual cost (e.g., makespan)

        print("-" * 30)
        print(f"Algorithm Finished")
        print(f"Best Individual Fitness (e.g., Makespan or Total Distance): {best_fitness}")
        # Since repair aims for conflict-free, we expect no conflicts.
        # A final check can still be useful for debugging the repair function.
        # final_conflicts = self._detect_conflicts(best_individual) # Need _detect_conflicts if you want this check
        # if final_conflicts:
        #      print(f"WARNING: Best solution found STILL contains {len(final_conflicts)} conflicts! Check repair function.")
        #      # print("Conflicts:", final_conflicts)
        # else:
        #      print("Best solution appears conflict-free (as expected).")

        # print("Best Individual Paths (Robot ID: Path Length):")
        # for robot_id, path in best_individual.items():
        #     print(f"  {robot_id}: {len(path)-1 if path else 0} steps")
        # Optional: print full paths
        # print("\nFull Paths:")
        # for robot_id, path in best_individual.items():
        #     print(f" R{robot_id}: {path}")

        return best_individual, logbook

    # Keep _detect_conflicts if you want to run a final check or for debugging repair
    def _detect_conflicts(self, individual):
        """
        Helper function to detect vertex and edge conflicts in a set of paths.
        (Used for final verification/debugging, not fitness)
        """
        conflicts = []
        individual_data = dict(individual) # Ensure it's a dictionary
        max_steps = max((len(p) for p in individual_data.values()), default=0)

        if max_steps == 0: return []

        vertex_occupation = defaultdict(dict) # V[t][node] = robot_id
        edge_traversal = defaultdict(dict)    # E[t][(u,v)] = robot_id (move ends at t)

        for t in range(max_steps):
            current_vertex_occupations = {} # Occupations at exactly time t
            current_edge_traversals = {} # Traversals ending exactly at time t

            # Process vertex occupations and waits at time t
            for robot_id, path in individual_data.items():
                node = self._get_location(path, t)
                if node is None: continue

                if node in current_vertex_occupations:
                    conflicts.append(('vertex', t, node, robot_id, current_vertex_occupations[node]))
                else:
                    current_vertex_occupations[node] = robot_id
            vertex_occupation[t] = current_vertex_occupations

            # Process edge traversals ending at time t
            if t > 0:
                for robot_id, path in individual_data.items():
                    current_node = self._get_location(path, t)
                    prev_node = self._get_location(path, t - 1)

                    if current_node is None or prev_node is None or current_node == prev_node:
                        continue # No move occurred

                    edge = (prev_node, current_node)
                    swap_edge = (current_node, prev_node)

                    # Check for direct swap conflict with moves ending at the *same time* t
                    if swap_edge in current_edge_traversals:
                         conflicts.append(('edge', t, edge, robot_id, current_edge_traversals[swap_edge]))

                    current_edge_traversals[edge] = robot_id
                edge_traversal[t] = current_edge_traversals

        return conflicts