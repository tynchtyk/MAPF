import random
import networkx as nx
from collections import defaultdict
from deap import base, creator, tools, algorithms
import deap
import copy
import numpy as np
import math # Import math for infinity

# Define a large constant for penalty (effectively infinity for practical path lengths)
CONFLICT_PENALTY_BASE = 1e9


class PathBasedEA_DEAP_MULTI():

    def __init__(self, graph, robots, population_size=50, generations=50, p_cross=0.8, p_mut=0.2):
        """
        Initializes the Path-Based Evolutionary Algorithm for multi-objective optimization.
        """
        self.graph = graph
        self.robots = robots
        self.robot_map = {r.robot_id: r for r in robots}
        self.population_size = population_size
        self.generations = generations
        self.p_crossover = p_cross
        self.p_mutation = p_mut

        # --- DEAP Setup for Multi-Objective Optimization ---
        # Create FitnessMulti: objective is to minimize both values
        # weights=(-1.0, -1.0) means minimize both objective values
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        # Individual is a dictionary mapping robot_id -> path (list of nodes)
        creator.create("Individual", dict, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        # Register functions for creating individuals and populations
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_fitness_multi) # New evaluation function
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutation)
        # Register selection operator for multi-objective optimization (NSGA-II)
        self.toolbox.register("select", tools.selNSGA2)

        self.mean_conflict_history = []
        self.mean_distance_history = []
        self.min_conflict_history = []
        self.min_distance_history = []

    def generate_individual(self):
        """Generates a single individual."""
        individual = {}
        temp_graph = self.randomize_graph_weights()
        for robot in self.robots:
            path = self.heuristic_path(robot, temp_graph)
            individual[robot.robot_id] = path
        repaired_individual = self.repair_individual(individual)
        return creator.Individual(repaired_individual)

    def randomize_graph_weights(self):
        """Creates a copy of the graph with randomized edge weights."""
        temp_graph = self.graph.G.copy()
        for u, v in temp_graph.edges():
            temp_graph[u][v]['weight'] = random.uniform(0.5, 2.0)
        return temp_graph

    def heuristic_path(self, robot, graph):
        """Generates a plausible path for a single robot using a nearest-target heuristic."""
        current_pos = robot.start
        path = [current_pos]
        targets_to_visit = robot.targets[:]

        while targets_to_visit:
            try:
                next_target = min(targets_to_visit,
                                  key=lambda t: nx.shortest_path_length(graph, current_pos, t, weight='weight'))
                segment = nx.shortest_path(graph, current_pos, next_target, weight='weight')
                path.extend(segment[1:])
                current_pos = next_target
                targets_to_visit.remove(next_target)
            except nx.NetworkXNoPath:
                print(f"Warning: Cannot find path for Robot {robot.robot_id}...")
                break
        return path

    def evaluate_fitness_multi(self, individual):
        """Evaluates the fitness of an individual based on two objectives:
        1. Number of conflicts (to be minimized).
        2. Total path length (to be minimized).
        """
        conflicts = self._detect_conflicts(individual)
        conflict_count = len(conflicts)

        total_distance = sum(len(p) - 1 for p in individual.values() if p)

        return (conflict_count, total_distance) # Return as a tuple for multi-objective

    def _detect_conflicts(self, individual):
        """Helper function to detect vertex and edge conflicts."""
        conflicts = []
        max_steps = 0
        for path in individual.values():
            if path:
                max_steps = max(max_steps, len(path))

        if max_steps == 0:
             return []

        vertex_occupation = defaultdict(dict)
        edge_traversal = defaultdict(dict)

        for t in range(max_steps):
            current_vertex_occupations = {}
            for robot_id, path in individual.items():
                if t < len(path):
                    current_node = path[t]
                    if current_node in current_vertex_occupations:
                        conflicts.append(('vertex', t, current_node, robot_id, current_vertex_occupations[current_node]))
                    else:
                         current_vertex_occupations[current_node] = robot_id
                else:
                     final_node = path[-1] if path else None
                     if final_node is not None:
                         if final_node in current_vertex_occupations:
                             conflicts.append(('vertex_wait', t, final_node, robot_id, current_vertex_occupations[final_node]))
                         else:
                             current_vertex_occupations[final_node] = robot_id

            vertex_occupation[t] = current_vertex_occupations

            if t > 0:
                current_edge_traversals = {}
                for robot_id, path in individual.items():
                    if t < len(path):
                        current_node = path[t]
                        prev_node = path[t-1]
                        edge = (prev_node, current_node)
                        swap_edge = (current_node, prev_node)

                        if swap_edge in current_edge_traversals:
                             conflicts.append(('edge', t, edge, robot_id, current_edge_traversals[swap_edge]))
                        current_edge_traversals[edge] = robot_id

                edge_traversal[t] = current_edge_traversals

        return conflicts

    def crossover(self, ind1, ind2):
        """Performs crossover between two individuals."""
        child1_data, child2_data = {}, {}
        ind1_data = dict(ind1)
        ind2_data = dict(ind2)

        for robot_id in self.robot_map.keys():
            path1 = ind1_data.get(robot_id, [])
            path2 = ind2_data.get(robot_id, [])

            if not path1 or not path2:
                child1_data[robot_id] = path1[:]
                child2_data[robot_id] = path2[:]
                continue

            common_nodes = list(set(path1[:-1]) & set(path2[:-1]))

            if common_nodes:
                crossover_vertex = random.choice(common_nodes)
                idx1 = path1.index(crossover_vertex)
                idx2 = path2.index(crossover_vertex)

                child1_path = path1[:idx1+1] + path2[idx2+1:]
                child2_path = path2[:idx2+1] + path1[idx1+1:]
            else:
                child1_path = path1[:]
                child2_path = path2[:]

            child1_data[robot_id] = child1_path
            child2_data[robot_id] = child2_path

        child1 = creator.Individual(child1_data)
        child2 = creator.Individual(child2_data)

        child1 = creator.Individual(self.repair_individual(child1))
        child2 = creator.Individual(self.repair_individual(child2))

        return child1, child2

    def mutation(self, individual):
        """Performs mutation on an individual."""
        mutant_data = dict(individual)
        robot_id_to_mutate = random.choice(list(self.robot_map.keys()))
        path = mutant_data.get(robot_id_to_mutate, [])

        if len(path) < 2:
             return creator.Individual(self.repair_individual(mutant_data)),

        mutation_type = random.choice(['rewire', 'insert_vertex', 'delete_vertex'])

        try:
            if mutation_type == 'rewire' and len(path) >= 3:
                idx1, idx2 = sorted(random.sample(range(1, len(path) - 1), 2))
                start_node = path[idx1]
                end_node = path[idx2]
                new_segment = self.graph.shortest_path(start_node, end_node)
                mutated_path = path[:idx1] + new_segment + path[idx2+1:]
                mutant_data[robot_id_to_mutate] = mutated_path

            elif mutation_type == 'insert_vertex':
                possible_nodes = list(self.graph.G.nodes - {path[0], path[-1]})
                if not possible_nodes:
                   return creator.Individual(self.repair_individual(mutant_data)),

                insert_node = random.choice(possible_nodes)
                insert_idx = random.randint(1, len(path) - 1)
                mutated_path = path[:insert_idx] + [insert_node] + path[insert_idx:]
                mutant_data[robot_id_to_mutate] = mutated_path

            elif mutation_type == 'delete_vertex' and len(path) >= 3:
                 delete_idx = random.randint(1, len(path) - 2)
                 mutated_path = path[:delete_idx] + path[delete_idx+1:]
                 mutant_data[robot_id_to_mutate] = mutated_path

            else:
                 mutated_path = path
                 mutant_data[robot_id_to_mutate] = mutated_path

        except nx.NetworkXNoPath:
             mutant_data[robot_id_to_mutate] = path
             print(f"Warning: Mutation 'rewire' failed...")
        except Exception as e:
             print(f"Warning: Error during mutation...")
             mutant_data[robot_id_to_mutate] = path

        repaired_mutant = self.repair_individual(mutant_data)
        return creator.Individual(repaired_mutant),

    def repair_individual(self, individual: dict) -> dict:
        initial_individual = individual.copy()
        cache = {}
        def safe_shortest(u, v):
            if (u, v) not in cache:
                cache[(u, v)] = nx.shortest_path(self.graph.G, u, v)
            return cache[(u, v)]

        repaired = {}
        for robot in self.robots:
            raw = individual[robot.robot_id]
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
        return repaired

    def run(self):
        """Executes the multi-objective evolutionary algorithm and plots the results."""
        # Initialize population
        pop = self.toolbox.population(n=self.population_size)

        # Hall of Fame for multi-objective (stores non-dominated solutions)
        hof = tools.ParetoFront()

        # Statistics to track during evolution for both objectives
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # --- Run the Multi-Objective EA (NSGA-II) ---
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
        avg_fitness = logbook.select("avg")
        min_fitness = logbook.select("min")

        self.mean_conflict_history = [avg[0] for avg in avg_fitness]
        self.mean_distance_history = [avg[1] for avg in avg_fitness]
        self.min_conflict_history = [min_val[0] for min_val in min_fitness]
        self.min_distance_history = [min_val[1] for min_val in min_fitness]

        print("-" * 30)
        print("Pareto Front (Non-dominated solutions):")
        for ind in hof:
            conflicts, distance = ind.fitness.values
            print(f"Conflicts: {conflicts}, Distance: {distance}")

        return hof[0], logbook
