import random
import networkx as nx
from algorithms.base import BaseEvolutionaryAlgorithm
from collections import defaultdict, deque
import copy

class PathBasedEA(BaseEvolutionaryAlgorithm):

    def __init__(self, graph, robots, population_size=50, generations=50):
        super().__init__(graph, robots, population_size, generations)
        self.fitness_history = []
        print(f"Initialized PathBasedEA with population_size={population_size}, generations={generations}")

    def initial_population(self):
        population = []
        methods = [self.heuristic_path, self.random_path]
        for i in range(self.population_size):
            individual = {}
            temp_graph = self.randomize_graph_weights()
            for robot in self.robots:
                method = random.choice(methods)
                raw_path = method(robot, temp_graph)
                individual[robot.robot_id] = raw_path
            repaired_individual = self.repair_individual(individual)
            population.append(repaired_individual)
            #print(f"Valid Individual {i+1}/{self.population_size} created")
        #print(f"Initial valid population created with size: {len(population)}")
        return population

    def randomize_graph_weights(self):
        temp_graph = self.graph.G.copy()
        for u, v in temp_graph.edges():
            temp_graph[u][v]['weight'] = random.uniform(0.5, 2.0)
        return temp_graph

    def heuristic_path(self, robot, graph):
        current_pos = robot.start
        path = [current_pos]
        targets = robot.targets[:]
        random.shuffle(targets)
        while targets:
            next_target = min(targets, key=lambda t: nx.shortest_path_length(graph, current_pos, t, weight='weight'))
            segment = nx.shortest_path(graph, current_pos, next_target, weight='weight')
            path.extend(segment[1:])
            current_pos = next_target
            targets.remove(next_target)
        #print(f"Heuristic path generated for robot {robot.robot_id}: {path}")
        return path

    def random_path(self, robot, graph):
        targets = robot.targets[:]
        random.shuffle(targets)
        path = [robot.start]
        for target in targets:
            segment = nx.shortest_path(graph, path[-1], target, weight='weight')
            path.extend(segment[1:])
        #print(f"Random path generated for robot {robot.robot_id}: {path}")
        return path

    def fitness(self, individual):
        vertex_t = defaultdict(set)
        edge_t = defaultdict(set)
        max_steps = max(len(path) for path in individual.values())

        for t in range(max_steps):
            for robot in self.robots:
                path = individual[robot.robot_id]
                if t >= len(path):
                    continue

                curr = path[t]
                prev = path[t - 1] if t > 0 else curr

                # Vertex conflict
                if curr in vertex_t[t]:
                    return -1  # Conflict
                vertex_t[t].add(curr)

                # Edge conflict (swap)
                if (curr, prev) in edge_t[t]:
                    return -1  # Conflict
                edge_t[t].add((prev, curr))

        # No conflict: compute total travel + makespan
        total_distance, makespan = 0, 0
        for path in individual.values():
            path_length = len(path) - 1
            total_distance += path_length
            makespan = max(makespan, path_length)

        return total_distance# + (1.5 * makespan)


    def selection(self, population):
        # Filter out individuals with -1 fitness
        valid_individuals = [(ind, self.fitness(ind)) for ind in population if self.fitness(ind) != -1]

        if not valid_individuals:
            # Fallback if all are invalid — choose randomly
            return random.choice(population)

        total_fitness = sum(fit for _, fit in valid_individuals)
        pick = random.uniform(0, total_fitness)
        current = 0

        for ind, fit in valid_individuals:
            current += fit
            if current >= pick:
                return ind

    def crossover(self, parent1, parent2):
        child = {}
        for robot in self.robots:
            path1, path2 = parent1[robot.robot_id], parent2[robot.robot_id]
            common = set(path1) & set(path2)
            branching = list(common - {path1[-1]})
            if branching:
                crossover_vertex = random.choice(branching)
                idx1, idx2 = path1.index(crossover_vertex), path2.index(crossover_vertex)
                child_path = path1[:idx1] + path2[idx2:]
                #print(f"Crossover at vertex {crossover_vertex} for robot {robot.robot_id}")
            else:
                child_path = path1[:]
                #print(f"No crossover point found; child path copied from parent for robot {robot.robot_id}")
            child[robot.robot_id] = child_path
        repaired_child = self.repair_individual(child)
        return repaired_child

    def mutation(self, individual):
        mutant = individual.copy()
        for robot in self.robots:
            path = mutant[robot.robot_id][:]
            mutation_type = random.choice(['rewire', 'vertex'])
            #print(f"Mutation type {mutation_type} chosen for robot {robot.robot_id}")

            if mutation_type == 'rewire' and len(path) > 3:
                idx1, idx2 = sorted(random.sample(range(len(path)), 2))
                segment = self.graph.shortest_path(path[idx1], path[idx2])
                path = path[:idx1] + segment + path[idx2+1:]
                #print(f"Segment rewired between indices {idx1} and {idx2} for robot {robot.robot_id}")

            elif mutation_type == 'vertex':
                if random.random() < 0.5 and len(path) > 2:
                    del_idx = random.randint(1, len(path)-2)
                    del path[del_idx]
                    #print(f"Vertex deleted at index {del_idx} for robot {robot.robot_id}")
                else:
                    insert_vertex = random.choice(list(self.graph.G.nodes))
                    insert_idx = random.randint(1, len(path)-1)
                    path.insert(insert_idx, insert_vertex)
                    #print(f"Vertex {insert_vertex} inserted at index {insert_idx} for robot {robot.robot_id}")

            mutant[robot.robot_id] = path
        repaired_mutant = self.repair_individual(mutant)
        return repaired_mutant

    def repair_individual(self, individual):
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
                    path.append(v)  # waiting at the same position is valid
                elif not self.graph.G.has_edge(path[-1], v):
                    path += safe_shortest(path[-1], v)[1:]
                else:
                    path.append(v)

            for t in robot.targets:
                if t not in path:
                    path += safe_shortest(path[-1], t)[1:]

            repaired[robot.robot_id] = path
        expanded_individual = copy.deepcopy(repaired)
        """max_steps = max(len(p) for p in repaired.values())
        max_iterations = 10000
        iterations = 0

        while iterations < max_iterations:
            conflict = False
            vertex_t = defaultdict(dict)
            edge_t = defaultdict(set)

            for t in range(max_steps):
                for robot in sorted(self.robots, key=lambda r: r.priority):
                    path = repaired[robot.robot_id]
                    if t >= len(path):
                        continue

                    curr = path[t]
                    prev = path[t - 1] if t > 0 else curr

                    # Check vertex conflict
                    for other_id, other_pos in vertex_t[t].items():
                        if curr == other_pos:
                            path.insert(t, prev)
                            conflict = True
                            break
                    if conflict:
                        break

                    # Check edge conflict
                    if (curr, prev) in edge_t[t]:
                        for other_id in vertex_t[t]:
                            other_path = repaired[other_id]
                            if t < len(other_path):
                                other_curr = other_path[t]
                                other_prev = other_path[t - 1] if t > 0 else other_curr
                                if other_curr == prev and other_prev == curr:
                                    if robot.priority > self._get_robot_by_id(other_id).priority:
                                        path.insert(t, prev)
                                    else:
                                        repaired[other_id].insert(t, other_prev)
                                    conflict = True
                                    break
                    if conflict:
                        break

                    vertex_t[t][robot.robot_id] = curr
                    edge_t[t].add((prev, curr))

                if conflict:
                    break

            if not conflict:
                break

            max_steps = max(len(p) for p in repaired.values())
            iterations += 1

        if iterations >= max_iterations:
            print("Repair function exceeded maximum iterations — possible infinite loop.", 
                               "\nInitial :", initial_individual,
                               "\nExpanded :", expanded_individual)
            self.repair_individual(individual)"""

        return repaired

    def _get_robot_by_id(self, robot_id):
        for r in self.robots:
            if r.robot_id == robot_id:
                return r
        raise ValueError(f"Robot ID {robot_id} not found")


    def evolve(self, elitism_size=5):
        population = self.initial_population()
        population = [self.repair_individual(ind) for ind in population]

        for generation in range(self.generations):
            population.sort(key=lambda ind: (self.fitness(ind) == -1, self.fitness(ind)))
            best_fitness = self.fitness(population[0])
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            self.fitness_history.append(best_fitness)

            # ✅ Apply Elitism — Keep top N individuals
            elites = [copy.deepcopy(ind) for ind in population[:elitism_size]]
            next_gen = elites[:]

            # 🧬 Generate the rest of the next generation
            while len(next_gen) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                mutant_child = self.mutation(child)
                repaired_child = self.repair_individual(mutant_child)
                next_gen.append(repaired_child)

            population = next_gen

        # ✅ Return the best valid individual
        valid_population = [ind for ind in population if self.fitness(ind) != -1]
        if not valid_population:
            print("⚠️ All individuals had conflicts. Returning a random one.")
            return random.choice(population)

        best_individual = min(valid_population, key=lambda ind: self.fitness(ind))
        print(f"Best individual found with fitness: {self.fitness(best_individual)}")
        return best_individual


