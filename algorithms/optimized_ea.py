import random
import networkx as nx
from algorithms.base import BaseEvolutionarylgorithm

class PathBasedEA(BaseEvolutionarylgorithm):

    def __init__(self, graph, robots, population_size=50, generations=1000):
        super().__init__(graph, robots, population_size, generations)
        self.graph = graph
        self.robots = robots
        self.fitness_history = []
        self.makespan_history = []
        self.conflict_history = []
        self.distance_history = []

    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for robot in self.robots:
                path = self.heuristic_path(robot)
                individual[robot.robot_id] = path
            population.append(individual)
        return population

    def heuristic_path(self, robot):
        current_pos = robot.start
        path = [current_pos]
        targets = robot.targets[:]
        random.shuffle(targets)
        while targets:
            next_target = min(targets, key=lambda t: self.graph.shortest_path_length(current_pos, t))
            segment = self.graph.shortest_path(current_pos, next_target)
            path.extend(segment[1:])
            current_pos = next_target
            targets.remove(next_target)
        return path

    def fitness(self, individual):
        total_distance = 0
        conflicts = 0
        makespan = 0
        visited_positions = {}

        for robot_id, path in individual.items():
            path_length = len(path) - 1
            total_distance += path_length
            makespan = max(makespan, path_length)

            for t, pos in enumerate(path):
                if (t, pos) in visited_positions:
                    conflicts += 1
                else:
                    visited_positions[(t, pos)] = robot_id

        fitness_value = total_distance + (20 * conflicts) + (1.5 * makespan)
        return fitness_value, total_distance, conflicts, makespan

    def selection(self, population):
        tournament_size = 3
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda ind: self.fitness(ind)[0])
        return tournament[0], tournament[1]

    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}
        for robot in self.robots:
            path1, path2 = parent1[robot.robot_id], parent2[robot.robot_id]
            common_vertices = set(path1) & set(path2)

            if len(common_vertices) > 1:
                branching_vertex = random.choice(list(common_vertices - {path1[-1]}))

                idx1 = path1.index(branching_vertex)
                idx2 = path2.index(branching_vertex)

                child1_path = path1[:idx1] + path2[idx2:]
                child2_path = path2[:idx2] + path1[idx1:]

                child1[robot.robot_id] = self.repair_path(child1_path, robot)
                child2[robot.robot_id] = self.repair_path(child2_path, robot)
            else:
                child1[robot.robot_id] = path1[:]
                child2[robot.robot_id] = path2[:]
        return child1, child2

    def mutation(self, individual):
        mutant = {}
        for robot in self.robots:
            path = individual[robot.robot_id][:]
            if len(path) < 4:
                mutant[robot.robot_id] = path
                continue

            idx1, idx2 = sorted(random.sample(range(1, len(path)-1), 2))
            start, end = path[idx1], path[idx2]
            new_segment = self.graph.shortest_path(start, end)

            if new_segment:
                mutant_path = path[:idx1] + new_segment + path[idx2+1:]
            else:
                mutant_path = path

            mutant[robot.robot_id] = self.repair_path(mutant_path, robot)

        return mutant

    def repair_path(self, path, robot):
        if not path:
            path = [robot.start]

        repaired_path = [path[0]]
        for next_vertex in path[1:]:
            if self.graph.G.has_edge(repaired_path[-1], next_vertex):
                repaired_path.append(next_vertex)
            else:
                segment = self.graph.shortest_path(repaired_path[-1], next_vertex)
                if segment:
                    repaired_path.extend(segment[1:])

        missing_targets = set(robot.targets) - set(repaired_path)
        current_pos = repaired_path[-1]

        for target in missing_targets:
            segment = self.graph.shortest_path(current_pos, target)
            if segment:
                repaired_path.extend(segment[1:])
                current_pos = target

        return repaired_path

    def evolve(self):
        population = self.initial_population()

        for generation in range(self.generations):
            population.sort(key=lambda ind: self.fitness(ind)[0])
            best_fitness, dist, conflicts, makespan = self.fitness(population[0])

            self.fitness_history.append(best_fitness)
            self.distance_history.append(dist)
            self.conflict_history.append(conflicts)
            self.makespan_history.append(makespan)

            next_gen = population[:10]  # Elitism

            while len(next_gen) < self.population_size:
                p1, p2 = self.selection(population)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutation(c1))
                if len(next_gen) < self.population_size:
                    next_gen.append(self.mutation(c2))

            population = next_gen

        return min(population, key=lambda ind: self.fitness(ind)[0])
