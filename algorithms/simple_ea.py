import random
from algorithms.base import BaseEvolutionarylgorithm
from algorithms.operators import (
    roulette_selection,
    one_point_crossover,
    swap_mutation,
)

class SimpleEvolutionaryAlgorithm(BaseEvolutionarylgorithm):
    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for robot in self.robots:
                path = [robot.start] + random.sample(robot.targets, len(robot.targets))
                individual[robot.robot_id] = path
            population.append(individual)
        return population

    def fitness(self, individual):
        total_cost = 0
        for robot_id, path in individual.items():
            total_cost += sum(
                self.graph.shortest_path(path[i], path[i+1]).__len__() - 1
                for i in range(len(path)-1)
            )
        return total_cost

    def selection(self, population):
        return roulette_selection(population, self.fitness)

    def crossover(self, parent1, parent2):
        return one_point_crossover(parent1, parent2)

    def mutation(self, individual):
        return swap_mutation(individual)
