from abc import ABC, abstractmethod

class BaseEvolutionaryAlgorithm(ABC):
    def __init__(self, graph, robots, population_size=50, generations=100):
        self.graph = graph
        self.robots = robots
        self.population_size = population_size
        self.generations = generations

    @abstractmethod
    def initial_population(self):
        pass

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def selection(self, population):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutation(self, individual):
        pass

    def evolve(self):
        population = self.initial_population()

        for generation in range(self.generations):
            population = sorted(population, key=self.fitness)
            next_gen = population[:10]  # elitism

            while len(next_gen) < self.population_size:
                p1, p2 = self.selection(population)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutation(c1))
                if len(next_gen) < self.population_size:
                    next_gen.append(self.mutation(c2))

            population = next_gen

        return min(population, key=self.fitness)
