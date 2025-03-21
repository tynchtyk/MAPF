import random
import copy

def roulette_selection(population, fitness_fn):
    weights = [1 / (fitness_fn(indiv) + 1e-6) for indiv in population]
    return random.choices(population, weights=weights, k=2)

def one_point_crossover(parent1, parent2):
    child1, child2 = {}, {}
    for robot_id in parent1:
        path1 = parent1[robot_id]
        path2 = parent2[robot_id]
        i = random.randint(1, len(path1) - 1)
        child1[robot_id] = path1[:i] + path2[i:]
        child2[robot_id] = path2[:i] + path1[i:]
    return child1, child2

def swap_mutation(individual):
    mutated = copy.deepcopy(individual)
    for robot_id in mutated:
        path = mutated[robot_id]
        if len(path) > 2:
            i, j = random.sample(range(1, len(path)), 2)
            path[i], path[j] = path[j], path[i]
    return mutated
