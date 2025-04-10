import yaml
from utils import visualize_solution,  load_scenario, show_graph_structure, show_graph_with_robots
from algorithms.simple_ea import SimpleEvolutionaryAlgorithm
from algorithms.optimized_ea import PathBasedEA
from map_graph import MapfGraph
from plot import plot_ea_metrics, plot_combined_metrics
import numpy as np


def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()

    # Generate scenario with multiple goals
    robots = load_scenario(config['generated_scenario_file'])
    print(robots)
    # Load map
    graph = MapfGraph(config['map_file'])

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = PathBasedEA(graph, robots)
    #algo.initial_population()
    solution = algo.evolve()

    # Visualize or output
    visualize_solution(graph, robots, solution)

    

    generations = np.arange(1, len(algo.fitness_history) + 1)
    plot_ea_metrics(generations, algo.fitness_history, algo.makespan_history, algo.conflict_history, algo.distance_history)
    plot_combined_metrics(generations, algo.fitness_history, algo.makespan_history, algo.conflict_history, algo.distance_history)

if __name__ == "__main__":
    main()
