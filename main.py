import yaml
from utils import visualize_solution,  load_scenario, show_graph_structure, show_graph_with_robots
from algorithms.simple_ea import SimpleEvolutionaryAlgorithm
from map_graph import MapfGraph

def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()

    # Generate scenario with multiple goals
    robots = load_scenario(config['generated_scenario_file'])

    # Load map
    graph = MapfGraph(config['map_file'])
    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = SimpleEvolutionaryAlgorithm(graph, robots)
    solution = algo.evolve()

    # Visualize or output
    visualize_solution(graph, robots, solution)

if __name__ == "__main__":
    main()
