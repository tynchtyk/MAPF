import yaml
from utils import *
from algorithms.p3 import P3
from algorithms.p3_2 import P3_2
from algorithms.optimized_ea import PathBasedEA_DEAP
from map_graph import MapfGraph
from plot import *
import numpy as np


def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)



"""def run_deap_ea_multi():
    config = load_config()

    # Generate scenario with multiple goals
    robots = load_scenario(config['generated_scenario_file'])
    print(robots)
    # Load map
    graph = MapfGraph(config['map_file'])

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = PathBasedEA_DEAP_MULTI(graph, 
                    robots, 
                    config['ea_params']['population_size'],
                    config['ea_params']['num_generations'],
                    config['ea_params']['mutation_rate'],
                    config['ea_params']['crossover_rate'],)
    #algo.initial_population()
    solution, logbook = algo.run()

    # Visualize or output
    visualize_solution(graph, robots, solution)
    plot_deap_statistics(logbook)

def run_non_deap_ea():
    config = load_config()

    # Generate scenario with multiple goals
    robots = load_scenario(config['generated_scenario_file'])
    print(robots)
    # Load map
    graph = MapfGraph(config['map_file'])

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = PathBasedEA(graph, 
                    robots, 
                    config['ea_params']['population_size'],
                    config['ea_params']['num_generations'],
                    config['ea_params']['mutation_rate'],
                    config['ea_params']['crossover_rate'],)
    #algo.initial_population()
    solution = algo.run()
    
    # Visualize or output
    visualize_solution(graph, robots, solution)
    generations = np.arange(1, len(algo.fitness_history) + 1)
    plot_combined_metrics(generations, algo.fitness_history, algo.makespan_history, algo.conflict_history, algo.distance_history)

def run_feasibile_only_ea():
    config = load_config()

    # Generate scenario with multiple goals
    robots = load_scenario(config['generated_scenario_file'])
    print("Robots", len(robots))
    # Load map
    graph = MapfGraph(config['map_file'])

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = PathBasedEA_DEAP_FeasibleOnly(graph, 
                    robots, 
                    config['ea_params']['population_size'],
                    config['ea_params']['num_generations'],
                    config['ea_params']['mutation_rate'],
                    config['ea_params']['crossover_rate'],)
    #algo.initial_population()
    solution, logbook = algo.run()

    # Visualize or output
    visualize_solution(graph, robots, solution)
    plot_deap_statistics(logbook)"""

def run_deap_ea():
    config = load_config()

    # Load agents and map
    graph, robots = load_map_and_robots(config['scenario_file'])
    print(robots)

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    algo = PathBasedEA_DEAP(graph, 
                    robots, 
                    config['ea_params']['population_size'],
                    config['ea_params']['num_generations'],
                    config['ea_params']['mutation_rate'],
                    config['ea_params']['crossover_rate'],)
    #algo.initial_population()
    solution, logbook = algo.run()

    # Visualize or output
    visualize_solution(graph, robots, solution)
    plot_deap_statistics(logbook)
#    plot_analysis(analysis)

def run_p3():
    config = load_config()

    # Load agents and map
    graph, robots = load_map_and_robots(config['scenario_file'])
    print(robots)

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    p3 = P3(graph, 
             robots, 
             config['ea_params']['num_generations'])

    best, cost = p3.run()
    print("Best fitness", cost)
    show_statistics(p3.best_hist, p3.avg_hist, p3.conf_hist)
    visualize_solution(graph, robots, best)

def run_p3_2():
    config = load_config()

    # Load agents and map
    graph, robots = load_map_and_robots(config['scenario_file'])
    print(robots)

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    p3_2 = P3_2(graph, 
             robots, 
             config['ea_params']['num_generations'])

    best, best_history, avg_history, conf_history = p3_2.run()
    print("Best fitness", best)
    show_statistics(best_history, avg_history, conf_history)
    visualize_solution(graph, robots, best)

def main():
    #run_feasibile_only_ea()
    #run_deap_ea()
    #run_p3()
    run_p3_2()
    #run_non_deap_ea()
    #run_deap_ea_multi()
    
if __name__ == "__main__":
    main()
