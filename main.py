import yaml
from utils import *
from algorithms.ga import TimeLogger
from algorithms.p3_dsm import P3_DSM
from algorithms.p3_dsm_robot_conflicts import P3_DSM_ROBOT_CONFLICRS
from algorithms.p3_dsm_hybrid import P3_DSM_HYBRID
from algorithms.p3_base import P3_Base
from algorithms.p3_cdgx import P3_CDGX
#from algorithms.optimized_ea import PathBasedEA_DEAP
from map_graph import MapfGraph
from plot import *
import numpy as np
import time

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

def run_p3_base():
    config = load_config()

    # Load agents and map
    graph, robots = load_map_and_robots(config['scenario_file'])
    print(robots)

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    p3 = P3_Base(graph, 
             robots, 
             config['ea_params']['num_generations'])

    best, best_history, avg_history, conf_history = p3.run()
    print("Best fitness", best)
    show_statistics(best_history, avg_history, conf_history)
    visualize_solution(graph, robots, best)

def run_p3_dsm():
    config = load_config()

    # Load agents and map
    graph, robots = load_map_and_robots(config['scenario_file'])
    print(robots)

    #show_graph_structure(graph)
    #show_graph_with_robots(graph, robots)

    # Plug in any algorithm here
    p3_DSM = P3_DSM_HYBRID(graph, 
             robots, 
             config['ea_params']['num_generations'])

    best, best_history, avg_history, conf_history = p3_DSM.run()
    print("Best fitness", best)
    show_statistics(best_history, avg_history, conf_history)
    visualize_solution(graph, robots, best)
    plot_time_log(p3_DSM.time_logger)


def collect_histories(N=5):
    config = load_config()
    seeds = list(range(N))

    all_p3_base, all_p3_dsm , algo_p3_cdgx = [], [], []
    times_p3_base, times_p3_dsm, times_p3_cdgx = [], [], []

    for seed in seeds:
        print(f"\nSeed {seed}")
        #Algo 1
        random.seed(seed); np.random.seed(seed)
        graph, robots = load_map_and_robots(config['scenario_file'])

        start = time.perf_counter()
        p3_base = P3_Base(graph, robots, config['ea_params']['num_generations'])
        _, b1, a1, c1 = p3_base.run()
        times_p3_base.append(time.perf_counter() - start)
        all_p3_base.append((b1, a1, c1))

        #Algo 2
        random.seed(seed); np.random.seed(seed)
        graph, robots = load_map_and_robots(config['scenario_file'])

        start = time.perf_counter()
        p3_dsm_cdgx = P3_DSM_ROBOT_CONFLICRS(graph, robots, config['ea_params']['num_generations'])
        _, b2, a2, c2 = p3_dsm_cdgx.run()
        times_p3_cdgx.append(time.perf_counter() - start)
        algo_p3_cdgx.append((b2, a2, c2))

        #Algo 3
        random.seed(seed); np.random.seed(seed)
        graph, robots = load_map_and_robots(config['scenario_file'])

        start = time.perf_counter()
        p3_dsm = P3_DSM_HYBRID(graph, robots, config['ea_params']['num_generations'])
        _, b3, a3, c3 = p3_dsm.run()
        times_p3_dsm.append(time.perf_counter() - start)
        all_p3_dsm.append((b3, a3, c3))


    print("\n=== Timing Summary ===")
    print(f"P3   : {np.mean(times_p3_base):.2f}s ± {np.std(times_p3_base):.2f}")
    print(f"P3_CDGX : {np.mean(times_p3_cdgx):.2f}s ± {np.std(times_p3_cdgx):.2f}")
    print(f"P3_DSM : {np.mean(times_p3_dsm):.2f}s ± {np.std(times_p3_dsm):.2f}")

    return all_p3_base, algo_p3_cdgx, all_p3_dsm, times_p3_base, times_p3_cdgx, times_p3_dsm

def main():
    #run_feasibile_only_ea()
    #run_deap_ea()
    #run_p3_base()
    #run_p3_cdgx()
    #run_p3_dsm()
    #run_non_deap_ea()
    #run_deap_ea_multi()

    all_p3_base, algo_dsm_robot_conflicts, all_p3_dsm_hybrid, times_p3_base, times_p3_dsm_robot_conflicts, times_p3_dsm_hybrid = collect_histories(2)
    show_statistics_mean_std(all_p3_base, algo_dsm_robot_conflicts, all_p3_dsm_hybrid)
    plot_runtime_comparison(times_p3_base, times_p3_dsm_robot_conflicts, times_p3_dsm_hybrid)
if __name__ == "__main__":
    main()
