import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ea_metrics(generations, fitness_history, makespan_history, conflict_history, distance_history):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot Fitness
    sns.lineplot(x=generations, y=fitness_history, ax=axs[0, 0], marker='o', color='blue')
    axs[0, 0].set_title('Fitness Convergence')
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Fitness (Weighted)')

    # Plot Makespan
    sns.lineplot(x=generations, y=makespan_history, ax=axs[0, 1], marker='o', color='green')
    axs[0, 1].set_title('Makespan Evolution')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Makespan (Steps)')

    # Plot Conflicts
    sns.lineplot(x=generations, y=conflict_history, ax=axs[1, 0], marker='o', color='red')
    axs[1, 0].set_title('Conflict Evolution')
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Number of Conflicts')

    # Plot Travel Distance
    sns.lineplot(x=generations, y=distance_history, ax=axs[1, 1], marker='o', color='purple')
    axs[1, 1].set_title('Travel Distance Evolution')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Total Travel Distance (Steps)')

    plt.tight_layout()
    plt.show()

def plot_ea_metrics(generations, fitness_history):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot Fitness
    sns.lineplot(x=generations, y=fitness_history, ax=axs[0, 0], marker='o', color='blue')
    axs[0, 0].set_title('Fitness Convergence')
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Fitness (Weighted)')

def plot_combined_metrics(generations, min_fitness_history, max_fitness_history, mean_fitness_history, conflict_count_history):
    plt.figure(figsize=(14, 8))

    # Plot fitness metrics
    plt.plot(generations, min_fitness_history, marker='o', label='Min Fitness', linewidth=2)
    plt.plot(generations, max_fitness_history, marker='s', label='Max Fitness', linewidth=2)
    plt.plot(generations, mean_fitness_history, marker='X', label='Mean Fitness', linewidth=2)

    # Create a twin axis to plot conflicts separately
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(generations, conflict_count_history, color='red', linestyle='--', marker='^', label='Conflicts', linewidth=2)

    # Titles and labels
    ax1.set_title('Evolution of Fitness and Conflict Metrics per Generation', fontsize=16)
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Fitness Metrics', fontsize=14)
    ax2.set_ylabel('Number of Conflicts', fontsize=14, color='red')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_deap_statistics(logbook,
                         title="Evolutionary Algorithm Fitness Statistics",
                         y_label="Fitness (Cost)",
                         use_log_scale_threshold=100):
    """
    Plots evolution of fitness statistics (min, max, avg, std) from DEAP's logbook.
    Assumes a minimization context where lower fitness values are better.

    Args:
        logbook (deap.logbook.Logbook): The logbook object returned by DEAP algorithms.
        title (str): The title for the plot.
        y_label (str): The label for the Y-axis.
        use_log_scale_threshold (float | None): If max(avg_fitness) / min(best_fitness) > threshold,
                                                automatically use a log scale for the Y-axis.
                                                Set to None or 0 to disable automatic log scale.
    """
    if plt is None or sns is None:
         print("Plotting libraries not available. Cannot generate plot.")
         return

    # --- Data Extraction and Validation ---
    required_keys = ["gen", "min", "max", "avg"]
    if not all(key in logbook.header for key in required_keys):
        missing = [key for key in required_keys if key not in logbook.header]
        print(f"Warning: Logbook missing required keys: {missing}. Cannot generate full plot.")
        return

    try:
        generations = np.array(logbook.select("gen"))
        # Assuming minimization: 'min' is best, 'max' is worst
        fit_best = np.array(logbook.select("min"))
        fit_worst = np.array(logbook.select("max"))
        fit_avg = np.array(logbook.select("avg"))
        #fit_std = np.array(logbook.select("std"))
    except Exception as e:
        print(f"Error extracting data from logbook: {e}")
        return

    if len(generations) == 0:
         print("Warning: No data points (generations) found in logbook.")
         return

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid") # Use seaborn styling

    # Plot Worst Fitness (Max Value) - often useful to see penalties or poor solutions
    plt.plot(generations, fit_worst, label="Worst Fitness (Max Cost)", marker='^', linestyle=':', color='salmon', markersize=5, alpha=0.7)

    # Plot Best Fitness (Min Value) - primary indicator of progress
    plt.plot(generations, fit_best, label="Best Fitness (Min Cost)", marker='s', linestyle='-', color='forestgreen', markersize=5, linewidth=1.5)

    # Plot Average Fitness
    plt.plot(generations, fit_avg, label="Mean Fitness", linestyle='--', color='royalblue', linewidth=1.5)

    # Plot Standard Deviation Range (Avg ± Std)
    #lower_bound = fit_avg - fit_std
    #upper_bound = fit_avg + fit_std
    #plt.fill_between(generations, lower_bound, upper_bound,
    #                 color='lightblue', alpha=0.4, label="Std Dev Range (Avg ± σ)")

    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best") # Let matplotlib decide the best location

    # --- Dynamic Y-axis Scaling ---
    min_positive_best_fit = fit_best[fit_best > 0] if np.any(fit_best > 0) else None
    use_log = False
    if use_log_scale_threshold and use_log_scale_threshold > 0 and min_positive_best_fit is not None and len(min_positive_best_fit) > 0:
         # Check if the range warrants log scale (e.g., max average is much larger than min best)
         if np.max(fit_avg) > np.min(min_positive_best_fit) * use_log_scale_threshold:
              use_log = True

    if use_log:
        print("Applying log scale to Y-axis.")
        plt.yscale('log')
        # Adjust bottom ylim for log scale, ensuring it's > 0 and slightly below the minimum positive value
        min_y_lim = max(1e-9, np.min(min_positive_best_fit) * 0.8) if min_positive_best_fit is not None and len(min_positive_best_fit) > 0 else 1e-9
        plt.ylim(bottom=min_y_lim)
    else:
        # For linear scale, set the bottom limit to be slightly below the minimum fitness
        min_y = np.min(fit_best)
        plt.ylim(bottom=min_y * 1.1 if min_y < 0 else min_y * 0.9)

    plt.tight_layout() # Adjust plot to prevent labels overlapping
    plt.show()

def plot_deap_statistics_multi(self):
        """Plots the average and minimum number of conflicts and total distance over generations."""
        generations = np.array(self.select("gen"))

        # Plot Average Conflicts and Distance
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(generations, self.mean_conflict_history, label='Average Conflicts')
        plt.plot(generations, self.min_conflict_history, label='Minimum Conflicts')
        plt.xlabel('Generation')
        plt.ylabel('Number of Conflicts')
        plt.title('Average and Minimum Number of Conflicts over Generations')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(generations, self.mean_distance_history, label='Average Total Distance')
        plt.plot(generations, self.min_distance_history, label='Minimum Total Distance')
        plt.xlabel('Generation')
        plt.ylabel('Total Path Length')
        plt.title('Average and Minimum Total Path Length over Generations')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot Pareto Front (if any solutions in hof)
        if self.hof:
            conflicts = [ind.fitness.values[0] for ind in self.hof]
            distances = [ind.fitness.values[1] for ind in self.hof]

            plt.figure(figsize=(8, 6))
            plt.scatter(conflicts, distances, label='Pareto Front')
            plt.xlabel('Number of Conflicts')
            plt.ylabel('Total Path Length')
            plt.title('Pareto Front of Non-dominated Solutions')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            print("No non-dominated solutions found in the Hall of Fame.")