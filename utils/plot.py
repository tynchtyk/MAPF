from turtle import pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker
from matplotlib import animation
from algorithms.ga import TimeLogger
import pandas as pd

ALGORITHM_COLORS = {
    "SPC": "orange",
    "CHHM": "green",
    "CM": "blue",
}

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

def plot_analysis(analysis_data):
    """Generates three analysis plots"""
    plt.figure(figsize=(15, 5))
    
    # Exploration plot
    plt.subplot(1, 3, 1)
    gens = [x['generation'] for x in analysis_data['exploration']]
    plt.plot(gens, [x['path_diversity'] for x in analysis_data['exploration']])
    plt.title('Solution Diversity')
    plt.xlabel('Generation')
    plt.ylabel('Unique Solutions Ratio')
    
    # Gradient plot
    plt.subplot(1, 3, 2)
    gens = [x['generation'] for x in analysis_data['gradient']]
    plt.plot(gens, [x['avg_improvement'] for x in analysis_data['gradient']])
    plt.title('Fitness Improvement')
    plt.xlabel('Generation')
    plt.ylabel('Average Improvement')
    
    # Conflict plot
    plt.subplot(1, 3, 3)
    gens = [x['generation'] for x in analysis_data['conflicts']]
    plt.plot(gens, [x['avg_conflicts'] for x in analysis_data['conflicts']])
    plt.plot(gens, [x['min_conflicts'] for x in analysis_data['conflicts']])
    plt.plot(gens, [x['max_conflicts'] for x in analysis_data['conflicts']])
    plt.title('Conflict Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Number of Conflicts')
    plt.legend(['Average', 'Minimum', 'Maximum'])
    
    plt.tight_layout()
    plt.show()

def plot_deap_statistics(logbook,
                         title="Evolutionary Algorithm Fitness Statistics",
                         y_label="Fitness (Cost)",
                         use_log_scale_threshold=1):
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
    """min_positive_best_fit = fit_best[fit_best > 0] if np.any(fit_best > 0) else None
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
        plt.ylim(bottom=min_y * 1.1 if min_y < 0 else min_y * 0.9)"""

    # --- Y-axis Formatting (no log scale) ---
    min_y = np.min(fit_best)
    plt.ylim(bottom=min_y * 1.1 if min_y < 0 else min_y * 0.9)
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))  # Always show direct numbers

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

def show_statistics(best_hist, avg_hist, conf_hist,
                       title="P3 Fitness Statistics", y_label="Fitness (Cost)"):
    """Replicates the look of your DEAP‐style plot for P3 arrays."""
    import matplotlib.ticker as mticker
    if not best_hist:
        print("No statistics to plot.")
        return
    gens = range(len(best_hist))
    plt.figure(figsize=(12,6))
    import seaborn as sns; sns.set(style="whitegrid")

    # Worst ≈ max fitness per generation (can be reconstructed)
    plt.plot(gens, best_hist,  label="Best Fitness",  linestyle='-', color='forestgreen', linewidth=1.5)
    plt.plot(gens, avg_hist,   label="Mean Fitness",  linestyle='--',color='royalblue', linewidth=1.5)

    # Conflicts on secondary axis
    ax1 = plt.gca(); ax2 = ax1.twinx()
    ax2.plot(gens, conf_hist, color='red', label='Avg Conflicts', linewidth=1.2)
    ax2.set_ylabel('# Conflicts', color='red'); ax2.tick_params(axis='y', colors='red')

    ax1.set_xlabel('Generation'); ax1.set_ylabel(y_label)
    ax1.set_title(title)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
    plt.tight_layout(); plt.show()

def plot_time_log(logger: TimeLogger):
    names = []
    totals = []
    avgs = []
    counts = []

    for name, data in logger.records.items():
        names.append(name)
        totals.append(data["total_time"])
        counts.append(data["count"])
        avgs.append(data["total_time"] / data["count"])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # --- Total Time Plot ---
    axs[0].barh(names, totals, color='skyblue', edgecolor='black')
    axs[0].set_title("Total Time (s)", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Seconds", fontsize=12)
    axs[0].grid(axis='x', linestyle='--', alpha=0.7)

    # --- Average Time Plot ---
    axs[1].barh(names, avgs, color='lightgreen', edgecolor='black')
    axs[1].set_title("Average Time per Call (s)", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Seconds", fontsize=12)
    axs[1].grid(axis='x', linestyle='--', alpha=0.7)

    # --- Call Count Plot ---
    axs[2].barh(names, counts, color='lightcoral', edgecolor='black')
    axs[2].set_title("Number of Calls", fontsize=14, fontweight='bold')
    axs[2].set_xlabel("Calls", fontsize=12)
    axs[2].grid(axis='x', linestyle='--', alpha=0.7)

    # --- Overall Layout ---
    plt.suptitle("Function Execution Time Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_runtime_comparison(times_p3, times_p3_cdgx, times_p3_dsm, label1="SPC", label2="RGCM", label3="CHHM"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set(style="whitegrid")

    seeds = np.arange(1, len(times_p3) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(seeds, times_p3,   label=label1, marker='o', linewidth=2, color='forestgreen')
    plt.plot(seeds, times_p3_cdgx, label=label2, marker='s', linewidth=2, color='darkorange')
    plt.plot(seeds, times_p3_dsm, label=label3, marker='p', linewidth=3, color='royalblue')

    plt.xlabel("Seed #")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime per Run")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def show_statistics_mean_std(all1, all2, all3,
    label1="SPC", label2="RGCM", label3="CHHM",
    title="Mean Best Fitness per Generation",
    color1="forestgreen", color2="darkorange", color3="royalblue",
    save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid", font_scale=1.2)

    def pad_histories(histories, idx):
        L = max(len(h[idx]) for h in histories)
        return np.array([h[idx] + [h[idx][-1]] * (L - len(h[idx])) for h in histories])

    # Expect each item in all1, all2 to be a tuple: (best_history, avg_history, conf_history)
    best1 = pad_histories(all1, 0)
    avg1  = pad_histories(all1, 1)

    best2 = pad_histories(all2, 0)
    avg2  = pad_histories(all2, 1)

    best3 = pad_histories(all3, 0)
    avg3  = pad_histories(all3, 1)

    gens = range(best1.shape[1])

    mean_best1 = best1.mean(axis=0)
    mean_avg1  = avg1.mean(axis=0)

    mean_best2 = best2.mean(axis=0)
    mean_avg2  = avg2.mean(axis=0)

    mean_best3 = best3.mean(axis=0)
    mean_avg3  = avg3.mean(axis=0)

    print("Algo 1: ", avg1)
    print("Algo 2: ", avg2)
    print("Algo 3: ", avg3)

    plt.figure(figsize=(10, 6))
    plt.plot(gens, mean_best1, label=f"{label1} (Best)", color=color1, linewidth=2)
    plt.plot(gens, mean_avg1, label=f"{label1} (Avg)", linestyle="--", color=color1, linewidth=2)

    plt.plot(gens, mean_best2, label=f"{label2} (Best)", color=color2, linewidth=2)
    plt.plot(gens, mean_avg2, label=f"{label2} (Avg)", linestyle="--", color=color2, linewidth=2)

    plt.plot(gens, mean_best3, label=f"{label3} (Best)", color=color3, linewidth=3)
    plt.plot(gens, mean_avg3, label=f"{label3} (Avg)", linestyle="--", color=color3, linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Fitness (Lower is Better)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()



def animate_solution(graph, robots, paths):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, graph.width-0.5); ax.set_ylim(-0.5, graph.height-0.5)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    # obstacles
    for (x,y) in graph.obstacles:
        ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1, color='black'))
    colors = plt.cm.tab10(range(len(robots)))
    circles = {}
    for r, col in zip(robots, colors):
        c = plt.Circle(r.start, 0.3, color=col, zorder=3)
        ax.add_patch(c); circles[r.robot_id] = c
    max_t = max(len(p) for p in paths.values())
    def update(frame):
        for r in robots:
            p = paths[r.robot_id]
            pos = p[frame] if frame < len(p) else p[-1]
            circles[r.robot_id].center = pos
        return circles.values()
    ani = animation.FuncAnimation(fig, update, frames=max_t, interval=250, blit=True)
    plt.show()

def _prepare_results(df: pd.DataFrame):
    """Prepare dataset: test_id, best_cost, rel_cost (no outlier removal)."""
    df = df.copy()
    df["test_id"] = df.groupby("scenario").ngroup() + 1
    df["best_cost"] = df.groupby("scenario")["cost"].transform("min")
    df["rel_cost"] = df["cost"] / df["best_cost"]
    return dict(df=df)

def plot_runtime_vs_cost(df: pd.DataFrame, savefile: str = "runtime_vs_cost.png"):
    """Scatter plot of runtime vs absolute cost (linear scale)."""
    pp = _prepare_results(df)
    df = pp["df"]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set1", df["alg"].nunique())
    sns.scatterplot(
        data=df,
        x="runtime",
        y="cost",
        hue="alg",
        style="alg",
        palette=palette,
        s=80
    )
    plt.xlabel("Runtime (s)")
    plt.ylabel("Cost")
    plt.title("Runtime vs Cost")
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()


def plot_mean_cost_per_alg(df: pd.DataFrame, savefile: str = "mean_cost_bar.png"):
    """Bar plot of mean relative cost per algorithm."""
    pp = _prepare_results(df)
    df = pp["df"]

    mean_rel = df.groupby("alg")["rel_cost"].mean().sort_values()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=mean_rel.index, y=mean_rel.values, palette="Set2")
    plt.ylabel("Mean Relative Cost (↓ is better)")
    plt.title("Average Quality by Algorithm")
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_mean_metric_per_algorithm(df, metric, ylabel, title, savefile, log_y=True):
    plt.figure(figsize=(8, 5))
    mean_vals = df.groupby("alg")[metric].mean().sort_values()
    alg_order = mean_vals.index.tolist()
    palette = [ALGORITHM_COLORS.get(alg, "gray") for alg in alg_order]

    sns.barplot(x=alg_order, y=mean_vals.values, palette=palette)
    plt.ylabel(ylabel)
    plt.title(title)

    if log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

    print(f"\nAverage {metric} per algorithm:")
    print(mean_vals.round(3).to_string())
    print("-" * 50)



def plot_distance_vs_conflicts(df, savefile="distance_vs_conflicts.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="distance",
        y="conflicts",
        hue="alg",
        style="alg",
        s=100
    )
    plt.xlabel("Total Distance")
    plt.ylabel("Number of Conflicts")
    plt.title("Distance–Conflicts Trade-off")
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_fitness_vs_problem_category(df, savefile="fitness_vs_problem_category.png", log_y=True):
    df = df.copy()
    df["problem_category"] = list(zip(df["agents"], df["goals"]))
    df["problem_category"] = df["problem_category"].apply(lambda x: (int(x[0]), float(x[1])))
    df = df.sort_values(by=["agents", "goals"])

    agg = df.groupby(["problem_category", "alg"]).agg(mean_cost=("cost", "mean")).reset_index()
    categories = sorted(agg["problem_category"].unique(), key=lambda x: (x[0], x[1]))
    x_indices = range(len(categories))

    plt.figure(figsize=(10, 6))

    for alg, sub in agg.groupby("alg"):
        sub = sub.set_index("problem_category").loc[categories].reset_index()
        color = ALGORITHM_COLORS.get(alg, "gray")
        plt.plot(x_indices, sub["mean_cost"], marker='o', color=color, label=alg, linewidth=2)

    plt.xticks(ticks=x_indices, labels=[f"{r}×{g:.1f}" for r, g in categories], rotation=45)
    plt.xlabel("Problem Category (#robots × #goals per robot)")
    plt.ylabel("Fitness (Cost)")
    if log_y:
        plt.yscale("log")  # Only apply log scale if requested
        plt.ylabel("Fitness (Cost, log scale)")
    plt.title("Algorithm Fitness vs Problem Category")
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()


def plot_cost_vs_runtime(df, savefile="cost_vs_runtime.png"):
    """
    Plots the trade-off between solution quality (cost) and runtime.

    Args:
        df (pd.DataFrame): DataFrame with results.
        savefile (str): Path to save the plot image.
    """
    # Aggregate data to get mean cost and runtime for each alg and problem size
    agg_df = df.groupby(['agents', 'alg']).agg(
        mean_cost=('cost', 'mean'),
        mean_runtime=('runtime', 'mean')
    ).reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # Create the scatter plot
    sns.scatterplot(
        data=agg_df,
        x='mean_runtime',
        y='mean_cost',
        hue='alg',
        style='agents',  # Different markers for different numbers of agents
        s=150,  # Marker size
        palette='muted'
    )

    plt.title('Performance Trade-off: Solution Cost vs. Runtime', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Runtime (seconds)', fontsize=12)
    plt.ylabel('Mean Cost', fontsize=12)
    plt.legend(title='Algorithm | # Agents')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_scalability(df, metric='cost', savefile="scalability_cost.png"):
    """
    Plots how a given metric (cost or runtime) scales with the number of agents.
    Includes shaded error bands for standard deviation.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric (str): The column to plot ('cost' or 'runtime').
        savefile (str): Path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # Use seaborn's lineplot to automatically calculate means and confidence intervals
    sns.lineplot(
        data=df,
        x='agents',
        y=metric,
        hue='alg',
        marker='o',
        linewidth=2.5,
        errorbar='sd', # Use standard deviation for the error band
        palette='muted'
    )

    title_metric = 'Solution Cost' if metric == 'cost' else 'Runtime'
    plt.title(f'{title_metric} Scalability vs. Number of Agents', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel(f'Mean {title_metric.capitalize()}', fontsize=12)
    if metric == 'cost':
        plt.yscale('log')
        plt.ylabel(f'Mean {title_metric.capitalize()} (Log Scale)', fontsize=12)

    plt.legend(title='Algorithm')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_convergence(df_generation_data, savefile="convergence.png"):
    """
    Plots the convergence of algorithms over generations.

    Args:
        df_generation_data (pd.DataFrame): DataFrame with per-generation data.
            Must contain 'generation', 'cost', 'alg', and a problem identifier column.
        savefile (str): Path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # This plot assumes the data is for a single, representative problem instance.
    # You could also average over multiple runs if the data is structured that way.
    sns.lineplot(
        data=df_generation_data,
        x='generation',
        y='cost',
        hue='alg',
        linewidth=2.5,
        palette='muted',
        errorbar='sd' # Shows variability across different runs of the same problem
    )

    plt.title('Algorithm Convergence Over Generations', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Cost Found', fontsize=12)
    plt.legend(title='Algorithm')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_test_vs_fitness(df: pd.DataFrame, savefile: str = "fitness_vs_test.png"):
    """Line plot of relative cost per test case."""
    pp = _prepare_results(df)
    df = pp["df"]

    agg = (df.groupby(["test_id", "alg"])
             .agg(mean_rel=("rel_cost", "mean"))
             .reset_index())

    plt.figure(figsize=(10, 5))
    for alg, sub in agg.groupby("alg"):
        plt.plot(sub["test_id"], sub["mean_rel"], "-o", label=alg)

    plt.xlabel("Test Number")
    plt.ylabel("Average Relative Cost (cost / scenario best)")
    plt.title("Algorithm Quality per Test Case")
    plt.xticks(sorted(df["test_id"].unique()))
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

def plot_summary(df):
    agg = (df.groupby(["scenario", "agents", "goals", "alg"])
             .agg(cost_mean=("cost", "mean"),
                  cost_sd  =("cost",  "std"),
                  rt_mean  =("runtime","mean"))
             .reset_index())

    plt.figure(figsize=(11, 6))
    ax = sns.scatterplot(data=agg,
                         x="agents", y="cost_mean",
                         hue="alg", style="goals", s=120)
    for _, row in agg.iterrows():
        ax.errorbar(row["agents"], row["cost_mean"],
                    yerr=row["cost_sd"], fmt="none",
                    ecolor="gray", alpha=0.4, capsize=3)

    ax.set_xlabel("# robots"); ax.set_ylabel("Final cost (mean ± sd)")
    ax.set_title("Scalability of P3 vs P3-LT on all scenarios")
    ax.legend(title="Algorithm / goals")
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=300)
    plt.show()

def plot_summary(df, metric="cost", savefile="summary_plot.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert metric in {"cost", "runtime"}

    # Aggregate mean ± std
    agg = (
        df.groupby(["scenario", "agents", "goals", "alg"])
          .agg(
              value_mean=(metric, "mean"),
              value_sd=(metric, "std")
          )
          .reset_index()
    )

    ylabel = "Final Cost" if metric == "cost" else "Runtime (s)"

    g = sns.FacetGrid(
        agg, col="scenario", col_wrap=3,
        hue="alg", height=4, aspect=1.3,
        palette="Set2", sharey=False
    )

    g.map_dataframe(sns.scatterplot,
                    x="agents", y="value_mean",
                    style="goals", s=100)

    # Error bars
    for ax, (_, sub) in zip(g.axes.flat, agg.groupby("scenario")):
        for _, row in sub.iterrows():
            ax.errorbar(row["agents"], row["value_mean"],
                        yerr=row["value_sd"],
                        fmt="none", capsize=3,
                        alpha=0.4, ecolor="gray")

    g.set_axis_labels("# Robots", ylabel)
    g.add_legend(title="Algorithm / Goals")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{ylabel} Comparison per Scenario", fontsize=16)

    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()