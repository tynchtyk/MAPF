# ================= batch_benchmark.py =============================
import argparse, glob, os, time, random, statistics, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid", font_scale=1.3)
import yaml
from utils import *
from algorithms.p3_dsm import P3_DSM
from algorithms.p3_dsm_hybrid import P3_DSM_HYBRID
from algorithms.p3_dsm_robot_conflicts import P3_DSM_ROBOT_CONFLICRS
from algorithms.p3_base import P3_Base
from algorithms.p3_cdgx import P3_CDGX
#from algorithms.optimized_ea import PathBasedEA_DEAP
from map_graph import MapfGraph
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# -----------------------------------------------------------------
# User-editable parameters
# -----------------------------------------------------------------
N_SEEDS         = 1           # independent runs per scenario / algorithm
NUM_GENERATIONS = 20          # can be read from your YAML config instead
OUT_CSV         = "benchmark_raw.csv"
OUT_FIG         = "summary_cost.png"
# -----------------------------------------------------------------

# -------------- Your existing helpers ----------------------------
# Assume these are already imported from your framework
# from your_module import load_map_and_robots, P3, P3_2, show_statistics, ...
# -----------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def run_one(AlgClass, scenario_file, seed):
    random.seed(seed); np.random.seed(seed)
    graph, robots = load_map_and_robots(scenario_file)
    ea = AlgClass(graph, robots, NUM_GENERATIONS)
    t0 = time.perf_counter()
    ea.run()                                # returns paths & histories
    dt = time.perf_counter() - t0
    return dict(cost=ea.best_cost, runtime=dt)


def robot_meta(robots):
    n_agents = len(robots)
    # assume each Robot object has .targets (list of goals)
    goals_per = statistics.mean(len(r.targets) for r in robots) if robots else 0
    return n_agents, goals_per


def _run_task(args):
    """Worker wrapper so ProcessPool can pickle the call"""
    AlgClass, alg_name, scenario_file, seed, num_generations = args
    # local imports inside the subprocess
    import random, time, numpy as np
    from utils import load_map_and_robots          # make sure resolvable
    ea_args = dict(generations=num_generations)    # adapt if API differs

    random.seed(seed); np.random.seed(seed)
    graph, robots = load_map_and_robots(scenario_file)
    ea = AlgClass(graph, robots, **ea_args)

    t0 = time.perf_counter()
    ea.run()
    dt = time.perf_counter() - t0
    return {
        "scenario": os.path.basename(scenario_file),
        "agents": len(robots),
        "goals": statistics.mean(len(r.targets) for r in robots),
        "alg": alg_name,
        "seed": seed,
        "cost": ea.best_cost,
        "distance": sum(len(p) for p in ea.best_individual.values()),
        "conflicts": len(ea._detect_conflicts(ea.best_individual)),
        "runtime": dt,
        "generations": ea.generations,
    }

# ----------------------------------------------------------------
# Parallel batch_folder
# ----------------------------------------------------------------
def batch_folder(folder):
    scenario_files = sorted(glob.glob(os.path.join(folder, "*.*")))
    scenario_files = [f for f in scenario_files
                      if pathlib.Path(f).suffix.lower() in
                         {".yaml", ".yml", ".json", ".txt"}]
    if not scenario_files:
        raise RuntimeError("No scenario files found in the folder!")

    seeds = list(range(N_SEEDS))
    algs  = [("P3_BASE", P3_Base), ("P3_DSM", P3_DSM_ROBOT_CONFLICRS), ("P3_DSM_HYBRID", P3_DSM_HYBRID),]

    # Build job list
    jobs = []
    cnt = 0
    for scen in scenario_files:
        #if scen[35] == '4':
        for alg_name, Alg in algs:
            for s in seeds:
                jobs.append((Alg, alg_name, scen, s, NUM_GENERATIONS))
        cnt += 1

    # Run in parallel
    results = []
    max_workers = min(multiprocessing.cpu_count(), len(jobs))
    print(f"Running {len(jobs)} tasks on {max_workers} processes …")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        fut2job = {pool.submit(_run_task, job): job for job in jobs}
        for fut in as_completed(fut2job):
            res = fut.result()
            results.append(res)
            print(f"✔ {res['scenario']:30s} | {res['alg']:7s} "
                  f"seed={res['seed']:<2}  cost={res['cost']:.1f} "
                  f"time={res['runtime']:.2f}s")

    return pd.DataFrame(results)


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

def _prepare_results(df: pd.DataFrame):
    """Prepare dataset: test_id, best_cost, rel_cost (no outlier removal)."""
    df = df.copy()
    df["test_id"] = df.groupby("scenario").ngroup() + 1
    df["best_cost"] = df.groupby("scenario")["cost"].transform("min")
    df["rel_cost"] = df["cost"] / df["best_cost"]
    return dict(df=df)


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

def plot_mean_metric_per_algorithm(df, metric, ylabel, title, savefile):
    plt.figure(figsize=(8, 5))
    mean_vals = df.groupby("alg")[metric].mean().sort_values()
    sns.barplot(x=mean_vals.index, y=mean_vals.values, palette="Set2")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

    # Print the computed averages to console
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

def plot_fitness_vs_problem_category(df, savefile="fitness_vs_problem_category.png"):
    df = df.copy()
    # Create tuple-based problem categories and sort them
    df["problem_category"] = list(zip(df["agents"], df["goals"]))
    df["problem_category"] = df["problem_category"].apply(lambda x: (int(x[0]), float(x[1])))
    df = df.sort_values(by=["agents", "goals"])

    # Aggregate mean cost per category and algorithm
    agg = (
        df.groupby(["problem_category", "alg"])
          .agg(mean_cost=("cost", "mean"))
          .reset_index()
    )

    # Prepare sorted x-ticks
    categories = sorted(agg["problem_category"].unique(), key=lambda x: (x[0], x[1]))
    x_indices = range(len(categories))

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    markers = ['o', '^', 's', 'D', '*', 'x']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']

    for idx, (alg, sub) in enumerate(agg.groupby("alg")):
        sub = sub.set_index("problem_category").loc[categories].reset_index()
        plt.plot(x_indices, sub["mean_cost"],
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 label=alg,
                 linewidth=2)

    # X-axis with tuple labels
    plt.xticks(ticks=x_indices, labels=[f"{r}×{g:.1f}" for r, g in categories], rotation=45)
    plt.yscale('linear')
    plt.xlabel("Problem Category (#robots × #goals per robot)")
    plt.ylabel("Fitness (Cost, log scale)")
    plt.title("Algorithm Fitness vs Problem Category")
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.show()

    
def main():
    parser = argparse.ArgumentParser(
        description="Run P3 and P3_2 on every scenario file in a folder")
    parser.add_argument("folder", help="Folder containing scenario files")
    args = parser.parse_args()

    #df = batch_folder(args.folder)
    #df.to_csv(OUT_CSV, index=False)
    #print("Saved raw results to", OUT_CSV)

    df = pd.read_csv("benchmark_raw.csv")
    #print(df.head())
    #plot_summary(df, metric="cost", savefile="cost_summary.png")
    #plot_test_vs_fitness(df,  savefile="cost_by_test.png")
    #plot_runtime_vs_cost(df)                     # scatter (log cost)
    #plot_mean_cost_per_alg(df)                   # bar (relative cost)

    # Plot 1: Mean Cost per Algorithm
    plot_mean_metric_per_algorithm(
        df,
        metric="cost",
        ylabel="Mean Final Cost",
        title="Mean Final Cost per Algorithm",
        savefile="mean_cost_per_algorithm.png"
    )


    # Plot 3: Mean Conflicts per Algorithm
    plot_mean_metric_per_algorithm(
        df,
        metric="conflicts",
        ylabel="Mean Number of Conflicts",
        title="Mean Number of Conflicts per Algorithm",
        savefile="mean_conflicts_per_algorithm.png"
    )

    plot_fitness_vs_problem_category(df, savefile="fitness_vs_problem_category.png")


if __name__ == "__main__":
    main()
