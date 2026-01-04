import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# ==========================================
#               CONFIGURATION
# ==========================================

ALG_ORDER = ["Base", "RGCM", "CHHM"]

ALG_STYLE = {
    "Base": {"color": "#1f77b4", "marker": "o", "label": "Base P3", "zorder": 10},
    "RGCM": {"color": "#ff7f0e", "marker": "s", "label": "RGCM", "zorder": 5},
    "CHHM": {"color": "#2ca02c", "marker": "^", "label": "CHHM", "zorder": 6}
}

MAX_RUNTIME = 180.0 
TIME_GRID_POINTS = 100 

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 12,
    "lines.linewidth": 2.5,
    "lines.markersize": 9,
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath):
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def get_map_name(scenario_id):
    s = str(scenario_id).lower()
    for key in ["warehouse", "room", "random", "empty", "maze", "city"]:
        if s.startswith(key): return key
    return s.split("_")[0]

def parse_history(history_str):
    try:
        if pd.isna(history_str) or str(history_str).strip() in ["[]", ""]: return []
        s_clean = str(history_str).replace("inf", "float('inf')").replace("nan", "float('nan')")
        return ast.literal_eval(s_clean)
    except: return []

def get_step_values(history, time_grid):
    if not history: return np.full_like(time_grid, np.nan)
    history = sorted(history, key=lambda x: x[0])
    times, costs = zip(*history)
    idx = np.searchsorted(times, time_grid, side='right') - 1
    return np.array([costs[i] if i >= 0 else costs[0] for i in idx])

# ==========================================
#      PART 1: GENERATE SUMMARY CSV
# ==========================================

def make_summary(raw_path, summary_path):
    print(f"--- PART 1: Generating Summary CSV ---")
    if not os.path.exists(raw_path): 
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    cols = ["agents", "goals", "cost", "success", "time_to_feasible", "conflicts"]
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["map"] = df["scenario_id"].apply(get_map_name) if "scenario_id" in df.columns else "unknown"

    summary_rows = []
    grouped = df.groupby(["map", "goals", "agents", "alg"])
    time_grid = np.linspace(0, MAX_RUNTIME, TIME_GRID_POINTS)

    for name, group in grouped:
        map_name, g, agents, alg = name
        row = {
            "map": map_name, "goals": g, "agents": agents, "alg": alg,
            "success_rate": group["success"].mean(),
            "soc_all_median": group["cost"].median(),
            "time_feas_median": group.loc[group["time_to_feasible"] > 0, "time_to_feasible"].median(),
            "conflicts_median": group.loc[group["success"] == 0, "conflicts"].median() if not group[group["success"]==0].empty else 0
        }
        
        if "best_time_history" in group.columns:
            curves = [get_step_values(parse_history(h), time_grid) for h in group["best_time_history"] if parse_history(h)]
            if curves:
                median_curve = np.nanmedian(np.array(curves), axis=0)
                row["convergence_curve"] = json.dumps([None if np.isnan(x) else x for x in median_curve.tolist()])
        
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Summary Saved: {summary_path}")

# ==========================================
#      PART 2: GLOBAL SUMMARY PLOT
# ==========================================

def generate_all_global_summaries(summary_path, outdir):
    print(f"--- PART 2: Generating Separate Global Summary Plots ---")
    df = pd.read_csv(summary_path)
    summary_data = []
    
    # Configuration-level grouping
    for (m, g, a), group in df.groupby(["map", "goals", "agents"]):
        base_row = group[group["alg"] == "Base"]
        if base_row.empty: continue
        base_soc = base_row.iloc[0]["soc_all_median"]
        
        for _, row in group.iterrows():
            # Improvement calculation
            improvement = ((base_soc - row["soc_all_median"]) / base_soc * 100) if base_soc > 0 else 0
            
            summary_data.append({
                "alg": row["alg"],
                "success": row["success_rate"],
                "improvement": improvement,
                # Crucial: Filter for conflicts only when the algorithm failed
                "conflicts": row["conflicts_median"] if row["success_rate"] < 1.0 else np.nan,
                "time": row["time_feas_median"] if row["success_rate"] > 0 else np.nan
            })

    summary_df = pd.DataFrame(summary_data)
    existing_algs = [a for a in ALG_ORDER if a in summary_df["alg"].unique()]
    
    # We use 'mean' for success/improvement, but for conflicts we use 
    # the mean of the medians from failed runs.
    final_stats = summary_df.groupby("alg").agg({
        "success": "mean",
        "improvement": "mean",
        "conflicts": "mean", # Mean of conflict counts in difficult scenarios
        "time": "mean"
    }).reindex(existing_algs)

    colors = [ALG_STYLE[a]['color'] for a in existing_algs]

    # --- Plot 1: Success Rate ---
    fig1, ax1 = plt.subplots()
    (final_stats["success"] * 100).plot(kind='bar', ax=ax1, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Overall Mean Success Rate (%)", fontweight='bold')
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(0, 105)
    plt.setp(ax1.get_xticklabels(), rotation=0)
    savefig(fig1, os.path.join(outdir, "0_global_success.png"))

    # --- Plot 2: SOC Improvement ---
    fig2, ax2 = plt.subplots()
    final_stats["improvement"].plot(kind='bar', ax=ax2, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title("Global SOC Improvement vs Base (%)", fontweight='bold')
    ax2.set_ylabel("Reduction in Path Cost (%)")
    ax2.axhline(0, color='black', linewidth=1)
    plt.setp(ax2.get_xticklabels(), rotation=0)
    savefig(fig2, os.path.join(outdir, "0_global_soc_improvement.png"))

    # --- Plot 3: Conflict Resilience (THE FIX) ---
    # This shows how many collisions remain when the algorithm can't solve it.
    fig3, ax3 = plt.subplots()
    final_stats["conflicts"].plot(kind='bar', ax=ax3, color=colors, edgecolor='black', alpha=0.8)
    ax3.set_title("Mean Residual Conflicts in Failed Runs", fontweight='bold')
    ax3.set_ylabel("Avg. Remaining Collisions (Lower is Better)")
    plt.setp(ax3.get_xticklabels(), rotation=0)
    savefig(fig3, os.path.join(outdir, "0_global_conflicts.png"))

    # --- Plot 4: Avg Time to Feasible ---
    fig4, ax4 = plt.subplots()
    final_stats["time"].plot(kind='bar', ax=ax4, color=colors, edgecolor='black', alpha=0.8)
    ax4.set_title("Mean Time to Find First Feasible Solution", fontweight='bold')
    ax4.set_ylabel("Time (seconds)")
    plt.setp(ax4.get_xticklabels(), rotation=0)
    savefig(fig4, os.path.join(outdir, "0_global_time.png"))

    print("\nGlobal Performance Summary Complete.")
    print(final_stats)

# ==========================================
#      PART 3: SCALABILITY & CONVERGENCE
# ==========================================

def plot_scalability_metrics(summary_path, outdir):
    print(f"--- PART 3: Generating Scalability & Convergence Plots ---")
    df = pd.read_csv(summary_path)
    time_grid = np.linspace(0, MAX_RUNTIME, TIME_GRID_POINTS)

    for (m, g), subset in df.groupby(["map", "goals"]):
        
        # 1. Success Rate (Z-Order handled)
        fig, ax = plt.subplots()
        for alg in ALG_ORDER:
            s = subset[subset["alg"] == alg].sort_values("agents")
            if not s.empty: ax.plot(s["agents"], s["success_rate"], **ALG_STYLE.get(alg, {}))
        ax.set_ylabel("Success Rate"); ax.set_xlabel("Agents"); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Success Rate: {m.capitalize()} (Goals: {int(g)})"); ax.legend(); savefig(fig, os.path.join(outdir, f"1_success_{m}_g{int(g)}.png"))

        # 2. Raw SOC (Log Scale)
        fig, ax = plt.subplots()
        for alg in ALG_ORDER:
            s = subset[subset["alg"] == alg].sort_values("agents")
            if not s.empty: ax.plot(s["agents"], s["soc_all_median"], **ALG_STYLE.get(alg, {}))
        ax.set_yscale("log"); ax.set_ylabel("Median SOC (Log)"); ax.set_xlabel("Agents")
        ax.set_title(f"SOC Scalability: {m.capitalize()}"); ax.legend(); savefig(fig, os.path.join(outdir, f"2_soc_raw_{m}_g{int(g)}.png"))

        # 3. Residual Conflicts
        fig, ax = plt.subplots()
        for alg in ALG_ORDER:
            s = subset[subset["alg"] == alg].sort_values("agents")
            if not s.empty: ax.plot(s["agents"], s["conflicts_median"], **ALG_STYLE.get(alg, {}))
        ax.set_ylabel("Median Residual Conflicts"); ax.set_xlabel("Agents")
        ax.set_title(f"Collision Analysis: {m.capitalize()}"); ax.legend(); savefig(fig, os.path.join(outdir, f"3_conflicts_{m}_g{int(g)}.png"))

    # Convergence
    conv_dir = os.path.join(outdir, "convergence"); ensure_dir(conv_dir)
    for (m, g, a), sub in df.groupby(["map", "goals", "agents"]):
        fig, ax = plt.subplots()
        plotted = False
        for alg in ALG_ORDER:
            row = sub[sub["alg"] == alg]
            if row.empty or pd.isna(row.iloc[0]["convergence_curve"]): continue
            curve = [np.nan if x is None else x for x in json.loads(row.iloc[0]["convergence_curve"])]
            if any(~np.isnan(curve)):
                ax.step(time_grid, curve, where='post', label=alg, color=ALG_STYLE[alg]["color"])
                plotted = True
        if plotted:
            ax.set_yscale("log"); ax.set_title(f"Convergence: {m.capitalize()} (A:{int(a)})")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("SOC (Log)"); ax.legend()
            savefig(fig, os.path.join(conv_dir, f"conv_{m}_a{int(a)}_g{int(g)}.png"))

# ==========================================
#                   MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="results.csv")
    parser.add_argument("--summary", type=str, default="results_summary.csv")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()
    
    ensure_dir(args.outdir)
    # make_summary(args.raw, args.summary)
    generate_all_global_summaries(args.summary, args.outdir)
    plot_scalability_metrics(args.summary, args.outdir)
    
    print(f"\nProcessing Complete. All plots saved in: {args.outdir}")