import glob
import os
import pathlib
import random
import statistics
import multiprocessing as mp
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# --- Algorithm Imports ---
from algorithms.p3_dsm_hybrid import P3_DSM_HYBRID
from algorithms.p3_dsm_robot_conflicts import P3_DSM_ROBOT_CONFLICTS
from algorithms.p3_base import P3_Base

# --- Utility Imports ---
from utils.utils import load_map_and_robots

# --- Constants ---
# Used ONLY if the algorithm code CRASHES (Exception raised)
CRASH_PENALTY = 1_000_000_000.0 

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------- Single-run execution (child process) --------------------

def _run_task_payload(job: Tuple[Any, str, str, int, float, int, int]) -> Dict[str, Any]:
    """
    Runs ONE (scenario, algorithm, seed) and returns a standardized dict.
    """
    AlgClass, alg_name, scenario_file, seed, max_seconds, local_steps, patience = job

    # deterministic randomness per run
    random.seed(seed)
    np.random.seed(seed)

    graph, robots = load_map_and_robots(scenario_file)

    ea = AlgClass(
        graph,
        robots,
        local_steps=local_steps,
        max_seconds=max_seconds,
        patience=patience,
    )

    t0 = time.perf_counter()
    try:
        # 
        # The algorithm is responsible for stopping itself via Soft Timeout.
        best, stats = ea.run() 
    except Exception as e:
        # If we reach here, the algorithm code CRASHED (bug, memory, etc.)
        # We MUST return something to keep the experiment running.
        print(f"DEBUG: CRASH in {alg_name} {os.path.basename(scenario_file)}: {e}")
        dt = time.perf_counter() - t0
        return {
            "scenario": os.path.basename(scenario_file),
            "scenario_id": os.path.splitext(os.path.basename(scenario_file))[0],
            "agents": len(robots),
            "goals": float(statistics.mean(len(r.targets) for r in robots)) if robots else 0.0,
            "alg": alg_name,
            "seed": seed,
            "cost": CRASH_PENALTY,  # Only used on CRASH
            "conflicts": -1,
            "success": 0,
            "runtime": dt,
            "external_runtime": dt,
        }

    dt = time.perf_counter() - t0

    # --- Derived metrics ---
    if best:
        distance = sum(max(0, len(p) - 1) for p in best.values())
        makespan = max((max(0, len(p) - 1) for p in best.values()), default=0)
    else:
        distance = 0
        makespan = 0

    # Retrieve cost directly. 
    # If the algorithm class was fixed correctly, this will NEVER be inf.
    # If it IS inf, we leave it as inf (so you can see the bug in the algo).
    final_cost = stats.get("cost", float('inf'))
    
    # Fallback: If algo returned inf/missing, try to calc fitness now
    if (final_cost == float('inf') or np.isinf(final_cost)) and best:
        try:
            final_cost = ea.fitness(best, modified_rids=None)
        except:
            pass # Keep it as inf/original value if calc fails

    # Ensure conflicts/success exist
    if "conflicts" not in stats:
        try:
            stats["conflicts"] = len(ea._detect_conflicts(best)) if best else -1
        except Exception:
            stats["conflicts"] = -1

    if "success" not in stats:
        stats["success"] = 1 if stats.get("conflicts", -1) == 0 else 0

    out = {
        "scenario": os.path.basename(scenario_file),
        "scenario_id": os.path.splitext(os.path.basename(scenario_file))[0],
        "agents": len(robots),
        "goals": float(statistics.mean(len(r.targets) for r in robots)) if robots else 0.0,
        "alg": alg_name,
        "seed": seed,
        "distance": distance,
        "makespan": makespan,
        "cost": final_cost, 
        "external_runtime": dt,
    }
    
    for k, v in stats.items():
        if k != "cost":
            out[k] = v
    
    return out


def _worker_entry(result_queue: "mp.Queue", job: Tuple[Any, str, str, int, float, int, int]) -> None:
    """
    Always puts exactly one result into the queue.
    """
    try:
        res = _run_task_payload(job)
        result_queue.put(res)
    except Exception as e:
        AlgClass, alg_name, scenario_file, seed, _, _, _ = job
        print(f"CRITICAL WORKER CRASH: {e}")
        res = {
            "scenario": os.path.basename(scenario_file),
            "alg": alg_name,
            "seed": seed,
            "cost": CRASH_PENALTY,
            "success": 0
        }
        result_queue.put(res)


def run_one_safe(
    ctx: mp.context.BaseContext,
    job: Tuple[Any, str, str, int, float, int, int],
) -> Dict[str, Any]:
    """
    Runs a single job in its own child process.
    WAITS INDEFINITELY for the child to finish (No Hard Kill).
    """
    q = ctx.Queue()
    p = ctx.Process(target=_worker_entry, args=(q, job))
    p.start()
    
    # Wait indefinitely for the process to finish
    p.join() 

    if not q.empty():
        return q.get()

    AlgClass, alg_name, scenario_file, seed, max_seconds, local_steps, patience = job
    return {
        "scenario": os.path.basename(scenario_file),
        "scenario_id": os.path.splitext(os.path.basename(scenario_file))[0],
        "alg": alg_name,
        "seed": seed,
        "cost": CRASH_PENALTY,
        "conflicts": -1,
        "success": 0,
        "runtime": -1.0,
        "external_runtime": -1.0,
    }


# -------------------- Global Worker Function --------------------

def _pool_worker(ctx, job_queue, out_queue):
    """
    Consumes jobs from job_queue, runs them safely, puts results in out_queue.
    """
    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            res = run_one_safe(ctx, job)
        except Exception as e:
            if job and len(job) >= 4:
                _, alg_name, scenario_file, seed, _, _, _ = job
                scen_name = os.path.basename(scenario_file)
            else:
                scen_name, alg_name, seed = "Unknown", "Unknown", -1

            print(f"RUNNER EXCEPTION for {scen_name}: {e}")
            res = {
                "scenario": scen_name,
                "alg": alg_name,
                "seed": seed,
                "cost": CRASH_PENALTY,
                "success": 0
            }
        out_queue.put(res)


# -------------------- Experiment runner --------------------

def run_experiment(config: dict) -> pd.DataFrame:
    ctx = mp.get_context("spawn")

    scenario_folder = config["scenario_folder"]
    scenario_files = sorted(glob.glob(os.path.join(scenario_folder, "*.*")))
    scenario_files = [
        f for f in scenario_files
        if pathlib.Path(f).suffix.lower() in {".yaml", ".yml", ".json", ".txt"}
    ]
    if not scenario_files:
        raise RuntimeError(f"No scenario files found in {scenario_folder}")

    seed_count = int(config["simulation_params"]["seed"])
    seeds = list(range(seed_count))

    max_seconds = float(config["simulation_params"].get("max_seconds", 180.0))
    local_steps = int(config["simulation_params"].get("local_steps", 5))
    patience = int(config["simulation_params"].get("patience", 1000))

    algs = [
        ("Base", P3_Base),
        ("RGCM", P3_DSM_ROBOT_CONFLICTS),
        ("CHHM", P3_DSM_HYBRID),
    ]

    # Build jobs
    jobs: List[Tuple[Any, str, str, int, float, int, int]] = []
    for scen in scenario_files:
        for alg_name, AlgClass in algs:
            for seed in seeds:
                jobs.append((AlgClass, alg_name, scen, seed, max_seconds, local_steps, patience))

    total = len(jobs)
    print(f"Running {total} tasks (max_seconds={max_seconds:.0f}s)...")
    print("Hard Timeouts: DISABLED (Waiting for completion)")

    max_workers = int(config["simulation_params"].get("parallel_workers", 8))
    print(f"Parallel workers: {max_workers}")

    job_queue = ctx.Queue()
    out_queue = ctx.Queue()

    for job in jobs:
        job_queue.put(job)
    for _ in range(max_workers):
        job_queue.put(None)

    # Start workers
    workers = [ctx.Process(target=_pool_worker, args=(ctx, job_queue, out_queue)) for _ in range(max_workers)]
    for w in workers:
        w.start()

    results: List[Dict[str, Any]] = []
    completed = 0

    while completed < total:
        try:
            res = out_queue.get()
            results.append(res)
            completed += 1

            scen = res.get("scenario", "?")
            alg = res.get("alg", "?")
            seed = res.get("seed", -1)
            cost = res.get("cost", float('inf'))
            succ = res.get("success", 0)
            dt = res.get("external_runtime", 0.0)
            status = "✔" if succ else "✘"
            
            # Display logic
            if cost == CRASH_PENALTY:
                cost_str = "CRASH"
            elif cost == float('inf'):
                cost_str = "inf"
            else:
                cost_str = f"{cost:.1f}"

            print(f"[{completed:>4}/{total}] {status} {scen[:28]:28s} | {alg:5s} seed={seed:<2} cost={cost_str} t={dt:.2f}s")

        except Exception as e:
            print(f"Error retrieving result from queue: {e}")

    for w in workers:
        w.join()

    # --- Clean up DataFrame before returning ---
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df.drop(columns=["error"])
        
    return df


def main():
    config = load_config()
    df = run_experiment(config)
    out_file = config["simulation_params"].get("result_file", "results.csv")
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()