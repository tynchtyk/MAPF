import random
import copy
import math
import numpy as np
from typing import Dict, List, Any
from algorithms.ga import GA

Individual = Dict[int, List[Any]]

class P3_DSM_ROBOT_CONFLICTS(GA):
    """
    Baseline B: RGCM (Robot Grouping based on Conflict Matrix).
    Updated to track Best SOC (Any) history.
    """

    def __init__(self, graph, robots, local_steps: int = 5, max_seconds: float = 3.0, patience: int = 1000):
        super().__init__(graph, robots, generations=0, local_steps=local_steps)
        self.max_seconds = float(max_seconds)
        self.patience = int(patience)
        self.pyramid: List[List[Individual]] = []
        self.rgcm_improvements = 0

    def run(self):
        self._set_deadline(self.max_seconds)
        
        time_to_feasible = None
        time_to_best_feasible = -1.0
        best_time_history = [] 
        
        self.best_individual = None
        self.best_cost = float("inf")
        best_feasible_individual = None
        best_feasible_cost = float("inf")
        
        no_improve_counter = 0

        try:
            if self._time_is_up(): 
                return self._build_stats(None, 0, -1, -1, [])

            indiv = self.generate_individual()
            
            start_cost = self.fitness(indiv, modified_rids=None)
            self.best_individual = copy.deepcopy(indiv)
            self.best_cost = start_cost
            best_time_history.append((0.0, self.best_cost))

            indiv = self.first_improvement_hill_climber(indiv, self.local_steps)
            
            indiv_cost = self.fitness(indiv, modified_rids=None)
            
            if indiv_cost < self.best_cost:
                self.best_cost = indiv_cost
                self.best_individual = copy.deepcopy(indiv)
                best_time_history.append((self._elapsed(), self.best_cost))

            conflicts = self._detect_conflicts(indiv)
            if not conflicts:
                best_feasible_individual = copy.deepcopy(indiv)
                best_feasible_cost = indiv_cost
                time_to_feasible = 0.0
                time_to_best_feasible = 0.0

            while not self._time_is_up():
                indiv = self.generate_individual()
                indiv = self.first_improvement_hill_climber(indiv, self.local_steps)
                if self._time_is_up(): break

                indiv_cost = self.fitness(indiv, modified_rids=None)

                # RGCM Mixing
                for level_pop in self.pyramid:
                    if self._time_is_up(): break
                    if not level_pop: continue
                    partner = random.choice(level_pop)
                    child = self._optimal_mix(indiv, partner, level_pop)
                    if self._time_is_up(): break
                    
                    ls_steps = max(1, self.local_steps // 4)
                    child = self.first_improvement_hill_climber(child, ls_steps)
                    if self._time_is_up(): break
                    
                    child_cost = self.fitness(child, modified_rids=None)
                    if child_cost < indiv_cost:
                        indiv = child
                        indiv_cost = child_cost

                if self._time_is_up(): break

                max_level_sz = 2 ** len(self.pyramid) if self.pyramid else 1
                if not self.pyramid or len(self.pyramid[-1]) >= max_level_sz:
                    self.pyramid.append([])
                self.pyramid[-1].append(copy.deepcopy(indiv))

                conflicts = self._detect_conflicts(indiv)
                t = self._elapsed()

                if indiv_cost < self.best_cost:
                    self.best_cost = indiv_cost
                    self.best_individual = copy.deepcopy(indiv)
                    best_time_history.append((t, self.best_cost)) 
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                if not conflicts:
                    if time_to_feasible is None:
                        time_to_feasible = t
                    if indiv_cost < best_feasible_cost:
                        best_feasible_cost = indiv_cost
                        best_feasible_individual = copy.deepcopy(indiv)
                        time_to_best_feasible = t

                if no_improve_counter >= self.patience:
                    break
        
        except Exception as e:
            print(f"DEBUG: Internal Exception in P3_DSM_ROBOT_CONFLICTS: {e}")

        return self._build_stats(best_feasible_individual, best_feasible_cost, time_to_feasible, time_to_best_feasible, best_time_history)

    def _build_stats(self, best_feasible_indiv, best_feasible_cost, time_to_feasible, time_to_best_feasible, best_time_history):
        """
        Only returns metrics required for:
        1. Success Rate
        2. SOC (Distance)
        3. Conflict Count
        4. Time to Solution
        5. Convergence Plots
        """
        final_best = best_feasible_indiv if best_feasible_indiv is not None else self.best_individual
        final_conflicts = -1
        
        # If we have a solution (feasible or not), calculate conflicts
        if final_best:
            final_conflicts = len(self._detect_conflicts(final_best))
        
        success = 1 if (best_feasible_indiv is not None) else 0

        # "distance" in the CSV usually maps to 'cost' here if feasible.
        # If infeasible, self.best_cost includes penalties, but that's fine for the 'cost' column.
        # We return the best COST found (which is SOC if feasible).
        reported_cost = best_feasible_cost if best_feasible_indiv else self.best_cost

        stats = {
            "cost": reported_cost,         # Kept for compatibility
            "conflicts": final_conflicts,  # Required for Conflict plots
            "success": success,            # Required for Success plots
            "time_to_feasible": time_to_feasible if time_to_feasible is not None else -1.0, # Required for Time plots
            "time_to_best_feasible": time_to_best_feasible if best_feasible_indiv else -1.0, # Required for Time plots
            "best_time_history": best_time_history, # Required for Convergence plots
        }
        
        safe_return = self.repair_individual(final_best, conflict_repair=False) if final_best else self.generate_individual()
        return safe_return, stats

    def _optimal_mix(self, receiver, donor, population):
        if not population or self._time_is_up(): return receiver
        dsm = self._build_dsm(population)
        if dsm is None or self._time_is_up(): return receiver
        linkage = self._linkage_tree(dsm)
        if self._time_is_up(): return receiver
        current = copy.deepcopy(receiver)
        f_cur = self.fitness(current, modified_rids=None)
        
        for cluster in linkage:
            if self._time_is_up(): break
            trial = copy.deepcopy(current)
            for rid_idx in cluster:
                # Safe bounds check
                if rid_idx < len(self.robots):
                    rid = self.robots[rid_idx].robot_id
                    trial[rid] = copy.deepcopy(donor[rid])
            
            if self._time_is_up(): break
            trial = self.repair_individual(trial, conflict_repair=True)
            f_trial = self.fitness(trial, modified_rids=None)
            
            if f_trial < f_cur:
                current, f_cur = trial, f_trial
                
        return current

    def _build_dsm(self, population):
        R = len(self.robots); m = len(population)
        if m == 0 or R <= 1: return None
        
        id_to_index = {robot.robot_id: idx for idx, robot in enumerate(self.robots)}
        pair_hits = np.zeros((R, R), dtype=int)
        
        # Optimize: Sample population if too large
        sample_size = min(len(population), 20)
        sampled_pop = random.sample(population, sample_size)

        for indiv in sampled_pop:
            if self._time_is_up(): return None
            seen = set()
            for _, _, _, r1, r2 in self._detect_conflicts(indiv):
                i = id_to_index.get(r1, r1); j = id_to_index.get(r2, r2)
                if i >= R or j >= R: continue
                i, j = sorted((i, j))
                if (i, j) not in seen:
                    pair_hits[i, j] += 1
                    seen.add((i, j))
        
        eps = 1e-12; H_row = np.zeros(R); dsm = np.ones((R, R), dtype=float)
        for i in range(R):
            for j in range(i + 1, R):
                p1 = pair_hits[i, j] / sample_size
                if p1 < eps or p1 > 1 - eps: H = 0.0
                else: H = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1)
                H_row[i] = max(H_row[i], H); H_row[j] = max(H_row[j], H)
                dsm[i, j] = dsm[j, i] = 1.0 - H
                
        for i in range(R):
            for j in range(i + 1, R):
                hmax = max(H_row[i], H_row[j])
                dsm[i, j] = dsm[j, i] = 1.0 if hmax < eps else dsm[i, j] / hmax
        return dsm

    def _linkage_tree(self, dsm):
        # Using Inverted Distance (1 - Dependency)
        num_vars = len(dsm)
        if num_vars <= 1: return [[0]]
        
        current_dist = np.full((num_vars, num_vars), math.inf)
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                current_dist[i, j] = current_dist[j, i] = 1.0 - dsm[i, j]
        np.fill_diagonal(current_dist, math.inf)
        
        active = list(range(num_vars))
        node_list = [[i] for i in range(num_vars)]
        children = {}; next_idx = num_vars
        
        for _ in range(num_vars - 1):
            if self._time_is_up(): break
            if len(active) < 2: break
            
            i_mat, j_mat = np.unravel_index(np.argmin(current_dist), current_dist.shape)
            if current_dist[i_mat, j_mat] == math.inf: break
            
            i_node, j_node = active[i_mat], active[j_mat]
            merged = node_list[i_node] + node_list[j_node]
            node_list.append(merged); children[next_idx] = (i_node, j_node)
            
            m = current_dist.shape[0]; new_dist = np.full((m - 1, m - 1), math.inf)
            keep = [k for k in range(m) if k not in (i_mat, j_mat)]
            
            for r_new, r_old in enumerate(keep):
                for c_new, c_old in enumerate(keep[r_new + 1 :], start=r_new + 1):
                    new_dist[r_new, c_new] = new_dist[c_new, r_new] = current_dist[r_old, c_old]
                    
            for idx, k in enumerate(keep):
                new_dist[idx, -1] = new_dist[-1, idx] = min(current_dist[i_mat, k], current_dist[j_mat, k])
                
            current_dist = new_dist; active = [active[k] for k in keep] + [next_idx]; next_idx += 1
            
        flat = []
        def dfs(idx):
            if idx in children: dfs(children[idx][0]); dfs(children[idx][1])
            flat.append(node_list[idx])
        if next_idx - 1 < len(node_list): dfs(next_idx - 1)
        else:
             for n in node_list: flat.append(n)
        return flat