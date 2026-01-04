import random
import copy
from typing import Dict, List, Any
from algorithms.ga import GA

Individual = Dict[int, List[Any]]

class P3_Base(GA):
    """
    Baseline A: P3 with Simple Splice Crossover.
    Updated to track Best SOC (Any) history for convergence analysis.
    """

    def __init__(self, graph, robots, local_steps: int = 25, max_seconds: float = 3.0, patience: int = 1000):
        super().__init__(graph, robots, generations=0, local_steps=local_steps)
        self.max_seconds = float(max_seconds)
        self.patience = int(patience)
        self.pyramid: List[List[Individual]] = []

    def crossover(self, pa: Individual, pb: Individual) -> Individual:
        if self._time_is_up(): return copy.deepcopy(pa)
        child: Individual = {}
        rids = list(self.robot_map.keys())
        for rid in rids:
            if self._time_is_up():
                if rid not in child: child[rid] = copy.deepcopy(pa.get(rid, []))
                continue
            a = pa.get(rid, []); b = pb.get(rid, [])
            if not a or not b:
                child[rid] = copy.deepcopy(a if len(a) >= len(b) else b)
                continue
            common = list(set(a[:-1]) & set(b[:-1]))
            if common:
                splice = random.choice(common)
                try: ia = a.index(splice); ib = b.index(splice)
                except ValueError: ia = len(a) // 2; ib = len(b) // 2
            else: ia = len(a) // 2; ib = len(b) // 2
            child[rid] = a[: ia + 1] + b[ib + 1 :]
        
        if not self._time_is_up():
            return self.repair_individual(child, conflict_repair=True)
        return child

    def run(self):
        self._set_deadline(self.max_seconds)

        # Metrics
        time_to_feasible = None
        time_to_best_feasible = -1.0
        best_time_history = []  
        
        # We don't need 'total_evals' for the plots, removed it.
        
        self.best_individual = None
        self.best_cost = float("inf")
        best_feasible_individual = None
        best_feasible_cost = float("inf")
        no_improve_counter = 0

        try:
            # === 1. INITIALIZATION ===
            if self._time_is_up(): 
                return self._build_stats(None, 0, -1, -1, [])

            indiv = self.generate_individual()
            
            # Initial evaluation
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

            # === 2. MAIN LOOP ===
            while not self._time_is_up():
                indiv = self.generate_individual()
                indiv = self.first_improvement_hill_climber(indiv, self.local_steps)
                if self._time_is_up(): break
                
                indiv_cost = self.fitness(indiv, modified_rids=None)

                # Pyramid Mixing
                for level_pop in self.pyramid:
                    if self._time_is_up(): break
                    if not level_pop: continue
                    partner = random.choice(level_pop)
                    child = self.crossover(indiv, partner)
                    if self._time_is_up(): break
                    
                    ls_steps = max(1, self.local_steps // 4)
                    child = self.first_improvement_hill_climber(child, ls_steps)
                    if self._time_is_up(): break
                    
                    child_cost = self.fitness(child, modified_rids=None)
                    if child_cost < indiv_cost:
                        indiv = child
                        indiv_cost = child_cost

                if self._time_is_up(): break
                
                # Add to Pyramid
                max_level_sz = 2 ** len(self.pyramid) if self.pyramid else 1
                if not self.pyramid or len(self.pyramid[-1]) >= max_level_sz:
                    self.pyramid.append([])
                self.pyramid[-1].append(copy.deepcopy(indiv))

                # === UPDATED TRACKING ===
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
            print(f"DEBUG: Internal Exception in P3_Base: {e}")

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