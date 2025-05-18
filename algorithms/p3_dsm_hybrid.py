import random
import copy
import math
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import networkx as nx
from algorithms.ga import GA, TimeLogger, log_time

# --- Type Hint Placeholders ---
GraphType = Any
RobotType = Any
NodeId = Any
Path = List[NodeId]
Individual = Dict[int, Path]

class P3_DSM_HYBRID(GA):
    """Parameter‑less Population Pyramid (P3) with a HYBRID dependency‑aware
    recombination operator.  
    Stage‑A copies entire paths for conflict‑coupled robots (robot–level DSM).
    Stage‑B immediately splices *only the conflict‑timesteps* inside those
    robots, borrowing segments from the donor.  All decisions are derived from
    the conflict list `_detect_conflicts`; no numerical hyper‑parameters are
    exposed to the user.
    """

    # ------------------------------------------------------------------
    def __init__(self, graph: GraphType, robots: List[RobotType],
                 generations: int = 200, local_steps: int = 5) -> None:
        super().__init__(graph, robots, generations, local_steps)
        self.pyramid: List[List[Individual]] = []
        self.LT_PAD: Optional[NodeId] = None  # sentinel when padding paths

    # ====================== MAIN EVOLUTIONARY LOOP ====================
    def run(self) -> Tuple[Optional[Individual], List[float], List[float], List[float]]:
        for gen in range(self.generations):
            indiv = self.generate_individual()
            indiv = self.first_improvement_hill_climber(indiv, self.local_steps)

            for level_pop in self.pyramid:
                if not level_pop:
                    continue
                partner = random.choice(level_pop)
                child = self._optimal_mix(indiv, partner, level_pop)  # <- HYBRID OPERATOR
                child = self.first_improvement_hill_climber(child, max(1, self.local_steps // 4))
                if self.fitness(child) < self.fitness(indiv):
                    indiv = child

            # -------- pyramid bookkeeping ----------
            max_level_sz = 2 ** len(self.pyramid) if self.pyramid else 1
            if not self.pyramid or len(self.pyramid[-1]) >= max_level_sz:
                self.pyramid.append([])
            self.pyramid[-1].append(indiv)

            f = self.fitness(indiv)
            if f < self.best_cost:
                self.best_cost, self.best_individual = f, copy.deepcopy(indiv)

            # statistics
            all_ind = [i for lvl in self.pyramid for i in lvl]
            if all_ind:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    c_arr = np.array([self.fitness(i) for i in all_ind])
                    conf_arr = np.array([len(self._detect_conflicts(i)) for i in all_ind])
                    self.avg_history.append(float(np.nanmean(c_arr)))
                    self.conf_history.append(float(np.nanmean(conf_arr)))
            else:
                self.avg_history.append(self.best_cost if self.best_cost != math.inf else math.nan)
                self.conf_history.append(0.0)
            self.best_history.append(self.best_cost)

            print(f"Gen {gen:3d} | Best {self.best_cost:.1f} | Avg {self.avg_history[-1]:.1f} | Conf {self.conf_history[-1]:.2f}")
        

        return self.best_individual, self.best_history, self.avg_history, self.conf_history

    # ================= HYBRID RECOMBINATION OPERATOR ==================
    @log_time
    def _optimal_mix(self, receiver: Individual, donor: Individual,
                     population: List[Individual]) -> Individual:
        """Return a child created by two‑stage optimal mixing.

        * Stage‑A: copy full paths of robots in each linkage cluster from donor
          to child, evaluate; keep if better.
        * Stage‑B: regardless of Stage‑A outcome, iterate over conflict hot
          windows and try the *opposite* parent slice (reintroduce receiver
          slice if donor path was accepted; insert donor slice if not). Accept
          window if it improves fitness, recomputing hotspots after each
          improvement to avoid stale windows and overlap issues.
        """
        if not population:
            return receiver

        linkage = self._linkage_tree(self._build_robot_dsm(population))
        current = copy.deepcopy(receiver)
        f_cur   = self.fitness(current)

        for cluster in linkage:  # leaf → root
            cluster_rids = [self.robots[idx].robot_id for idx in cluster]

            # backup receiver and donor slices for cluster robots
            recv_backup = {rid: current[rid][:] for rid in cluster_rids}
            donor_paths = {rid: donor[rid][:]   for rid in cluster_rids}

            # ---------------- Stage‑A : whole‑path donor copy ------------
            trialA = copy.deepcopy(current)
            for rid in cluster_rids:
                trialA[rid] = donor_paths[rid][:]
            fA = self.fitness(trialA)
            if fA < f_cur:
                current, f_cur = trialA, fA     # accept donor macro
                opposite_source = recv_backup   # Stage‑B will test revert
            else:
                opposite_source = donor_paths   # donor macro rejected → Stage‑B tests donor windows only

            # -------------- Stage‑B : window‑level mixing ---------------
            improved = True
            while improved:
                improved = False
                # hotspots recomputed each outer pass to stay fresh
                cluster_confs = [c for c in self._detect_conflicts(current)
                                  if c[3] in cluster_rids or c[4] in cluster_rids]
                hot = self._hot_positions(cluster_confs)

                for rid in cluster_rids:
                    windows = self._windows(current[rid], hot.get(rid, set()))
                    for a, b in windows:  # small → big
                        # guard unequal lengths
                        if b >= len(opposite_source[rid]):
                            continue
                        if current[rid][a:b+1] == opposite_source[rid][a:b+1]:
                            continue
                        cand = copy.deepcopy(current)
                        cand[rid][a:b+1] = opposite_source[rid][a:b+1]
                        cand = self.repair_individual(cand, conflict_repair=False)
                        f_cand = self.fitness(cand)
                        if f_cand < f_cur:
                            current, f_cur = cand, f_cand
                            improved = True  # restart with fresh hotspots
                            break  # break inner window loop
                    if improved:
                        break  # break rid loop to recompute hotspots

        return current


    # ---------------- helper: build robot‑level DSM -------------------
    def _build_robot_dsm(self, population: List[Individual]) -> np.ndarray:
        R = len(self.robots)
        m = len(population)
        dsm = np.ones((R, R), dtype=float)
        if m == 0 or R <= 1:
            return dsm

        pair_hits = np.zeros((R, R), dtype=int)
        for indiv in population:
            seen = set()
            for _, _, _, r1, r2 in self._detect_conflicts(indiv):
                i, j = sorted((r1, r2))
                if (i, j) not in seen:
                    pair_hits[i, j] += 1
                    seen.add((i, j))

        eps = 1e-12
        H_row = np.zeros(R)
        for i in range(R):
            for j in range(i + 1, R):
                p1 = pair_hits[i, j] / m
                if p1 < eps or p1 > 1 - eps:
                    H = 0.0
                else:
                    H = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1)
                H_row[i] = max(H_row[i], H)
                H_row[j] = max(H_row[j], H)
                dsm[i, j] = dsm[j, i] = 1.0 - H
        for i in range(R):
            for j in range(i + 1, R):
                hmax = max(H_row[i], H_row[j])
                dsm[i, j] = dsm[j, i] = 1.0 if hmax < eps else dsm[i, j] / hmax
        return dsm

    # --------------- helper: linkage tree (unchanged) -----------------
    def _linkage_tree(self, dsm: np.ndarray) -> List[List[int]]:
        num_vars = len(dsm)
        if num_vars == 0:
            return []
        if num_vars == 1:
            return [[0]]

        current = dsm.copy().astype(float)
        np.fill_diagonal(current, math.inf)
        active = list(range(num_vars))
        node_list: List[Any] = [[i] for i in range(num_vars)]
        children: Dict[int, Tuple[int, int]] = {}
        next_idx = num_vars

        for _ in range(num_vars - 1):
            if len(active) < 2:
                break
            i_mat, j_mat = np.unravel_index(np.argmin(current), current.shape)
            if current[i_mat, j_mat] == math.inf:
                break
            i_node, j_node = active[i_mat], active[j_mat]
            merged = node_list[i_node] + node_list[j_node]
            node_list.append(merged)
            children[next_idx] = (i_node, j_node)

            # update distance matrix (single linkage)
            m = current.shape[0]
            new = np.full((m - 1, m - 1), math.inf)
            new_dists = []
            keep = [k for k in range(m) if k not in (i_mat, j_mat)]
            for k in keep:
                new_dists.append(min(current[i_mat, k], current[j_mat, k]))
            for r, old_r in enumerate(keep):
                for c, old_c in enumerate(keep[r + 1:], start=r + 1):
                    new[r, c] = new[c, r] = current[old_r, old_c]
            if new_dists:
                new[:-1, -1] = new_dists
                new[-1, :-1] = new_dists
            np.fill_diagonal(new, math.inf)
            current = new

            active = [active[k] for k in keep] + [next_idx]
            next_idx += 1

        flat = []
        root = next_idx - 1
        visited = set()

        def dfs(idx):
            if idx in visited:
                return
            visited.add(idx)
            if idx in children:
                dfs(children[idx][0])
                dfs(children[idx][1])
            flat.append(node_list[idx])

        dfs(root)
        return flat

    # ======================= STAGE‑B HELPERS ==========================
    def _hot_positions(self, conflicts):
        H = defaultdict(set)
        for _, t, _, r1, r2 in conflicts:
            H[r1].add(t)
            H[r2].add(t)
        return H  # robot -> set(timestep)

    def _windows(self, path: Path, hot_set: set) -> List[Tuple[int, int]]:
        if not hot_set:
            return []
        marks = sorted(hot_set)
        spans = {(max(0, t - 1), min(len(path) - 1, t + 1)) for t in marks}
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        # smallest window first
        merged.sort(key=lambda ab: (ab[1] - ab[0] + 1))
        return merged

    def _mix_robot(self, sol: Individual, donor: Individual, rid: int, windows: List[Tuple[int, int]]) -> Individual:
        if not windows:
            return sol
        best = sol
        f_best = self.fitness(best)
        recv_path = sol[rid]
        donor_path = donor[rid]
        for a, b in windows:
            candidate_path = recv_path[:a] + donor_path[a:b + 1] + recv_path[b + 1:]
            cand = copy.deepcopy(best)
            cand[rid] = candidate_path
            cand = self.repair_individual(cand, conflict_repair=True)
            f_cand = self.fitness(cand)
            if f_cand < f_best:
                best, f_best, recv_path = cand, f_cand, candidate_path
        return best
