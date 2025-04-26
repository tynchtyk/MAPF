import random
import copy
import math
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
#from sklearn.cluster import KMeans
import networkx as nx

class P3:
    def __init__(
        self,
        graph,
        robots,
        generations: int = 400,
        seed: int | None = None,
    ) -> None:
        self.graph = graph
        self.robots = robots
        self.rids = [r.robot_id for r in robots]
        self.generations = generations
        self.rng = random.Random(seed)
        
        # Dynamic penalty initialization
        maxsum = sum(max(graph.shortest_path_length(r.start, t) for t in r.targets) for r in robots)
        self.PENALTY = 10 * maxsum
        self.conflict_history = []
        
        # Pyramid data
        self.pop_levels: List[List[Dict[int, List[int]]]] = []
        self.linkage_levels: List[List[List[int]]] = []
        
        # Stats
        self.best_fit = math.inf
        self.best_indiv: Dict[int, List[int]] | None = None
        self.best_hist: List[float] = []
        self.avg_hist: List[float] = []
        self.conf_hist: List[float] = []

    # ----------------------------- Core Functions -----------------------------
    def detect_conflicts(self, paths: Dict[int, List[int]]):
        conflicts = []
        horizon = max((len(p) for p in paths.values()), default=0)
        occ_v = [dict() for _ in range(horizon)]
        occ_e = [dict() for _ in range(horizon)]

        for rid, path in paths.items():
            for t, node in enumerate(path):
                if node in occ_v[t]:
                    conflicts.append(("v", t, node, rid, occ_v[t][node]))
                occ_v[t][node] = rid

                if t > 0:
                    e = (path[t-1], node)
                    swap = (node, path[t-1])
                    if swap in occ_e[t]:
                        conflicts.append(("e", t, e, rid, occ_e[t][swap]))
                    occ_e[t][e] = rid
        return conflicts

    def connectivity_repair(self, paths: Dict[int, List[int]]):
        cache = {}
        def sp(u, v):
            if (u, v) not in cache:
                cache[(u, v)] = self.graph.shortest_path(u, v)
            return cache[(u, v)]

        fixed = {}
        for r in self.robots:
            p = paths.get(r.robot_id, [r.start]) or [r.start]
            seq = [p[0]]

            for v in p[1:]:
                if self.graph.G.has_edge(seq[-1], v):
                    seq.append(v)
                else:
                    seq.extend(sp(seq[-1], v)[1:])

            for tgt in r.targets:
                if tgt not in seq:
                    seq.extend(sp(seq[-1], tgt)[1:])

            fixed[r.robot_id] = seq
        return fixed

    # ----------------------------- Improved Initialization -----------------------------
    def _solve_tsp(self, start, targets):
        if not targets:
            return [start]
        
        nodes = [start] + targets
        dist = [[self.graph.shortest_path_length(u, v) for v in nodes] for u in nodes]
        tour = nx.approximation.traveling_salesman_problem(nx.Graph(dist), cycle=False)
        return [nodes[i] for i in tour]

    def _seed_individual(self):
        # nearest‑neighbour chain for each robot
        paths = {}
        for r in self.robots:
            cur = r.start
            path = [cur]
            rem = r.targets[:]
            while rem:
                nxt = min(rem, key=lambda t: self.graph.shortest_path_length(cur, t))
                path.extend(self.graph.shortest_path(cur, nxt)[1:])
                cur = nxt
                rem.remove(nxt)
            paths[r.robot_id] = path
        
        # Conflict-aware refinement
        """for _ in range(3):
            conflicts = self.detect_conflicts(paths)
            if not conflicts:
                break
            for c in conflicts:
                rid1, rid2 = c[3], c[4]
                if len(paths[rid1]) > len(paths[rid2]):
                    self._reroute_robot(paths, rid1, c[1])
                else:
                    self._reroute_robot(paths, rid2, c[1])"""
        
        return self.connectivity_repair(paths)

    def _reroute_robot(self, paths, rid, t):
        if t == 0 or t >= len(paths[rid])-1:
            return
        alt_path = self.graph.shortest_path(paths[rid][t-1], paths[rid][t+1])
        if alt_path:
            paths[rid] = paths[rid][:t] + alt_path[1:-1] + paths[rid][t+2:]

    # ----------------------------- Enhanced Local Search -----------------------------
    def _fihc(self, indiv):
        fit = self._evaluate(indiv)
        improved = True
        while improved:
            improved = False
            
            # 2-opt swaps
            for rid in self.rids:
                path = indiv[rid]
                if len(path) < 4:
                    continue
                
                for i in range(1, len(path)-2):
                    for j in range(i+1, len(path)-1):
                        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                        cand = copy.deepcopy(indiv)
                        cand[rid] = new_path
                        cand = self.connectivity_repair(cand)
                        f = self._evaluate(cand)
                        if f < fit:
                            indiv, fit, improved = cand, f, True
                            break
                    if improved:
                        break
            
            # Target reordering
            if not improved:
                for rid in self.rids:
                    cand = self._reorder_targets(copy.deepcopy(indiv), rid)
                    f = self._evaluate(cand)
                    if f < fit:
                        indiv, fit, improved = cand, f, True
                        break
        
        return indiv, fit

    def _reorder_targets(self, indiv, rid):
        path = indiv[rid]
        targets = [v for v in path if v in next(r for r in self.robots if r.robot_id == rid).targets]
        if len(targets) <= 1:
            return indiv
        
        best_path = path
        best_fit = self._evaluate(indiv)
        
        for _ in range(3):
            new_order = targets.copy()
            self.rng.shuffle(new_order)
            new_path = [path[0]]
            for t in new_order:
                new_path.extend(self.graph.shortest_path(new_path[-1], t)[1:])
            
            cand = copy.deepcopy(indiv)
            cand[rid] = new_path
            cand = self.connectivity_repair(cand)
            f = self._evaluate(cand)
            if f < best_fit:
                best_path, best_fit = new_path, f
        
        indiv[rid] = best_path
        return indiv

    # ----------------------------- Linkage Learning -----------------------------
    def _build_dsm(self, pop):
        n = len(self.rids)
        counts = np.zeros((n, n), dtype=float)
        total = np.zeros((n,), dtype=float)
        idx = {rid: k for k, rid in enumerate(self.rids)}
        
        for ind in pop:
            involved = defaultdict(int)
            for c in self.detect_conflicts(ind):
                involved[c[3]] = 1
                if len(c) > 4:
                    involved[c[4]] = 1
            for a in involved:
                ia = idx[a]
                total[ia] += 1
                for b in involved:
                    ib = idx[b]
                    if ia != ib:
                        counts[ia, ib] += 1
        
        M = np.zeros_like(counts)
        for i in range(n):
            for j in range(n):
                if i == j or total[i] == 0:
                    continue
                M[i, j] = counts[i, j] / total[i]
        return M

    def _dsm_linkage_tree(self, M):
        clusters = [[rid] for rid in self.rids]
        sim = lambda A, B: max(M[self.rids.index(a), self.rids.index(b)] for a in A for b in B)
        subsets = []
        
        while len(clusters) > 1:
            best, pair = -1, None
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    s = sim(clusters[i], clusters[j])
                    if s > best:
                        best, pair = s, (i, j)
            i, j = pair
            merge = clusters[i] + clusters[j]
            subsets.append(merge)
            clusters = [clusters[k] for k in range(len(clusters)) if k not in pair] + [merge]
        
        subsets.sort(key=len)
        return subsets

    # ----------------------------- Optimal Mixing -----------------------------
    def _optimal_mixing(self, child, donor, linkage_sets):
        fit_child = self._evaluate(child)
        for subset in self.rng.sample(linkage_sets, len(linkage_sets)):
            trial = copy.deepcopy(child)
            for rid in subset:
                if len(donor[rid]) > 3 and len(child[rid]) > 3:
                    start = self.rng.randint(1, len(child[rid])-2)
                    end = self.rng.randint(start+1, len(child[rid])-1)
                    trial[rid] = child[rid][:start] + donor[rid][start:end] + child[rid][end:]
                else:
                    trial[rid] = donor[rid]
            
            trial = self.connectivity_repair(trial)
            f_trial = self._evaluate(trial)
            if f_trial < fit_child:
                child, fit_child = trial, f_trial
        
        return child, fit_child

    # ----------------------------- Dynamic Penalty -----------------------------
    def _evaluate(self, indiv):
        conflicts = self.detect_conflicts(indiv)
        num_conflicts = len(conflicts)
        
        # Track conflicts for adaptive penalty (limit history to 10 gens)
        self.conflict_history.append(num_conflicts)
        if len(self.conflict_history) > 10:
            self.conflict_history.pop(0)
        
        # Calculate flowtime (sum of path lengths)
        flowtime = sum(len(p) - 1 for p in indiv.values())
        
        # Safely calculate penalty (with upper bound)
        if num_conflicts > 0:
            current_penalty = min(self.PENALTY * num_conflicts, 1e6)  # Cap at 1 million
        else:
            current_penalty = 0
    
        return flowtime + current_penalty

    # ----------------------------- Main Loop -----------------------------
    def run(self, log_every: int = 1):
        for gen in range(self.generations):
            # (1) Create & optimize new individual
            indiv = self._seed_individual()
            indiv, fit = self._fihc(indiv)
            self._update_best(indiv, fit)

            # (2) Pyramid climb
            level = 0
            while True:
                if level >= len(self.pop_levels):
                    self.pop_levels.append([])
                    self.linkage_levels.append([])
                
                pop = self.pop_levels[level]
                pop.append(indiv)
                
                if not self.linkage_levels[level]:
                    M = self._build_dsm(pop)
                    self.linkage_levels[level] = self._dsm_linkage_tree(M)
                
                donor = self.rng.choice(pop)
                child, child_fit = self._optimal_mixing(indiv, donor, self.linkage_levels[level])
                self._update_best(child, child_fit)
                
                if child_fit < fit:
                    indiv, fit = child, child_fit
                    level += 1
                else:
                    break

            # (3) Restart mechanism
            if gen > 20 and len(set(self.best_hist[-20:])) == 1:
                self.pop_levels = [[]]
                self.linkage_levels = [[]]

            # Logging
            if gen % log_every == 0 or gen == self.generations - 1:
                all_inds = [i for lv in self.pop_levels for i in lv]
                avg_fit = sum(self._evaluate(i) for i in all_inds) / len(all_inds) if all_inds else 0
                avg_conf = sum(len(self.detect_conflicts(i)) for i in all_inds) / len(all_inds) if all_inds else 0
                
                self.best_hist.append(self.best_fit)
                self.avg_hist.append(avg_fit)
                self.conf_hist.append(avg_conf)
                
                print(
                    f"Gen {gen:4d} "
                    f"Best {self.best_fit:.1f} | Avg {avg_fit:.1f} | "
                    f"Conflicts {avg_conf:.2f}"
                )
        
        return self.best_indiv, self.best_fit

    def _update_best(self, indiv, fit):
        if fit < self.best_fit:
            self.best_fit = fit
            self.best_indiv = indiv