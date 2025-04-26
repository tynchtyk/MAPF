import random
import copy
import math
from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np


###############################################################
# Parameterless Population Pyramid EA                          #
###############################################################

class P3_2:
    """PPP evolutionary algorithm for multi‑robot path‑finding.

    * Stops after a fixed **generations** budget (like the DEAP version).
    * Maintains per‑generation statistics (**avg, min, max**) of population
      cost values, stored in a `logbook` list of dicts to mimic DEAP’s output.
    """

    ###################################################################
    # Construction                                                     #
    ###################################################################

    def __init__(
        self,
        graph,
        robots,
        generations: int = 200,
        local_steps: int = 25,
    ) -> None:
        self.graph = graph
        self.robots = robots
        self.robot_map = {r.robot_id: r for r in robots}
        self.generations = generations
        self.local_steps = local_steps

        # Conflict‑penalty magnitude
        self.CONFLICT_PENALTY_BASE = self._calculate_penalty_base()

        # Pyramid holds all individuals generated so far
        self.pyramid: List[List[Dict[int, List[int]]]] = []

        # Elitism
        self.best_individual: Dict[int, List[int]] | None = None
        self.best_cost: float = math.inf

        # Statistics histories
        self.avg_history: List[float] = []
        self.best_history: List[float] = []
        self.conf_history: List[float] = []

    ###################################################################
    # Public API                                                       #
    ###################################################################

    def run(self):
        for gen in range(self.generations):
            # 1. create new solution & hill‑climb
            indiv = self.generate_individual()
            indiv = self.local_search(indiv, self.local_steps)

            # 2. climb the pyramid via improvement‑only crossovers
            for population in self.pyramid:
                partner = random.choice(population)
                child = self.crossover(indiv, partner)
                child = self.local_search(child, self.local_steps // 4)
                if self.fitness(child) < self.fitness(indiv):
                    indiv = child

            # 3. insert individual, create new level if necessary
            if not self.pyramid or len(self.pyramid[-1]) >= 2 ** len(self.pyramid):
                self.pyramid.append([])
            self.pyramid[-1].append(indiv)

            # 4. elitism update
            cost = self.fitness(indiv)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_individual = indiv

            # 5. Record statistics histories
            all_indivs = [ind for lvl in self.pyramid for ind in lvl]
            costs = [self.fitness(ind) for ind in all_indivs]
            confl = [len(self._detect_conflicts(ind)) for ind in all_indivs]
            self.avg_history.append(float(np.mean(costs)))
            self.best_history.append(self.best_cost)
            self.conf_history.append(float(np.mean(confl)))

            print(
                    f"Gen {gen:4d} "
                    f"Best {self.best_cost:.1f} | Avg {float(np.mean(costs)):.1f} | "
                    f"Conflicts {float(np.mean(confl)):.2f}"
                )

        return self.best_individual, self.best_history, self.avg_history, self.conf_history

    ###################################################################
    # Evolutionary operators                                           #
    ###################################################################

    def generate_individual(self):
        methods = [self.heuristic_path, self.random_path]
        indiv: Dict[int, List[int]] = {}
        temp_g = self.randomize_graph_weights()
        for r in self.robots:
            indiv[r.robot_id] = random.choice(methods)(r, temp_g)
        return self.repair_individual(indiv, conflict_repair=False)

    def local_search(self, indiv: Dict[int, List[int]], steps: int):
        current, cur_cost = indiv, self.fitness(indiv)
        for _ in range(steps):
            mutant = self.mutation(current)
            m_cost = self.fitness(mutant)
            if m_cost < cur_cost:
                current, cur_cost = mutant, m_cost
        return current

    def crossover(self, pa: Dict[int, List[int]], pb: Dict[int, List[int]]):
        child: Dict[int, List[int]] = {}
        for rid in self.robot_map.keys():
            a, b = pa[rid], pb[rid]
            if not a or not b:
                child[rid] = copy.deepcopy(a if len(a) >= len(b) else b)
                continue
            common = list(set(a[:-1]) & set(b[:-1]))
            if common:
                splice = random.choice(common)
                ia, ib = a.index(splice), b.index(splice)
            else:
                ia, ib = len(a) // 2, len(b) // 2
            child[rid] = a[: ia + 1] + b[ib + 1 :]
        return self.repair_individual(child, conflict_repair=True)

    def mutation(self, indiv: Dict[int, List[int]]):
        mutant = copy.deepcopy(indiv)
        rid = random.choice(list(self.robot_map.keys()))
        path = mutant[rid]
        if len(path) < 2:
            return mutant
        mtype = random.choice(["rewire", "insert", "delete"])
        try:
            if mtype == "rewire" and len(path) >= 3:
                i1, i2 = sorted(random.sample(range(1, len(path) - 1), 2))
                seg = self.graph.shortest_path(path[i1], path[i2])
                path = path[: i1 + 1] + seg[1:] + path[i2 + 1 :]
            elif mtype == "insert":
                idx = random.randint(1, len(path) - 2)
                path.insert(idx, path[idx])
            elif mtype == "delete" and len(path) >= 3:
                idx = random.randint(1, len(path) - 2)
                del path[idx]
        except nx.NetworkXNoPath:
            pass
        mutant[rid] = path
        return self.repair_individual(mutant, conflict_repair=False)

    ###################################################################
    # Fitness & conflicts                                              #
    ###################################################################

    def fitness(self, indiv: Dict[int, List[int]]):
        dist = sum(len(p) - 1 for p in indiv.values() if p)
        conflicts = len(self._detect_conflicts(indiv))
        return dist if conflicts == 0 else dist + conflicts * self.CONFLICT_PENALTY_BASE

    def _detect_conflicts(self, indiv):
        cs = []
        horizon = max((len(p) for p in indiv.values()), default=0) + 1
        occ = defaultdict(dict)
        for t in range(horizon):
            nodes = {}
            for rid, p in indiv.items():
                v = p[t] if t < len(p) else p[-1]
                if v in nodes:
                    cs.append(("vertex", t, v, rid, nodes[v]))
                nodes[v] = rid
            occ[t] = nodes
            if t:
                edges = {}
                for rid, p in indiv.items():
                    if t < len(p):
                        u, v = p[t - 1], p[t]
                        if (v, u) in edges:
                            cs.append(("edge", t, (u, v), rid, edges[(v, u)]))
                        edges[(u, v)] = rid
        return cs

    ###################################################################
    # Repair & helpers                                                #
    ###################################################################

    def repair_individual(self, indiv: Dict[int, List[int]], conflict_repair=False):
        cache = {}

        def ssp(u, v):
            if (u, v) not in cache:
                try:
                    cache[(u, v)] = self.graph.shortest_path(u, v)
                except nx.NetworkXNoPath:
                    cache[(u, v)] = [u]
            return cache[(u, v)]

        repaired: Dict[int, List[int]] = {}
        for r in self.robots:
            raw = indiv.get(r.robot_id, []) or [r.start]
            path = [raw[0]]
            for v in raw[1:]:
                if v == path[-1]:
                    path.append(v)
                elif not self.graph.G.has_edge(path[-1], v):
                    path += ssp(path[-1], v)[1:]
                else:
                    path.append(v)
            for trg in r.targets:
                if trg not in path:
                    path += ssp(path[-1], trg)[1:]
            repaired[r.robot_id] = path

        if not conflict_repair:
            return repaired

        # Simple delay‑only conflict repair
        """schedule = defaultdict(dict)
        for r in sorted(self.robots, key=lambda rr: rr.robot_id):
            rid = r.robot_id
            path = repaired[rid]
            delay = 0
            while True:
                bad = False
                for t, node in enumerate(path):
                    tt = t + delay
                    if node in schedule[tt]:
                        bad = True
                        break
                    if t and schedule.get(tt - 1, {}).get(node):
                        other = schedule[tt - 1][node]
                        if repaired[other][tt - 1 - delay] == path[t - 1]:
                            bad = True
                            break
                if bad:
                    delay += 1
                else:
                    for t, node in enumerate(path):
                        schedule[t + delay][node] = rid
                    break"""
        return repaired

    ###################################################################
    # Misc utilities                                                  #
    ###################################################################

    def randomize_graph_weights(self):
        g = copy.deepcopy(self.graph)
        for u, v in g.G.edges():
            g.G[u][v]["weight"] = random.uniform(0.5, 2.0)
        return g

    def heuristic_path(self, r, g):
        cur = r.start
        path = [cur]
        remaining = r.targets[:]
        while remaining:
            try:
                nxt = min(remaining, key=lambda t: g.shortest_path_length(cur, t))
                seg = g.shortest_path(cur, nxt)
                path += seg[1:]
                cur = nxt
                remaining.remove(nxt)
            except nx.NetworkXNoPath:
                break
        return path

    def random_path(self, r, g):
        nodes = list(g.G.nodes())
        must = {r.start, *r.targets}
        extra_n = random.randint(1, max(1, len(r.targets)))
        extras = random.sample(list(set(nodes) - must), min(extra_n, len(nodes) - len(must)))
        wp = r.targets[:] + extras
        random.shuffle(wp)
        if wp and wp[-1] not in r.targets:
            wp[-1] = random.choice(r.targets)
        sequence = [r.start] + wp
        full = []
        for i in range(len(sequence) - 1):
            try:
                seg = g.shortest_path(sequence[i], sequence[i + 1])
                full += seg[1:] if full else seg
            except nx.NetworkXNoPath:
                continue
        return full

    def _calculate_penalty_base(self):
        total = 0
        for r in self.robots:
            try:
                total += max(self.graph.shortest_path_length(r.start, trg) for trg in r.targets)
            except nx.NetworkXNoPath:
                total += nx.diameter(self.graph.G)
        return 1000 * total
