import random
import copy
import math
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import networkx as nx
import time
import functools

# --- Type Hint Placeholders ---
GraphType = Any
RobotType = Any
NodeId = Any
Path = List[NodeId]
Individual = Dict[int, Path]

# --- Time Logger Definition ---
class TimeLogger:
    def __init__(self):
        self.records = {}

    def log(self, name, duration):
        if name not in self.records:
            self.records[name] = {"count": 0, "total_time": 0.0}
        self.records[name]["count"] += 1
        self.records[name]["total_time"] += duration

    def report(self):
        lines = []
        for name, data in self.records.items():
            avg = data["total_time"] / data["count"]
            lines.append(f"[TIME] {name}: called {data['count']} times, total {data['total_time']:.6f}s, avg {avg:.6f}s")
        return "\n".join(lines)

    def get_history(self):
        return self.records.copy()

# --- Decorator for Method Timing ---
def log_time(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        if hasattr(self, 'time_logger'):
            self.time_logger.log(method.__qualname__, duration)
        return result
    return wrapper

class GA:
    def __init__(
        self,
        graph: GraphType,
        robots: List[RobotType],
        generations: int = 200,
        local_steps: int = 5,
    ) -> None:
        self.graph = graph
        self.robots = robots
        self.robot_map = {r.robot_id: r for r in robots}
        self.generations = generations
        self.local_steps = local_steps
        self.time_logger = TimeLogger()
        self.CONFLICT_PENALTY_BASE = self._calculate_penalty_base()
        self.best_individual: Dict[int, List[int]] | None = None
        self.best_cost: float = math.inf
        self.avg_history: List[float] = []
        self.best_history: List[float] = []
        self.conf_history: List[float] = []
        self.time_log_history: List[Dict[str, Any]] = []
        self._fitness_cache: Dict[Tuple, float] = {}
        self._path_lengths: Dict[int, Dict[int, int]] = {}
    
    def _set_deadline(self, max_seconds: float):
        self._t0 = time.perf_counter()
        self._deadline = self._t0 + float(max_seconds)

    def _time_is_up(self) -> bool:
        """Returns True if the deadline has passed, False otherwise."""
        return time.perf_counter() >= self._deadline

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _calculate_penalty_base(self) -> float:
        """
        Conflict penalty weight W_conflict.
        """
        total_lb_soc = 0.0

        for r in self.robots:
            if not getattr(r, "targets", None):
                continue

            current = r.start
            remaining = list(r.targets)

            # Greedy chaining as a cheap SOC lower-bound proxy
            while remaining:
                best_t = None
                best_d = None
                for t in remaining:
                    try:
                        d = self.graph.shortest_path_length(current, t)
                    except Exception:
                        d = None
                    if d is None:
                        continue
                    if best_d is None or d < best_d:
                        best_d = d
                        best_t = t

                if best_t is None:
                    try:
                        best_d = nx.diameter(self.graph.G)
                    except Exception:
                        best_d = 100
                    total_lb_soc += best_d
                    break

                total_lb_soc += best_d
                current = best_t
                remaining.remove(best_t)

        if total_lb_soc <= 0:
            total_lb_soc = 100.0 * max(1, len(self.robots))

        num_agents = max(1, len(self.robots))
        size_factor = 1.0 + 0.05 * (num_agents - 1)
        W_conflict = 2.0 * total_lb_soc * size_factor

        return float(W_conflict)

    @log_time
    def randomize_graph_weights(self):
        g = copy.deepcopy(self.graph)
        # --- FIX: Check for MultiGraph to avoid unpacking error ---
        if g.G.is_multigraph():
            for u, v, k in g.G.edges(keys=True):
                g.G[u][v][k]["weight"] = random.uniform(0.5, 2.0)
        else:
            for u, v in g.G.edges():
                g.G[u][v]["weight"] = random.uniform(0.5, 2.0)
        return g

    @log_time
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

    @log_time
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

    @log_time
    def fitness(self, indiv: Individual, modified_rids: Optional[Set[int]] = None) -> float:
        indiv_key = tuple((rid, tuple(path)) for rid, path in sorted(indiv.items()))
        if modified_rids is None and indiv_key in self._fitness_cache:
            return self._fitness_cache[indiv_key]
        dist = 0
        for rid, path in indiv.items():
            if rid not in self._path_lengths:
                self._path_lengths[rid] = {}
            if modified_rids is None or rid in modified_rids:
                path_len = len(path) - 1 if path else 0
                self._path_lengths[rid][indiv_key] = path_len
            else:
                if indiv_key in self._path_lengths[rid]:
                    path_len = self._path_lengths[rid][indiv_key]
                else:
                    path_len = len(path) - 1 if path else 0
                    self._path_lengths[rid][indiv_key] = path_len
            dist += path_len
        conflicts = len(self._detect_conflicts(indiv, modified_rids=modified_rids))
        fitness = dist if conflicts == 0 else dist + conflicts * self.CONFLICT_PENALTY_BASE
        if modified_rids is None:
            self._fitness_cache[indiv_key] = fitness
        return fitness

    @log_time
    def _detect_conflicts(self, indiv: Individual, modified_rids: Optional[Set[int]] = None) -> List[Tuple]:
        cs = []
        horizon = max((len(p) for p in indiv.values()), default=0) + 1
        if modified_rids is None:
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
        else:
            for t in range(horizon):
                nodes = {}
                for rid in modified_rids:
                    if t < len(indiv[rid]):
                        v = indiv[rid][t] if t < len(indiv[rid]) else indiv[rid][-1]
                        for other_rid, p in indiv.items():
                            if other_rid in modified_rids and other_rid <= rid:
                                continue
                            other_v = p[t] if t < len(p) else p[-1]
                            if v == other_v:
                                cs.append(("vertex", t, v, rid, other_rid))
                if t > 0:
                    for rid in modified_rids:
                        if t < len(indiv[rid]):
                            u, v = indiv[rid][t - 1], indiv[rid][t]
                            for other_rid, p in indiv.items():
                                if other_rid in modified_rids and other_rid <= rid:
                                    continue
                                if t < len(p):
                                    other_u, other_v = p[t - 1], p[t]
                                    if (u, v) == (other_v, other_u):
                                        cs.append(("edge", t, (u, v), rid, other_rid))
        return cs

    @log_time
    def repair_individual(self, indiv: Individual, conflict_repair: bool = False) -> Individual:
        cache = {}
        def ssp(u, v):
            if (u, v) not in cache:
                try:
                    cache[(u, v)] = self.graph.shortest_path(u, v)
                except nx.NetworkXNoPath:
                    cache[(u, v)] = [u]
            return cache[(u, v)]
        repaired = {}
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
        return repaired

    @log_time
    def generate_individual(self) -> Individual:
        methods = [self.heuristic_path, self.random_path]
        indiv: Individual = {}
        for r in self.robots:
            temp_g = self.randomize_graph_weights()
            indiv[r.robot_id] = random.choice(methods)(r, temp_g)
        return self.repair_individual(indiv, conflict_repair=False)

    @log_time
    def first_improvement_hill_climber(self, individual: Individual, samples: int = 5) -> Individual:
        """
        Modified to gracefully return 'current' if time expires, instead of raising exception.
        """
        samples = max(1, samples)
        current = individual

        if self._time_is_up():
            return current

        f_cur = self.fitness(current, modified_rids=None)
        tested_robots: Set[int] = set()

        while True:
            # Graceful exit check 1
            if self._time_is_up():
                return current

            rids = list(self.robot_map.keys())
            random.shuffle(rids)
            improvement_found = False

            for rid in rids:
                # Graceful exit check 2
                if self._time_is_up():
                    return current

                if rid in tested_robots:
                    continue

                original_path = current[rid]
                path_len = len(original_path)

                if path_len < 2:
                    tested_robots.add(rid)
                    continue

                for mtype in random.sample(["rewire", "insert", "delete"], 3):
                    # Graceful exit check 3
                    if self._time_is_up():
                        return current

                    if (mtype == "rewire" and path_len < 4) or (mtype == "delete" and path_len < 3):
                        continue

                    for _ in range(samples):
                        # Graceful exit check 4
                        if self._time_is_up():
                            return current

                        mutated_path = self._mutate_path(original_path, mtype)
                        if mutated_path == original_path:
                            continue

                        trial = current.copy()
                        trial[rid] = mutated_path

                        trial = self.repair_individual(trial, conflict_repair=False)
                        f_trial = self.fitness(trial, modified_rids={rid})

                        if f_trial < f_cur:
                            current = trial
                            f_cur = f_trial
                            tested_robots.clear()
                            improvement_found = True
                            break

                    if improvement_found:
                        break

                if not improvement_found:
                    tested_robots.add(rid)

            if not improvement_found:
                break

        return current

    @log_time
    def _mutate_path(self, path: Path, mtype: str) -> Path:
        n = len(path)
        if n < 2:
            return path
        new_path = path
        try:
            if mtype == "rewire" and n >= 4:
                idx_range = range(1, n - 1)
                if len(idx_range) >= 2:
                    i1, i2 = sorted(random.sample(idx_range, 2))
                    segment = self.graph.shortest_path(path[i1], path[i2])
                    new_path = path[:i1 + 1] + segment[1:] + path[i2 + 1:]
            elif mtype == "insert" and n >= 1:
                idx = random.randint(0, n - 1)
                new_path = path[:idx] + [path[idx]] + path[idx:]
            elif mtype == "delete" and n >= 3:
                idx = random.randint(1, n - 2)
                u, w, v = path[idx - 1], path[idx], path[idx + 1]
                neighbors_of_w = list(self.graph.G.neighbors(w))
                if not neighbors_of_w:
                    return path
                random.shuffle(neighbors_of_w)
                for neighbor_node in neighbors_of_w[:min(len(neighbors_of_w), 5)]:
                    try:
                        seg1 = self.graph.shortest_path(u, neighbor_node)
                        seg2 = self.graph.shortest_path(neighbor_node, v)
                        new_path = path[:idx - 1] + seg1 + seg2[1:] + path[idx + 2:]
                        break
                    except nx.NetworkXNoPath:
                        continue
        except (nx.NetworkXNoPath, ValueError, IndexError, Exception) as e:
            warnings.warn(f"Error during mutation {mtype}: {e}")
            return path
        if not new_path and path:
            return path
        return new_path

    def print_time_log_report(self):
        print(self.time_logger.report())

    def save_time_log(self):
        snapshot = self.time_logger.get_history()
        self.time_log_history.append(snapshot)