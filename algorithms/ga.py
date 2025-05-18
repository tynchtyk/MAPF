import random
import copy
import math
from collections import defaultdict
from typing import Dict, List
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional # Added Optional
import time
import functools

# --- Type Hint Placeholders ---
# Replace these with actual classes or Protocols if available
GraphType = Any
RobotType = Any
NodeId = Any # Type for graph node identifiers (e.g., int, str, tuple)
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
        self.time_logger = TimeLogger()  # Instance-specific time logger


        # Conflict‑penalty magnitude
        self.CONFLICT_PENALTY_BASE = self._calculate_penalty_base()
        self.best_individual: Dict[int, List[int]] | None = None
        self.best_cost: float = math.inf

        # Statistics histories
        self.avg_history: List[float] = []
        self.best_history: List[float] = []
        self.conf_history: List[float] = []
        self.time_log_history: List[Dict[str, Any]] = []

    def _calculate_penalty_base(self):
        total = 0
        for r in self.robots:
            try:
                total += max(self.graph.shortest_path_length(r.start, trg) for trg in r.targets)
            except:
                total += nx.diameter(self.graph.G)
        return 1000 * total

    @log_time
    def randomize_graph_weights(self):
        g = copy.deepcopy(self.graph)
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
    def fitness(self, indiv):
        dist = sum(len(p) - 1 for p in indiv.values() if p)
        conflicts = len(self._detect_conflicts(indiv))
        return dist if conflicts == 0 else dist + conflicts * self.CONFLICT_PENALTY_BASE

    @log_time
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

    @log_time
    def repair_individual(self, indiv, conflict_repair=False):
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

            # Ensure all targets are included
            for trg in r.targets:
                if trg not in path:
                    path += ssp(path[-1], trg)[1:]

            repaired[r.robot_id] = path

        return repaired

    @log_time
    def generate_individual(self):
        methods = [self.heuristic_path, self.random_path]
        indiv: Individual = {}
        for r in self.robots:
            temp_g = self.randomize_graph_weights()
            indiv[r.robot_id] = random.choice(methods)(r, temp_g)
        return self.repair_individual(indiv, conflict_repair=False)

    # ----- hill-climber -------------------------------------------------
    @log_time
    def first_improvement_hill_climber(self, individual: Individual, samples: int = 5) -> Individual:
        samples = max(1, samples)
        current = individual  # No initial copy (we'll copy only if modified)
        f_cur = self.fitness(current)
        tested_robots = set()  # Track robots that failed to improve
        
        while True:
            rids = list(self.robot_map.keys())
            random.shuffle(rids)
            improvement_found = False

            for rid in rids:
                if rid in tested_robots:
                    continue

                original_path = current[rid]
                path_len = len(original_path)
                
                # Skip robots with paths too short for any mutation
                if path_len < 2:
                    tested_robots.add(rid)
                    continue

                # Try mutations in random order
                for mtype in random.sample(["rewire", "insert", "delete"], 3):
                    # Early skip for impossible mutations
                    if (mtype == "rewire" and path_len < 4) or \
                    (mtype == "delete" and path_len < 3):
                        continue

                    for _ in range(samples):
                        mutated_path = self._mutate_path(original_path, mtype)
                        if mutated_path == original_path:
                            continue  # Skip if no change

                        # Minimal copy: modify only the changed robot
                        trial = current.copy()  # Shallow copy
                        trial[rid] = mutated_path
                        trial = self.repair_individual(trial, conflict_repair=False)

                        f_trial = self.fitness(trial)
                        if f_trial < f_cur:
                            current = trial
                            f_cur = f_trial
                            tested_robots = set()  # Reset since we improved
                            improvement_found = True
                            break  # First improvement found
                    
                    if improvement_found:
                        break
                # End of mutation types

                if not improvement_found:
                    tested_robots.add(rid)
            # End of robot loop

            if not improvement_found:
                break  # No improvement in full pass

        return current


    # ----- mutation helper (single path) -------------------------------
    @log_time
    def _mutate_path(self, path: Path, mtype: str) -> Path:
        """
        Applies a mutation operation to a single robot's path.
        (Full docstring as provided previously)
        """
        n = len(path)
        if n < 2:
            return path # Cannot mutate paths with 0 or 1 node

        new_path = path # Default to original path if mutation fails/not applicable

        try:
            if mtype == "rewire" and n >= 4:
                idx_range = range(1, n - 1)
                # FIX: Check if range has at least 2 elements for sampling
                if len(idx_range) >= 2:
                    i1, i2 = sorted(random.sample(idx_range, 2))
                    # Check if nodes are already adjacent (rewire less useful)
                    # if i2 == i1 + 1 and self.graph.G.has_edge(path[i1], path[i2]):
                    #     return path # Optional: skip if already directly connected
                    segment = self.graph.shortest_path(path[i1], path[i2])
                    new_path = path[: i1 + 1] + segment[1:] + path[i2 + 1 :]
                # else: cannot sample 2 distinct indices

            elif mtype == "insert" and n >= 1:
                idx = random.randint(0, n - 1)
                new_path = path[:idx] + [path[idx]] + path[idx:]


            elif mtype == "delete" and n >= 3:
                # Index range for deletion is 1 to n-2 inclusive
                idx = random.randint(1, n - 2)
                u, w, v = path[idx - 1], path[idx], path[idx + 1]

                # FIX: Ensure neighbors are sampled correctly (handle nodes with few/no neighbors)
                neighbors_of_w = list(self.graph.G.neighbors(w))
                if not neighbors_of_w: return path # Cannot reroute if w has no neighbors

                random.shuffle(neighbors_of_w)
                # FIX: Limit number of neighbors to try (e.g., 5) for efficiency
                for neighbor_node in neighbors_of_w[:min(len(neighbors_of_w), 5)]:
                    try:
                        # Find paths u -> neighbor and neighbor -> v
                        seg1 = self.graph.shortest_path(u, neighbor_node)
                        seg2 = self.graph.shortest_path(neighbor_node, v)
                        # Correct reconstruction: path up to u, seg1, seg2[1:], path after v
                        new_path = path[:idx - 1] + seg1 + seg2[1:] + path[idx + 2 :]
                        break # Found a valid replacement path
                    except nx.NetworkXNoPath:
                        continue # Try next neighbor
                    except Exception as e:
                         warnings.warn(f"Error during pathfinding in delete mutation ({u}->{neighbor_node}->{v}): {e}")
                         continue # Try next neighbor

        # FIX: Add broader exception handling for robustness
        except nx.NetworkXNoPath:
            return path # Return original path if shortest_path fails
        except ValueError as e:
             # Handles potential random.sample errors if range is too small
             # Or if randint range is invalid (e.g., n=2 for insert)
             # warnings.warn(f"ValueError during mutation {mtype}: {e}")
             return path
        except IndexError as e:
             # Handle potential index errors if path structure is unexpected
             # warnings.warn(f"IndexError during mutation {mtype}: {e}")
             return path
        except Exception as e:
             warnings.warn(f"Unexpected error during mutation {mtype}: {e}")
             return path

        # FIX: Ensure the returned path is not empty if original wasn't
        if not new_path and path:
             return path # Revert if mutation somehow resulted in empty path

        return new_path

    # --- Log Reporting ---
    def print_time_log_report(self):
        print(self.time_logger.report())

    def save_time_log(self):
        snapshot = self.time_logger.get_history()
        self.time_log_history.append(snapshot)