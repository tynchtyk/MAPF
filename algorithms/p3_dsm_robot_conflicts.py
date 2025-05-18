import random
import copy
import math
import warnings # Import warnings module to handle warnings
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional # Added Optional
import numpy as np
#from sklearn.cluster import KMeans # Not used
import networkx as nx
from algorithms.ga import GA

# --- Type Hint Placeholders ---
# Replace these with actual classes or Protocols if available
GraphType = Any
RobotType = Any
NodeId = Any # Type for graph node identifiers (e.g., int, str, tuple)
Path = List[NodeId]
Individual = Dict[int, Path]

class P3_DSM_ROBOT_CONFLICRS(GA):
    """
    Implements the Parameter-less Population Pyramid (P3) algorithm
    enhanced with Dependency Structure Matrix (DSM) Linkage-Tree mixing
    for solving Multi-Agent Path Finding (MAPF) problems.

    (Full docstring as provided previously)
    """

    # ----- construction --------------------------------------------------
    def __init__(
        self,
        graph: GraphType,
        robots: List[RobotType],
        generations: int = 200,
        local_steps: int = 5,
    ) -> None:
        super().__init__(graph, robots, generations, local_steps)
        self.pyramid: List[List[Individual]] = []

        # Constants for Linkage Tree Mixing
        self.LT_PAD: Optional[NodeId] = None # Sentinel for padding in flattened representation (Make Optional explicit)
        self.LT_Q: float = 0.5  # Acceptance probability for cluster swap in _optimal_mix

    # ----- public API --------------------------------------------
    def run(self) -> Tuple[Optional[Individual], List[float], List[float], List[float]]:
        """
        Executes the main evolutionary loop of the P3 algorithm.
        (Full docstring as provided previously)
        """
        for gen in range(self.generations):
            indiv = self.first_improvement_hill_climber(self.generate_individual(), self.local_steps)

            for level_pop in self.pyramid:
                if not level_pop:
                    continue
                partner = random.choice(level_pop)
                child = self._optimal_mix(indiv, partner, level_pop)

                child = self.first_improvement_hill_climber(child, max(1, self.local_steps // 4))

                if self.fitness(child) < self.fitness(indiv):
                    indiv = child # Update indiv if child is better

            # 3) Insert into pyramid
            current_max_level_size = 2 ** len(self.pyramid) if self.pyramid else 1
            if not self.pyramid or len(self.pyramid[-1]) >= current_max_level_size:
                self.pyramid.append([]) # Add a new level
            self.pyramid[-1].append(indiv) # Add individual to the last level

            # 4) Elitism: Track the best solution found globally
            f = self.fitness(indiv)
            if f < self.best_cost:
                self.best_cost = f
                self.best_individual = copy.deepcopy(indiv)

            # 5) Stats collection
            all_ind = [i for lvl in self.pyramid for i in lvl]
            if all_ind:
                 with warnings.catch_warnings():
                      warnings.simplefilter("ignore", category=RuntimeWarning)
                      costs = np.array([self.fitness(i) for i in all_ind])
                      confs = np.array([len(self._detect_conflicts(i)) for i in all_ind])
                      avg_cost = float(np.nanmean(costs)) # Use nanmean
                      avg_conf = float(np.nanmean(confs)) # Use nanmean
                 self.avg_history.append(avg_cost)
                 self.conf_history.append(avg_conf)
            else:
                 # Append current best cost or NaN if no individuals exist yet
                 self.avg_history.append(self.best_cost if self.best_cost != math.inf else math.nan)
                 self.conf_history.append(0.0) # No conflicts if no individuals
            self.best_history.append(self.best_cost)


            # Print progress (consider using logging module for more control)
            print(
                f"Gen {gen:3d} | "
                f"Best {self.best_cost:.1f} | "
                f"Avg {self.avg_history[-1]:.1f} | "
                f"Conf {self.conf_history[-1]:.2f}"
            )

        return self.best_individual, self.best_history, self.avg_history, self.conf_history

   
    # ----- BB-Mix with vertex-tape linkage tree ------------------------
    # LT_PAD, LT_Q defined in __init__

    def _optimal_mix(self, receiver: Individual, donor: Individual, population: List[Individual]) -> Individual:
        if not population:
            return receiver 

        dsm = self._build_dsm(population)
        lt = self._linkage_tree(dsm)            
        current, f_cur = receiver, self.fitness(receiver)

        for cluster in lt:                     
            trial = copy.deepcopy(current)
            for rid_idx in cluster:            
                rid = self.robots[rid_idx].robot_id
                trial[rid] = copy.deepcopy(donor[rid])

            trial = self.repair_individual(trial, conflict_repair=True)
            f_trial = self.fitness(trial)
            if f_trial < f_cur:                 
                current, f_cur = trial, f_trial

        return current


    def _build_dsm(self, population: List[Individual]) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Return an R×R DSM where R = #robots.
        d(i,j) is 0 if robots i and j ALWAYS collide,
                1 if they NEVER collide,
                in between according to mutual information of the binary
                variable  Z_{ij} = 1⇔(i,j) collide in the plan.
        """
        R = len(self.robots)
        m = len(population)
        if m == 0 or R <= 1:
            return None                            # nothing to do

        # --- count how many plans show a collision for each robot pair ----
        pair_hits = np.zeros((R, R), dtype=int)
        for indiv in population:
            seen = set()
            for _, _, _, r1, r2 in self._detect_conflicts(indiv):
                i, j = sorted((r1, r2))
                if (i, j) not in seen:            # count at most once per indiv
                    pair_hits[i, j] += 1
                    seen.add((i, j))

        # --- convert to mutual-information distance -----------------------
        eps = 1e-12
        H_row = np.zeros(R)
        dsm   = np.ones((R, R), dtype=float)

        for i in range(R):
            for j in range(i + 1, R):
                p1 = pair_hits[i, j] / m          # probability of collision
                if p1 < eps or p1 > 1 - eps:
                    H = 0.0                       # fully deterministic
                else:
                    H = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1)
                H_row[i] = max(H_row[i], H)
                H_row[j] = max(H_row[j], H)
                dsm[i, j] = dsm[j, i] = 1.0 - H   # provisional

        # normalise by max(H_i, H_j)
        for i in range(R):
            for j in range(i + 1, R):
                hmax = max(H_row[i], H_row[j])
                dsm[i, j] = dsm[j, i] = 1.0 if hmax < eps else dsm[i, j] / hmax

        return dsm

    def _linkage_tree(self, dsm: np.ndarray) -> List[List[int]]:
        """
        Builds a linkage tree (hierarchical clustering) from a DSM using
        a custom Nearest Neighbor Chain heuristic.
        NOTE: This implementation differs from standard libraries like scipy.
              It aims to produce a flat list ordered leaf->root via DFS.
        (Docstring reflects the custom nature)
        """
        num_vars = len(dsm)
        if num_vars == 0: return []
        if num_vars == 1: return [[0]]

        # Check for NaNs or Infs in DSM, which can break clustering
        if np.any(np.isnan(dsm)) or np.any(np.isinf(dsm)):
            warnings.warn("NaN or Inf detected in DSM, cannot build linkage tree.")
            # Try to clean DSM or return empty list
            # dsm = np.nan_to_num(dsm, nan=1.0, posinf=1.0, neginf=1.0) # Option: replace bad values
            return [] # Return empty, OM will skip

        # Copy DSM to avoid modifying original, ensure float type
        current_dist_matrix = dsm.copy().astype(float)
        # Set diagonal to infinity to avoid self-clustering
        np.fill_diagonal(current_dist_matrix, math.inf)

        # clusters is a list where each element represents an active cluster (list of original indices)
        clusters: List[List[int]] = [[i] for i in range(num_vars)]
        # Map from the index in the *current* distance matrix back to the index in the `clusters` list
        active_indices = list(range(num_vars))

        # Store the hierarchy for DFS traversal later
        # node_list stores cluster contents (initially singletons)
        node_list: List[Any] = [[i] for i in range(num_vars)]
        # children[i] will store the two children indices (in node_list) that merged to form node i
        children: Dict[int, Tuple[int, int]] = {}
        next_node_idx = num_vars # Index for newly created internal nodes

        for _ in range(num_vars - 1): # Perform n-1 merges
             if len(active_indices) < 2: break # Stop if fewer than 2 clusters remain

             # Find the minimum distance in the current distance matrix
             # NOTE: This is closer to standard single/complete linkage finding the global minimum
             # The original "chain" logic seemed more complex. This is more standard.
             min_dist = np.min(current_dist_matrix)
             # Check if min_dist is valid
             if min_dist == math.inf:
                  warnings.warn("Could not find finite distance pair in linkage tree merge. Stopping early.")
                  break

             # Find indices (in the *current* matrix) of the minimum distance pair
             merge_matrix_idx1, merge_matrix_idx2 = np.unravel_index(np.argmin(current_dist_matrix), current_dist_matrix.shape)

             # Map matrix indices back to indices in the `clusters` list / `active_indices` list
             # These indices point to the nodes in node_list being merged
             node_idx1 = active_indices[merge_matrix_idx1]
             node_idx2 = active_indices[merge_matrix_idx2]

             # Create new cluster content and add to node_list
             merged_content = node_list[node_idx1] + node_list[node_idx2]
             node_list.append(merged_content)
             new_node_internal_idx = next_node_idx
             children[new_node_internal_idx] = (node_idx1, node_idx2) # Record children
             next_node_idx += 1

             # --- Update Distance Matrix (Single Linkage Example) ---
             # Remove rows/columns corresponding to merged clusters
             # Create new row/column for the merged cluster
             m = current_dist_matrix.shape[0]
             new_dist_matrix = np.full((m - 1, m - 1), math.inf)

             # Calculate distances from the new cluster to all others (Single Linkage)
             new_distances = []
             idx_map = {} # Map old matrix index to new matrix index
             new_idx_counter = 0
             for i in range(m):
                  if i != merge_matrix_idx1 and i != merge_matrix_idx2:
                      # Distance is min(dist(new, old_i)) = min(dist(c1, old_i), dist(c2, old_i))
                      dist1 = current_dist_matrix[merge_matrix_idx1, i]
                      dist2 = current_dist_matrix[merge_matrix_idx2, i]
                      new_dist = min(dist1, dist2)
                      new_distances.append(new_dist)
                      idx_map[i] = new_idx_counter
                      new_idx_counter += 1

             # Populate the new matrix
             new_m = m - 1
             for r in range(new_m -1): # Fill upper triangle of existing clusters part
                  old_r = [k for k, v in idx_map.items() if v == r][0]
                  for c in range(r + 1, new_m -1):
                       old_c = [k for k, v in idx_map.items() if v == c][0]
                       new_dist_matrix[r, c] = new_dist_matrix[c, r] = current_dist_matrix[old_r, old_c]

             # Add distances for the newly merged cluster (last row/col)
             if new_distances: # Check if new_distances is not empty
                 new_dist_matrix[:-1, -1] = new_distances
                 new_dist_matrix[-1, :-1] = new_distances

             np.fill_diagonal(new_dist_matrix, math.inf) # Keep diagonal infinite
             current_dist_matrix = new_dist_matrix

             # Update active_indices: remove merged, add new internal node index
             new_active_indices = [active_indices[i] for i in range(len(active_indices)) if i != merge_matrix_idx1 and i != merge_matrix_idx2]
             new_active_indices.append(new_node_internal_idx)
             active_indices = new_active_indices


        # ---- Flatten the hierarchy using Depth First Search (leaf-first post-order) ----
        flat_ordered_clusters = []
        visited = set()
        # Start DFS from the root node (last added node)
        root_node_idx = next_node_idx - 1

        def dfs(node_idx):
            if node_idx in visited:
                return
            visited.add(node_idx)
            # If it's an internal node (has children recorded)
            if node_idx in children:
                child1_idx, child2_idx = children[node_idx]
                dfs(child1_idx)
                dfs(child2_idx)
            # Append the content of the cluster (list of original var indices) after visiting children
            flat_ordered_clusters.append(node_list[node_idx])

        if root_node_idx >= 0 : # Check if any merges happened
             dfs(root_node_idx)
        elif num_vars > 0: # Handle case with only one variable
            flat_ordered_clusters = [[0]]


        # The list should contain clusters from leaves up to the root
        return flat_ordered_clusters


   