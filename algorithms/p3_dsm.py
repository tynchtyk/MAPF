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

class P3_DSM(GA):
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
            # 1) Create a fresh solution + initial hill climb
            indiv = self.first_improvement_hill_climber(self.generate_individual(), self.local_steps)

            # 2) Climb each level with Optimal Mixing (OM)
            for level_pop in self.pyramid:
                # FIX: Skip empty levels to prevent random.choice error
                if not level_pop:
                    continue
                partner = random.choice(level_pop)
                child = self._optimal_mix(indiv, partner, level_pop)
                # Hill climb the result of mixing (using fewer steps is reasonable)
                #child = self.first_improvement_hill_climber(child, max(1, self.local_steps // 4))
                # Compare fitness after hill climbing
                # FIX: Check if child is better than the *current* indiv, not the initial one for this loop
                if self.fitness(child) < self.fitness(indiv):
                    indiv = child # Update indiv if child is better

            # 3) Insert into pyramid
            # FIX: Ensure pyramid level growth logic is robust if pyramid starts empty
            current_max_level_size = 2 ** len(self.pyramid) if self.pyramid else 1
            if not self.pyramid or len(self.pyramid[-1]) >= current_max_level_size:
                self.pyramid.append([]) # Add a new level
            self.pyramid[-1].append(indiv) # Add individual to the last level

            # 4) Elitism: Track the best solution found globally
            f = self.fitness(indiv)
            if f < self.best_cost:
                self.best_cost = f
                # Use deepcopy to store the best individual independently
                self.best_individual = copy.deepcopy(indiv)

            # 5) Stats collection
            all_ind = [i for lvl in self.pyramid for i in lvl]
            # FIX: Handle case where all_ind is empty before calculating means
            if all_ind:
                 # Use numpy for potentially better NaN handling, but catch warnings
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
        """
        Performs Optimal Mixing (OM) crossover using a Linkage Tree derived
        from a Dependency Structure Matrix (DSM).
        (Full docstring as provided previously)
        """
        if not population:
            return receiver # Cannot mix without context

        try:
            dsm, dims = self._build_dsm(population)
            if dsm is None or dims[1] == 0: # Handle empty DSM or zero tape length
                return receiver

            lt = self._linkage_tree(dsm)
            if not lt: # Handle empty linkage tree
                return receiver

            r_vec = self._flatten(receiver, dims)
            d_vec = self._flatten(donor, dims)
            tape_len = len(r_vec) # Use actual tape length

            current_best_receiver = receiver
            current_best_vec = r_vec
            current_best_fitness = self.fitness(receiver)

            for cluster_indices in lt:
                if not cluster_indices: continue # Skip empty clusters if they occur

                # Convert to numpy array for potential boolean indexing later
                cluster = np.array(cluster_indices, dtype=int)

                # FIX: Ensure indices are within bounds *before* accessing donor vector
                valid_mask = (cluster >= 0) & (cluster < tape_len)
                valid_cluster_indices = cluster[valid_mask]

                # Skip if cluster is empty after filtering or probabilistically
                if valid_cluster_indices.size == 0 or random.random() >= self.LT_Q:
                    continue

                # Create trial vector by swapping valid indices
                trial_vec = current_best_vec.copy()
                try:
                     # Assign values from donor vector at valid indices
                     trial_vec[valid_cluster_indices] = d_vec[valid_cluster_indices]
                except IndexError:
                     # This might happen if d_vec is unexpectedly shorter (though dims should match)
                     warnings.warn(f"IndexError during Optimal Mix cluster swap. Tape lengths: R={len(r_vec)}, D={len(d_vec)}. Cluster indices max={cluster.max()}. Skipping swap.")
                     continue # Skip this swap attempt

                # Convert trial vector back and evaluate
                # Use current_best_receiver as template for robustness
                trial_ind = self._unflatten(trial_vec, dims, current_best_receiver)
                trial_fitness = self.fitness(trial_ind)

                if trial_fitness < current_best_fitness:
                    # Accept improvement
                    current_best_receiver = trial_ind
                    current_best_vec = trial_vec
                    current_best_fitness = trial_fitness

        # FIX: Add broad exception handling for robustness during mixing process
        except Exception as e:
             warnings.warn(f"Error during _optimal_mix: {e}. Returning original receiver.")
             # Log the full traceback here if possible using logging module
             # import traceback; traceback.print_exc(); # For debugging
             return receiver # Return original receiver on error

        return current_best_receiver


    # --- DSM + linkage-tree helpers -----------------------------------
    def _flatten(self, indiv: Individual, dims: Tuple[int, int]) -> np.ndarray:
        """
        Flattens an individual (dict of paths) into a 1D NumPy array ("tape").
        (Full docstring as provided previously)
        """
        R, S = dims
        if R == 0 or S == 0: return np.array([], dtype=object) # Handle zero dimensions

        tape = np.full(R * S, self.LT_PAD, dtype=object)
        for r_idx, rob in enumerate(self.robots):
            # Ensure robot ID exists in individual (robustness)
            path = indiv.get(rob.robot_id, [])
            path_len = len(path)
            start_idx = r_idx * S
            # Cap length at S
            effective_len = min(path_len, S)
            end_idx = start_idx + effective_len
            if effective_len > 0:
                # Assign path segment, ensuring it fits
                tape[start_idx : end_idx] = path[:effective_len]
        return tape

    def _unflatten(self, tape: np.ndarray, dims: Tuple[int, int], template: Individual) -> Individual:
        """
        Reconstructs an individual (dict of paths) from its flattened 1D tape representation.
        (Full docstring as provided previously)
        """
        R, S = dims
        if R == 0 or S == 0: return {} # Handle zero dimensions

        indiv: Individual = {}
        for r_idx, rob in enumerate(self.robots):
            start_idx = r_idx * S
            end_idx = (r_idx + 1) * S
            # Ensure indices are within tape bounds
            if start_idx >= len(tape): continue
            end_idx = min(end_idx, len(tape))
            segment = tape[start_idx : end_idx]

            # Filter padding values
            # FIX: Check type before comparison if nodes can be non-hashable/complex
            path = [node for node in segment if node is not self.LT_PAD]

            # FIX: Robust fallback if path becomes empty
            if not path:
                 fallback_path = template.get(rob.robot_id, [rob.start]) # Default to start node
                 # Ensure fallback is not empty list itself
                 path = fallback_path[:1] if fallback_path else [rob.start]

            # Ensure path starts correctly, even if fallback was used
            # (Repair function also handles this, but good to be robust here)
            if not path or path[0] != rob.start:
                 # Avoid adding start if it's already the first element after filtering
                 if not path or path[0] != rob.start:
                      path = [rob.start] + [p for p in path if p != rob.start]


            indiv[rob.robot_id] = path

        # Repair ensures targets are met and path is connected
        return self.repair_individual(indiv, conflict_repair=False)

    def _tape_dims(self, population: List[Individual]) -> Tuple[int, int]:
        """
        Calculates the dimensions required for the flattened tape representation.
        (Full docstring as provided previously)
        """
        if not self.robots: return (0, 0) # No robots, zero dimensions
        num_robots = len(self.robots)

        if not population:
            # Use a default max length if population is empty, e.g., 1 or based on graph size
            return num_robots, 1

        max_len = 0
        for ind in population:
            if not isinstance(ind, dict): continue # Skip invalid individuals
            for path in ind.values():
                 if isinstance(path, list): # Check if path is a list
                      max_len = max(max_len, len(path))

        # Ensure max_len is at least 1
        max_len = max(max_len, 1)
        return num_robots, max_len

    def _build_dsm(self, population: List[Individual]) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Builds the Dependency Structure Matrix (DSM) based on mutual information.
        (Full docstring as provided previously)
        """
        if not population:
            return None, (len(self.robots), 0)

        dims = self._tape_dims(population)
        R, S = dims
        if R == 0 or S == 0: return None, dims # Handle zero dimensions

        nvar = R * S
        pop_size = len(population)

        try:
             pop_matrix = np.stack([self._flatten(ind, dims) for ind in population])
        except ValueError as e:
             warnings.warn(f"Error stacking flattened individuals in _build_dsm (potential dimension mismatch?): {e}")
             return None, dims # Cannot proceed if stacking fails

        H = np.zeros(nvar)
        # Use small epsilon to avoid log(0)
        epsilon = 1e-9
        for i in range(nvar):
            counts = Counter(pop_matrix[:, i])
            entropy = 0.0
            for allele_count in counts.values():
                p = allele_count / pop_size
                # Add epsilon inside log for stability
                entropy -= p * math.log2(p + epsilon)
            H[i] = max(0, entropy) # Ensure entropy is non-negative

        dsm = np.zeros((nvar, nvar))
        for i in range(nvar):
            counts_i = Counter(pop_matrix[:, i])
            for j in range(i + 1, nvar):
                counts_j = Counter(pop_matrix[:, j])
                joint_counts = Counter(zip(pop_matrix[:, i], pop_matrix[:, j]))
                mutual_info = 0.0

                for (xi, xj), joint_count in joint_counts.items():
                    pxy = joint_count / pop_size
                    px = counts_i[xi] / pop_size
                    py = counts_j[xj] / pop_size

                    # Check for non-zero probabilities before division/log
                    if pxy > epsilon and px > epsilon and py > epsilon:
                        mutual_info += pxy * math.log2(pxy / (px * py))

                # Normalize MI
                max_H = max(H[i], H[j])
                if max_H < epsilon: # Use epsilon check for floating point comparison
                    distance = 1.0 # Max distance if entropies are zero
                else:
                    # FIX: Clamp normalized MI robustly before calculating distance
                    # MI can theoretically exceed max(H[i], H[j]) slightly due to precision
                    normalized_mi = 0.0
                    if max_H > epsilon: # Avoid division by zero/small number
                         normalized_mi = mutual_info / max_H
                    # Clamp value between 0 and 1
                    clamped_nmi = min(max(0.0, normalized_mi), 1.0)
                    distance = 1.0 - clamped_nmi

                dsm[i, j] = dsm[j, i] = distance

        return dsm, dims

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

