import random
import copy
import math
from collections import defaultdict
from typing import Dict, List
from algorithms.ga import GA

import networkx as nx
import numpy as np


###############################################################
# Parameterless Population Pyramid EA                          #
###############################################################

class P3_CDGX(GA):
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
        super().__init__(graph, robots, generations)
        # Pyramid holds all individuals generated so far
        self.pyramid: List[List[Dict[int, List[int]]]] = []

    def crossover(self,
        receiver: Dict[int, List[int]],   # ← “source”  (will be improved)
        donor:     Dict[int, List[int]]   # ← “partner” (supplies blocks)
    ) -> Dict[int, List[int]]:
        """
        Improvement-only crossover that moves entire *conflict groups*
        from donor → receiver.

        1. Build transitive robot groups that collide in *either* parent.
        2. Walk each group (in random order is fine):
            • copy all paths of the group from donor into a trial child
            • repair   (connectivity + optional delay scheduler)
            • if   fitness(trial) < fitness(receiver):   commit the change
        3. Return the (possibly unchanged) receiver.
        """

        # ---------- 1. detect conflict pairs in BOTH parents ------------
        conflict_pairs: set[tuple[int, int]] = set()
        for indiv in (receiver, donor):
            for _, _, _, r1, r2 in self._detect_conflicts(indiv):
                conflict_pairs.add(tuple(sorted((r1, r2))))

        # ---------- 2. Union–Find to build transitive groups ------------
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in conflict_pairs:
            union(a, b)
        for rid in self.robot_map.keys():      # ensure singletons exist
            find(rid)

        groups = {}
        for rid in self.robot_map:
            groups.setdefault(find(rid), set()).add(rid)

        # optional random order for diversity
        group_list = list(groups.values())
        random.shuffle(group_list)

        # ---------- 3. optimal-mixing loop ------------------------------
        current = copy.deepcopy(receiver)
        f_current = self.fitness(current)

        for g in group_list:
            trial = copy.deepcopy(current)
            for rid in g:
                trial[rid] = copy.deepcopy(donor[rid])

            # fast repair (no conflict scheduler here – scheduler inside fitness)
            trial = self.repair_individual(trial, conflict_repair=False)
            f_trial = self.fitness(trial)

            if f_trial < f_current:            # improvement-only acceptance
                current, f_current = trial, f_trial

        return current

    def run(self):
        for gen in range(self.generations):
            # 1. create new solution & hill‑climb
            indiv = self.generate_individual()
            #print("Before FICH : ", self.fitness(indiv))
            indiv = self.first_improvement_hill_climber(indiv, 5)
            #print("After FICH : ", self.fitness(indiv))
            
            #indiv = self.local_search(indiv, self.local_steps)

            # 2. climb the pyramid via improvement‑only crossovers
            for population in self.pyramid:
                partner = random.choice(population)
                child = self.crossover(indiv, partner)
                child = self.first_improvement_hill_climber(child, self.local_steps // 4)
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
