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

class P3_Base(GA):

    def __init__(
        self,
        graph,
        robots,
        generations: int = 200,
        local_steps: int = 25,
    ) -> None:
        super().__init__(graph, robots, generations, local_steps)

        # Pyramid holds all individuals generated so far
        self.pyramid: List[List[Dict[int, List[int]]]] = []

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
        repair_best = self.repair_individual(self.best_individual, conflict_repair=False)
        return self.best_individual, self.best_history, self.avg_history, self.conf_history
