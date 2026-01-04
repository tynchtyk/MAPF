import networkx as nx

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class MapfGraph:
    def __init__(self, dimensions, obstacles):
        self.height, self.width = dimensions
        self.obstacles = set(map(tuple, obstacles))
        self.grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.obstacles:
            self.grid[y][x] = '@'
        self.G = nx.Graph()
        self.build_graph()
        self._precomputed_paths = None
        self._precomputed_lengths = None
        #self.precompute_all_pairs_astar()

    def build_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == '.':
                    self.G.add_node((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < self.width and 0 <= ny_ < self.height and self.grid[ny_][nx_] == '.':
                            self.G.add_edge((x, y), (nx_, ny_))

    def precompute_all_pairs_astar(self):
        print("Precomputing all-pairs A* shortest paths...")
        self._precomputed_paths = {}
        self._precomputed_lengths = {}

        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            self._precomputed_paths[u] = {}
            self._precomputed_lengths[u] = {}
            for v in nodes:
                if u == v:
                    self._precomputed_paths[u][v] = [u]
                    self._precomputed_lengths[u][v] = 0
                    continue
                try:
                    path = nx.astar_path(self.G, u, v, heuristic=manhattan_distance)
                    length = len(path) - 1
                    self._precomputed_paths[u][v] = path
                    self._precomputed_lengths[u][v] = length
                except nx.NetworkXNoPath:
                    self._precomputed_paths[u][v] = []
                    self._precomputed_lengths[u][v] = float('inf')
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(nodes)} rows computed")
        print("Precomputation done.")

    def _get_precomputed_path(self, start, goal):
        if self._precomputed_paths is None:
            return self.shortest_path(start, goal)
        return self._precomputed_paths.get(tuple(start), {}).get(tuple(goal), [])

    def _get_precomputed_length(self, start, goal):
        if self._precomputed_lengths is None:
            return self.shortest_path_length(start, goal)
        return self._precomputed_lengths.get(tuple(start), {}).get(tuple(goal), float('inf'))

    def shortest_path(self, start, goal):
        try:
            return nx.astar_path(self.G, source=tuple(start), target=tuple(goal), heuristic=manhattan_distance)
        except nx.NetworkXNoPath:
            return []

    def shortest_path_length(self, start, goal):
        try:
            return nx.astar_path_length(self.G, source=tuple(start), target=tuple(goal), heuristic=manhattan_distance)
        except nx.NetworkXNoPath:
            return float('inf')
