import networkx as nx

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class MapfGraph:
    def __init__(self, dimensions, obstacles):
        self.height, self.width = dimensions
        self.obstacles = set(map(tuple, obstacles))
        self.grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.obstacles:
            self.grid[y][x] = '@'  # Match format of old ASCII maps
        self.G = nx.Graph()
        self.build_graph()

    def build_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == '.':  # Passable tile
                    self.G.add_node((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if (
                            0 <= nx_ < self.width and
                            0 <= ny_ < self.height and
                            self.grid[ny_][nx_] == '.'
                        ):
                            self.G.add_edge((x, y), (nx_, ny_))

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

