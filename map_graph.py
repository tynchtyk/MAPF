import networkx as nx
from utils import load_map


class MapfGraph:
    def __init__(self, map_file):
        self.grid, self.width, self.height = load_map(map_file)
        self.G = nx.Graph()
        self.build_graph()

    def build_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == '.':  # Passable tile
                    self.G.add_node((x, y))
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx_, ny_ = x + dx, y + dy
                        if (
                            0 <= nx_ < self.width and
                            0 <= ny_ < self.height and
                            self.grid[ny_][nx_] == '.'
                        ):
                            self.G.add_edge((x, y), (nx_, ny_))

    def shortest_path(self, start, goal):
        try:
            return nx.shortest_path(self.G, source=start, target=goal)
        except nx.NetworkXNoPath:
            return []

    def shortest_path_length(self, start, goal):
        try:
            return nx.shortest_path_length(self.G, source=start, target=goal)
        except nx.NetworkXNoPath:
            return float('inf')
