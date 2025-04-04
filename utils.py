import os
import random
from robot import Robot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx

def load_map(map_file):
    with open(map_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Parse map format (expects: type, height, width, map header)
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid_start_index = lines.index("map") + 1
    grid = [list(line) for line in lines[grid_start_index:]]

    return grid, width, height

def generate_new_scenario(original_scenario_file, output_scenario_file, min_goals=2, max_goals=5):
    """
    Generate a new MAPF scenario file with multiple goals per robot.
    
    :param original_scenario_file: Path to the input scenario file.
    :param output_scenario_file: Path to save the new scenario file.
    :param min_goals: Minimum number of goals per robot.
    :param max_goals: Maximum number of goals per robot.
    """
    robots = {}
    goal_positions = []  # Stores all goal positions for random selection

    with open(original_scenario_file, 'r') as f:
        lines = f.readlines()

    # Ensure output folder exists
    output_folder = os.path.dirname(output_scenario_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created directory: {output_folder}")

    header = lines[0] if lines[0].startswith("version") else None
    map_name, map_width, map_height = None, None, None 
    scenario_data = []

    for line in lines:
        if line.startswith("version"):  # Skip header
            continue

        # Extract values
        values = line.split()
        if map_name is None:
            map_name = values[1]
            map_width = values[2]
            map_height = values[3]
        robot_id = int(values[0])  # Unique robot ID
        start_x, start_y = int(values[4]), int(values[5])
        goal_x, goal_y = int(values[6]), int(values[7])

        # Add goal position to the global pool
        goal_positions.append((goal_x, goal_y))

        # Store robot start position
        if robot_id not in robots:
            robots[robot_id] = {"start": (start_x, start_y), "targets": []}

        # Assign the original goal
        robots[robot_id]["targets"].append((goal_x, goal_y))

    # Ensure goal_positions are unique before selecting additional goals
    goal_positions = list(set(goal_positions))

    # Assign additional random goals to each robot
    for robot_id, robot_data in robots.items():
        num_goals = random.randint(min_goals, max_goals)

        # Ensure at least all original goals are used and distributed
        additional_goals = random.sample(goal_positions, min(num_goals - 1, len(goal_positions)))
        robot_data["targets"].extend(additional_goals)

    # Generate new scenario content
    with open(output_scenario_file, "w") as f:
        if header:
            f.write(header)  # Write the "version 1" header if it exists

        for robot_id, robot_data in robots.items():
            start_x, start_y = robot_data["start"]

            for goal_x, goal_y in robot_data["targets"]:
                # Write scenario in original format with default map name, width, height
                f.write(f"{robot_id} {map_name} {map_width} {map_height} {start_x} {start_y} {goal_x} {goal_y} 0.00000000\n")

    print(f"✅ New scenario file saved as: {output_scenario_file}")

def load_scenario(output_scenario_file):
    robot_list = []
    with open(output_scenario_file, 'r') as f:
        for line in f:
            if line.startswith("version"):
                continue
            values = line.split()
            robot_id = int(values[0])
            start = (int(values[4]), int(values[5]))
            goal = (int(values[6]), int(values[7]))

            if not any(r.robot_id == robot_id for r in robot_list):
                robot_list.append(Robot(robot_id, start, []))
            for r in robot_list:
                if r.robot_id == robot_id:
                    r.targets.append(goal)

    return robot_list

def expand_solution_paths(graph, solution):
    """Ensure each step between positions is one cell apart."""
    expanded = {}
    for robot_id, path in solution.items():
        full_path = []
        for i in range(len(path) - 1):
            segment = nx.shortest_path(graph.G, path[i], path[i+1])
            if i > 0:
                segment = segment[1:]  # Avoid duplicate
            full_path.extend(segment)
        expanded[robot_id] = full_path
    return expanded

def visualize_solution(graph, robots, solution, frame_interval=500):
    """
    Visualizes robot movement on a map, ensuring each move is one cell per frame.
    Targets disappear once reached by robots.
    """
    # Expand all solution paths so that each step is 1 cell
    solution = expand_solution_paths(graph, solution)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(graph.width))
    ax.set_yticks(np.arange(graph.height))
    ax.set_xticks(np.arange(-0.5, graph.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, graph.height, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Walls
    grid_display = np.zeros((graph.height, graph.width))
    for y in range(graph.height):
        for x in range(graph.width):
            if graph.grid[y][x] == '@':
                grid_display[y, x] = 1
    ax.imshow(grid_display, cmap="Greys", origin="upper")

    # Colors
    colors = plt.cm.get_cmap("tab10", len(robots))
    robot_patches = {}

    # Goal markers
    target_patches = {}
    for robot in robots:
        color = colors(robot.robot_id % 10)
        patches = []
        for gx, gy in robot.targets:
            patch, = ax.plot(gx, gy, 'X', color=color, markersize=20, zorder=2)
            patches.append(((gx, gy), patch))
        target_patches[robot.robot_id] = patches

    # Robot dot markers
    for robot in robots:
        color = colors(robot.robot_id % 10)
        patch, = ax.plot([], [], 'o', color=color, markersize=20, zorder=3)
        robot_patches[robot.robot_id] = patch

    # Determine max length
    max_time = max(len(p) for p in solution.values())

    def update(frame):
        for robot in robots:
            path = solution[robot.robot_id]
            if frame < len(path):
                x, y = path[frame]
                robot_patches[robot.robot_id].set_data(x, y)

                # Check and remove targets if reached
                new_patches = []
                for (tx, ty), patch in target_patches[robot.robot_id]:
                    if (tx, ty) == (x, y):
                        patch.remove()
                    else:
                        new_patches.append(((tx, ty), patch))
                target_patches[robot.robot_id] = new_patches

        return list(robot_patches.values())

    ani = animation.FuncAnimation(
        fig, update, frames=max_time, interval=frame_interval, blit=True
    )

    ax.set_title("Multi-Agent Path Planning (One Step per Frame)")
    ax.set_xlim(-0.5, graph.width - 0.5)
    ax.set_ylim(-0.5, graph.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()

def show_graph_structure(graph):
    """
    Visualizes the underlying NetworkX graph of the map.
    Passable cells are nodes, edges show valid moves.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(graph.width))
    ax.set_yticks(np.arange(graph.height))
    ax.set_xticks(np.arange(-0.5, graph.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, graph.height, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Walls
    grid_display = np.zeros((graph.height, graph.width))
    for y in range(graph.height):
        for x in range(graph.width):
            if graph.grid[y][x] == '@':
                grid_display[y][x] = 1
    ax.imshow(grid_display, cmap="Greys", origin="upper")

    ax.set_title("Map + Graph")
    ax.set_xlim(-0.5, graph.width - 0.5)
    ax.set_ylim(-0.5, graph.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()

def show_graph_with_robots(graph, robots):
    """
    Visualizes:
    - The map (walls and free cells)
    - Robot start positions (circles)
    - Robot goals (X markers)
    Fully aligned with imshow-style visualization (origin='upper').
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(graph.width))
    ax.set_yticks(np.arange(graph.height))
    ax.set_xticks(np.arange(-0.5, graph.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, graph.height, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Walls
    grid_display = np.zeros((graph.height, graph.width))
    for y in range(graph.height):
        for x in range(graph.width):
            if graph.grid[y][x] == '@':
                grid_display[y][x] = 1
    ax.imshow(grid_display, cmap="Greys", origin="upper")

    colors = plt.cm.get_cmap("tab10", len(robots))

    # Robots and goals
    for robot in robots:
        color = colors(robot.robot_id % 10)
        sx, sy = robot.start
        ax.plot(sx, sy, 'o', color=color, markersize=20, label=f"R{robot.robot_id}", zorder=3)
        for gx, gy in robot.targets:
            ax.plot(gx, gy, 'X', color=color, markersize=20, zorder=3)

    ax.set_title("Map + Robots Start & Goals")
    ax.set_xlim(-0.5, graph.width - 0.5)
    ax.set_ylim(-0.5, graph.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))
    plt.show()
