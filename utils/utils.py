import os
import random
from utils.robot import Robot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils.map_graph import MapfGraph
import networkx as nx
import yaml

def load_map(map_file):
    with open(map_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Parse map format (expects: type, height, width, map header)
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid_start_index = lines.index("map") + 1
    grid = [list(line) for line in lines[grid_start_index:]]

    return grid, width, height

def generate_new_scenario(original_scenario_file, output_scenario_file, robot_count, min_goals=2, max_goals=5):
    """
    Generate a new MAPF scenario file with multiple goals per robot.
    
    :param original_scenario_file: Path to the input scenario file.
    :param output_scenario_file: Path to save the new scenario file.
    :param robot_count: Number of robots to include in the generated scenario.
    :param min_goals: Minimum number of goals per robot.
    :param max_goals: Maximum number of goals per robot.
    """
    robots = {}
    goal_positions = []

    with open(original_scenario_file, 'r') as f:
        lines = f.readlines()

    # Ensure output folder exists
    output_folder = os.path.dirname(output_scenario_file)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created directory: {output_folder}")

    header = lines[0] if lines[0].startswith("version") else None
    map_name, map_width, map_height = None, None, None 

    for line in lines:
        if line.startswith("version"):
            continue

        values = line.split()
        if map_name is None:
            map_name, map_width, map_height = values[1], values[2], values[3]

        robot_id = int(values[0])
        start_x, start_y = int(values[4]), int(values[5])
        goal_x, goal_y = int(values[6]), int(values[7])

        goal_positions.append((goal_x, goal_y))

        if robot_id not in robots and len(robots) < robot_count:
            robots[robot_id] = {"start": (start_x, start_y), "targets": [(goal_x, goal_y)]}
        elif robot_id in robots:
            robots[robot_id]["targets"].append((goal_x, goal_y))

    # Limit to requested number of robots
    selected_robots = dict(list(robots.items())[:robot_count])
    goal_positions = list(set(goal_positions))  # unique goals

    for robot_id, robot_data in selected_robots.items():
        current_goals = set(robot_data["targets"])
        num_goals = random.randint(min_goals, max_goals)

        available_goals = list(set(goal_positions) - current_goals)
        random.shuffle(available_goals)
        additional_goals = available_goals[:max(0, num_goals - len(current_goals))]

        robot_data["targets"].extend(additional_goals)

    # Write new scenario
    with open(output_scenario_file, "w") as f:
        if header:
            f.write(header)

        for robot_id, robot_data in selected_robots.items():
            start_x, start_y = robot_data["start"]
            for goal_x, goal_y in robot_data["targets"]:
                f.write(f"{robot_id} {map_name} {map_width} {map_height} {start_x} {start_y} {goal_x} {goal_y} 0.00000000\n")

    print(f"✅ New scenario file saved as: {output_scenario_file}")

def generate_multigoal_scenario_from_file(original_scenario_file, output_scenario_file, robot_count, num_goals_per_robot):
    """
    Generate a new MAPF scenario file with a fixed number of goals per robot,
    treating each line as a unique robot and ignoring robot_id in the file.

    :param original_scenario_file: Path to the input scenario file.
    :param output_scenario_file: Path to save the new scenario file.
    :param robot_count: Number of robots to include in the generated scenario.
    :param num_goals_per_robot: Exact number of goals to assign per robot.
    """
    agents = []
    goal_positions = []

    with open(original_scenario_file, 'r') as f:
        lines = f.readlines()

    # Ensure output folder exists
    output_folder = os.path.dirname(output_scenario_file)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created directory: {output_folder}")

    header = lines[0] if lines[0].startswith("version") else None
    map_name, map_width, map_height = None, None, None

    # Extract agent start-goal pairs and goal positions
    for line in lines:
        if line.startswith("version"):
            continue

        values = line.split()
        if map_name is None:
            map_name, map_width, map_height = values[1], values[2], values[3]

        start_x, start_y = int(values[4]), int(values[5])
        goal_x, goal_y = int(values[6]), int(values[7])

        agents.append({"start": (start_x, start_y), "original_goal": (goal_x, goal_y)})
        goal_positions.append((goal_x, goal_y))

    # Limit to requested number of agents
    selected_agents = agents[:robot_count]
    goal_pool = list(set(goal_positions))  # Unique goals

    # Assign multiple goals per agent
    for agent in selected_agents:
        assigned_goals = {agent["original_goal"]}
        available_goals = list(set(goal_pool) - assigned_goals)
        random.shuffle(available_goals)
        additional_goals = available_goals[:max(0, num_goals_per_robot - 1)]
        agent["goals"] = list(assigned_goals) + additional_goals

    # Write to output .scen file
    with open(output_scenario_file, "w") as f:
        if header:
            f.write(header)

        for i, agent in enumerate(selected_agents):
            start_x, start_y = agent["start"]
            for goal_x, goal_y in agent["goals"]:
                f.write(f"{i} {map_name} {map_width} {map_height} {start_x} {start_y} {goal_x} {goal_y} 0.00000000\n")

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

            # Priority is inverse of robot_id (or assign differently if needed)
            if not any(r.robot_id == robot_id for r in robot_list):
                priority = robot_id  # lower robot_id = higher priority
                robot_list.append(Robot(robot_id, start, [], priority=priority))

            for r in robot_list:
                if r.robot_id == robot_id:
                    r.targets.append(goal)

    return robot_list

def load_map_and_robots(yaml_file_path: str):
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    # Validate map section
    if not isinstance(data, dict) or 'map' not in data or not isinstance(data['map'], dict):
        raise ValueError(f"Invalid or missing 'map' section in {yaml_file_path}")

    # Load dimensions
    dimensions = tuple(data['map'].get('dimensions', []))
    if not dimensions:
        raise ValueError(f"Missing or empty 'dimensions' in 'map' section of {yaml_file_path}")

    # Load obstacles safely
    obstacles_raw = data['map'].get('obstacles') or []
    if not isinstance(obstacles_raw, list):
        raise ValueError(f"'obstacles' must be a list in {yaml_file_path}")

    obstacles = [tuple(reversed(o)) for o in obstacles_raw]

    graph = MapfGraph(dimensions, obstacles)

    # Validate agents section
    agents_data = data.get('agents')
    if not isinstance(agents_data, list):
        raise ValueError(f"Missing or invalid 'agents' section in {yaml_file_path}")

    # Load robots
    robots = []
    for idx, agent in enumerate(agents_data):
        start = tuple(reversed(agent['start']))
        goals = [tuple(reversed(g)) for g in agent.get('goals', [])]
        robots.append(Robot(robot_id=idx, start=start, targets=goals, priority=idx))

    return graph, robots



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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_solution(graph, robots, solution, frame_interval=500):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(graph.width))
    ax.set_yticks(np.arange(graph.height))
    ax.set_xticks(np.arange(-0.5, graph.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, graph.height, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Draw static walls
    grid_display = np.zeros((graph.height, graph.width))
    for y in range(graph.height):
        for x in range(graph.width):
            if graph.grid[y][x] == '@':
                grid_display[y, x] = 1
    ax.imshow(grid_display, cmap="Greys", origin="upper")

    # Prepare colors
    colors = plt.cm.get_cmap("tab10", len(robots))
    robot_circles = {}
    robot_paths = {}
    initial_marker_states = {}
    for robot in robots:
        color = colors(robot.robot_id % 10)
        circle = plt.Circle((0, 0), 0.3, color=color, zorder=3)
        ax.add_patch(circle)
        robot_circles[robot.robot_id] = circle
        robot_paths[robot.robot_id] = solution[robot.robot_id]

        initial_marker_states[robot.robot_id] = []
        for tx, ty in reversed(robot.targets):
            marker, = ax.plot(tx, ty, 'X', color=color, markersize=15, zorder=2)
            initial_marker_states[robot.robot_id].append({'pos': (tx, ty), 'marker': marker, 'visible': True})

    marker_states = {rid: [dict(state) for state in states] for rid, states in initial_marker_states.items()}

    max_frames = max(len(p) for p in robot_paths.values())

    def interpolate(p0, p1, alpha):
        return (1 - alpha) * p0[0] + alpha * p1[0], (1 - alpha) * p0[1] + alpha * p1[1]

    def reset_markers():
        for rid, states in initial_marker_states.items():
            for i, state in enumerate(states):
                marker_states[rid][i]['visible'] = True
                marker_states[rid][i]['marker'].set_visible(True)

    def update(frame):
        if frame == 0:
            reset_markers()

        alpha = frame % 10 / 10.0
        tick = frame // 10

        for robot in robots:
            rid = robot.robot_id
            path = robot_paths[rid]

            # Move robot
            if tick < len(path) - 1:
                pos1, pos2 = path[tick], path[tick + 1]
                x, y = interpolate(pos1, pos2, alpha)
                robot_circles[rid].center = (x, y)
            elif tick < len(path):
                robot_circles[rid].center = path[tick]

            # Update marker visibility
            for state in marker_states[rid]:
                if tick < len(path) and path[tick] == state['pos']:
                    state['visible'] = False
                state['marker'].set_visible(state['visible'])

        elements = list(robot_circles.values())
        for states in marker_states.values():
            elements += [s['marker'] for s in states if s['visible']]

        return elements

    ani = animation.FuncAnimation(
        fig, update, frames=max_frames * 10, interval=frame_interval // 10, blit=True, repeat=True
    )

    ax.set_title("Smooth Multi-Agent Path Planning Visualization")
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
