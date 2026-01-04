import os
import random
import yaml
from typing import List, Tuple

def parse_octile_map(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    width = int(lines[2].split()[1])
    height = int(lines[1].split()[1])
    map_start = next(i for i, line in enumerate(lines) if line.strip() == "map") + 1

    obstacles = []
    free_cells = []

    for y in range(height):
        line = lines[map_start + y].strip()
        for x in range(width):
            if line[x] == 'T':
                obstacles.append([x, y])
            else:
                free_cells.append((x, y))

    return {
        'dimensions': [width, height],
        'obstacles': obstacles,
        'free_cells': free_cells
    }

def generate_scenario(map_data: dict, num_agents: int, targets_per_agent: int) -> dict:
    free_cells = map_data['free_cells']
    required_cells = num_agents * (1 + targets_per_agent)

    if len(free_cells) < required_cells:
        raise ValueError(f"Not enough free cells. Need {required_cells}, available {len(free_cells)}")

    available_cells = list(free_cells)
    random.shuffle(available_cells)

    scenario = {
        'map': {
            'dimensions': map_data['dimensions'],
            'obstacles': map_data['obstacles']
        },
        'agents': []
    }

    idx = 0
    for agent_id in range(num_agents):
        start = available_cells[idx]
        idx += 1

        targets = []
        for _ in range(targets_per_agent):
            targets.append(list(available_cells[idx]))
            idx += 1

        scenario['agents'].append({
            'name': f'agent{agent_id}',
            'start': list(start),
            'goals': targets
        })

    return scenario

def save_scenario(scenario: dict, output_file: str):
    with open(output_file, 'w') as f:
        yaml.dump(scenario, f, sort_keys=False, default_flow_style=None)

if __name__ == "__main__":
    CONFIG = {
        'map_file': 'room-64-64-8.map',
        'output_folder': 'maze_64-64',
        'test_cases': [
            (5, 3),  (5, 6), (5, 9), (5, 12),
            (10, 3), (10, 6), (10, 9), (10, 12),
            (25, 3), (25, 6), (25, 9), (25, 12),
            (50, 3), (50, 6), (50, 9), (50, 12),
        ],
        'variations_per_case': 5,
        'seed': 42  # Optional: set to None for non-reproducible randomness
    }

    os.makedirs(CONFIG['output_folder'], exist_ok=True)

    try:
        print("Parsing map...")
        map_data = parse_octile_map(CONFIG['map_file'])
        map_data['free_cells'] = [tuple(pos) for pos in map_data['free_cells']]

        random.seed(CONFIG['seed'])

        for (num_agents, targets_per_agent) in CONFIG['test_cases']:
            for version in range(CONFIG['variations_per_case']):
                try:
                    scenario = generate_scenario(map_data, num_agents, targets_per_agent)
                    filename = f"warehouse_agents{num_agents}_targets{targets_per_agent}_v{version + 1}.yaml"
                    output_path = os.path.join(CONFIG['output_folder'], filename)
                    save_scenario(scenario, output_path)

                    print(f"✅ Generated: {filename}")
                except Exception as e:
                    print(f"❌ Failed: agents={num_agents}, targets={targets_per_agent}, v={version + 1}")
                    print(f"Reason: {e}")

        print("✅ All scenarios generated.")
    except Exception as e:
        print(f"Fatal Error: {e}")
