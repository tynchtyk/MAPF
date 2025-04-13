import yaml
from utils import generate_new_scenario,generate_multigoal_scenario_from_file

def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def main():
    config = load_config()
    #generate_new_scenario(config["scenario_file"], config["generated_scenario_file"], 5, config["min_goals"], config["max_goals"])
    generate_multigoal_scenario_from_file(config["scenario_file"], config["generated_scenario_file"], config["robot_count"], config["goal_count"])
    
if __name__ == "__main__":
    main()
