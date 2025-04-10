# robot.py

class Robot:
    def __init__(self, robot_id, start, targets, priority=0):
        """
        :param robot_id: Unique identifier for the robot
        :param start: Tuple (x, y) representing the starting location
        :param targets: List of tuples [(x1, y1), (x2, y2), ...] representing goals
        :param priority: Integer priority (lower value = higher priority)
        """
        self.robot_id = robot_id
        self.start = start
        self.targets = targets
        self.priority = priority

    def __repr__(self):
        return f"Robot({self.robot_id}, start={self.start}, targets={self.targets}, priority={self.priority})"