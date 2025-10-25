from typing import List, Dict, Tuple, Optional, Union


class RobotConfig:
    """Configuration for a single robot in a multi-robot scenario."""
    
    def __init__(self, robot_id: str, start: Union[Tuple[int, int], int],
                 goal: Union[Tuple[int, int], int], start_time: int = 0,
                 priority: float = 1.0, safety_radius: float = 0.5, expected_duration: Optional[int] = None):
        """
        Initialize robot configuration.
        
        Args:
            robot_id: Unique identifier for the robot
            start: Start position (grid coords or node index)
            goal: Goal position (grid coords or node index)
            priority: Priority weight for this robot (higher = more important)
            safety_radius: Safety radius for collision avoidance
        """
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.priority = priority
        self.safety_radius = safety_radius
        self.start_time = start_time
        self.T = expected_duration  # By default is none and can be calculated by the problem, but you can predefine it

        # Dynamic state tracking
        self.current_position = start
        self.path = []
        self.goal_reached = False
        self.active = True  # Whether robot is actively planning
        
    def is_at_goal(self) -> bool:
        """Check if robot has reached its goal."""
        return self.current_position == self.goal
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'robot_id': self.robot_id,
            'start': self.start,
            'goal': self.goal,
            'priority': self.priority,
            'safety_radius': self.safety_radius,
            'current_position': self.current_position,
            'goal_reached': self.goal_reached,
            'active': self.active
        }
