from typing import List, Dict, Tuple, Optional, Union
from quantum.utils.coordinates import to_matrix_rc, to_robotics_xy


class RobotConfig:
    """Configuration for a single robot in a multi-robot scenario."""

    def __init__(self, robot_id: str, start: Union[Tuple[int, int], int],
                 goal: Union[Tuple[int, int], int], start_time: int = 0,
                 priority: float = 1.0, safety_radius: float = 0.5, expected_duration: Optional[int] = None,
                 coordinate_format: str = "matrix"):
        """
        Initialize robot configuration.

        Args:
            robot_id: Unique identifier for the robot
            start: Start position (grid coords or node index)
            goal: Goal position (grid coords or node index)
            priority: Priority weight for this robot (higher = more important)
            safety_radius: Safety radius for collision avoidance
            coordinate_format: "matrix" (Spooky's native (row, col), the default) or
                "cartesian" (robotics/Y-up (x, y)) — the format `start`/`goal` are
                given in. Internally everything still ends up stored as (row, col);
                see resolve_coordinates(). Node-index positions (graph mode, plain
                ints) are never affected regardless of this setting.
        """
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.priority = priority
        self.safety_radius = safety_radius
        self.start_time = start_time
        self.T = expected_duration  # By default is none and can be calculated by the problem, but you can predefine it
        self.coordinate_format = coordinate_format
        self.num_rows = None  # set by resolve_coordinates(); also doubles as the "already resolved" signal

        # Dynamic state tracking
        self.current_position = start
        self.path = []
        self.active = True  # Whether robot is actively planning

    def resolve_coordinates(self, num_rows: int):
        """
        One-time input conversion: turn start/goal/current_position from
        coordinate_format into Spooky's native matrix (row, col), now that the
        grid's row count is known. No-op if already resolved or already matrix.
        Positions given as plain ints (graph node indices) are left untouched.
        """
        if self.num_rows is not None:
            return
        if self.coordinate_format == "cartesian":
            if isinstance(self.start, (tuple, list)):
                self.start = to_matrix_rc(*self.start, num_rows)
            if isinstance(self.goal, (tuple, list)):
                self.goal = to_matrix_rc(*self.goal, num_rows)
            if isinstance(self.current_position, (tuple, list)):
                self.current_position = to_matrix_rc(*self.current_position, num_rows)
        self.num_rows = num_rows

    def format_position(self, pos):
        """
        Output conversion: given one of this robot's native matrix (row, col)
        positions, return it in this robot's coordinate_format. Read live off
        coordinate_format every call — not gated by resolve_coordinates.
        """
        if self.coordinate_format == "cartesian" and isinstance(pos, (tuple, list)):
            return to_robotics_xy(pos[0], pos[1], self.num_rows)
        return pos

    def is_at_goal(self) -> bool:
        """Check if robot has reached its goal."""
        return self.current_position == self.goal

    def to_dict(self) -> Dict:
        """Convert to dictionary representation, formatted per coordinate_format."""
        return {
            'robot_id': self.robot_id,
            'start': self.format_position(self.start),
            'goal': self.format_position(self.goal),
            'priority': self.priority,
            'safety_radius': self.safety_radius,
            'current_position': self.format_position(self.current_position),
            'path': [(*self.format_position((i, j)), t) for i, j, t in self.path],
            'active': self.active
        }
