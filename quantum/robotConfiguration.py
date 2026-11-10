from typing import List, Dict, Tuple, Optional, Union
from quantum.utils.coordinates import to_matrix_rc, to_robotics_xy, world_to_grid_cell, grid_cell_to_world


class RobotConfig:
    """Configuration for a single robot in a multi-robot scenario."""

    def __init__(self, robot_id: str, start: Union[Tuple[int, int], int],
                 goal: Union[Tuple[int, int], int], start_time: int = 0,
                 priority: float = 1.0, robot_radius: float = 0.5, inflation: float = 0.0,
                 expected_duration: Optional[int] = None,
                 coordinate_format: str = "matrix",
                 fixed_path: Optional[List] = None):
        """
        Initialize robot configuration.

        Args:
            robot_id: Unique identifier for the robot
            start: Start position (grid coords or node index)
            goal: Goal position (grid coords or node index)
            priority: Priority weight for this robot (higher = more important)
            robot_radius: Physical body radius, metres (largest half-extent for
                a non-circular robot). Two such robots touch at centre-to-centre
                distance robot_radius_a + robot_radius_b.
            inflation: Safety margin, metres — the minimum gap the planner keeps
                between this robot's edge and any obstacle or other robot's
                edge. See quantum/utils/clearance.py for how the two combine
                (additive vs obstacles, max-of-the-pair vs another robot).
            coordinate_format: "matrix" (Spooky's native (row, col), the default),
                "cartesian" (robotics/Y-up (x, y)), or "world" (real-world meters,
                ROS map frame — requires the map to carry an `origin`/`resolution`,
                see quantum/maps/pgm2HDF5.py) — the format `start`/`goal` are given
                in. Internally everything still ends up stored as (row, col); see
                resolve_coordinates(). Node-index positions (graph mode, plain
                ints) are never affected regardless of this setting.
            fixed_path: Committed positions, one per step from start_time, in
                coordinate_format; fixed_path[0] must be the start. The planner
                keeps them, plans on from the last one and plans every other
                robot around them. Ending at the goal fixes the robot for the
                whole plan (parked or already-committed robot). See
                apply_fixed_path().
        """
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.priority = priority
        self.robot_radius = robot_radius
        self.inflation = inflation
        self.start_time = start_time
        self.T = expected_duration  # By default is none and can be calculated by the problem, but you can predefine it
        self.coordinate_format = coordinate_format
        self.fixed_path = list(fixed_path) if fixed_path else None
        # Set by apply_fixed_path(): the committed positions (start..last)
        # and the time the first of them is held.
        self.fixed_prefix = None
        self.prefix_start_time = start_time
        self.num_rows = None  # set by resolve_coordinates(); also doubles as the "already resolved" signal
        self.origin = None  # set by resolve_coordinates() when coordinate_format == "world"
        self.resolution = None  # set by resolve_coordinates() when coordinate_format == "world"

        # Dynamic state tracking
        self.current_position = start
        self.path = []
        self.active = True  # Whether robot is actively planning

    @property
    def clearance(self) -> float:
        """Per-robot keep-out from obstacles (metres): body radius plus safety
        margin. Derived, never stored. NB robot-vs-robot separation is not
        clearance_a + clearance_b — see quantum/utils/clearance.py."""
        return self.robot_radius + self.inflation

    def resolve_coordinates(self, num_rows: int, origin=None, resolution=None):
        """
        One-time input conversion: turn start/goal/current_position from
        coordinate_format into Spooky's native matrix (row, col), now that the
        grid's row count (and, for "world", its origin/resolution) is known.
        No-op if already resolved or already matrix. Positions given as plain
        ints (graph node indices) are left untouched.
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
            if self.fixed_path:
                self.fixed_path = [
                    to_matrix_rc(*p, num_rows) if isinstance(p, (tuple, list)) else p
                    for p in self.fixed_path
                ]
        elif self.coordinate_format == "world":
            if origin is None or resolution is None:
                raise ValueError(
                    f"Robot '{self.robot_id}' uses coordinate_format='world' but no "
                    f"origin/resolution was provided — this map has no real-world frame."
                )
            if isinstance(self.start, (tuple, list)):
                self.start = world_to_grid_cell(*self.start[:2], num_rows, origin, resolution)
            if isinstance(self.goal, (tuple, list)):
                self.goal = world_to_grid_cell(*self.goal[:2], num_rows, origin, resolution)
            if isinstance(self.current_position, (tuple, list)):
                self.current_position = world_to_grid_cell(*self.current_position[:2], num_rows, origin, resolution)
            if self.fixed_path:
                self.fixed_path = [
                    world_to_grid_cell(*p[:2], num_rows, origin, resolution)
                    if isinstance(p, (tuple, list)) else p
                    for p in self.fixed_path
                ]
            self.origin = origin
            self.resolution = resolution

        # Spooky's native grid indices must be plain ints (used as array/dict
        # keys downstream). An HTTP caller's JSON numbers arrive as Python
        # floats even for exact integer cells — e.g. RobotSpec.start is
        # list[float] so "world"'s fractional meters are accepted too — so
        # normalize here regardless of coordinate_format, including the
        # "matrix" pass-through case above. Node-index positions (graph mode,
        # plain ints) aren't tuple/list and pass through untouched.
        self.start = self._to_int_cell(self.start)
        self.goal = self._to_int_cell(self.goal)
        self.current_position = self._to_int_cell(self.current_position)
        if self.fixed_path:
            self.fixed_path = [self._to_int_cell(p) for p in self.fixed_path]
        self.num_rows = num_rows

    def apply_fixed_path(self):
        """Turn fixed_path into a planning start: the robot is planned from
        its last committed position at start_time + len - 1, with the
        committed positions kept in fixed_prefix. Idempotent; call after
        resolve_coordinates()."""
        if not self.fixed_path or self.fixed_prefix is not None:
            return
        if self.fixed_path[0] != self.start:
            raise ValueError(
                f"Robot '{self.robot_id}' fixed_path starts at {self.fixed_path[0]}, "
                f"not at its start {self.start}."
            )
        self.fixed_prefix = list(self.fixed_path)
        self.prefix_start_time = self.start_time
        self.start = self.current_position = self.fixed_prefix[-1]
        self.start_time += len(self.fixed_prefix) - 1
        self.active = not self.fully_fixed  # nothing left to plan

    @property
    def fully_fixed(self) -> bool:
        """Committed all the way to its goal: never re-planned, holds it."""
        return bool(self.fixed_prefix) and self.fixed_prefix[-1] == self.goal

    @staticmethod
    def _to_int_cell(pos):
        if isinstance(pos, (tuple, list)):
            return tuple(int(round(v)) for v in pos)
        return pos

    def format_position(self, pos):
        """
        Output conversion: given one of this robot's native matrix (row, col)
        positions, return it in this robot's coordinate_format. Read live off
        coordinate_format every call — not gated by resolve_coordinates.
        """
        if not isinstance(pos, (tuple, list)):
            return pos
        if self.coordinate_format == "cartesian":
            return to_robotics_xy(pos[0], pos[1], self.num_rows)
        if self.coordinate_format == "world":
            return grid_cell_to_world(pos[0], pos[1], self.num_rows, self.origin, self.resolution)
        return pos

    def is_at_goal(self) -> bool:
        """Check if robot has reached its goal."""
        return self.current_position == self.goal

    def reset(self) -> None:
        """Restore dynamic state to the robot's initial start position."""
        self.path = []
        self.current_position = self.start
        self.active = not self.fully_fixed

    def to_dict(self) -> Dict:
        """Convert to dictionary representation, formatted per coordinate_format."""
        return {
            'robot_id': self.robot_id,
            'start': self.format_position(self.start),
            'goal': self.format_position(self.goal),
            'priority': self.priority,
            'robot_radius': self.robot_radius,
            'inflation': self.inflation,
            'current_position': self.format_position(self.current_position),
            'path': [(*self.format_position((i, j)), t) for i, j, t in self.path],
            'active': self.active
        }
