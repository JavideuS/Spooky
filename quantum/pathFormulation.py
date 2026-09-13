import quantum.map as map
import numpy as np
from quantum.robotConfiguration import RobotConfig
from quantum.utils.logger import get_logger


class InfeasibleProblemError(ValueError):
    """A robot's start/goal is out of bounds, on an obstacle, an unknown
    graph node, or leaves it no time to plan within the given horizon."""


class PathfindingProblem:
    def __init__(
        self,
        robots,
        grid=None,
        graph=None,
        T=None,
        name="unnamed",
        separation_factor=None,
        clearance_enabled=True,
    ):
        # Support both grid and graph formats
        self.logger = get_logger()
        self.grid = grid
        self.graph = graph

        # Clearance config (see quantum/utils/clearance.py). One feature, one
        # switch: obstacle keep-out and robot-robot separation both derive from
        # each robot's (robot_radius, inflation); they only compose differently.
        #   separation_factor: multiplier on the max-inflation term of
        #     robot-robot separation; None => clearance module default (1.0).
        #   clearance_enabled: when False, get_clearance_table()
        #     and get_obstacle_keepout() return empty and _validate_clearance_
        #     config() is skipped. In this case the problem is planned as pure MAPF
        #     even if robot_radius / inflation are set (useful as a baseline, or when
        #     the map is already an inflated cost map).
        #
        #     At the defaults every clearance quantity is a no-op regardless, so this
        #     only matters once real radii are set.
        self.separation_factor = separation_factor
        self.clearance_enabled = clearance_enabled

        # Validate that at least one format is provided
        if self.grid is None and self.graph is None:
            raise ValueError("Either grid or graph must be provided")

        self.robots = {}
        if isinstance(robots, dict):
            self.robots = robots
        elif isinstance(robots, RobotConfig):
            self.robots[robots.robot_id] = robots
        elif isinstance(robots, list):
            for robot in robots:
                self.robots[robot.robot_id] = robot
        self.num_robots = len(self.robots)

        # Resolve each robot's own coordinate_format into native matrix (row, col)
        # now that the grid (if any) is known. Internally, everything downstream
        # (builders, solvers, adjacency checks) only ever sees matrix coordinates.
        num_rows = self.grid.M if self.grid is not None else None
        for robot in self.robots.values():
            if robot.coordinate_format in ("cartesian", "world") and num_rows is None:
                raise ValueError(
                    f"Robot '{robot.robot_id}' uses coordinate_format='{robot.coordinate_format}' "
                    f"but this problem has no grid to derive a row count from — "
                    f"{robot.coordinate_format} conversion requires a grid."
                )
            origin = self.grid.origin if self.grid is not None else None
            resolution = self.grid.resolution if self.grid is not None else None
            robot.resolve_coordinates(num_rows, origin=origin, resolution=resolution)

        self._validate_robot_positions()

        if T is None:
            T = self.calculate_timeline()
        else:
            # Set individual robot times if not already set, sized so start_time + T
            # still fits within the global horizon T for staggered starts.
            for robot in self.robots.values():
                if robot.T is None:
                    remaining = T - robot.start_time
                    if remaining <= 0:
                        raise InfeasibleProblemError(
                            f"Robot '{robot.robot_id}' start_time={robot.start_time} "
                            f"leaves no time to plan within horizon T={T}."
                        )
                    robot.T = remaining
        self.T = T
        self.T = T
        self.name = name
        # Lazily built and cached; invalidated whenever the robot set changes
        # (add_robot). See get_clearance_table() / get_obstacle_keepout().
        self._clearance_table = None
        self._obstacle_keepout = None
        self._validate_clearance_config()

    def get_clearance_table(self):
        """{(a_id, b_id): D_ab} robot-robot separation offset sets for every
        ordered pair of robots (see quantum/utils/clearance.py). Built once,
        then cached. {} when clearance_enabled is False, or in pure graph
        mode (self.grid is None): D_ab's offsets are Chebyshev grid-cell
        arithmetic against a physical resolution, and a graph's node `pos`
        isn't guaranteed to be a uniform grid, so clearance stays exact-only
        there rather than applying grid-cell math to whatever pos a graph
        happens to carry. A "both"-format problem (grid and graph present)
        still gets real clearance, from the grid's resolution. Otherwise every
        D_ab is {(0, 0)} (callers fall back to exact same-cell / swap checks)
        unless some pair needs a footprint bigger than one cell."""
        if not self.clearance_enabled or self.grid is None:
            return {}
        if self._clearance_table is None:
            from quantum.utils import clearance as _clr

            resolution = self.grid.resolution
            factor = (
                self.separation_factor
                if self.separation_factor is not None
                else _clr.DEFAULT_SEPARATION_FACTOR
            )
            self._clearance_table = _clr.build_offset_table(
                self.robots, resolution, factor
            )
        return self._clearance_table

    def get_obstacle_keepout(self):
        """{robot_id: frozenset[(i, j)]} -> grid cells a robot may not occupy as
        a leader because its footprint (clearance = robot_radius + inflation)
        would cover an obstacle.

        Robot's own *start* is exempted (it may be parked tight to a wall and
        must still be representable at t=0; this is the start-in-inflation warning
        case in _validate_clearance_config).

        Goal is not exempted and doesn't need to be, as a goal in the inflated
        band is already a hard InfeasibleProblemError at construction. Every
        set is empty when clearance_enabled is False, in graph mode, or for a
        radius-0 footprint (the default). Lazy + cached; invalidated by
        add_robot. Builders consume this their own way (CBS: subtract from
        legal_cells; ILP: fix x to 0; QUBO: mask reachable sets)."""
        if self._obstacle_keepout is None:
            self._obstacle_keepout = self._build_obstacle_keepout()
        return self._obstacle_keepout

    def _build_obstacle_keepout(self):
        if self.grid is None or not self.clearance_enabled:
            return {rid: frozenset() for rid in self.robots}
        from quantum.utils import clearance as _clr

        res = self.grid.resolution
        obstacles = {tuple(o) for o in self.grid.obstacles}
        M, N = self.grid.M, self.grid.N
        band_by_radius = {}  # radius_cells -> frozenset (shared across equal footprints)
        out = {}
        for rid, robot in self.robots.items():
            r = _clr.footprint_radius_cells(robot.clearance, res)
            if r == 0:
                out[rid] = frozenset()
                continue
            if r not in band_by_radius:
                band_by_radius[r] = (
                    _clr.inflated_obstacle_cells(obstacles, M, N, robot.clearance, res)
                    - obstacles
                )
            out[rid] = frozenset(band_by_radius[r] - {tuple(robot.start)})
        return out

    def _validate_clearance_config(self):
        """Clearance-aware feasibility checks, run once construction is far
        enough along that each robot's T (hence deadline) is known. No-op in
        graph mode or when clearance_enabled is False.

        - goal within its own clearance of an obstacle -> InfeasibleProblemError
          (a non-human local planner can't complete the final approach).
        - start within its own clearance of an obstacle -> warning only (the
          robot is physically there already and the planners exempt it).
        - two robots whose starts are within their combined separation AND
          share a start_time -> InfeasibleProblemError. A robot's start is
          hard-pinned to an exact cell at an exact instant.
          Two robots pinned at the same instant, too close, is a direct
          contradiction in every possible solution -- exactly as certain as the
          goal case below, just gated on shared start_time instead of shared
          deadline. Differing start_times are NOT checked, for the same
          reason differing deadlines aren't checked for goals below: whether
          one robot (say, already parked from an earlier start) still occupies
          that cell by the time the other starts depends on an arrival time not
          known before solving (left to the solver's no_solution).
        - two robots whose goals are within their combined separation ->
          InfeasibleProblemError. In an abstract MAPF sense a robot that
          arrives, waits, and despawns before the other arrives could make
          non-overlapping windows work, but clearance_enabled is the
          deployment switch and real robots sit at their goal for good, so
          this blocks unconditionally. (Turn clearance_enabled off for the
          despawn semantics.)
        """
        if self.grid is None or not self.clearance_enabled:
            return
        from quantum.utils import clearance as _clr

        res = self.grid.resolution
        obstacles = {tuple(o) for o in self.grid.obstacles}
        for robot in self.robots.values():
            if not isinstance(robot.goal, (tuple, list)):
                continue
            r = _clr.footprint_radius_cells(robot.clearance, res)
            if r == 0:
                continue
            if _clr.cell_in_obstacle_footprint(robot.goal, obstacles, r):
                raise InfeasibleProblemError(
                    f"Robot '{robot.robot_id}' goal {tuple(robot.goal)} is within its "
                    f"clearance ({robot.clearance} m) of an obstacle — no collision-free "
                    f"final approach exists at this resolution ({res} m/cell)."
                )
            if _clr.cell_in_obstacle_footprint(robot.start, obstacles, r):
                self.logger.standard(
                    f"⚠ Robot '{robot.robot_id}' start {tuple(robot.start)} is within its "
                    f"clearance of an obstacle; planning will let it leave but the first "
                    f"steps run tight to the obstacle."
                )

        table = self.get_clearance_table()
        ids = list(self.robots)
        for x in range(len(ids)):
            a = self.robots[ids[x]]
            for y in range(x + 1, len(ids)):
                b = self.robots[ids[y]]
                D = table.get((ids[x], ids[y]))
                if not D or D == frozenset({(0, 0)}):
                    continue

                if (
                    isinstance(a.start, (tuple, list))
                    and isinstance(b.start, (tuple, list))
                    and a.start_time == b.start_time
                ):
                    off = (a.start[0] - b.start[0], a.start[1] - b.start[1])
                    if off in D:
                        raise InfeasibleProblemError(
                            f"Robots '{ids[x]}' and '{ids[y]}' start at {tuple(a.start)} / "
                            f"{tuple(b.start)} at the same start_time ({a.start_time}), "
                            f"closer than their required separation — they cannot both "
                            f"be there at once."
                        )

                if isinstance(a.goal, (tuple, list)) and isinstance(
                    b.goal, (tuple, list)
                ):
                    off = (a.goal[0] - b.goal[0], a.goal[1] - b.goal[1])
                    if off in D:
                        raise InfeasibleProblemError(
                            f"Robots '{ids[x]}' and '{ids[y]}' have goals {tuple(a.goal)} / "
                            f"{tuple(b.goal)} closer than their required separation — they "
                            f"cannot both park there."
                        )

    def _validate_robot_positions(self):
        """
        Fail fast on an infeasible start/goal (out of bounds, on an
        obstacle, or an unknown graph node) instead of letting it surface
        later as a cryptic var_limit error, a silently-unsatisfiable QUBO,
        or a failed benchmark run. Positions are matrix (row, col) tuples
        (grid mode) or plain node ints (graph mode) by this point -- see
        resolve_coordinates(). A tuple position is only meaningful with a
        grid present, and an int node id only with a graph present -- see
        get_graph_robot_current_goal() for how a tuple is later resolved to
        a node id in "both"-format problems.
        """
        for robot in self.robots.values():
            for label, pos in (("start", robot.start), ("goal", robot.goal)):
                if isinstance(pos, (tuple, list)) and self.grid is not None:
                    i, j = pos
                    if not (0 <= i < self.grid.M and 0 <= j < self.grid.N):
                        raise InfeasibleProblemError(
                            f"Robot '{robot.robot_id}' {label} {pos} is out of bounds "
                            f"for a {self.grid.M}x{self.grid.N} grid."
                        )
                    if (i, j) in self.grid.obstacles:
                        raise InfeasibleProblemError(
                            f"Robot '{robot.robot_id}' {label} {pos} is on an obstacle."
                        )
                elif isinstance(pos, int) and self.graph is not None:
                    if not (0 <= pos < len(self.graph.nodes)):
                        raise InfeasibleProblemError(
                            f"Robot '{robot.robot_id}' {label} node id {pos} does not exist "
                            f"in graph '{self.graph.name}' with {len(self.graph.nodes)} nodes."
                        )

    @classmethod
    def general_init(
        cls,
        start,
        end,
        grid=None,
        graph=None,
        T=None,
        name="unnamed",
        coordinate_format="matrix",
    ):
        # In this case we simply create a default robot configuration for single robot
        robot = RobotConfig("Lucia", start, end, coordinate_format=coordinate_format)

        return cls(robot, grid, graph, T, name)

    @classmethod
    def from_grid_dict(cls, grid, problem_dict):
        """
        Create a PathfindingProblem instance from a grid and dictionary.
        It receives problem section from config file
        and extracts problem parameters.
        The grid is expected since you probably will also be extracting it from config previously.
        """
        start = tuple(problem_dict["start"])
        end = tuple(problem_dict["goal"])
        T = problem_dict.get("T", None)
        coordinate_format = problem_dict.get("coordinate_format", "matrix")
        return cls.general_init(
            start, end, grid=grid, T=T, coordinate_format=coordinate_format
        )

    @classmethod
    def from_graph_data(
        cls, graph_data, start_node, end_node, T=None, name="graph_problem"
    ):
        """
        Create a PathfindingProblem instance from graph data.

        Args:
            graph_data: Dictionary with 'nodes' and 'edges' keys or Graph instance
            start_node: Starting node index
            end_node: Goal node index
            T: Time horizon (optional)
            name: Problem name
        """
        # Convert dict to Graph instance if needed
        if isinstance(graph_data, dict):
            graph = map.Graph.from_hdf5_data(graph_data, name)
        else:
            graph = graph_data

        if isinstance(start_node, (list, tuple)):
            start_node = graph.get_node_from_position(start_node)
        if isinstance(end_node, (list, tuple)):
            end_node = graph.get_node_from_position(end_node)

        return cls.general_init(start_node, end_node, graph=graph, T=T, name=name)

    @classmethod
    def from_unified_data(
        cls,
        h5_source,
        start,
        end,
        materials_data=None,
        T=None,
        name=None,
        coordinate_format="matrix",
    ):
        """
        Create a unified PathfindingProblem instance with both grid and graph data.
        This is the main function for loading synthetic maps that support both approaches.

        Args:
            h5_source: HDF5 file path or file-like object
            start: Start position (i,j) for grid or node_id for graph
            end: End position (i,j) for grid or node_id for graph
            materials_data: Optional materials data for Grid object
            T: Time horizon (optional)
            name: Problem name (optional, will use map name if not provided)

        Returns:
            PathfindingProblem: Unified problem with both grid and graph representations
        """
        from quantum.config.hdf5parser import load_both_from_hdf5

        # Load both data types
        data = load_both_from_hdf5(h5_source)

        # Use provided name or map name
        problem_name = name or data["name"]

        # Create grid if available
        grid = None
        if data["has_map"] and data["map_data"]:
            grid = map.Grid.from_hdf5_data(
                data["map_data"], materials_data=materials_data, name=problem_name
            )

        # Create graph if available
        graph = None
        if data["has_graph"] and data["graph_data"]:
            graph = map.Graph.from_hdf5_data(data["graph_data"], name=problem_name)

        # Create unified problem
        problem = cls.general_init(
            start=start,  # Keep original start for grid
            end=end,  # Keep original end for grid
            grid=grid,
            graph=graph,
            T=T,
            name=problem_name,
            coordinate_format=coordinate_format,
        )

        return problem

    @classmethod
    def from_h5(cls, h5_path, robots, materials_data=None, T=None, name=None):
        """
        Build a multi-robot PathfindingProblem straight from an .h5 map, with
        no companion problems YAML -- for callers (CLI flags, argOS/FastAPI
        requests) that supply robot start/goal at request time instead of
        from a committed config. `from_map_config`'s multi-robot branch is a
        thin wrapper around this that adds YAML parsing on top.

        Args:
            h5_path: Path to the map .h5 file (extension optional)
            robots: RobotConfig instance, or list/dict of them
            materials_data: Optional materials data for Grid object
            T: Time horizon (optional)
            name: Problem name (optional, defaults to the map's file stem)

        Returns:
            PathfindingProblem: Unified problem instance
        """
        h5_path = str(h5_path)
        if not h5_path.endswith(".h5"):
            h5_path = f"{h5_path}.h5"

        from quantum.config.hdf5parser import load_both_from_hdf5

        data = load_both_from_hdf5(h5_path)
        problem_name = name or data["name"]

        grid = None
        if data["has_map"] and data["map_data"]:
            grid = map.Grid.from_hdf5_data(
                data["map_data"], materials_data=materials_data, name=problem_name
            )

        graph = None
        if data["has_graph"] and data["graph_data"]:
            graph = map.Graph.from_hdf5_data(data["graph_data"], name=problem_name)

        return cls(robots=robots, grid=grid, graph=graph, T=T, name=problem_name)

    @classmethod
    def from_map_config(
        cls,
        map_path,
        problem_name="baseline",
        materials_data=None,
        coordinate_format="matrix",
    ):
        """
        Fast initialization from map path and problem configuration.
        This is a convenience method that combines H5 loading and YAML config parsing.
        Supports both single-robot (legacy) and multi-robot configurations.

        Args:
            map_path: Path to map file (with or without .h5/.yaml extension)
                     e.g., "maps/synthetic/5x5/obs5x5_medium" or "maps/synthetic/5x5/obs5x5_medium.h5"
            problem_name: Name of the problem configuration in the YAML file (default: "baseline")
            materials_data: Optional materials data for Grid object

        Returns:
            PathfindingProblem: Unified problem instance

        Example:
            >>> # Single robot (legacy format)
            >>> problem = PathfindingProblem.from_map_config(
            ...     "maps/synthetic/5x5/obs5x5_medium",
            ...     problem_name="baseline"
            ... )

            >>> # Multi-robot format
            >>> problem = PathfindingProblem.from_map_config(
            ...     "maps/synthetic/10x10/no_obs10x10",
            ...     problem_name="two_robots"
            ... )
        """
        import quantum.config.parser as config_parser
        from pathlib import Path

        # Normalize path (remove extension if present)
        map_path = str(map_path)
        if map_path.endswith(".h5"):
            base_path = map_path[:-3]
        elif map_path.endswith(".yaml"):
            base_path = map_path[:-5]
        else:
            base_path = map_path

        h5_path = f"{base_path}.h5"
        yaml_path = f"{base_path}.yaml"

        # Load problem configuration from YAML
        config = config_parser.load_config(yaml_path, sections=["problems"])

        if "problems" not in config or problem_name not in config["problems"]:
            raise ValueError(
                f"Problem '{problem_name}' not found in {yaml_path}. "
                f"Available problems: {list(config.get('problems', {}).keys())}"
            )

        problem_config = config["problems"][problem_name]
        time_limit = problem_config.get("time_limit", None)

        # Check if this is a multi-robot problem
        if "robots" in problem_config:
            # Multi-robot configuration
            robots = []
            for robot_id, robot_data in problem_config["robots"].items():
                robot = RobotConfig(
                    robot_id=robot_id,
                    start=tuple(robot_data["start"])
                    if isinstance(robot_data["start"], list)
                    else robot_data["start"],
                    goal=tuple(robot_data["goal"])
                    if isinstance(robot_data["goal"], list)
                    else robot_data["goal"],
                    start_time=robot_data.get("start_time", 0),
                    priority=robot_data.get("priority", 1.0),
                    # "safety_radius" accepted as a legacy alias for pre-rename
                    # problem YAMLs.
                    robot_radius=robot_data.get(
                        "robot_radius", robot_data.get("safety_radius", 0.5)
                    ),
                    inflation=robot_data.get("inflation", 0.0),
                    expected_duration=robot_data.get("expected_duration", None),
                    coordinate_format=robot_data.get(
                        "coordinate_format", coordinate_format
                    ),
                )
                robots.append(robot)

            problem_full_name = f"{Path(base_path).stem}_{problem_name}"
            return cls.from_h5(
                h5_path,
                robots=robots,
                materials_data=materials_data,
                T=time_limit,
                name=problem_full_name,
            )
        else:
            # Single robot (legacy format)
            start = (
                tuple(problem_config["start"])
                if isinstance(problem_config["start"], list)
                else problem_config["start"]
            )
            goal = (
                tuple(problem_config["goal"])
                if isinstance(problem_config["goal"], list)
                else problem_config["goal"]
            )

            # Use from_unified_data to create the problem
            return cls.from_unified_data(
                h5_source=h5_path,
                start=start,
                end=goal,
                materials_data=materials_data,
                T=time_limit,
                name=f"{Path(base_path).stem}_{problem_name}",
                coordinate_format=problem_config.get(
                    "coordinate_format", coordinate_format
                ),
            )

    def add_robot(self, robot: RobotConfig, keep_time=False):
        """Add a robot to the problem."""
        self.robots[robot.robot_id] = robot
        self.num_robots += 1
        # robot set changed; rebuild clearance caches on next request
        self._clearance_table = None
        self._obstacle_keepout = None
        if not keep_time:
            self.T = self.calculate_timeline()

    def manhattan_distance(self, start, end):
        """Calculate Manhattan distance for grid coordinates."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def euclidean_distance(self, start, end):
        """Calculate Euclidean distance for graph coordinates."""
        return np.sqrt(
            (start[0] - end[0]) * (start[0] - end[0])
            + (start[1] - end[1]) * (start[1] - end[1])
        )

    # def is_valid_move(self, robot, from_pos, to_pos):
    #     """Check if a move is valid."""
    #     if self.grid is not None:
    #         return self.grid.is_valid_move(robot, from_pos, to_pos)
    #     else:
    #         return self.graph.is_valid_move(robot, from_pos, to_pos)

    def set_robot_time(self):
        """Set time horizon T for each robot if not already set."""
        for robot in self.robots.values():
            if robot.T is None:
                if self.grid is not None:
                    # Heuristic: 2x Manhattan distance + 4 steps buffer
                    # This handles congestion/detours better than 1.5x, especially for short paths
                    dist = self.manhattan_distance(robot.current_position, robot.goal)
                    robot.T = int(dist * 2.0) + 4  # Better for multirobot and deroutes
                    self.logger.debug(
                        f"Calculated heuristic T for robot {robot.robot_id} with dist {dist}, T={robot.T}"
                    )

                else:  # graph format
                    # For graphs, I need to implement some heuristic like straight line from start to node
                    # And make a conversion from like meters to time steps and some extra margin
                    robot.T = 10

    def calculate_timeline(self):
        total_time = 0
        self.set_robot_time()
        for robot in self.robots.values():
            final_robot_time = robot.start_time + robot.T
            if final_robot_time > total_time:
                total_time = final_robot_time
        return total_time

    def get_robot_per_timestep(self):
        """
        Get a dictiorinary mapping each robot to that global timestep
        If the robot is inactive for that timestep, it will not appear in the list
        """
        robot_per_timestep = {}
        for t in range(self.T):
            robot_per_timestep[t] = []
            for robot in self.robots.values():
                if robot.start_time <= t < robot.start_time + robot.T:
                    robot_per_timestep[t].append(robot.robot_id)
        return robot_per_timestep

    def get_robot_nums(self):
        """
        Get the numberr associated to each robot id
        This works when retrieving variables from the QUBO
        """
        robot_num = {}
        for idx, robot_id in enumerate(self.robots.keys()):
            robot_num[robot_id] = idx
        return robot_num

    def get_format_type(self):
        """Return the format type: 'grid', 'graph', or 'both'."""
        if self.grid is not None and self.graph is not None:
            return "both"
        elif self.grid is not None:
            return "grid"
        else:
            return "graph"

    def get_graph_robot_current_goal(self, robot_id):
        """Get graph-specific current_position and goal node indices from a robot."""
        if self.graph is not None:
            # Convert coordinates to node indices if not already done
            robot = self.robots[robot_id]
            start_node = (
                robot.current_position
                if isinstance(robot.current_position, int)
                else self.graph.get_node_from_position(robot.current_position)
            )

            end_node = (
                robot.goal
                if isinstance(robot.goal, int)
                else self.graph.get_node_from_position(robot.goal)
            )
            return start_node, end_node
        else:
            return None, None

    def can_use_grid(self):
        """Check if grid representation is available."""
        return self.grid is not None

    def can_use_graph(self):
        """Check if graph representation is available."""
        return self.graph is not None

    def as_grid_only(self):
        """Return a new problem instance restricted to the grid representation."""
        if self.grid is None:
            raise ValueError("Grid representation not available in this problem")
        return PathfindingProblem(
            robots=self.robots,
            grid=self.grid,
            graph=None,
            T=self.T,
            name=self.name,
            separation_factor=self.separation_factor,
            clearance_enabled=self.clearance_enabled,
        )

    def as_graph_only(self):
        """Return a new problem instance restricted to the graph representation."""
        if self.graph is None:
            raise ValueError("Graph representation not available in this problem")
        return PathfindingProblem(
            robots=self.robots,
            grid=None,
            graph=self.graph,
            T=self.T,
            name=self.name,
            separation_factor=self.separation_factor,
            clearance_enabled=self.clearance_enabled,
        )

    def to_dict(self):
        """
        Convert the problem instance to a dictionary representation.
        """
        result = {
            "name": self.name,
            "T": self.T,
            "robots": {
                robot_id: robot.to_dict() for robot_id, robot in self.robots.items()
            },
        }

        if self.grid is not None:
            result["grid"] = self.grid.to_dict()

        if self.graph is not None:
            result["graph"] = self.graph.to_dict()

        return result
