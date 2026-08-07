import networkx as nx
from quantum.utils.logger import get_logger
from quantum.builder.ILPBuilder import bfs_reachable_sets, reverse_adjacency


class BaseCBSBuilder:
    """
    Mirrors BaseILPBuilder's shape (penalty-set stub metadata, common constructor shape, local_index(v))
    but build() constructs a plain networkx.Graph instead of a Pyomo model:
    CBS has no decision variables/constraints to declare, just a graph for
    its space-time A* low-level solver to search over. Concrete grid/graph
    subclasses implement build(), vars_per_time, and local_index(v).
    """

    def __init__(self, problem, name="cbs", verbose_level=2):
        self.problem = problem
        self.name = name
        self.graph = None
        self.legal_cells = None
        self.verbose_level = verbose_level
        self.logger = get_logger()
        # No penalty weights — CBS uses hard constraints (the low-level
        # search's reservation table) instead
        self.penalties = {
            "name": "cbs_hard_constraints",
            "constraints": ["reservation_table", "crash", "swap"],
        }

    def build(self, preprocess=True):
        """Build the search graph for the current problem state and return it."""
        raise NotImplementedError

    def local_index(self, v):
        """Map a builder-native vertex (grid tuple or graph node id) to the
        position component of the flat (robot, time, position) variable
        index — same contract as BaseILPBuilder.local_index(), and for the
        same reason: CBSSolver packs its result into that shared flat index
        so decode_path()/BaseSolver post-processing work unchanged."""
        raise NotImplementedError

    def get_robot_start_goal(self, robot_id):
        """Return (start_node_id, goal_node_id) in self.graph's node space
        for the given robot — the uniform lookup CBSSolver uses regardless
        of grid vs graph format."""
        raise NotImplementedError

    def reset_problem(self):
        """Restore every robot to its initial start state (mirrors BaseILPBuilder.reset_problem)."""
        for robot in self.problem.robots.values():
            robot.reset()


class GridCBSBuilder(BaseCBSBuilder):
    """CBS builder for grid pathfinding problems."""

    def __init__(self, problem, name="cbs_grid", verbose_level=2):
        if problem.grid is None:
            raise ValueError("Grid representation not available in this problem")
        super().__init__(problem, name=name, verbose_level=verbose_level)
        self.vars_per_time = problem.grid.M * problem.grid.N

    def local_index(self, v):
        i, j = v
        return i * self.problem.grid.N + j

    def get_robot_start_goal(self, robot_id):
        robot = self.problem.robots[robot_id]
        return self.local_index(robot.current_position), self.local_index(robot.goal)

    def build(self, preprocess=True):
        problem = self.problem
        grid = problem.grid
        robots = problem.robots

        graph = nx.Graph()
        for i in range(grid.M):
            for j in range(grid.N):
                if (i, j) not in grid.obstacles:
                    graph.add_node(self.local_index((i, j)), pos=(i, j))
        for (i, j), neighbors in grid.adjacency.items():
            if (i, j) in grid.obstacles:
                continue
            u = self.local_index((i, j))
            for ni, nj in neighbors:
                if (ni, nj) in grid.obstacles:
                    continue
                graph.add_edge(u, self.local_index((ni, nj)))

        self.graph = graph

        # Forward-from-start and backward-from-goal BFS reachability per
        # robot, exactly as GridILPBuilder does
        # Used to restrict SpaceTimeAStar's neighbor expansion, not to fix
        # ILP-style variables (CBS has none), but the reduction is the same
        # in spirit: fewer cells the low-level search has to consider.
        in_window_vars = 0
        final_vars = 0
        self.legal_cells = {} if preprocess else None
        if preprocess:
            reachable = {v: list(graph.neighbors(v)) + [v] for v in graph.nodes}
            reverse_reachable = reverse_adjacency(reachable)
            for robot_id, robot in robots.items():
                start_id = self.local_index(robot.current_position)
                goal_id = self.local_index(robot.goal)
                max_steps = robot.T - 1
                forward = bfs_reachable_sets(reachable, start_id, max_steps)
                backward = bfs_reachable_sets(reverse_reachable, goal_id, max_steps)
                legal = [forward[k] & backward[max_steps - k] for k in range(robot.T)]
                self.legal_cells[robot_id] = legal
                in_window_vars += self.vars_per_time * robot.T
                final_vars += sum(len(s) for s in legal)
        else:
            for robot in robots.values():
                in_window_vars += self.vars_per_time * robot.T
            final_vars = in_window_vars

        reduced = in_window_vars - final_vars
        self.bfs_stats = {
            "window": 0,
            "preprocess": preprocess,
            "initial_variables": in_window_vars,
            "variables_reduced": reduced,
            "final_variables": final_vars,
            "reduction_ratio": round(reduced / in_window_vars, 4)
            if in_window_vars
            else 0,
        }

        self.logger.standard(
            f"CBS graph built: {len(robots)} robots, {graph.number_of_nodes()} free cells"
            f"\nBFS Stats: {self.bfs_stats}"
        )
        return self.graph


class GraphCBSBuilder(BaseCBSBuilder):
    """CBS builder for graph pathfinding problems. Same structure as
    GridCBSBuilder, written over the graph's native vertex set instead of
    (i, j) grid cells — see BaseCBSBuilder's docstring for what differs."""

    def __init__(self, problem, name="cbs_graph", verbose_level=2):
        if problem.graph is None:
            raise ValueError("Graph representation not available in this problem")
        super().__init__(problem, name=name, verbose_level=verbose_level)
        self.vars_per_time = len(problem.graph.nodes)

    def local_index(self, v):
        return v

    def get_robot_start_goal(self, robot_id):
        return self.problem.get_graph_robot_current_goal(robot_id)

    def build(self, preprocess=True):
        problem = self.problem
        graph_data = problem.graph
        robots = problem.robots

        graph = nx.Graph()
        for node_id in range(len(graph_data.nodes)):
            graph.add_node(node_id, pos=tuple(graph_data.nodes[node_id]))
        for u, neighbors in graph_data.adjacency.items():
            for v, _weight in neighbors:
                graph.add_edge(u, v)

        self.graph = graph

        # Robot start/goal may be stored as raw node ids or as coordinates,
        # depending on how the problem was built
        start_goal = {a: problem.get_graph_robot_current_goal(a) for a in robots}

        in_window_vars = 0
        final_vars = 0
        self.legal_cells = {} if preprocess else None
        if preprocess:
            reachable = {v: list(graph.neighbors(v)) + [v] for v in graph.nodes}
            reverse_reachable = reverse_adjacency(reachable)
            for robot_id, robot in robots.items():
                start_id, goal_id = start_goal[robot_id]
                max_steps = robot.T - 1
                forward = bfs_reachable_sets(reachable, start_id, max_steps)
                backward = bfs_reachable_sets(reverse_reachable, goal_id, max_steps)
                legal = [forward[k] & backward[max_steps - k] for k in range(robot.T)]
                self.legal_cells[robot_id] = legal
                in_window_vars += self.vars_per_time * robot.T
                final_vars += sum(len(s) for s in legal)
        else:
            for robot in robots.values():
                in_window_vars += self.vars_per_time * robot.T
            final_vars = in_window_vars

        reduced = in_window_vars - final_vars
        self.bfs_stats = {
            "window": 0,
            "preprocess": preprocess,
            "initial_variables": in_window_vars,
            "variables_reduced": reduced,
            "final_variables": final_vars,
            "reduction_ratio": round(reduced / in_window_vars, 4)
            if in_window_vars
            else 0,
        }

        self.logger.standard(
            f"CBS graph built: {len(robots)} robots, {graph.number_of_nodes()} nodes"
            f"\nBFS Stats: {self.bfs_stats}"
        )
        return self.graph
