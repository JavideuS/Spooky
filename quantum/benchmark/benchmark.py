from pathlib import Path
from datetime import datetime
import json
import time


class BenchmarkRunner:
    def __init__(
        self, qubobuilder, solver, num_runs=10, output_dir="results/benchmarks"
    ):
        """
        Run benchmark on a given solver and problem.

        Args:
            problem (PathfindingProblem): A fully initialized problem
            solver (QUBOSolver): A solver implementing `.solve(Q)`
            penalty_set (dict): Penalty dictionary {K_hot: ..., K_adj: ...}
            num_runs (int): Number of times to run the solver
            output_dir (str): Where to save results
        """
        self.builder = qubobuilder
        self.problem = qubobuilder.problem
        self.penalty_set = qubobuilder.penalties
        self.solver = solver
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "problem": self.problem.to_dict(),
            "solver": self.solver.to_dict(),
            "penalty_set": self.penalty_set,
            "runs": []
        }

    def run_build(self):
        """Run the benchmark multiple times and store results"""
        print(f"\nBenchmarking Problem: {self.problem.name}")
        print(f"Using Solver: {self.solver.name}")
        print(f"Penalty Set: {self.penalty_set.get('name', 'unnamed')}")
        print("-" * 60)

        # Run multiple trials
        for run_id in range(1, self.num_runs + 1):
            self.builder.reset_problem()
            build_start = time.time()
            self.builder.build()
            build_duration = time.time() - build_start
        
            solve_start = time.time()
            # solution = self.solver.solve_qubo(self.builder, False)
            solution = self.solver.solve_qubo_smart(self.builder, False)
            solve_duration = time.time() - solve_start

            print(
                f"Build time: {build_duration:.4f}s, "
                f"Solve time: {solve_duration:.4f}s"
            )

            path = self.solver.decode_path(solution["solution"], self.problem)
            validation = is_solution_valid(path, self.problem)

            result = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "solution": solution,
                "validation": validation,
                "energy": solution["energy"],
                # "success": validation["valid"],
                "execution_time_sec": round(solve_duration, 3),
            }

            self.results["runs"].append(result)

            status = "✅ Valid" if validation["valid"] else "❌ Invalid"
            if not validation["valid"]:
                print("Validation details:", validation["details"])
                print("Reason:", validation["reason"])
                print("Message:", validation["message"])
            print(
                f"Run {run_id}: {status} | Time: {solve_duration:.2f}s | "
                f"Energy: {self.solver.total_energy(solution):.4f}"
            )
            print(f"Path: {path}")
            # print(f"Robot paths: {self.solver.get_robot_paths(path)}")
            for robot in self.problem.robots.values():
                print(f" Robot {robot.robot_id} path: {robot.path}")
            # print("Raw solution:", solution["solution"])

        self.save_results()
        return self.results

    def save_results(self):
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.results["penalty_set"], f, indent=2, default=str)
            # Use `default=str` to serialize tuples
            json.dump(self.results["runs"], f, indent=2, default=str)
        print(f"\nBenchmark complete. Results saved to {filepath}")


def is_solution_valid(solution, problem):
    """
    Checks if the decoded path represents a valid path from start to goal.
    Accepts a list of ((i, j, t), robot_num) tuples (decoded path), or a list of such paths.
    Args:
        solution (list): List of ((i, j, t), robot_num) tuples, 
            or list of such lists.
        problem: Problem instance (grid or graph)
    Returns:
        dict: Validation result with 'valid' flag and optional error details
    """
    problem_type = problem.get_format_type()
    
    if problem_type != "grid" and problem_type != "graph":
        return {
            "valid": False,
            "reason": "unsupported_problem_type",
            "message": f"Unsupported problem type: {problem_type}",
        }
    
    result = {"valid": True, "details": {}}

    # If input is a list of paths (list of lists of tuples)
    if solution and isinstance(solution[0], list):
        return [is_solution_valid(path, problem) for path in solution]

    # If input is a single path (list of tuples)
    positions = list(solution)
    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = "❌ No path positions found. Invalid sample"
        return result

    result["details"]["path"] = positions

    # Group positions by robot
    robot_positions = {}
    for (i, j, t), robot_num in positions:
        if robot_num not in robot_positions:
            robot_positions[robot_num] = []
        robot_positions[robot_num].append((i, j, t))

    # Validate each robot's path using unified validation
    for robot_num, robot_path in robot_positions.items():
        robot_id = list(problem.robots.keys())[robot_num]
        robot_result = _validate_single_robot_path_unified(
            robot_path, problem, robot_id, robot_num)
        if not robot_result["valid"]:
            result["valid"] = False
            result["reason"] = f"robot_{robot_num}_invalid"
            result["message"] = f"❌ Robot {robot_num}: {robot_result['message']}"
            result["details"][f"robot_{robot_num}"] = robot_result
            return result
        result["details"][f"robot_{robot_num}"] = robot_result

    # multi-robot vertex conflict check (same cell at same time)
    occupancy = {}  # (i, j, t) -> [robot_num, ...]
    for robot_num, robot_path in robot_positions.items():
        for (i, j, t) in robot_path:
            key = (i, j, t)
            occupancy.setdefault(key, []).append(robot_num)

    # Find collisions where two or more robots occupy same cell at same time
    conflicts = []
    for (i, j, t), robots in occupancy.items():
        if len(robots) > 1:
            conflicts.append({
                "cell": (i, j),
                "time": t,
                "robots": sorted(robots)
            })

    if conflicts:
        # Sort conflicts by time then cell for deterministic output
        conflicts.sort(key=lambda c: (c["time"], c["cell"]))
        result["valid"] = False
        result["reason"] = "vertex_conflict"
        result["message"] = (
            f"❌ Vertex conflict detected: {len(conflicts)} collision(s) found"
        )
        result["details"]["conflicts"] = conflicts
        return result

    result["valid"] = True
    result["message"] = "✅ Solution is valid"
    return result

def _validate_single_robot_path_unified(positions, problem, robot_id, 
                                        robot_num):
    """
    Unified validation function for single robot paths in both grid and graph 
    problems.
    
    Args:
        positions: List of (i, j, t) tuples for one robot
        problem: Problem instance (grid or graph)
        robot_id: Robot identifier/name
        robot_num: Robot number for error messages
        
    Returns:
        dict: Validation result
    """
    result = {"valid": True, "details": {}}
    
    # Get notation abstraction based on problem type
    notation = _get_notation_abstraction(problem)

    T = problem.robots[robot_id].T + problem.robots[robot_id].start_time
    
    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = f"❌ Robot {robot_num}: No path positions found"
        return result

    # 1. Sort by time step
    positions.sort(key=lambda x: x[2])

    # 2. Get expected time range and check all time steps are present
    expected_times = notation.get_expected_time_range(problem, robot_id, T)
    all_times = set(t for _, _, t in positions)
    missing_times = expected_times - all_times
    extra_times = all_times - expected_times

    if missing_times:
        result["valid"] = False
        result["reason"] = "missing_time_steps"
        result["message"] = (f"❌ Robot {robot_num}: Missing time steps: "
                             f"{missing_times}")
        result["details"]["missing_times"] = list(missing_times)
        result["details"]["extra_times"] = list(extra_times)
        return result

    if extra_times:
        result["valid"] = False
        result["reason"] = "extra_time_steps"
        result["message"] = (f"❌ Robot {robot_num}: Extra time steps found: "
                             f"{extra_times}")
        result["details"]["extra_times"] = list(extra_times)
        return result

    # 3. One-hot constraint per time step
    time_to_positions = {}
    for i, j, t in positions:
        time_to_positions.setdefault(t, []).append((i, j))

    for t, cells in time_to_positions.items():
        if len(cells) > 1:
            result["valid"] = False
            result["reason"] = "multiple_positions_per_time"
            result["message"] = f"❌ Robot {robot_num}: Multiple positions at time {t}: {cells}"
            result["details"]["conflicts"] = {t: cells}
            return result

    # 4. Start position check
    start_time = notation.get_start_time(problem, robot_id)
    start_positions = [(i, j) for i, j, t in positions if t == start_time]
    if not start_positions:
        result["valid"] = False
        result["reason"] = "wrong_start"
        result["message"] = f"❌ Robot {robot_num}: No start position found at time {start_time}"
        return result
    
    expected_start = notation.get_start_position(problem, robot_id)
    actual_start = notation.get_position_representation(problem, start_positions[0])
    if actual_start != expected_start:
        result["valid"] = False
        result["reason"] = "wrong_start"
        result["message"] = f"❌ Robot {robot_num}: Wrong start position. Expected {expected_start}, got {actual_start}"
        result["details"]["expected_start"] = expected_start
        result["details"]["actual_start"] = actual_start
        return result

    # 5. Goal position check
    expected_goal = notation.get_goal_position(problem, robot_id)
    goal_reached = any(notation.get_position_representation(problem, (i, j)) == expected_goal for i, j, t in positions)
    if not goal_reached:
        result["valid"] = False
        result["reason"] = "goal_not_reached"
        result["message"] = f"❌ Robot {robot_num}: Goal position {expected_goal} never reached"
        result["details"]["goal"] = expected_goal
        return result

    # 6. Movement must be valid (adjacent positions only)
    last_pos = None
    for i, j, t in positions:
        current_pos = (i, j)
        
        # Check for obstacles (grid only)
        if notation.has_obstacle_check():
            obstacles = notation.get_obstacles(problem)
            if current_pos in obstacles:
                result["valid"] = False
                result["reason"] = "obstacle_collision"
                result["message"] = f"❌ Robot {robot_num}: Path goes through obstacle at {current_pos} at time {t}"
                result["details"]["collisions"] = [(*current_pos, t)]
                return result
        
        if last_pos is not None:
            last_i, last_j, last_t = last_pos
            
            # Goal-lock bypass: if both current and previous are goal, allow staying
            if (notation.get_position_representation(problem, (last_i, last_j)) == expected_goal and 
                notation.get_position_representation(problem, current_pos) == expected_goal):
                last_pos = (i, j, t)
                continue
            
            # Valid move check using notation-specific adjacency
            if not notation.is_valid_move(problem, (last_i, last_j), current_pos):
                result["valid"] = False
                result["reason"] = "invalid_move"
                result["message"] = (
                    f"❌ Robot {robot_num}: Invalid move from ({last_i}, {last_j}, {last_t}) to ({i}, {j}, {t})"
                )
                result["details"]["invalid_moves"] = [(last_i, last_j, last_t, i, j, t)]
                return result
        last_pos = (i, j, t)

    # 7. Final check: Goal times
    goal_times = [t for i, j, t in positions if notation.get_position_representation(problem, (i, j)) == expected_goal]
    result["details"]["goal_times"] = goal_times

    result["valid"] = True
    result["reason"] = "valid_path"
    result["message"] = f"✅ Robot {robot_num}: Solution is valid"
    return result

def _get_notation_abstraction(problem):
    """Get the appropriate notation abstraction for the problem type."""
    problem_type = problem.get_format_type()
    
    if problem_type == "grid":
        return GridNotation()
    elif problem_type in ["graph", "both"]:
        return GraphNotation()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


class BaseNotation:
    """Base notation abstraction with common methods."""
    
    def get_expected_time_range(self, problem, robot_id, T):
        """Get expected time range for robot."""
        robot = problem.robots[robot_id]
        return set(range(robot.start_time, T))
    
    def get_start_time(self, problem, robot_id):
        """Get start time for robot."""
        robot = problem.robots[robot_id]
        return robot.start_time


class GridNotation(BaseNotation):
    """Notation abstraction for grid problems."""
    
    def get_start_position(self, problem, robot_id):
        """Get start position for robot."""
        robot = problem.robots[robot_id]
        return robot.start
    
    def get_goal_position(self, problem, robot_id):
        """Get goal position for robot."""
        robot = problem.robots[robot_id]
        return robot.goal
    
    def get_position_representation(self, problem, position):
        """Get position representation (coordinates for grid)."""
        return position
    
    def has_obstacle_check(self):
        """Whether this notation checks for obstacles."""
        return True
    
    def get_obstacles(self, problem):
        """Get obstacles for the problem."""
        return problem.grid.obstacles
    
    def is_valid_move(self, problem, from_pos, to_pos):
        """Check if move is valid using grid adjacency."""
        return to_pos in problem.grid.adjacency[from_pos]


class GraphNotation(BaseNotation):
    """Notation abstraction for graph problems."""
    
    def get_start_position(self, problem, robot_id):
        """Get start node for robot."""
        robot = problem.robots[robot_id]
        start_node = (robot.start if isinstance(robot.start, int)
                      else problem.graph.get_node_from_position(robot.start))
        return start_node
    
    def get_goal_position(self, problem, robot_id):
        """Get goal node for robot."""
        _, goal_node = problem.get_graph_robot_current_goal(robot_id)
        return goal_node
    
    def get_position_representation(self, problem, position):
        """Get position representation (node ID for graph)."""
        return problem.graph.get_node_from_position(position)
    
    def has_obstacle_check(self):
        """Whether this notation checks for obstacles (graphs don't)."""
        return False
    
    def get_obstacles(self, problem):
        """Get obstacles for the problem (not used for graphs)."""
        return set()
    
    def is_valid_move(self, problem, from_pos, to_pos):
        """Check if move is valid using graph adjacency."""
        from_node = problem.graph.get_node_from_position(from_pos)
        to_node = problem.graph.get_node_from_position(to_pos)
        # Adjacency list stores (neighbor_node, weight) tuples
        return any(neighbor_node == to_node for (neighbor_node, weight) in problem.graph.adjacency[from_node])


def convert_tuple_keys_to_str(obj):
    """
    Recursively convert tuple keys in dictionaries to strings.
    This is useful for serializing dictionaries with tuple keys to JSON.
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: convert_tuple_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuple_keys_to_str(i) for i in obj]
    else:
        return obj
