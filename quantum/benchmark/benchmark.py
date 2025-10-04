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
            solution = self.solver.solve_qubo_smart(self.builder, False)
            solve_duration = time.time() - solve_start

            print(
                f"Build time: {build_duration:.4f}s, "
                f"Solve time: {solve_duration:.4f}s"
            )

            path = self.solver.decode_path(solution["solution"], self.problem)
            # path = self.solver.merge_path_segments(path)
            # print("Decoded Path:", path)
            validation = is_solution_valid(path, self.builder)
            # print("Validation:", validation)

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
            # print("Raw solution:", solution["solution"])

        self.save_results()
        return self.results

    def save_results(self):
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.results["penalty_set"], f, indent=2, default=str)
            json.dump(self.results["runs"], f, indent=2, default=str)  # Use `default=str` to serialize tuples
        print(f"\nBenchmark complete. Results saved to {filepath}")



def is_solution_valid(solution, builder):
    """
    Checks if the decoded path represents a valid path from start to goal.
    Accepts a list of (i, j, t) tuples (decoded path), or a list of such paths.
    Args:
        solution (list): List of (i, j, t) tuples, or list of such lists.
        problem (PathfindingProblem): Problem definition including grid/graph,
            start, goal, T
    Returns:
        dict: Validation result with 'valid' flag and optional error details
    """

    problem_type = builder.problem.get_format_type()

    if problem_type == "grid":
        return _is_grid_solution_valid(solution, builder.problem, builder)
    elif problem_type == "graph" or problem_type == "both":
        return _is_graph_solution_valid(solution, builder.problem, builder)
    else:
        return {
            "valid": False,
            "reason": "unsupported_problem_type",
            "message": f"Unsupported problem type: {problem_type}",
        }

def _is_grid_solution_valid(solution, problem, builder):
    grid = problem.grid
    T = problem.T - builder.iter + 1  # You need to substract one per window 
    start = builder.initial_pos
    goal = problem.end
    obstacles = grid.obstacles

    result = {"valid": True, "details": {}}

    # If input is a list of paths (list of lists of tuples)
    if solution and isinstance(solution[0], list):
        return [is_solution_valid(path, problem, builder) for path in solution]

    # If input is a single path (list of tuples)
    positions = list(solution)
    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = "❌ No path positions found. Invalid sample"
        return result

    result["details"]["path"] = positions

    # 1. Sort by time step
    positions.sort(key=lambda x: x[2])

    # 2. Check all time steps from 0 to T-1 are present
    all_times = set(t for _, _, t in positions)
    expected_times = set(range(T))
    missing_times = expected_times - all_times
    extra_times = all_times - expected_times

    if missing_times:
        result["valid"] = False
        result["reason"] = "missing_time_steps"
        result["message"] = f"❌ Missing time steps: {missing_times}"
        result["details"]["missing_times"] = list(missing_times)
        result["details"]["extra_times"] = list(extra_times)
        return result

    if extra_times:
        result["valid"] = False
        result["reason"] = "extra_time_steps"
        result["message"] = f"❌ Extra time steps found: {extra_times}"
        result["details"]["extra_times"] = list(extra_times)
        return result

    # 3. One-hot constraint per time step
    time_to_positions = {}
    for i, j, t in positions:
        time_to_positions.setdefault(t, []).append((i, j))

    for t, cells in time_to_positions.items():
        if len(cells) != 1:
            result["valid"] = False
            result["reason"] = "multiple_cells_per_step"
            result["message"] = f"❌ More than one cell selected at time {t}: {cells}"
            result["details"]["time_to_positions"] = time_to_positions
            return result

    # 4. Start at correct position
    first_time_cells = time_to_positions.get(0, [])
    if not first_time_cells or first_time_cells[0] != start:
        result["valid"] = False
        result["reason"] = "start_mismatch"
        result["message"] = f"❌ Start position mismatch: Expected {start}, got {first_time_cells}"
        return result

    # 5. Goal reached at some time step
    goal_reached = any(i == goal[0] and j == goal[1] for i, j, t in positions)
    if not goal_reached:
        result["valid"] = False
        result["reason"] = "goal_not_reached"
        result["message"] = f"❌ Goal {goal} was never reached"
        return result

    # 6. Movement must be valid (adjacent cells only)
    last_pos = None
    for i, j, t in positions:
        if last_pos is not None:
            li, lj, lt = last_pos
            # Obstacle collision check
            if (i, j) in obstacles:
                result["valid"] = False
                result["reason"] = "obstacle_collision"
                result["message"] = f"❌ Path goes through obstacle at ({i}, {j}) at time {t}"
                result["details"]["collisions"] = [(i, j, t)]
                return result
            # Goal-lock bypass: if both current and previous are goal, allow staying
            if (li, lj) == goal and (i, j) == goal:
                last_pos = (i, j, t)
                continue
            # Valid move check using adjacency map from Grid class
            if (i, j) not in grid.adjacency[(li, lj)]:
                result["valid"] = False
                result["reason"] = "invalid_move"
                result["message"] = (
                    f"❌ Invalid move from ({li}, {lj}, {lt}) to ({i}, {j}, {t}) "
                    f"(Adjacency: {grid.adjacency.get((li, lj), 'N/A')})"
                )
                result["details"]["invalid_moves"] = [(li, lj, lt, i, j, t)]
                return result
        last_pos = (i, j, t)

    # 7. Final check: Did we reach the goal at the right time?
    goal_times = [t for i, j, t in positions if i == goal[0] and j == goal[1]]
    result["details"]["goal_times"] = goal_times

    result["valid"] = True
    result["reason"] = "valid_path"
    result["message"] = "✅ Solution is valid"

    return result

def _is_graph_solution_valid(solution, problem, builder):
    graph = problem.graph
    T = problem.T - builder.iter + 1
    
    start_node = builder.initial_pos
    goal_node = problem.get_graph_start_end()[1]

    result = {"valid": True, "details": {}}

    if solution and isinstance(solution[0], list):
        return [is_solution_valid(path, problem, builder) for path in solution]

    positions = list(solution)
    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = "❌ No path positions found. Invalid sample"
        return result
    
    result["details"]["path"] = positions

    # 1. Sort by time step (node_idx, time)
    positions.sort(key=lambda x: x[2])

    # 2. Check all time steps from 0 to T-1 are present
    all_times = set(t for _, _, t in positions)
    expected_times = set(range(T))
    missing_times = expected_times - all_times
    extra_times = all_times - expected_times

    if missing_times:
        result["valid"] = False
        result["reason"] = "missing_time_steps"
        result["message"] = f"❌ Missing time steps: {missing_times}"
        result["details"]["missing_times"] = list(missing_times)
        result["details"]["extra_times"] = list(extra_times)
        return result

    if extra_times:
        result["valid"] = False
        result["reason"] = "extra_time_steps"
        result["message"] = f"❌ Extra time steps found: {extra_times}"
        result["details"]["extra_times"] = list(extra_times)
        return result

    # 3. One-hot constraint per time step
    time_to_nodes = {}
    for i, j, t in positions:
        node_i = problem.graph.get_node_from_position((i, j))
        time_to_nodes.setdefault(t, []).append(node_i)

    for t, nodes in time_to_nodes.items():
        if len(nodes) != 1:
            result["valid"] = False
            result["reason"] = "multiple_nodes_per_step"
            result["message"] = f"❌ More than one node selected at time {t}: {nodes}"
            result["details"]["time_to_nodes"] = time_to_nodes
            return result
    
    # 4. Start at correct position
    first_time_node = time_to_nodes.get(0, [])
    if not first_time_node or first_time_node[0] != start_node:
        result["valid"] = False
        result["reason"] = "start_mismatch"
        result["message"] = f"❌ Start node mismatch: Expected {start_node}, got {first_time_node}"
        return result

    # 5. Goal reached at some time step
    goal_reached = any(problem.graph.get_node_from_position((i, j)) == goal_node for i, j, t in positions)
    if not goal_reached:
        result["valid"] = False
        result["reason"] = "goal_not_reached"
        result["message"] = f"❌ Goal node {goal_node} was never reached"
        return result

    # 6. Movement must be valid (adjacent nodes only)
    last_node_t = None
    for i, j, t in positions:
        node_i = problem.graph.get_node_from_position((i, j))
        if last_node_t is not None:
            last_node, last_t = last_node_t
            
            # Goal-lock bypass: if both current and previous are goal, allow staying
            if last_node == goal_node and node_i == goal_node:
                last_node_t = (node_i, t)
                continue
            
            # Valid move check using adjacency map from Graph class
            # Adjacency list stores (neighbor_node, weight) tuples
            if not any(neighbor_node == node_i for (neighbor_node, weight) in graph.adjacency[last_node]):
                result["valid"] = False
                result["reason"] = "invalid_move"
                result["message"] = (
                    f"❌ Invalid move from node {last_node} at time {last_t} to node {node_i} at time {t} "
                    f"(Adjacency: {graph.adjacency.get(last_node, 'N/A')})"
                )
                result["details"]["invalid_moves"] = [(last_node, last_t, node_i, t)]
                return result
        last_node_t = (node_i, t)

    # 7. Final check: Did we reach the goal at the right time?
    goal_times = [t for node_i, _, t in positions if node_i == goal_node]
    result["details"]["goal_times"] = goal_times

    result["valid"] = True
    result["reason"] = "valid_path"
    result["message"] = "✅ Solution is valid"

    return result


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
