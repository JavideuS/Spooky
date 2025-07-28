from pathlib import Path
from datetime import datetime
import json
import time


class BenchmarkRunner:
    def __init__(self, qubobuilder, solver, num_runs=10, output_dir="results/benchmarks"):
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

    def run(self):
        """Run the benchmark multiple times and store results"""
        print(f"\nBenchmarking Problem: {self.problem.name}")
        print(f"Using Solver: {self.solver.name}")
        print(f"Penalty Set: {self.penalty_set.get('name', 'unnamed')}")
        print("-" * 60)

        # Build QUBO once
        Q = self.builder.build()

        # Run multiple trials
        for run_id in range(1, self.num_runs + 1):
            start_time = time.time()
            solution = self.solver.solve_qubo(Q)
            duration = time.time() - start_time

            validation = is_solution_valid(solution["solution"], self.problem)

            result = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "solution": solution,
                "validation": validation,
                "energy": solution["energy"],
                "success": validation["valid"],
                "execution_time_sec": round(duration, 3),
            }

            self.results["runs"].append(result)

            status = "✅ Valid" if validation["valid"] else "❌ Invalid"
            print(f"Run {run_id}: {status} | Time: {duration:.2f}s | Energy: {solution['energy']:.4f}")
            print(f"Path: {self.solver.decode_path(solution['solution'], self.problem)}")

        self.save_results()
        return self.results
    
    def run_build(self):
        """Run the benchmark multiple times and store results"""
        print(f"\nBenchmarking Problem: {self.problem.name}")
        print(f"Using Solver: {self.solver.name}")
        print(f"Penalty Set: {self.penalty_set.get('name', 'unnamed')}")
        print("-" * 60)

        # Run multiple trials
        for run_id in range(1, self.num_runs + 1):
            build_start = time.time()
            self.builder.build()
            build_duration = time.time() - build_start

            solve_start = time.time()
            solution = self.solver.solve_qubo(self.builder)
            solve_duration = time.time() - solve_start

            print(f"Build time: {build_duration:.4f}s, Solve time: {solve_duration:.4f}s")

            path = self.solver.decode_path(solution["solution"], self.problem)
            validation = is_solution_valid(path, self.problem)
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
            print(f"Run {run_id}: {status} | Time: {solve_duration:.2f}s | Energy: {self.solver.total_energy(solution):.4f}")
            print(f"Path: {path}")

        self.save_results()
        return self.results

    def save_results(self):
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.results["penalty_set"], f, indent=2, default=str)
            json.dump(self.results["runs"], f, indent=2, default=str)  # Use `default=str` to serialize tuples
        print(f"\nBenchmark complete. Results saved to {filepath}")



def is_solution_valid(solution, problem):
    """
    Checks if the decoded path represents a valid path from start to goal.
    Accepts a list of (i, j, t) tuples (decoded path), or a list of such paths.
    Args:
        solution (list): List of (i, j, t) tuples, or list of such lists.
        problem (PathfindingProblem): Problem definition including grid, start,
            goal, T
    Returns:
        dict: Validation result with 'valid' flag and optional error details
    """
    grid = problem.grid
    M = grid.M
    N = grid.N
    T = problem.T
    start = problem.start
    goal = problem.end
    obstacles = grid.obstacles

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
                result["message"] = f"❌ Invalid move from ({li}, {lj}, {lt}) to ({i}, {j}, {t})"
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
