import yaml
import ast
from pathlib import Path
from . import hdf5parser


# PARSER FUNCTIONS
def parse_value(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "none":
            return None
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def convert_values(data):
    """
    Recursively convert strings like 'None' or '(2,0)' into real values.
    Also convert lists of length 2 with ints to tuples (for obstacles).
    """
    if isinstance(data, dict):
        return {k: convert_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Convert lists of length 2 with ints to tuples (for obstacles)
        if len(data) == 2 and all(isinstance(x, int) for x in data):
            return tuple(data)
        return [convert_values(item) for item in data]
    else:
        return parse_value(data)


# VERIFICATION FUNCTIONS
def validate_problem(problem):
    # if not isinstance(problem.get("grid", {}).get("M"), int) or problem["grid"]["M"] <= 0:
    #     raise ValueError("Grid M must be a positive integer")
    # if not isinstance(problem.get("grid", {}).get("N"), int) or problem["grid"]["N"] <= 0:
    #     raise ValueError("Grid N must be a positive integer")
    if not isinstance(problem.get("start"), tuple) or len(problem["start"]) != 2:
        raise ValueError("Start must be a 2-tuple (i, j)")
    if not isinstance(problem.get("goal"), tuple) or len(problem["goal"]) != 2:
        raise ValueError("Goal must be a 2-tuple (i, j)")
    if "T" in problem and (problem["T"] is not None and (not isinstance(problem["T"], int) or problem["T"] < 1)):
        raise ValueError("T must be an integer greater than 0 (>=1) or None")


def validate_solver(solver):
    backend = solver.get("backend")
    if not isinstance(backend, str) or backend not in ["dwave", "qiskit", "pennylane"]:
        raise ValueError("Solver backend must be one of: dwave, qiskit, pennylane")
    if not isinstance(solver.get("normalization_scale", 1.0), (int, float)):
        raise ValueError("normalization_scale must be a number")
    if not isinstance(solver.get("num_reads", 10), int) or solver["num_reads"] <= 0:
        raise ValueError("num_reads must be a positive integer")


def validate_penalty_set(name, penalty_set):
    required_keys = ["K_hot", "K_adj", "K_start", "K_goal", "K_lock"]
    for key in required_keys:
        if not isinstance(penalty_set.get(key), (int, float)):
            raise ValueError(f"Penalty set '{name}' missing or invalid value for {key}")


def validate_benchmark(benchmark):
    if not isinstance(benchmark.get("num_runs_per_config", 10), int) or benchmark["num_runs_per_config"] <= 0:
        raise ValueError("num_runs_per_config must be a positive integer")


# LOADER FUNCTION
def load_config(config_path="config.yaml", sections=None):
    # If it's a string or Path, open it
    if isinstance(config_path, (str, Path)):
        with open(config_path, "r") as f:
            raw_data = yaml.safe_load(f)
    else:
        # Assume it's a file-like object (has .read())
        try:
            # Ensure we're at the start
            if hasattr(config_path, 'seek'):
                config_path.seek(0)
            raw_data = yaml.safe_load(config_path)
        except Exception as e:
            raise ValueError(f"Failed to parse file-like config: {str(e)}")

    # Load only requested sections
    if sections is None:
        parsed_data = raw_data
    else:
        parsed_data = {section: raw_data.get(section) for section in sections}

    # Convert special strings like 'None', tuples, etc.
    parsed_data = convert_values(parsed_data)

    # Optional validation
    if "problems" in parsed_data:
        for name, problem in parsed_data["problems"].items():
            validate_problem(problem)

    if "solver" in parsed_data:
        for name, solver in parsed_data["solver"].items():
            validate_solver(solver)

    if "penalty_sets" in parsed_data:
        for name, pset in parsed_data["penalty_sets"].items():
            validate_penalty_set(name, pset)

    if "benchmark" in parsed_data:
        validate_benchmark(parsed_data["benchmark"])

    return parsed_data
