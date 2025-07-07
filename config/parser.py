import yaml
import ast


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
    """Recursively convert strings like 'None' or '(2,0)' into real values"""
    if isinstance(data, dict):
        return {k: convert_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_values(item) for item in data]
    else:
        return parse_value(data)


# VERIFICATION FUNCTIONS
def validate_problem(problem):
    if not isinstance(problem.get("grid", {}).get("M"), int) or problem["grid"]["M"] <= 0:
        raise ValueError("Grid M must be a positive integer")
    if not isinstance(problem.get("grid", {}).get("N"), int) or problem["grid"]["N"] <= 0:
        raise ValueError("Grid N must be a positive integer")
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
    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f)

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


# config = load_config(sections=["solver"])
# config = load_config()  # To load the entire config file
# print(config)
# print(config["solver"]["dwave_qa"])

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    
    print("Loaded config:")
    for name, problem in config.get("problems", {}).items():
        print(f"\nProblem '{name}':")
        print("  Raw:", problem)
        try:
            validate_problem(problem)
            print("  Validated ✅")
        except Exception as e:
            print("  ❌ Validation failed:", e)
