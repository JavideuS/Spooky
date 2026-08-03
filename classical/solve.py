import pyomo.environ as pyo

from model import build_model


def solve(problem, solver_name="appsi_highs", **solver_options):
    """
    Build and solve the ILP model for a (grid-only) PathfindingProblem.

    Returns:
        (model, results) — the solved Pyomo model and the solver's results object.
    """
    model = build_model(problem)
    solver = pyo.SolverFactory(solver_name)
    results = solver.solve(model, **solver_options)
    return model, results


def decode_paths(model, problem):
    """
    Extract robot paths from a solved model.

    Returns:
        Dict[robot_id, List[(i, j, t)]] — matches RobotConfig.path's convention.
    """
    paths = {}
    for a in model.A:
        path = []
        for t in model.T:
            for v in model.V:
                if pyo.value(model.x[a, v, t]) > 0.5:
                    path.append((v[0], v[1], t))
                    break
        paths[a] = path
    return paths


if __name__ == "__main__":
    from quantum.pathFormulation import PathfindingProblem

    problem = PathfindingProblem.from_map_config(
        # "../quantum/maps/synthetic/5x5/obs5x5_medium", problem_name="baseline"
        "../quantum/maps/synthetic/10x10/obs10x10_hard",
        problem_name="four_robots",
    ).as_grid_only()

    model, results = solve(problem)
    print(results.solver.termination_condition)

    paths = decode_paths(model, problem)
    for robot_id, path in paths.items():
        print(robot_id, path)
