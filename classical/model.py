import pyomo.environ as pyo


def build_model(problem):
    """
    Build a time-expanded ILP model for the grid MAPF problem.
    Mirrors the (robot, time, position) indexing used by QUBOBuilder,
    but replaces penalty terms with hard constraints.
    """
    grid = problem.grid
    robots = problem.robots

    model = pyo.ConcreteModel()

    model.A = pyo.Set(initialize=list(robots.keys()))
    model.V = pyo.Set(
        initialize=[
            (i, j)
            for i in range(grid.M)
            for j in range(grid.N)
            if (i, j) not in grid.obstacles
        ]
    )
    model.T = pyo.RangeSet(0, problem.T - 1)
    model.T_minus = pyo.RangeSet(0, problem.T - 2)

    # ILP allows staying in place; QUBO's grid.adjacency is kept strict-neighbor-only
    # since it relies on that for its own penalty terms, so extend it locally here.
    reachable = {v: grid.adjacency[v] + [v] for v in model.V}

    # Decision variables x[robot, position, time]
    model.x = pyo.Var(model.A, model.V, model.T, within=pyo.Binary)

    # exactly one cell per robot per timestep
    model.one_hot = pyo.Constraint(
        model.A, model.T, rule=lambda m, a, t: sum(m.x[a, v, t] for v in m.V) == 1
    )

    # if at v at time t, must move to a neighbor of v at time t+1
    model.adjacency = pyo.Constraint(
        model.A, model.V, model.T_minus,
        rule=lambda m, a, i, j, t: m.x[a, (i, j), t] <= sum(
            m.x[a, vp, t + 1] for vp in reachable[(i, j)]
        )
    )

    # at start position at each robot's own start time
    model.start = pyo.Constraint(
        model.A,
        rule=lambda m, a: m.x[a, robots[a].current_position, robots[a].start_time] == 1
    )

    # at goal position at each robot's own goal time
    model.goal = pyo.Constraint(
        model.A,
        rule=lambda m, a: m.x[a, robots[a].goal, robots[a].start_time + robots[a].T - 1] == 1
    )

    # once at goal, stay at goal
    model.goal_lock = pyo.Constraint(
        model.A, model.T_minus,
        rule=lambda m, a, t: m.x[a, robots[a].goal, t] <= m.x[a, robots[a].goal, t + 1]
    )

    # minimize total timesteps spent away from goal (equivalent to sum of arrival times,
    # given goal_lock forces x[a, g_a, ·] to be monotone)
    model.obj = pyo.Objective(
        sense=pyo.minimize,
        expr=sum(
            model.x[a, v, t]
            for a in model.A for t in model.T
            for v in model.V if v != robots[a].goal
        )
    )

    return model
