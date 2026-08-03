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

    # Each robot only exists for its own [start_time, start_time + T - 1] window,
    # matching the QUBO: no variables before start_time, none after the robot's
    # own goal is reached (regardless of how long the shared horizon runs).
    active_range = {a: range(robots[a].start_time, robots[a].start_time + robots[a].T)
                     for a in robots}

    # Decision variables x[robot, position, time]
    model.x = pyo.Var(model.A, model.V, model.T, within=pyo.Binary)

    for a in model.A:
        for v in model.V:
            for t in model.T:
                if t not in active_range[a]:
                    model.x[a, v, t].fix(0)

    # exactly one cell per robot per timestep, only while the robot exists
    def one_hot_rule(m, a, t):
        if t not in active_range[a]:
            return pyo.Constraint.Skip
        return sum(m.x[a, v, t] for v in m.V) == 1
    model.one_hot = pyo.Constraint(model.A, model.T, rule=one_hot_rule)

    # if at v at time t, must move to a neighbor of v at time t+1, only while
    # both t and t+1 fall inside the robot's own window
    def adjacency_rule(m, a, i, j, t):
        if t not in active_range[a] or (t + 1) not in active_range[a]:
            return pyo.Constraint.Skip
        return m.x[a, (i, j), t] <= sum(m.x[a, vp, t + 1] for vp in reachable[(i, j)])
    model.adjacency = pyo.Constraint(model.A, model.V, model.T_minus, rule=adjacency_rule)

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

    # once at goal, stay at goal, only within the robot's own window
    def goal_lock_rule(m, a, t):
        if t not in active_range[a] or (t + 1) not in active_range[a]:
            return pyo.Constraint.Skip
        return m.x[a, robots[a].goal, t] <= m.x[a, robots[a].goal, t + 1]
    model.goal_lock = pyo.Constraint(model.A, model.T_minus, rule=goal_lock_rule)

    # at most one robot per cell per timestep
    model.crash = pyo.Constraint(
        model.V, model.T,
        rule=lambda m, i, j, t: sum(m.x[a, (i, j), t] for a in m.A) <= 1
    )

    # no two robots may swap positions across an edge between t and t+1
    model.swap = pyo.ConstraintList()
    robot_list = list(robots.keys())
    for t in model.T_minus:
        for idx_i in range(len(robot_list)):
            for idx_j in range(idx_i + 1, len(robot_list)):
                ai, aj = robot_list[idx_i], robot_list[idx_j]
                for v in model.V:
                    for w in grid.adjacency[v]:
                        model.swap.add(
                            model.x[ai, v, t] + model.x[aj, w, t] +
                            model.x[ai, w, t + 1] + model.x[aj, v, t + 1] <= 3
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
