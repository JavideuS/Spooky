def is_solution_valid(solution, problem):
    """
    Checks if the binary solution vector represents a valid path from start to goal.
    
    Args:
        solution (dict): Binary solution dict {idx: 0/1}
        problem (PathfindingProblem): Problem definition including grid, start, goal, T
    
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
    moves = grid.moves  # List of allowed move deltas, e.g., [(-1,0), (1,0), ...]

    result = {"valid": True, "details": {}}

    # 1. Get active indices
    active_indices = [idx for idx, val in solution.items() if val == 1]
    if not active_indices:
        result["valid"] = False
        result["reason"] = "no_active_bits"
        result["message"] = "❌ No active variables found. Invalid sample"
        return result

    # 2. Decode positions
    def decode_position(idx):
        t = idx // (M * N)
        pos = idx % (M * N)
        i = pos // N
        j = pos % N
        return i, j, t

    try:
        positions = [decode_position(idx) for idx in active_indices]
    except Exception as e:
        result["valid"] = False
        result["reason"] = "invalid_index"
        result["message"] = f"❌ Error decoding indices: {e}"
        return result

    result["details"]["path"] = positions

    # 3. Sort by time step
    positions.sort(key=lambda x: x[2])

    # 4. Check all time steps from 0 to T-1 are present
    all_times = set(t for _, _, t in positions)
    expected_times = set(range(T))  # We expect exactly T time steps

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

    # 5. One-hot constraint per time step
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

    # 6. Start at correct position
    first_time_cells = time_to_positions.get(0, [])
    if not first_time_cells or first_time_cells[0] != start:
        result["valid"] = False
        result["reason"] = "start_mismatch"
        result["message"] = f"❌ Start position mismatch: Expected {start}, got {first_time_cells}"
        return result

    # 7. Goal reached at some time step
    goal_reached = any(i == goal[0] and j == goal[1] for i, j, t in positions)
    if not goal_reached:
        result["valid"] = False
        result["reason"] = "goal_not_reached"
        result["message"] = f"❌ Goal {goal} was never reached"
        return result

    # 8. Movement must be valid (adjacent cells only)
    last_pos = None
    for i, j, t in sorted(positions, key=lambda x: x[2]):
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

    # 9. Final check: Did we reach the goal at the right time?
    goal_times = [t for i, j, t in positions if i == goal[0] and j == goal[1]]
    result["details"]["goal_times"] = goal_times

    result["valid"] = True
    result["reason"] = "valid_path"
    result["message"] = "✅ Solution is valid"

    return result
