"""
Phase 1 sanity checks for binary-encoding plumbing in BaseQUBO: bit-width
computation, binary_var_index allocation, and roundtrip decode via
paths.decode_position_binary. No penalty/constraint logic is exercised here.
"""
import numpy as np
import pytest

from quantum.map import Grid, Graph
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.builder.GraphQUBO import GraphQUBO
from quantum.utils import paths

PENALTIES = {
    "K_hot": 9,
    "K_adj": 4.8,
    "K_start": 6.5,
    "K_goal": 3,
    "K_lock": 4,
    "K_bt": 2.3,
    "K_tp": 1.2,
    "K_goal_approx": 0.7,
    "K_obs": 0,
    "K_crash": 3,
}


def make_single_robot_problem():
    """Trivial single-robot, no-obstacle 2x2 grid problem: (1,0) -> (0,1), T=3."""
    grid = Grid(M=2, N=2, obstacles=[])
    robot = RobotConfig("r0", start=(1, 0), goal=(0, 1), expected_duration=3)
    return PathfindingProblem([robot], grid=grid, T=3)


def test_default_encoding_is_one_hot():
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES)
    assert builder.encoding == "one_hot"


def test_invalid_encoding_rejected():
    problem = make_single_robot_problem()
    with pytest.raises(ValueError):
        GridQUBOBuilder(problem, PENALTIES, encoding="not_a_real_encoding")


def test_bit_width_matches_grid_size():
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, encoding="binary")
    # 2x2 grid -> 4 positions -> B = ceil(log2(4)) = 2
    assert builder.B == 2
    assert paths.bit_width(4) == 2
    assert paths.bit_width(1) == 0
    assert paths.bit_width(5) == 3


def test_binary_var_count_matches_robots_times_T_times_B():
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, encoding="binary")

    num_robots = problem.num_robots
    T = problem.T
    B = builder.B

    allocated = {
        builder.binary_var_index(r, t, b)
        for r in range(num_robots)
        for t in range(T)
        for b in range(B)
    }
    assert len(allocated) == num_robots * T * B
    assert allocated == set(range(num_robots * T * B))


def test_binary_var_index_formula():
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, encoding="binary")
    T, B = builder.total_t, builder.B
    for r in range(problem.num_robots):
        for t in range(T):
            for b in range(B):
                assert builder.binary_var_index(r, t, b) == r * T * B + t * B + b


def test_decode_position_binary_roundtrip():
    problem = make_single_robot_problem()
    N = problem.grid.N  # 2
    B = 2
    for i in range(problem.grid.M):
        for j in range(N):
            code = i * N + j
            bits = [(code >> b) & 1 for b in range(B)]  # bits[b] is coefficient of 2**b (LSB-first)
            assert paths.bits_to_code(bits) == code
            assert paths.decode_position_binary(bits, problem) == (i, j)


# --- Stage 1: build()/windowing/reduction branching (no constraint logic) ---

def make_5x5_problem():
    """Single-robot, no-obstacle 5x5 grid, long horizon so windowing kicks in."""
    grid = Grid(M=5, N=5, obstacles=[])
    robot = RobotConfig("r0", start=(4, 0), goal=(0, 4), expected_duration=20)
    return PathfindingProblem([robot], grid=grid, T=20)


def test_binary_build_empty_constraints_does_not_crash():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    assert builder.build(constraints_to_apply=[]) == {}


def test_binary_build_default_constraints_not_implemented():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    with pytest.raises(NotImplementedError):
        builder.build()


def test_one_hot_build_default_constraints_unaffected():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="one_hot")
    Q = builder.build()
    assert len(Q) > 0


def test_binary_get_logical_variables_fixes_start_bits():
    """Start (4,0) on a 5x5 grid = code 20 = 0b10100. Only the start
    timestep's B bits are fixed (no partial-reachability analog exists for
    binary); active_cells stays empty since binary doesn't use it."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    # Instance-level override so this test is independent of whatever
    # _FIX_BINARY_START is currently set to for manual comparison testing
    # (see base_qubo.py) -- this test is specifically about the "on" behavior.
    builder._FIX_BINARY_START = True
    fixed_ones, active_cells = builder.get_logical_variables()
    expected = {
        builder.binary_var_index(0, 0, 0): 0,
        builder.binary_var_index(0, 0, 1): 0,
        builder.binary_var_index(0, 0, 2): 1,
        builder.binary_var_index(0, 0, 3): 0,
        builder.binary_var_index(0, 0, 4): 1,
    }
    assert fixed_ones == expected
    assert active_cells == {}


def test_binary_diag_fixed_vars_is_noop():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    assert builder.diag_fixed_vars() == {}


def test_binary_window_grows_flat_one_hot_grows_with_connectivity():
    """Core hypothesis check: for the same var_limit, binary's flat B-per-
    timestep growth should allow a larger window than one-hot's
    connectivity^step growth on a 5x5 grid with no BFS reduction applied to
    either (stage 1 has no reduction yet)."""
    problem = make_5x5_problem()
    binary_builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    one_hot_builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="one_hot")

    assert binary_builder.max_window_size() > one_hot_builder.max_window_size()


# --- Same stage-1 checks for GraphQUBO ---

def make_5x5_graph_problem():
    """Single-robot 5x5 grid graph (4-connected), long horizon so windowing kicks in."""
    nodes = [(i, j) for i in range(5) for j in range(5)]
    edges = []
    for i in range(5):
        for j in range(5):
            n = i * 5 + j
            if j + 1 < 5:
                edges.append((n, n + 1))
            if i + 1 < 5:
                edges.append((n, n + 5))
    graph = Graph(nodes, edges)
    robot = RobotConfig("r0", start=0, goal=24, expected_duration=20)
    return PathfindingProblem([robot], graph=graph, T=20)


def test_graph_binary_build_empty_constraints_does_not_crash():
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    assert builder.build(constraints_to_apply=[]) == {}


def test_graph_binary_build_default_constraints_not_implemented():
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    with pytest.raises(NotImplementedError):
        builder.build()


def test_graph_binary_get_logical_variables_fixes_start_bits():
    """Start node 0 -> all-zero code; only the start timestep's B bits are
    fixed, active_cells stays empty (binary doesn't use it)."""
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    builder._FIX_BINARY_START = True
    fixed_ones, active_cells = builder.get_logical_variables()
    expected = {builder.binary_var_index(0, 0, b): 0 for b in range(builder.B)}
    assert fixed_ones == expected
    assert active_cells == {}


def test_binary_get_logical_variables_noop_when_start_fix_disabled():
    """_FIX_BINARY_START=False (e.g. for A/B testing against the LS
    adjacency fit in isolation) reverts get_logical_variables() to the
    original no-op -- start is then only soft-biased via
    apply_start_penalty_binary's K_start term, never hard-pinned."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    builder._FIX_BINARY_START = False
    fixed_ones, active_cells = builder.get_logical_variables()
    assert fixed_ones == {}
    assert active_cells == {}


def test_graph_binary_window_grows_flat_one_hot_grows_with_connectivity():
    problem = make_5x5_graph_problem()
    binary_builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    one_hot_builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="one_hot")

    assert binary_builder.max_window_size() > one_hot_builder.max_window_size()


# --- Stage 2: implemented binary constraints (start, goal, adjacency-reward
# LS fit) and the documented LP dead end. See base_qubo.py's
# _BINARY_ADJACENCY_FIT_METHOD docstring: an exact LP max-margin fit of the
# adjacency block is confirmed infeasible for full grid/graph adjacency, so
# LS regression is the current default and the only one exercised for
# correctness here — the LP path is only checked to still fail the way it's
# documented to.

GRID_UNIMPLEMENTED_BINARY_CONSTRAINTS = [
    "one_hot",
    "adjacency_penalty",
    "goal_fix",
    "goal_early",
    "lock",
    "backtracking",
    "tp",
    "terrain",
    "elevation",
    "obstacle",
    "multi_robot",
]

GRAPH_UNIMPLEMENTED_BINARY_CONSTRAINTS = [
    "one_hot",
    "adjacency",
    "lock",
    "backtracking",
    "multi_robot_collision",
]


def test_start_penalty_binary_closed_form():
    """(1,0) -> code 2 = 0b10 on a 2x2 grid: bit0=0, bit1=1."""
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, encoding="binary")
    K_start = PENALTIES["K_start"]

    Q = builder.build(constraints_to_apply=["start"])

    idx_b0 = builder.binary_var_index(0, 0, 0)
    idx_b1 = builder.binary_var_index(0, 0, 1)
    assert Q == {
        (idx_b0, idx_b0): K_start,       # bit 0 of code 2 is 0 -> K_start*(1-0)
        (idx_b1, idx_b1): -K_start,      # bit 1 of code 2 is 1 -> K_start*(1-2)
    }


def test_goal_later_penalty_binary_closed_form():
    """(0,1) -> code 1 = 0b01: bit0=1, bit1=0. Applied at t=1,2 (start+1..end)
    with growing weight K_goal*(1+t/T)."""
    problem = make_single_robot_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, encoding="binary")
    K_goal = PENALTIES["K_goal"]
    T = builder.total_t

    Q = builder.build(constraints_to_apply=["goal_later"])

    expected = {}
    for t in (1, 2):
        time_factor = 1 + t / T
        for b, g_b in ((0, 1), (1, 0)):
            idx = builder.binary_var_index(0, t, b)
            expected[(idx, idx)] = K_goal * time_factor * (1 - 2 * g_b)

    assert Q.keys() == expected.keys()
    for key, val in expected.items():
        assert Q[key] == pytest.approx(val)


def test_adjacency_reward_binary_block_shape_and_upper_triangular():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    builder.build(constraints_to_apply=["adjacency_reward"])

    block = builder._adjacency_binary_block
    assert block.shape == (2 * builder.B, 2 * builder.B)
    assert np.all(np.tril(block, -1) == 0)


def test_adjacency_reward_binary_block_cached_across_builds():
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    assert builder._adjacency_binary_block is None

    builder.build(constraints_to_apply=["adjacency_reward"])
    first_block = builder._adjacency_binary_block
    assert first_block is not None

    builder.build(constraints_to_apply=["adjacency_reward"])
    assert builder._adjacency_binary_block is first_block


def _block_energy(block, bits):
    n_bits = len(bits)
    energy = 0.0
    for i in range(n_bits):
        energy += block[i, i] * bits[i]
        for j in range(i + 1, n_bits):
            energy += block[i, j] * bits[i] * bits[j]
    return energy


def test_adjacency_reward_binary_ls_fit_favors_real_neighbors():
    """The LS fit is a soft regression, not an exact separator (LP proved no
    exact separator exists — see test_adjacency_lp_fit_raises_on_full_grid),
    so this only checks the fit points the right direction: real adjacent
    transitions (target 0) should score lower on average than non-adjacent
    ones, across the full 5x5 truth table."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    builder.build(constraints_to_apply=["adjacency_reward"])
    block = builder._adjacency_binary_block

    N = problem.grid.N
    adjacency = problem.grid.adjacency

    def neighbor_codes(code):
        i, j = divmod(code, N)
        return [k * N + l for k, l in adjacency.get((i, j), [])]

    num_positions = problem.grid.M * N
    bits_list, targets = builder._binary_transition_truth_table(num_positions, neighbor_codes)

    valid_energies = [
        _block_energy(block, bits) for bits, t in zip(bits_list, targets) if t == 0
    ]
    invalid_energies = [
        _block_energy(block, bits) for bits, t in zip(bits_list, targets) if t != 0
    ]
    assert valid_energies and invalid_energies
    assert sum(valid_energies) / len(valid_energies) < sum(invalid_energies) / len(invalid_energies)


def test_binary_transition_truth_table_ghost_target():
    """25 real positions need B=5 bits (2^5=32); codes 25..31 don't decode to
    any real cell and must all collapse to one fixed target: one worse than
    the largest real BFS hop-distance on the grid."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")

    N = problem.grid.N
    adjacency = problem.grid.adjacency

    def neighbor_codes(code):
        i, j = divmod(code, N)
        return [k * N + l for k, l in adjacency.get((i, j), [])]

    num_positions = problem.grid.M * N
    dist = builder._all_pairs_bfs_distance(num_positions, neighbor_codes)
    ghost_target = max(dist.values()) + 1

    bits_list, targets = builder._binary_transition_truth_table(num_positions, neighbor_codes)
    B = builder.B
    ghost_rows = [
        t
        for bits, t in zip(bits_list, targets)
        if paths.bits_to_code(bits[:B]) >= num_positions or paths.bits_to_code(bits[B:]) >= num_positions
    ]
    assert ghost_rows  # 5x5 isn't a power of 2, so ghost codes exist
    assert all(t == ghost_target for t in ghost_rows)


def test_adjacency_lp_fit_raises_on_full_grid():
    """Documents the confirmed-infeasible LP path (see
    BaseQUBO._fit_binary_pairwise_block_lp's docstring): no exact quadratic
    separator exists for full 5x5 grid adjacency, so the LP fit must raise
    rather than silently return a useless block."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")

    N = problem.grid.N
    adjacency = problem.grid.adjacency

    def neighbor_codes(code):
        i, j = divmod(code, N)
        return [k * N + l for k, l in adjacency.get((i, j), [])]

    num_positions = problem.grid.M * N
    with pytest.raises(RuntimeError):
        builder._fit_binary_pairwise_block_lp(num_positions, neighbor_codes)


def _make_grid_lp_input(M, N):
    """Fresh no-obstacle MxN grid problem + the neighbor_codes closure
    _fit_binary_pairwise_block_lp needs, decoupled from the 5x5 fixture so
    the boundary test below can sweep sizes."""
    grid = Grid(M=M, N=N, obstacles=[])
    robot = RobotConfig("r0", start=(M - 1, 0), goal=(0, N - 1), expected_duration=(M + N) * 2)
    problem = PathfindingProblem([robot], grid=grid, T=(M + N) * 2)
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=9999, encoding="binary")

    adjacency = grid.adjacency

    def neighbor_codes(code):
        i, j = divmod(code, N)
        return [k * N + l for k, l in adjacency.get((i, j), [])]

    return builder, neighbor_codes


def test_adjacency_lp_separability_boundary():
    """Characterizes *where* the LP max-margin fit succeeds vs. fails,
    rather than just asserting failure at one size. Empirically the
    boundary is not simply "small grid = separable, big grid = infeasible":
    2x2 (B=2, exactly 2^B positions, no slack ghost states) already fails,
    while 2x3 and 3x3 (a few spare ghost codes to absorb the discrimination
    burden) succeed, and everything from 3x4 upward fails again. This locks
    in that non-monotonic shape as a known fact so a future change to the
    fit (or to B's bit-width formula) that shifts the boundary gets noticed
    instead of silently drifting. See _fit_binary_pairwise_block_lp's
    docstring for why: the (2B x 2B) block is shared/tiled across every
    "from" cell, and its O(B^2) free coefficients run out of capacity to
    separate O(num_positions^2) transition targets as the grid grows.

    Uses _solve_binary_pairwise_lp_margin directly (the non-raising core)
    instead of catching RuntimeError from _fit_binary_pairwise_block_lp, so
    a failing case is pinned to the right *reason*: scipy always reports
    success=True here (the LP is classifiable — it solves cleanly to
    optimality), it's the margin itself that comes back exactly 0. That
    distinguishes "no quadratic form exists" from "the solver choked" —
    if this ever starts asserting success=False instead, that's a
    genuinely different failure mode worth looking at, not the same old
    known limitation."""
    cases = [
        (2, 2, False),
        (2, 3, True),
        (3, 3, True),
        (2, 4, False),
        (3, 4, False),
        (4, 4, False),
        (5, 5, False),
    ]
    for M, N, expected_separable in cases:
        builder, neighbor_codes = _make_grid_lp_input(M, N)
        num_positions = M * N
        success, message, margin, block = builder._solve_binary_pairwise_lp_margin(
            num_positions, neighbor_codes
        )
        assert success, f"{M}x{N}: LP was not even classifiable ({message})"
        assert block.shape == (2 * builder.B, 2 * builder.B)
        if expected_separable:
            assert margin > 1e-9, f"{M}x{N}: expected a positive separating margin, got {margin}"
        else:
            assert margin == pytest.approx(0.0, abs=1e-9), (
                f"{M}x{N}: expected margin exactly 0 (classifiable, no separator), got {margin}"
            )


def test_adjacency_lp_fit_lp_raises_specifically_on_zero_margin():
    """Pins the public _fit_binary_pairwise_block_lp entry point to raising
    for the "no separating margin" reason specifically (not a generic
    RuntimeError that could also mean scipy-level infeasibility) on the
    5x5 grid used elsewhere in this file."""
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")

    N = problem.grid.N
    adjacency = problem.grid.adjacency

    def neighbor_codes(code):
        i, j = divmod(code, N)
        return [k * N + l for k, l in adjacency.get((i, j), [])]

    num_positions = problem.grid.M * N
    with pytest.raises(RuntimeError, match="no separating margin"):
        builder._fit_binary_pairwise_block_lp(num_positions, neighbor_codes)


@pytest.mark.parametrize("constraint", GRID_UNIMPLEMENTED_BINARY_CONSTRAINTS)
def test_binary_unimplemented_constraints_raise(constraint):
    problem = make_5x5_problem()
    builder = GridQUBOBuilder(problem, PENALTIES, var_limit=60, encoding="binary")
    with pytest.raises(NotImplementedError):
        builder.build(constraints_to_apply=[constraint])


# --- Same stage-2 checks for GraphQUBO ---

def test_graph_start_penalty_binary_closed_form():
    """start node 0 -> code 0 (all bits 0), so every bit contributes
    +K_start (the (1 - 2*0) collapse)."""
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    K_start = PENALTIES["K_start"]

    Q = builder.build(constraints_to_apply=["start"])

    expected = {
        (builder.binary_var_index(0, 0, b), builder.binary_var_index(0, 0, b)): K_start
        for b in range(builder.B)
    }
    assert Q == expected


def test_graph_goal_penalty_binary_closed_form():
    """goal node 24 -> code 24 = 0b11000: bits 3 and 4 set. Applied at every
    t in start+1..end with growing weight K_goal*(1+t/T)."""
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    K_goal = PENALTIES["K_goal"]
    T = builder.total_t
    goal_bits = {b: (24 >> b) & 1 for b in range(builder.B)}

    Q = builder.build(constraints_to_apply=["goal"])

    start_time = 0
    end = min(builder.t_max, problem.robots["r0"].T)
    expected = {}
    for t in range(start_time + 1, end):
        time_factor = 1 + t / T
        for b, g_b in goal_bits.items():
            idx = builder.binary_var_index(0, t, b)
            expected[(idx, idx)] = K_goal * time_factor * (1 - 2 * g_b)

    assert Q.keys() == expected.keys()
    for key, val in expected.items():
        assert Q[key] == pytest.approx(val)


def test_graph_adjacency_reward_binary_block_shape_and_cached():
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    assert builder._adjacency_binary_block is None

    builder.build(constraints_to_apply=["adjacency_reward"])
    block = builder._adjacency_binary_block
    assert block.shape == (2 * builder.B, 2 * builder.B)

    builder.build(constraints_to_apply=["adjacency_reward"])
    assert builder._adjacency_binary_block is block


@pytest.mark.parametrize("constraint", GRAPH_UNIMPLEMENTED_BINARY_CONSTRAINTS)
def test_graph_binary_unimplemented_constraints_raise(constraint):
    problem = make_5x5_graph_problem()
    builder = GraphQUBO(problem, PENALTIES, var_limit=60, encoding="binary")
    with pytest.raises(NotImplementedError):
        builder.build(constraints_to_apply=[constraint])
