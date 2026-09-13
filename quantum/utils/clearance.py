"""Robot-footprint clearance geometry, shared by CBS / ILP / QUBO.

See quantum/docs/clearance.md for the full derivation and notation this
module implements, and for why the separate "trailing" constraint the
D_ab table also drives (CBS's crossing check, ILPBuilder.model.trailing,
QUBO's planned K_trail) is a distinct hazard, not a generalisation of swap.

Each robot has two real-world scalars (metres):

* ``robot_radius`` — physical body radius (largest half-extent for a
  non-circular robot; robots are treated as circular for now — see the
  bottom of this docstring for what a non-circular footprint would need).
* ``inflation`` — safety margin: the minimum gap the planner keeps between
  this robot's *edge* and any obstacle or other robot's edge.

**Robot vs obstacle** (per robot, additive): a leader cell is illegal for
robot ``k`` when its body + margin would cover an obstacle, i.e. within
``clearance_k = robot_radius_k + inflation_k`` of an obstacle cell.

**Robot vs robot** (per pair, tighter than additive): the two centres must
stay at least

    robot_radius_a + robot_radius_b + factor * max(inflation_a, inflation_b)

apart. The margin is ``max`` not a sum — each local planner just wants *its*
comfort gap between the two edges, not both stacked — with ``factor`` (>= 1)
buying a little extra for odometry / lidar slop (try 1.1–1.2; default 1.0,
never silently baked into the geometry). This is encoded as

    D_ab = { off : chebyshev(off) * resolution < R_pair_metres }

and two robots collide at a timestep iff ``(pos_a - pos_b) in D_ab``.

Sizing everywhere is Chebyshev (square footprints): integer, no ``sqrt``,
conservative by at most ``sqrt(2)`` vs a true disc. The ``< R`` is strict, so
the default ``robot_radius`` of 0.5 at unit resolution with no inflation
gives ``D_ab == {(0, 0)}`` — plain same-cell checking, no obstacle inflation,
i.e. every pre-existing config and result is unchanged until someone sets a
sub-cell resolution or a real ``robot_radius`` / ``inflation``.

For circular robots ``D_ab`` is always a single Chebyshev disc, computed
directly by :func:`pair_separation_offsets` from four scalars -- no set
algebra needed. A non-circular ``robot_footprint`` would need each robot's
own offset set ``S_k`` and the general (and generally *asymmetric* --
subscript order matters) Minkowski difference ``D_ab = S_b (+) (-S_a)``,
tested as ``(pos_a - pos_b) in D_ab``; neither exists here since nothing
needs it yet.
"""

import math
from itertools import product

# Multiplier on the max-inflation term of the robot-robot separation. 1.0 =
# exactly the requested margin; 1.1–1.2 leaves slack for odom/lidar error.
DEFAULT_SEPARATION_FACTOR = 1.0


def footprint_radius_cells(clearance, resolution):
    """Largest integer strictly less than ``clearance / resolution`` (>= 0) —
    the Chebyshev radius by which obstacles are dilated for a robot of this
    clearance."""
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    return max(math.ceil(clearance / resolution) - 1, 0)


def _square(radius_cells):
    """Every ``(di, dj)`` with Chebyshev norm <= radius_cells. Always ``(0,0)``."""
    r = max(int(radius_cells), 0)
    return frozenset(product(range(-r, r + 1), repeat=2))


def pair_separation_offsets(
    radius_a, inflation_a, radius_b, inflation_b, resolution, factor
):
    """``D_ab``: every ``pos_a - pos_b`` cell offset closer than the required
    centre-to-centre separation
    ``radius_a + radius_b + factor * max(inflation_a, inflation_b)``."""
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    reach_m = radius_a + radius_b + factor * max(inflation_a, inflation_b)
    # strict: offsets with chebyshev * resolution < reach_m
    radius_cells = math.ceil(reach_m / resolution) - 1
    return _square(radius_cells)


def build_offset_table(robots, resolution, factor=DEFAULT_SEPARATION_FACTOR):
    """``{(a_id, b_id): D_ab}`` for every ordered pair of distinct robots
    (``robots`` is ``{id: RobotConfig}``). Every entry is ``{(0, 0)}`` — and
    callers fall back to exact same-cell / swap checks — when no pair needs a
    footprint bigger than one cell."""
    table = {}
    ids = list(robots)
    for a in ids:
        ra = robots[a]
        for b in ids:
            if a == b:
                continue
            rb = robots[b]
            table[(a, b)] = pair_separation_offsets(
                ra.robot_radius,
                ra.inflation,
                rb.robot_radius,
                rb.inflation,
                resolution,
                factor,
            )
    return table


def cell_in_obstacle_footprint(cell, obstacle_set, radius_cells):
    """True if ``cell`` is within Chebyshev ``radius_cells`` of any obstacle in
    ``obstacle_set`` (a set of ``(i, j)`` tuples). ``radius_cells == 0`` reduces
    to "is an obstacle"."""
    ci, cj = cell
    if radius_cells <= 0:
        return (ci, cj) in obstacle_set
    return any(
        max(abs(ci - oi), abs(cj - oj)) <= radius_cells for (oi, oj) in obstacle_set
    )


def inflated_obstacle_cells(obstacles, M, N, clearance, resolution):
    """Every in-bounds cell within a robot's footprint radius of an obstacle —
    the cells it may not occupy as a leader. Plain obstacle set (no dilation)
    for a zero-radius footprint."""
    r = footprint_radius_cells(clearance, resolution)
    obstacle_set = {tuple(o) for o in obstacles}
    if r == 0:
        return obstacle_set
    blocked = set()
    for oi, oj in obstacle_set:
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = oi + di, oj + dj
                if 0 <= i < M and 0 <= j < N:
                    blocked.add((i, j))
    return blocked
