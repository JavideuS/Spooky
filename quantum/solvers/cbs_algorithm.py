"""
Conflict-Based Search (CBS) for multi-robot path finding over a plain
networkx.Graph of integer node ids. Pure algorithm — no BaseSolver/builder
coupling, see CBS_solver.py for the adapter that plugs this into the rest of
the pipeline.

Two-level search:
- Low level: SpaceTimeAStar, single-robot shortest path in space-time,
  avoiding a set of forbidden (node, time) vertices and (from, to, time)
  edge transitions, honoring an optional per-step "legal cells" restriction.
- High level: ConflictBasedSearch, a best-first search over a constraint
  tree. Each node holds one path per robot and a growing set of per-robot
  constraints; on a conflict between two robots, branch into two children,
  each forbidding one robot from the offending vertex/edge, and replan just
  that robot.

Paths returned by the high level are padded with repeated goal positions
from first arrival through each robot's own deadline (start_time + T - 1) —
this matches ILPSolver's goal_lock semantics (a robot occupies/blocks its
goal cell for the rest of its own active window, not indefinitely), which is
what CBSSolver's cost-equivalence with ILPSolver depends on. Cost (sum of
arrival offsets) is computed from the trimmed, unpadded arrival time, not
the padded path length, for the same reason.
"""

import heapq
import time as timing
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


def stable_hash(value: str) -> int:
    """Deterministic string->int hash, unlike Python's built-in hash() which
    is randomized per-process (PYTHONHASHSEED) for str/bytes since 3.3 —
    using that here would make CBS's tie-breaking (and therefore its
    results) non-reproducible across runs/processes, defeating the point of
    a reproducible benchmark sweep."""
    return zlib.crc32(value.encode())


@dataclass(order=True)
class AStarNode:
    f_score: float
    g_score: int = field(compare=False)
    node: int = field(compare=False)
    time: int = field(compare=False)
    parent: Optional["AStarNode"] = field(default=None, compare=False)


class SpaceTimeAStar:
    """Single-robot space-time A* with a reservation-avoidance interface
    (forbidden vertices/edges passed in per call, not stored on self) so one
    instance can be reused across all of CBS's replanning calls."""

    # Smaller than the smallest possible true cost gap between two distinct-
    # cost paths (every move costs exactly 1). Applied once per pushed node,
    # not summed with the parent's bias (f_score is tentative_g + heuristic
    # + bias(neighbor), and tentative_g is the true, unbiased path cost) —
    # so it never compounds across a path's depth, it only ever perturbs
    # ordering among candidates whose true g_score is exactly tied.
    _TIE_BREAK_EPSILON = 1e-4

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._pos = nx.get_node_attributes(graph, "pos")

    def _heuristic(self, a: int, b: int) -> float:
        if a in self._pos and b in self._pos:
            pa, pb = self._pos[a], self._pos[b]
            return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
        return 0.0

    def _tie_break_bias(self, seed: int, node: int) -> float:
        # stable_hash, not builtin hash() — this module defines stable_hash
        # specifically to avoid hash()'s per-process randomization for
        # str/bytes; hash() on an int tuple happens to not be randomized
        # either, but using it right next to stable_hash() here would be a
        # trap for whoever next touches this (looks unsafe even though it
        # isn't, in a module where that distinction actually matters).
        return (stable_hash(f"{seed}:{node}") % 10007) / 10007 * self._TIE_BREAK_EPSILON

    @staticmethod
    def _can_hold_goal(
        goal: int,
        arrival_time: int,
        start_time: int,
        deadline: int,
        forbidden_vertices: Set[Tuple[int, int]],
        legal_cells: Optional[List[Set[int]]],
    ) -> bool:
        """Whether arriving at goal at arrival_time is actually a valid
        stopping point — i.e. the robot can then sit there through its own
        deadline (what the caller's padding step does) without violating
        any constraint at any of those held timesteps, not just the
        constraints that apply during the search up to arrival_time itself.
        legal_cells is indexed by step count from start_time (same
        convention as the neighbor-expansion loop below), not absolute
        time, hence the `t - start_time`. Edge constraints don't need
        checking here: every edge constraint this codebase ever generates
        has from_node != to_node (see ConflictBasedSearch._all_conflicts's
        edge-conflict branch), so a "stay in place" transition — the padded
        portion — can never match one."""
        return all(
            (goal, t) not in forbidden_vertices
            and (
                legal_cells is None
                or (t - start_time < len(legal_cells) and goal in legal_cells[t - start_time])
            )
            for t in range(arrival_time, deadline + 1)
        )

    def find_path(
        self,
        start: int,
        goal: int,
        start_time: int,
        deadline: int,
        forbidden_vertices: Set[Tuple[int, int]],
        forbidden_edges: Set[Tuple[int, int, int]],
        legal_cells: Optional[List[Set[int]]] = None,
        tie_break_seed: int = 0,
    ) -> Optional[List[Tuple[int, int]]]:
        """Returns [(node, t), ...] from (start, start_time) to the first
        arrival at goal, or None if no path exists within the deadline.
        Not padded — the caller pads to the robot's own deadline if needed.

        tie_break_seed: perturbs which of several *equal-cost* paths gets
        returned (see _TIE_BREAK_EPSILON) without ever affecting the actual
        shortest cost found. Pass a different seed per robot (e.g. hash of
        robot_id) so multiple robots planned independently in a symmetric
        map don't all default to the identical canonical route — reduces
        how often CBS's root starts out with easily-avoidable conflicts
        baked in from a shared tie-break, purely a search-space diversity
        heuristic, not required for correctness."""
        if (start, start_time) in forbidden_vertices:
            return None

        start_node = AStarNode(
            f_score=self._heuristic(start, goal), g_score=0, node=start, time=start_time
        )
        open_heap = [start_node]
        best_g: Dict[Tuple[int, int], int] = {(start, start_time): 0}

        while open_heap:
            current = heapq.heappop(open_heap)
            state = (current.node, current.time)
            if best_g.get(state, current.g_score + 1) < current.g_score:
                continue  # stale heap entry, a cheaper path to this state was already found

            if current.node == goal and self._can_hold_goal(
                goal, current.time, start_time, deadline, forbidden_vertices, legal_cells
            ):
                return self._reconstruct(current)

            if current.time >= deadline:
                continue

            next_time = current.time + 1
            for neighbor in list(self.graph.neighbors(current.node)) + [current.node]:
                if legal_cells is not None:
                    step = next_time - start_time
                    if step >= len(legal_cells) or neighbor not in legal_cells[step]:
                        continue
                if (neighbor, next_time) in forbidden_vertices:
                    continue
                if (current.node, neighbor, current.time) in forbidden_edges:
                    continue

                tentative_g = current.g_score + 1
                state = (neighbor, next_time)
                if tentative_g < best_g.get(state, tentative_g + 1):
                    best_g[state] = tentative_g
                    heapq.heappush(
                        open_heap,
                        AStarNode(
                            f_score=tentative_g
                            + self._heuristic(neighbor, goal)
                            + self._tie_break_bias(tie_break_seed, neighbor),
                            g_score=tentative_g,
                            node=neighbor,
                            time=next_time,
                            parent=current,
                        ),
                    )

        return None

    @staticmethod
    def _reconstruct(node: AStarNode) -> List[Tuple[int, int]]:
        path = []
        current = node
        while current is not None:
            path.append((current.node, current.time))
            current = current.parent
        path.reverse()
        return path


@dataclass(order=True)
class CTNode:
    """One node of CBS's constraint tree. Ordered by (cost, conflict_count)
    — on equal cost, the node with fewer remaining conflicts is explored
    first (a standard CBS tie-break: it steers the search toward a
    conflict-free node with less branching than an arbitrary equal-cost
    order would). conflicts caches the full scan from when this node was
    created (see ConflictBasedSearch._all_conflicts), so solve()'s main
    loop doesn't have to re-scan an unchanged solution just to look up
    what it already computed at push time."""

    cost: float
    conflict_count: int = 0
    vertex_constraints: Dict[str, Set[Tuple[int, int]]] = field(
        compare=False, default_factory=dict
    )
    edge_constraints: Dict[str, Set[Tuple[int, int, int]]] = field(
        compare=False, default_factory=dict
    )
    solution: Dict[str, List[Tuple[int, int]]] = field(compare=False, default_factory=dict)
    conflicts: List[tuple] = field(compare=False, default_factory=list)


class ConflictBasedSearch:
    """CBS over a plain networkx.Graph of integer node ids."""

    def __init__(self, graph: nx.Graph, node_limit: int = 5000, time_limit: Optional[float] = None):
        self.graph = graph
        self.node_limit = node_limit
        self.time_limit = time_limit
        self._astar = SpaceTimeAStar(graph)

    @staticmethod
    def _pad(path: List[Tuple[int, int]], goal: int, deadline: int) -> List[Tuple[int, int]]:
        padded = list(path)
        last_t = path[-1][1]
        for t in range(last_t + 1, deadline + 1):
            padded.append((goal, t))
        return padded

    @staticmethod
    def _arrival_offset(path: List[Tuple[int, int]], goal: int, start_time: int) -> int:
        t_arrive = path[-1][1]
        for node, t in reversed(path):
            if node != goal:
                break
            t_arrive = t
        return t_arrive - start_time

    def _low_level(
        self,
        robot_id: str,
        robots_meta: Dict[str, dict],
        ct_node: CTNode,
        legal_cells: Optional[Dict[str, List[Set[int]]]],
    ) -> Optional[List[Tuple[int, int]]]:
        meta = robots_meta[robot_id]
        # CBS's low level only ever sees the explicit per-robot constraints
        # accumulated from root to this CT node (added by _all_conflicts's
        # branching, below) — never the other robots' current paths in this
        # node's solution. Treating those as hard obstacles would silently
        # turn every replan into prioritized planning: the replanned robot
        # becomes trivially conflict-free against whichever paths happen to
        # be in ct_node.solution right now, so the first branch typically
        # pops "conflict-free" immediately (a prioritized-planning result
        # wearing an "optimal" label), and an over-constrained replan
        # returns None and silently prunes a subtree CBS could have solved
        # — up to reporting "no_solution"/"infeasible" on feasible
        # instances. Conflicts against other robots are exactly what
        # _all_conflicts() checks for on the *joint* solution after every
        # replan; the low level must stay ignorant of them.
        forbidden_vertices = set(ct_node.vertex_constraints.get(robot_id, set()))
        forbidden_edges = set(ct_node.edge_constraints.get(robot_id, set()))

        raw = self._astar.find_path(
            meta["start"],
            meta["goal"],
            meta["start_time"],
            meta["deadline"],
            forbidden_vertices,
            forbidden_edges,
            legal_cells=legal_cells.get(robot_id) if legal_cells else None,
            tie_break_seed=stable_hash(robot_id),
        )
        if raw is None:
            return None
        return self._pad(raw, meta["goal"], meta["deadline"])

    def _total_cost(self, solution: Dict[str, List[Tuple[int, int]]], robots_meta: Dict[str, dict]) -> float:
        return sum(
            self._arrival_offset(path, robots_meta[rid]["goal"], robots_meta[rid]["start_time"])
            for rid, path in solution.items()
        )

    @staticmethod
    def _all_conflicts(solution: Dict[str, List[Tuple[int, int]]]) -> List[tuple]:
        """Every conflict in solution (vertex + edge), sorted by time — the
        full scan, not just the earliest one. Used both to pick the
        conflict solve() branches on (conflicts[0]) and, via len(), as
        CTNode.conflict_count for tie-breaking equal-cost nodes. Each
        conflict is either (t, "vertex", node, [robot_ids]) or
        (t, "edge", robot_a, robot_b, node_a_at_t, node_a_at_t1) — matching
        the two conflict types quantum/benchmark/benchmark.py's
        is_solution_valid() checks (same-cell/same-time, and swap)."""
        occupancy: Dict[Tuple[int, int], List[str]] = {}
        for robot_id, path in solution.items():
            for node, t in path:
                occupancy.setdefault((node, t), []).append(robot_id)

        conflicts = [
            (t, "vertex", node, robots)
            for (node, t), robots in occupancy.items()
            if len(robots) > 1
        ]

        robot_ids = list(solution.keys())
        for i in range(len(robot_ids)):
            a = robot_ids[i]
            pos_a = {t: n for n, t in solution[a]}
            for j in range(i + 1, len(robot_ids)):
                b = robot_ids[j]
                pos_b = {t: n for n, t in solution[b]}
                for t in set(pos_a) & set(pos_b):
                    if t + 1 not in pos_a or t + 1 not in pos_b:
                        continue
                    if (
                        pos_a[t] == pos_b[t + 1]
                        and pos_b[t] == pos_a[t + 1]
                        and pos_a[t] != pos_a[t + 1]
                    ):
                        conflicts.append((t, "edge", a, b, pos_a[t], pos_a[t + 1]))

        conflicts.sort(key=lambda c: c[0])
        return conflicts

    def solve(
        self,
        robots_meta: Dict[str, dict],
        legal_cells: Optional[Dict[str, List[Set[int]]]] = None,
    ) -> Tuple[Dict[str, List[Tuple[int, int]]], dict]:
        """robots_meta[robot_id] = {"start": node, "goal": node, "start_time": int, "deadline": int}.
        Returns (solution, meta) where meta["termination_condition"] is one of:
          - "optimal": solution is a genuine, conflict-free, minimum-cost
            joint plan.
          - "infeasible": some robot has no path even with no other robots
            to avoid — checked before any branching starts, a hard
            impossibility independent of budget.
          - "no_solution": the entire constraint tree was exhausted (heap
            emptied) without ever finding a conflict-free node. Given CBS's
            completeness guarantee, this is a proof the joint problem has no
            solution — not a budget cutoff.
          - "node_limit_exceeded" / "time_limit_exceeded": search was cut
            off before resolving feasibility either way. Genuinely
            inconclusive — could be feasible or not.

        Unlike ILPSolver, CBS has no "best incumbent so far" to fall back on
        for the non-"optimal" cases: every node this search ever pops either
        has zero conflicts (returned immediately as "optimal") or at least
        one (branched on) — there is no such thing as a partially-resolved
        CBS solution, a colliding joint plan isn't a "worse but usable"
        answer the way a suboptimal-but-feasible ILP incumbent is. So
        solution is {} (empty) for every termination_condition except
        "optimal" — callers must check termination_condition before trusting
        solution/energy, not treat a budget cutoff as if it were a
        worse-but-valid result."""
        root = CTNode(cost=0.0)
        for robot_id in robots_meta:
            # Root has no constraints for anyone yet, so each robot's path
            # here is genuinely independent of every other robot — _low_level
            # only ever looks at ct_node's constraint sets (empty at root),
            # never at ct_node.solution, so passing `root` itself (already
            # partially filled in by earlier robots in this loop) is exactly
            # as independent as a fresh empty node would be.
            path = self._low_level(robot_id, robots_meta, root, legal_cells)
            if path is None:
                return {}, {"termination_condition": "infeasible", "nodes_expanded": 0}
            root.solution[robot_id] = path
        root.cost = self._total_cost(root.solution, robots_meta)
        root.conflicts = self._all_conflicts(root.solution)
        root.conflict_count = len(root.conflicts)

        open_heap = [root]
        nodes_expanded = 0
        wall_start = timing.time()

        while open_heap:
            if nodes_expanded >= self.node_limit:
                return {}, {
                    "termination_condition": "node_limit_exceeded",
                    "nodes_expanded": nodes_expanded,
                }
            if self.time_limit is not None and (timing.time() - wall_start) > self.time_limit:
                return {}, {
                    "termination_condition": "time_limit_exceeded",
                    "nodes_expanded": nodes_expanded,
                }

            node = heapq.heappop(open_heap)
            nodes_expanded += 1

            # node.conflicts was already computed when this node was
            # created (root, above, or as a child below) — no need to
            # re-scan an unchanged solution just to look it up again.
            if not node.conflicts:
                return node.solution, {
                    "termination_condition": "optimal",
                    "nodes_expanded": nodes_expanded,
                }
            conflict = node.conflicts[0]

            t = conflict[0]
            if conflict[1] == "vertex":
                _, _, cell_node, robots_in_conflict = conflict
                branch_robots = robots_in_conflict[:2]
                for robot_id in branch_robots:
                    child = CTNode(
                        cost=0.0,
                        vertex_constraints={k: set(v) for k, v in node.vertex_constraints.items()},
                        edge_constraints={k: set(v) for k, v in node.edge_constraints.items()},
                        solution=dict(node.solution),
                    )
                    child.vertex_constraints.setdefault(robot_id, set()).add((cell_node, t))
                    new_path = self._low_level(robot_id, robots_meta, child, legal_cells)
                    if new_path is None:
                        continue
                    child.solution[robot_id] = new_path
                    child.cost = self._total_cost(child.solution, robots_meta)
                    child.conflicts = self._all_conflicts(child.solution)
                    child.conflict_count = len(child.conflicts)
                    heapq.heappush(open_heap, child)
            else:
                _, _, robot_a, robot_b, node_at_t, node_at_t1 = conflict
                branches = [(robot_a, node_at_t, node_at_t1), (robot_b, node_at_t1, node_at_t)]
                for robot_id, from_node, to_node in branches:
                    child = CTNode(
                        cost=0.0,
                        vertex_constraints={k: set(v) for k, v in node.vertex_constraints.items()},
                        edge_constraints={k: set(v) for k, v in node.edge_constraints.items()},
                        solution=dict(node.solution),
                    )
                    child.edge_constraints.setdefault(robot_id, set()).add((from_node, to_node, t))
                    new_path = self._low_level(robot_id, robots_meta, child, legal_cells)
                    if new_path is None:
                        continue
                    child.solution[robot_id] = new_path
                    child.cost = self._total_cost(child.solution, robots_meta)
                    child.conflicts = self._all_conflicts(child.solution)
                    child.conflict_count = len(child.conflicts)
                    heapq.heappush(open_heap, child)

        # Heap exhausted without ever popping a conflict-free node — given
        # CBS's completeness guarantee, this proves the joint problem has no
        # solution (see solve()'s docstring). No incumbent to fall back on
        # here either, same reasoning as the two budget-cutoff returns above.
        return {}, {"termination_condition": "no_solution", "nodes_expanded": nodes_expanded}
