"""GraphQUBO on a synthetic map loaded as a graph."""

from quantum.builder import GraphQUBO
from quantum.pathFormulation import PathfindingProblem
from quantum.tests.test_qubo_clearance import PENALTIES
from quantum.utils import preprocess as pm


def test_safe_reachability_returns_node_ids():
    """graph.adjacency holds (neighbour, weight) pairs; the safe BFS used to
    add the pairs themselves, so every bfs_safe graph window crashed."""
    problem = PathfindingProblem.from_map_config(
        "quantum/maps/synthetic/5x5/obs5x5_easy", "two_robots"
    ).as_graph_only()
    builder = GraphQUBO(problem, penalties=PENALTIES)
    _, cells = builder.get_logical_variables(pm.bfs_variant(pm.BFS_SAFE))
    assert cells and all(isinstance(n, int) for nodes in cells.values() for n in nodes)
    builder._active_cells = cells
    builder.build()  # used to raise TypeError
    assert builder.Q
