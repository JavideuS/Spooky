"""
Conversion between Spooky's native matrix convention and robotics/Cartesian
convention.

Spooky's core — map.py, pathFormulation.py, the builders, and every solver's
QUBO index encoding/decoding — works exclusively in (row, col) matrix indices:
row 0 is the top row, row increases downward, col increases rightward. That
internal invariant never changes, regardless of what convention a caller uses.

Robotics/Cartesian convention (Y-up) instead has y increase upward from an
origin at the bottom-left. Converting between the two requires knowing the
grid's row count (M) to flip the vertical axis; column and x are equivalent.

Callers may still specify/receive positions in cartesian — see
`RobotConfig.coordinate_format` (quantum/robotConfiguration.py), which calls
`to_matrix_rc` once on ingest (`resolve_coordinates`) and `to_robotics_xy` on
every read (`format_position`), and `quantum/visualizer.py`'s `convention`
param for display. These functions are the primitives those call; use them
directly for anything outside that path (e.g. converting a plain path list
before handing it to a robotics stack that expects Y-up).
"""
from typing import List, Sequence, Tuple


def to_robotics_xy(row: int, col: int, num_rows: int) -> Tuple[int, int]:
    """Convert a single (row, col) matrix cell to (x, y) robotics/Y-up coordinates."""
    x = col
    y = (num_rows - 1) - row
    return x, y


def to_matrix_rc(x: int, y: int, num_rows: int) -> Tuple[int, int]:
    """Convert a single (x, y) robotics/Y-up cell back to (row, col) matrix coordinates."""
    row = (num_rows - 1) - y
    col = x
    return row, col


def path_to_robotics_xy(path: Sequence[Sequence[int]], num_rows: int) -> List[Tuple[int, int]]:
    """Convert a path of [row, col] (or [row, col, t]) cells to (x, y) robotics/Y-up tuples."""
    return [to_robotics_xy(cell[0], cell[1], num_rows) for cell in path]


def path_to_matrix_rc(path: Sequence[Sequence[int]], num_rows: int) -> List[Tuple[int, int]]:
    """Convert a path of [x, y] robotics/Y-up cells back to (row, col) matrix tuples."""
    return [to_matrix_rc(cell[0], cell[1], num_rows) for cell in path]


def flip_region_cartesian_to_matrix(
    start: Sequence[int], end: Sequence[int], num_rows: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert an axis-aligned rectangle's corners from cartesian (x, y) to matrix
    (row, col). Flipping the vertical axis inverts which corner has the smaller
    row, so the two converted corners are re-sorted into a valid (start, end)
    pair with start <= end on both axes — needed for e.g. map YAML `region`
    blocks (see quantum/maps/yaml2HDF5.py's flip_map_config_to_matrix), which
    are iterated as an inclusive range from start to end.
    """
    r0, c0 = to_matrix_rc(start[0], start[1], num_rows)
    r1, c1 = to_matrix_rc(end[0], end[1], num_rows)
    return (min(r0, r1), min(c0, c1)), (max(r0, r1), max(c0, c1))
