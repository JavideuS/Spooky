"""
Conversion between Spooky's native matrix convention and robotics/Cartesian
convention.

Spooky addresses grid cells purely as (row, col) matrix indices: row 0 is the
top row, row increases downward, col increases rightward. This is the
convention used throughout map.py, pathFormulation.py, robotConfiguration.py,
and every builder/solver — there is no separate x/y-with-up-axis concept
anywhere in the core library (see visualizer.py, which explicitly inverts its
plotly y-axis to compensate for this when rendering).

Robotics/Cartesian convention (Y-up) instead has y increase upward from an
origin at the bottom-left. Converting between the two requires knowing the
grid's row count (M) to flip the vertical axis; column and x are equivalent.

These helpers are not wired into any solver, builder, or API endpoint —
Spooky's public surface stays in (row, col) matrix convention. Call these
explicitly at whichever boundary needs the other convention (e.g. before
handing a path to a robotics stack that expects Y-up).
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
