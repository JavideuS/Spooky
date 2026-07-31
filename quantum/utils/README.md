# Quantum Utilities

A collection of helper functions and common tools used across the Quantum Navigation stack.

## Modules

### `paths.py`
QUBO index decoding shared by every solver: `decode_position(idx, problem)` reverses
the flat variable index into `(row, col, t, robot_num)` (or a node id for graph
problems), and `merge_paths()` stitches consecutive windowed-solve segments into one
path. See `BaseSolver.decode_path()` for the main caller.

### `coordinates.py`
Conversion between Spooky's native matrix `(row, col)` convention and robotics/
Cartesian `(x, y)` Y-up convention. The core (`map.py`, `pathFormulation.py`, every
builder/solver) always works in matrix indices — that never changes. Callers that
want cartesian in/out use this at two places:

- `RobotConfig.coordinate_format` (`"matrix"` or `"cartesian"`, per-robot) — calls
  `to_matrix_rc` once on ingest (`resolve_coordinates`) and `to_robotics_xy` on every
  read (`format_position`). This is what `/v1/plan`'s `coordinate_format` field and
  `qubo_cli.py --coordinate-format` are backed by.
- `quantum/visualizer.py`'s `convention` param (`"matrix"` or `"robotics"`) — same
  idea, purely for how a plot's axes are drawn; obstacles and paths get the identical
  conversion so the rendered picture never becomes inconsistent with itself.

Maps themselves are the one place this *isn't* a runtime choice — a `.h5`/YAML map
has no live `coordinate_format` attribute, since it's a baked array, not something
reformatted on every read. Authoring a map in cartesian is a one-time, one-directional
re-indexing done at generation time instead: see `coordinate_format: cartesian` in
`quantum/maps/template.yaml` and `yaml2HDF5.flip_map_config_to_matrix()`.

Call the four functions here (`to_robotics_xy` / `to_matrix_rc` / `path_to_robotics_xy`
/ `path_to_matrix_rc`) directly for anything outside those two paths — e.g. converting
a plain path list before handing it to an external robotics stack.

### `validation.py`
`is_valid_move()` checks whether a move between two positions is adjacent given the
problem's grid/graph; used by solvers' post-processing to detect and truncate invalid
paths.

### `logger.py`
`VerboseLogger` — a global singleton with levels 0–3 (Silent/Minimal/Standard/Debug).
Call `set_verbose_level(n)` once at startup, then `get_logger()` anywhere to log at the
current level without threading a logger instance through every call.

## Usage

These utilities are primarily for internal use by the `builder` and `solver` modules
but can be imported for custom extensions.

```python
from quantum.utils.coordinates import to_robotics_xy

# Convert a native (row, col) cell to robotics (x, y) for a 5-row grid
x, y = to_robotics_xy(row=0, col=2, num_rows=5)
```
