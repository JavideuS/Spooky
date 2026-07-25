# fastapi_app

HTTP layer over Spooky. Exposes two independent endpoint families that share
the same map/solver config but don't share state:

- **`/v1/*` — stateless planner.** For external callers (e.g. a Go control
  circuit). No robot registration, no per-call map upload. Maps and solver
  instances live in server-side in-memory registries; a call is just
  `map_id + solver + robots in, paths + cost out`.
- **`/robots/*` — stateful, per-robot sessions.** For driving Spooky itself
  (animation tests, interactive experimentation from within the FastAPI
  layer). A robot is registered, maps are uploaded into its own namespace,
  and `/robots/{id}/plan` reuses that per-robot state across calls.

Both call the same underlying solve pipeline
(`builder.build()` → `solver.solve_qubo_smart(builder, False)` →
`solver.decode_path(...)`) — see `quantum/qubo_cli.py` for the reference
version of that pipeline (kept deliberately separate: `qubo_cli.py`'s solve
path is coupled to `argparse.Namespace` and isn't a good import target).

## Running

```bash
pip install -e ".[fastapi]"   # fastapi, uvicorn, python-multipart
cd fastapi_app/
uvicorn api:app --reload
# Then open http://127.0.0.1:8000/docs
```

Must be run from `fastapi_app/` — config paths (`config/solvers.yaml`,
`config/maps.yaml`, `../quantum/config/config.yaml`) are resolved relative to
the working directory at startup.

## Config

- `config/solvers.yaml` — named solver profiles (`dwave.general`,
  `pennylane.qaoa_QNG`, ...), keyed `backend.name`. No aliases — one
  canonical name per solver. Loaded into `global_solver_configs` at startup
  (`config_api.py`).
- `config/maps.yaml` — named map registry (`map_id -> {path, description}`),
  paths relative to `quantum/`. Loaded into the in-memory map registry at
  startup (`registry.py`).
- `../quantum/config/config.yaml` — penalty sets (`crash`, `swap`, ...),
  shared with the core library. `crash` is the default for `/v1/plan`.

## Map registry (`registry.py`)

Maps are **not** loaded at startup, only indexed. Both representations
(`Grid` and `Graph`) are parsed from HDF5 together, in one file read, lazily
on first `GET`/`plan` request that references a given `map_id` — a map
that's never requested is never loaded (matters for the 1000x1000 synthetic
map). Synthetic maps generally carry both representations in the same HDF5
file, so requesting either one loads both.

- `GET /v1/maps` — list every registered `map_id` (curated `maps.yaml`
  entries + runtime uploads), with `loaded: bool` and, once loaded,
  `has_grid` / `has_graph`. Before first load, `has_grid`/`has_graph` show
  `false` regardless of what the underlying file actually has — they only
  reflect what's been parsed, not what's parseable.
- `POST /v1/maps/{map_id}` — upload a new HDF5 (+ optional materials YAML) at
  runtime, added to the same in-memory registry. Not persisted back to
  `maps.yaml` — it only lives for this process.

Solver instances are cached the same way (`registry.get_solver`), built once
per solver key on first use and reused across `/v1/plan` calls.

## Grid vs. graph (`/v1/plan`'s `format` field)

`/v1/plan` accepts `format: "grid"` (default) or `format: "graph"`, selecting
`QUBOBuilder` vs. `GraphQUBO`. Positions in each entry of the `robots` list
(`start`/`goal`) are **always** `[row, col]`, even in graph mode —
the endpoint resolves them to node ids server-side via
`Graph.get_node_from_position` before building `RobotConfig` objects
(`GraphQUBO` and the shared windowing code in `base_qubo.py` expect
`RobotConfig.start`/`.goal` to already be node ids for graph problems, unlike
grid problems where they stay `(row, col)` tuples — see
`PathfindingProblem.from_graph_data` for the same conversion done for
single-robot CLI use). A position that isn't a node in the map's graph
returns a 400, not a silent grid-only fallback. Decoded paths come back the
same shape either way — `[[row, col], ...]` — since `decode_position`
already maps graph node ids back to their stored position.

## Coordinate convention

Spooky is **Y-down (matrix/image convention)** throughout: positions are
`(row, col)`, row 0 is the top row, row increases downward. There is no
separate x/y-with-up-axis concept anywhere in the core library — confirmed by
`quantum/visualizer.py`, which explicitly inverts its plotly y-axis to
compensate for this when rendering.

`/v1/plan` and `/robots/{id}/plan` both return paths in this native
`(row, col)` convention — no conversion happens at the API boundary. If a
caller needs robotics/Cartesian (Y-up) coordinates instead, convert
explicitly using `quantum/utils/coordinates.py`
(`to_robotics_xy` / `to_matrix_rc` / `path_to_robotics_xy` /
`path_to_matrix_rc`), which needs the grid's row count to flip the axis.

## Endpoints

| Method & path | Purpose |
|---|---|
| `GET /solvers` | List configured solver profiles. |
| `GET /v1/maps` | List the map registry (curated + uploaded). |
| `POST /v1/maps/{map_id}` | Upload/register a map at runtime. |
| `POST /v1/plan` | Stateless plan: `robots: [{id?, start, goal, ...}, ...]` (one entry for single-robot, more for multi-robot), `format: "grid"\|"graph"`. Returns `{paths: [{robot_id, path}], cost, ...}`. |
| `POST /robots` | Register a robot session. |
| `GET /robots` / `GET /robots/{id}` | List / inspect robot sessions. |
| `POST /robots/{id}/maps/{map_id}` | Upload a map into a robot's own namespace. |
| `GET /robots/{id}/maps[/{map_id}]` | List / inspect a robot's maps. |
| `DELETE /robots/{id}/maps/{map_id}` | Remove a map from a robot's namespace. |
| `POST /robots/{id}/plan` | Stateful single-robot plan using the robot's active/uploaded map + solver. |

## `web_page.py`

A standalone Dash dashboard prototype, not wired into `api.py` or the
FastAPI app — scratch/demo code, not part of the served API surface.
