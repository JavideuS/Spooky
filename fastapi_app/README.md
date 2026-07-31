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
(`builder.build()` → `solver.solve_qubo(builder)` →
`solver.decode_path(...)`) — see `quantum/qubo_cli.py` for the reference
version of that pipeline (kept deliberately separate: `qubo_cli.py`'s solve
path is coupled to `argparse.Namespace` and isn't a good import target).

## Running

```bash
pip install -e ".[fastapi]"   # fastapi, uvicorn, python-multipart
cd fastapi_app/
uvicorn api:app --reload
```

- **http://127.0.0.1:8000/demo** — interactive demo UI: pick a map and
  solver, add robots, plan, and watch the solved paths animate. Easiest way
  to try the service without writing any code — see "Demo page" below.
- **http://127.0.0.1:8000/docs** — Swagger UI for the raw `/v1/*` and
  `/robots/*` endpoints, for programmatic callers.

`/` redirects to `/demo` (needed for HF Spaces' Docker SDK, which iframes
whatever the container serves at `/`) — this app is a backend/demo service
without its own landing page, so `/demo` is the closest thing to one.

### Docker

```bash
docker build -t spooky-fastapi .        # or -f Dockerfile.gpu for CUDA
docker run -p 7860:7860 spooky-fastapi  # http://localhost:7860/demo
```

Live demo: [huggingface.co/spaces/JavideuS/Spooky](https://huggingface.co/spaces/JavideuS/Spooky).

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

`GET /v1/maps/{map_id}/preview` renders the map's grid (obstacles + terrain,
no robots/paths) via `quantum/visualizer.py`. Two `embed` modes: `html`
(default) — a self-contained Plotly fragment, plotly.js via CDN, good for a
direct browser open or a Swagger link; `json` — `{"data": [...], "layout":
{...}}` for a page that already has `Plotly` loaded and wants to call
`Plotly.newPlot`/`Plotly.react` itself (this is what `/demo` uses, so it can
later update the same figure with a solved path instead of re-embedding a
whole new document). Grid-only — a graph-only map returns 400, since the
visualizer has no graph rendering.

`GET /v1/penalty-sets` lists the penalty_set names available to `/v1/plan`
(from `../quantum/config/config.yaml`), with `crash` as the documented
default.

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

Spooky's core is **always matrix (row, col)** internally — row 0 is the top
row, row increases downward — regardless of what convention a caller uses.
That never changes; every builder, solver, and QUBO index encoding/decoding
works exclusively in matrix indices.

At the API boundary, though, `coordinate_format` is a per-robot, request-time
choice: `"matrix"` (default) or `"cartesian"` (robotics/Y-up: origin
bottom-left, y increasing upward). Set it per entry in `/v1/plan`'s `robots`
list (`RobotSpec.coordinate_format`), or once on `/robots/{id}/plan`
(`PlanRequest.coordinate_format`):

- **Input**: that robot's `start`/`goal` are read in the declared convention
  and converted to matrix once, before solving
  (`RobotConfig.resolve_coordinates`).
- **Output**: that robot's returned `path` is converted back to the same
  convention (`RobotConfig.format_position` / `BaseSolver.format_output_path`)
  — each `RobotPathResult`/`PlanResponse` echoes the `coordinate_format` it
  used, so a caller never has to guess which frame a path is in.
- **Graph mode** doesn't support `"cartesian"` — positions there resolve
  directly to node ids server-side, which have no coordinate frame of their
  own to convert; a cartesian request against `format: "graph"` returns 400.
- `GET /v1/maps/{map_id}/preview` and `/v1/plan`'s `render: true` figure both
  take/reflect the same `coordinate_format` too, but purely as a **display**
  choice passed to `quantum/visualizer.py`'s `convention` param — it changes
  axis labels/origin/direction, not the underlying grid data. Both obstacles
  and paths go through the same conversion, so the rendered picture is
  self-consistent in either mode (a wall doesn't visually move when you
  relabel the axes describing it).

This is different from a **map's** own convention, which is fixed at the file
level, not a per-request choice — see `../quantum/maps/README.md`.

For anything outside this path (e.g. converting a plain path list before
handing it to an external robotics stack), use
`quantum/utils/coordinates.py` directly (`to_robotics_xy` / `to_matrix_rc` /
`path_to_robotics_xy` / `path_to_matrix_rc`), which needs the grid's row
count to flip the axis.

## Endpoints

| Method & path | Purpose |
|---|---|
| `GET /demo` | Self-contained demo UI — map/solver pickers, a robot form (or raw JSON), a live Plotly view. See below. |
| `GET /solvers` | List configured solver profiles. |
| `GET /v1/penalty-sets` | List penalty_set names available to `/v1/plan`. |
| `GET /v1/maps` | List the map registry (curated + uploaded). |
| `GET /v1/maps/{map_id}/preview` | Render the map's grid (obstacles/terrain) via Plotly. `?embed=html\|json&coordinate_format=matrix\|cartesian`. |
| `POST /v1/maps/{map_id}` | Upload/register a map at runtime. |
| `POST /v1/plan` | Stateless plan: `robots: [{id?, start, goal, coordinate_format?, ...}, ...]` (one entry for single-robot, more for multi-robot), `format: "grid"\|"graph"`, `render: bool`. Returns `{paths: [{robot_id, path, coordinate_format}], cost, ..., figure?}`. |
| `POST /robots` | Register a robot session. |
| `GET /robots` / `GET /robots/{id}` | List / inspect robot sessions. |
| `POST /robots/{id}/maps/{map_id}` | Upload a map into a robot's own namespace. |
| `GET /robots/{id}/maps[/{map_id}]` | List / inspect a robot's maps. |
| `DELETE /robots/{id}/maps/{map_id}` | Remove a map from a robot's namespace. |
| `POST /robots/{id}/plan` | Stateful single-robot plan using the robot's active/uploaded map + solver. |

## Demo page (`GET /demo`, `static/demo.html`)

A single self-contained HTML/JS page (no build step, no framework —
`plotly.js` via CDN `<script>` in `<head>`), served straight from disk by the
`/demo` route. It's a thin client over the existing `/v1/*` endpoints, not a
new code path.

**How to use it:**

1. Open `http://127.0.0.1:8000/demo`.
2. Pick a **map** (upper-left, above the preview) — it loads immediately.
3. Pick the **coordinate format** just below it — `matrix` (native, row/col)
   or `cartesian` (robotics Y-up, x/y). This drives three things at once: the
   preview/solved-plot axes, how the robot form's start/goal fields are
   labeled, and the `coordinate_format` sent with every robot in the request.
   Switching **Format** (below) to `graph` locks this back to `matrix`, since
   graph mode has no coordinate frame to convert (positions resolve straight
   to node ids).
4. Pick a **solver** in the sidebar; profiles tagged `"general"` are
   preselected as the recommended default. Tags and description show below
   the picker.
5. Add one or more **robots** (start/goal, labeled row/col or x/y depending on
   step 3) via the form rows, or switch to the "Raw JSON" tab to hand-edit the
   exact `/v1/plan` request body (start_time, priority, safety_radius, or
   anything the form doesn't expose) — whichever tab is active when you click
   "Plan path" is what gets sent.
6. Click **Plan path**. The result strip shows cost / planning time / solver;
   each robot's result row also shows the `coordinate_format` its path came
   back in. The visualization animates the solved paths with play/pause and a
   timestep slider (drag it to scrub forward or back). Single-robot problems
   render as Scooby; 2–4 robots get the ninja pack, matched by robot name
   (name your robot `"kai"`, `"jay"`, `"lloyd"`, `"zane"`, or `"cole"` to pin
   its character, otherwise they're assigned in pool order) — this comes from
   `visualizer.py`'s `create_animated_plot`, not page-specific code.

Details on the wiring: map + solver `<select>`s are populated from
`GET /v1/maps` / `GET /solvers` (solvers grouped by backend — `dwave` /
`pennylane` / `qiskit`, qiskit split out since it's remote hardware, not a
local simulator). Picking a map (or toggling coordinate format) fetches
`/v1/maps/{id}/preview?embed=json&coordinate_format=...` and renders it with
`Plotly.newPlot`. Planning posts to `/v1/plan` with `render: true`; the
response's `figure` (`data` + `layout` + `frames`) is drawn with
`Plotly.newPlot(...).then(() => Plotly.addFrames(...))` — a plain
`Plotly.react` call would silently drop the animation frames, since it only
takes `data`/`layout`.

There's no manual solver-recommendation logic beyond the `"general"` tag
match — a map-size/robot-count-aware recommendation is a possible future
addition, not something this page attempts.

## `web_page.py`

A standalone Dash dashboard prototype, not wired into `api.py` or the
FastAPI app — scratch/demo code, not part of the served API surface. `/demo`
above is the actively maintained one.
