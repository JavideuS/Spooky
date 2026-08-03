from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import uvicorn
import quantum.config.hdf5parser as h5parser
import quantum.config.parser as config_parser
from quantum import map
from quantum.builder import QUBOBuilder, GraphQUBO
from quantum.robotConfiguration import RobotConfig
from quantum.visualizer import QuantumRoboticsVisualizer
import quantum.pathFormulation as pathfinding
import registry
from profiles.robot import Robot, RegisterRobotRequest
from profiles.models import (
    MapInfo,
    RobotMapsResponse,
    PlanRequest,
    PlanResponse,
    MapRegistryResponse,
    MapUploadResponse,
    StatelessPlanRequest,
    StatelessPlanResponse,
    RobotPathResult,
)
from config_api import (
    load_solver_configs,
    global_solver_configs,
    global_penalties_params,
)
from typing import Dict, Optional
import datetime
import time

# This app's own assets (config/, web/), addressed relative to this file so
# uvicorn can be launched from any directory. Library-owned data (penalty sets,
# maps) lives under registry.QUANTUM_ROOT instead.
APP_ROOT = Path(__file__).resolve().parent

# Global registry: robot_id → Robot instance
robots: Dict[str, Robot] = {}

# This should be retrieved from config/robot_templates.yaml
# Atlhough I still don't have structure for this, need more data
TEMPLATES = {
    "default": {},
    "mobile-robot": {"active_solver": "simulated_annealing"},
    "quantum-agent": {"active_solver": "dwave.general"},
    "research-qaoa": {"active_solver": "pennylane.qaoa_QNG"},
}


def _warm_start_gpu_devices():
    """
    Pay PennyLane's lightning.gpu CUDA/cuStateVec cold-start cost (~2s, a
    one-time cost per process) here at startup instead of on whichever
    request happens to be the first to use a GPU solver profile. The cost is
    process-wide, not per solver instance, so a single throwaway device
    covers every pennylane.* profile configured with device=lightning.gpu.
    """
    needs_gpu = any(
        cfg.get("backend") == "pennylane" and cfg.get("device") == "lightning.gpu"
        for cfg in global_solver_configs.values()
    )
    if not needs_gpu:
        return
    try:
        import pennylane as qml

        t0 = time.time()
        qml.device("lightning.gpu", wires=2)
        print(f"Warmed lightning.gpu device ({time.time() - t0:.2f}s).")
    except Exception as e:
        print(f"Skipping lightning.gpu warm-start: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager to handle startup and shutdown events.
    """
    try:
        solvers_config = config_parser.load_config(
            APP_ROOT / "config/solvers.yaml", sections=["solvers"]
        )
        solvers = solvers_config.get("solvers", {})

        load_solver_configs(solvers)
        print("Solver configurations loaded.")
        _warm_start_gpu_devices()

        penalties_conf = config_parser.load_config(
            registry.QUANTUM_ROOT / "config/config.yaml", sections=["penalty_sets"]
        )
        global_penalties_params.update(penalties_conf["penalty_sets"])

        maps_conf = config_parser.load_config(
            APP_ROOT / "config/maps.yaml", sections=["maps"]
        )
        registry.load_map_registry(maps_conf.get("maps") or {})
        print(
            f"Map registry loaded ({len(maps_conf.get('maps') or {})} entries, lazy-loaded on first use)."
        )

        yield  # Application is ready to handle requests
    finally:
        # Cleanup if needed
        global_solver_configs.clear()
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)

DEMO_HTML_PATH = APP_ROOT / "web" / "demo.html"


@app.get("/", include_in_schema=False)
def root():
    """Redirect to /demo — matters for HF Spaces' Docker SDK, which iframes '/'."""
    return RedirectResponse(url="/demo")


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """
    Self-contained demo UI: map/solver pickers, a robot form, and a live Plotly view.
    No-store: this file changes often during development and carries no ETag/
    Last-Modified, so without an explicit directive some browsers will serve a
    stale cached copy on a plain reload instead of refetching.
    """
    return HTMLResponse(
        content=DEMO_HTML_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/solvers")
def list_solvers() -> Dict[str, dict]:
    """
    List all available solvers and their configurations.

    Returns:
        Dict mapping solver_id → configuration
    """
    return global_solver_configs


@app.get("/v1/penalty-sets")
def list_penalty_sets() -> Dict[str, object]:
    """List available penalty_set names for /v1/plan (see quantum/config/config.yaml)."""
    return {"penalty_sets": list(global_penalties_params.keys()), "default": "crash"}


@app.post("/robots/{robot_id}/maps/{map_id}")
async def upload_map(
    robot_id: str,
    map_id: str,
    file: UploadFile,
    materials_file: Optional[UploadFile] = None,
):
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")

    robot = robots[robot_id]
    try:
        file.file.seek(0)
        map_conf = h5parser.load_map_from_hdf5(file.file)

        materials_conf = None
        if materials_file:
            materials_conf = config_parser.load_config(materials_file.file)["materials"]

        map_obj = map.Grid.from_hdf5_data(map_conf, materials_conf)

        robot.maps[map_id] = map_obj
        if robot.active_map is None:
            robot.active_map = map_id  # auto-activate first map

        return {"status": "map_uploaded", "map_id": map_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Map loading failed: {str(e)}")


@app.get("/robots/{robot_id}/maps", response_model=RobotMapsResponse)
def list_robot_maps(robot_id: str) -> Dict[str, dict]:
    """
    List all maps loaded for a specific robot.

    Returns:
        Dict mapping map_id → metadata (name, grid size, materials, etc.)
    """
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")

    robot = robots[robot_id]
    result = {}

    for map_id, map_obj in robot.maps.items():
        try:
            # Extract metadata from your MapObject (adjust based on your class)
            result[map_id] = MapInfo(
                name=getattr(map_obj, "name", map_id),
                grid_size=f"{getattr(map_obj, 'M', 'unknown')}x{getattr(map_obj, 'N', 'unknown')}",
                resolution=getattr(map_obj, "resolution", "unknown"),
                materials=getattr(map_obj, "materials", []),
                loaded=True,
                is_active=(map_id == robot.active_map),
            )
        except Exception as e:
            result[map_id] = {"error": f"Failed to read metadata: {str(e)}"}

    return {"robot_id": robot_id, "map_count": len(result), "maps": result}


@app.get("/robots/{robot_id}/maps/{map_id}")
def get_robot_map_info(robot_id: str, map_id: str):
    """
    Get detailed info about a specific map for a robot.
    """
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")

    robot = robots[robot_id]

    if map_id not in robot.maps:
        raise HTTPException(404, "Map not found for this robot")

    map_obj = robot.maps[map_id]

    # You can expose more attributes based on your MapObject
    return {
        "robot_id": robot_id,
        "map_id": map_id,
        "name": getattr(map_obj, "name", map_id),
        "grid_size": [getattr(map_obj, "M", None), getattr(map_obj, "N", None)],
        "resolution": getattr(map_obj, "resolution", None),
        "materials": getattr(map_obj, "materials", []),
        "is_active": map_id == robot.active_map,
        # Optional: expose internal flags
        "has_terrain": map_obj.terrain is not None,
        "has_elevation": map_obj.elevation is not None,
        "metadata": "Map from HDF5 with custom layers",
    }


@app.delete("/robots/{robot_id}/maps/{map_id}")
def delete_robot_map(robot_id: str, map_id: str):
    """
    Delete a specific map from a robot's storage.
    If the active map is deleted, active_map_id is set to None.
    """
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")

    robot = robots[robot_id]

    if map_id not in robot.maps:
        raise HTTPException(404, "Map not found for this robot")

    # Remove map
    del robot.maps[map_id]

    # If it was the active map, clear active_map_id
    if robot.active_map == map_id:
        if robot.maps:
            # Set to another map if available
            robot.active_map = next(iter(robot.maps))
        else:
            robot.active_map = None

    return {"status": "deleted", "robot_id": robot_id, "map_id": map_id}


# ROBOTS


@app.post("/robots")
def register_robot(request: RegisterRobotRequest):
    if not request.robot_id:
        raise HTTPException(400, "robot_id is required")

    if request.robot_id in robots:
        # Optional: return existing
        return {
            "status": "already_registered",
            "robot": robots[request.robot_id].to_dict(),
        }

    # Validate template
    if request.template not in TEMPLATES:
        raise HTTPException(400, f"Unknown template: {request.template}")

    # New robot
    robot = Robot(robot_id=request.robot_id, template=request.template)

    # Apply template defaults (if any)
    template_config = TEMPLATES[request.template]
    if "active_solver" in template_config:
        robot.active_solver = template_config["active_solver"]

    # Save
    robots[request.robot_id] = robot

    return {"status": "registered", "robot": robot.to_dict()}


@app.get("/robots/{robot_id}")
def get_robot(robot_id: str):
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")
    return {"robot": robots[robot_id].to_dict()}


@app.get("/robots")
def list_robots():
    return {"robots": [robot.to_dict() for robot in robots.values()]}


# PLANNING


@app.post("/robots/{robot_id}/plan", response_model=PlanResponse)
def plan_path(robot_id: str, request: PlanRequest):
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")

    robot = robots[robot_id]

    # --- 1. Resolve map ---
    if request.map_id not in robot.maps:
        raise HTTPException(404, "Map not found for this robot")
    map_obj = robot.maps[request.map_id]

    # --- 2. Resolve solver (smart: uses active if not specified) ---
    try:
        solver = robot.get_solver(request.solver)  # None → uses active_solver
    except Exception as e:
        raise HTTPException(400, str(e))

    # --- 3. Run planning ---
    try:
        robot_config = RobotConfig(
            robot_id=robot_id, start=tuple(request.start), goal=tuple(request.goal),
            coordinate_format=request.coordinate_format,
        )
        problem = pathfinding.PathfindingProblem(robot_config, grid=map_obj)
        builder = QUBOBuilder(
            problem, penalties=global_penalties_params["crash"], name="standard"
        )
        start_time = time.time()
        builder.build()
        solution = solver.solve(builder)
        planning_time = time.time() - start_time
        raw_path = solver.decode_path(solution["solution"], problem)
        if request.clip_at_goal:
            # Clip on matrix-native coords (against robot.goal) before
            # formatting, so the response stops at goal instead of showing
            # it parked there.
            clipped_robot_paths = solver.clip_paths_at_goal(
                solver.get_robot_paths(raw_path), problem
            )
            raw_path = [
                ((i, j, t), robot_num)
                for robot_num, coords in clipped_robot_paths.items()
                for (i, j, t) in coords
            ]
        formatted_path = solver.format_output_path(raw_path, problem)
        decoded_path = [[i, j] for (i, j, t), _ in formatted_path]
        energy = float(solver.total_energy(solution))
        print("Energy", energy)
        response = PlanResponse(
            path=decoded_path,
            coordinate_format=request.coordinate_format,
            cost=energy,
            # success=path["success"],
            map_id=request.map_id,
            # solve_time_ms=path["solve_time_ms"],
            solver_used=request.solver or robot.active_solver,
            metrics={
                "start": request.start,
                "goal": request.goal,
                "planning_time": planning_time,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
        if request.details:
            response.solver_details = solver.to_dict()

        return response
    except Exception as e:
        raise HTTPException(500, f"Planning failed: {str(e)}")


# STATELESS PLANNER (v1) — no per-robot session, maps/solvers already in memory


@app.get("/v1/maps", response_model=MapRegistryResponse)
def list_registered_maps():
    """List every map_id in the registry (curated + runtime-uploaded), and whether it's loaded yet."""
    maps = registry.list_maps()
    return {"map_count": len(maps), "maps": maps}


@app.get("/v1/maps/{map_id}/preview")
def preview_map(map_id: str, embed: str = "html", coordinate_format: str = "matrix"):
    """
    Render map_id's grid (obstacles + terrain, no robots/paths).
    embed="html" (default): a standalone-ish Plotly HTML fragment, plotly.js via
    CDN — good for a direct browser open or Swagger link, not for injecting into
    a page that wants to update the same figure later (script tags in an
    innerHTML-injected fragment don't execute).
    embed="json": {"data": [...], "layout": {...}} — for pages that already load
    plotly.js themselves and want to call Plotly.newPlot/react directly (this is
    what /demo uses, so it can later update the same figure with a solved path).
    coordinate_format="matrix" (default) or "cartesian" — purely a display choice
    (axis labels/origin/direction); the underlying grid data is unaffected.
    Grid-only: graph-only maps have no visualizer support and return 400.
    """
    if embed not in ("html", "json"):
        raise HTTPException(400, f"Unknown embed: {embed}. Must be 'html' or 'json'.")
    if coordinate_format not in ("matrix", "cartesian"):
        raise HTTPException(
            400, f"Unknown coordinate_format: {coordinate_format}. Must be 'matrix' or 'cartesian'."
        )

    try:
        grid = registry.get_map(map_id, format="grid")
    except KeyError:
        raise HTTPException(404, f"Unknown map_id: {map_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to load map '{map_id}': {e}")

    visualizer = QuantumRoboticsVisualizer(
        grid_size=(grid.M, grid.N), title=f"Map preview: {map_id}",
        convention="robotics" if coordinate_format == "cartesian" else "matrix",
    )
    problem_stub = SimpleNamespace(grid=grid)
    fig = visualizer.create_static_plot(obstacles=grid.obstacles, problem=problem_stub)

    if embed == "json":
        return json.loads(fig.to_json())

    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return HTMLResponse(content=html)


@app.post("/v1/maps/{map_id}", response_model=MapUploadResponse)
async def upload_registered_map(
    map_id: str, file: UploadFile, materials_file: Optional[UploadFile] = None
):
    """
    Register a new map at runtime by uploading its HDF5 file.
    Stored in the same in-memory registry as the curated maps.yaml entries,
    but not persisted back to maps.yaml — it only lives for this process.
    Both grid and graph representations are parsed if present in the file.
    """
    try:
        file.file.seek(0)
        data = h5parser.load_both_from_hdf5(file.file)

        materials_conf = None
        if materials_file:
            materials_conf = config_parser.load_config(materials_file.file)["materials"]

        grid = None
        if data["has_map"] and data["map_data"]:
            grid = map.Grid.from_hdf5_data(data["map_data"], materials_conf)

        graph = None
        if data["has_graph"] and data["graph_data"]:
            graph = map.Graph.from_hdf5_data(data["graph_data"])

        if grid is None and graph is None:
            raise ValueError(
                "HDF5 file contains neither a grid ('map_structure') nor a graph representation"
            )

        registry.register_uploaded_map(map_id, grid=grid, graph=graph)

        return MapUploadResponse(
            status="registered",
            map_id=map_id,
            grid_size=f"{grid.M}x{grid.N}" if grid else None,
            has_graph=graph is not None,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Map loading failed: {str(e)}")


@app.post("/v1/plan", response_model=StatelessPlanResponse)
def plan_stateless(request: StatelessPlanRequest):
    """
    Stateless planning: map_id + solver + robots in, paths + cost out.
    No robot registration, no per-call map upload — map and solver instances are
    resolved from the in-memory registries. `robots` always takes a list — one
    entry for a single robot, more for multi-robot — and both the grid and
    graph builder are supported (request.format).

    Positions are always given as [row, col] — even in graph mode, where they're
    resolved to node ids server-side via Graph.get_node_from_position. Paths are
    returned the same way: [[row, col], ...], Spooky's native (row, col) matrix
    convention — see quantum/utils/coordinates.py to convert to robotics (x, y)
    Y-up if needed.
    """
    if request.format not in ("grid", "graph"):
        raise HTTPException(
            400, f"Unknown format: {request.format}. Must be 'grid' or 'graph'."
        )

    # --- 1. Resolve map ---
    try:
        env = registry.get_map(request.map_id, format=request.format)
    except KeyError:
        raise HTTPException(404, f"Unknown map_id: {request.map_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to load map '{request.map_id}': {e}")

    # --- 2. Resolve solver ---
    try:
        solver = registry.get_solver(request.solver)
    except KeyError:
        raise HTTPException(400, f"Unknown solver: {request.solver}")

    # --- 3. Resolve penalties ---
    if request.penalty_set not in global_penalties_params:
        raise HTTPException(
            400,
            f"Unknown penalty_set: {request.penalty_set}. Available: {list(global_penalties_params.keys())}",
        )
    penalties = global_penalties_params[request.penalty_set]

    # --- 4. Build robot configs (single or multi) ---
    def resolve_position(pos: list[int], coordinate_format: str):
        # Graph builders index robots by node id, not (row, col) — RobotConfig
        # must hold the node id directly (see PathfindingProblem.from_graph_data).
        # Node ids have no coordinate frame of their own, so cartesian input isn't
        # supported in graph mode — convert to matrix before the position lookup
        # would be needed elsewhere, but here we just reject it outright.
        if coordinate_format == "cartesian" and request.format == "graph":
            raise HTTPException(
                400,
                "coordinate_format='cartesian' is not supported in graph mode — "
                "positions there resolve directly to node ids.",
            )
        if request.format != "graph":
            return tuple(pos)
        node_id = env.get_node_from_position(tuple(pos))
        if node_id is None:
            raise HTTPException(
                400, f"Position {pos} is not a node in map '{request.map_id}'"
            )
        return node_id

    robot_configs = [
        RobotConfig(
            robot_id=r.id or f"robot_{i}",
            start=resolve_position(r.start, r.coordinate_format),
            goal=resolve_position(r.goal, r.coordinate_format),
            start_time=r.start_time,
            priority=r.priority,
            safety_radius=r.safety_radius,
            coordinate_format=r.coordinate_format,
        )
        for i, r in enumerate(request.robots)
    ]

    # --- 5. Solve ---
    try:
        if request.format == "graph":
            problem = pathfinding.PathfindingProblem(
                robot_configs, graph=env, T=request.T
            )
            builder = GraphQUBO(problem, penalties=penalties, name="v1_plan")
        else:
            problem = pathfinding.PathfindingProblem(
                robot_configs, grid=env, T=request.T
            )
            builder = QUBOBuilder(problem, penalties=penalties, name="v1_plan")
        start_time = time.time()
        builder.build()
        solution = solver.solve(builder)
        planning_time = time.time() - start_time

        decoded = solver.decode_path(solution["solution"], problem)
        robot_paths = solver.get_robot_paths(
            decoded
        )  # {robot_num: [(i, j, t), ...] sorted by t} — matrix, for the visualizer below
        response_path = decoded
        if request.clip_at_goal:
            # Clip on matrix-native coords (against robot.goal) before
            # formatting, so the response stops at goal instead of showing
            # it parked there. robot_paths (visualizer, above) stays unclipped.
            clipped_robot_paths = solver.clip_paths_at_goal(robot_paths, problem)
            response_path = [
                ((i, j, t), robot_num)
                for robot_num, coords in clipped_robot_paths.items()
                for (i, j, t) in coords
            ]
        formatted_robot_paths = solver.get_robot_paths(
            solver.format_output_path(response_path, problem)
        )  # same, but each robot's path in its own coordinate_format — for the response
        num_to_id = {num: rid for rid, num in problem.get_robot_nums().items()}

        paths = [
            RobotPathResult(
                robot_id=num_to_id.get(robot_num, str(robot_num)),
                path=[[i, j] for (i, j, t) in coords],
                coordinate_format=problem.robots[num_to_id[robot_num]].coordinate_format,
            )
            for robot_num, coords in formatted_robot_paths.items()
        ]

        response = StatelessPlanResponse(
            paths=paths,
            cost=float(solver.total_energy(solution)),
            map_id=request.map_id,
            solver_used=request.solver,
            metrics={
                "planning_time": planning_time,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
        if request.details:
            response.solver_details = solver.to_dict()

        if request.render and request.format == "grid":
            # The figure is one shared plot for all robots, so a single display
            # convention is used — the first robot's coordinate_format (the demo
            # UI sets the same value for every robot via one toggle).
            render_format = request.robots[0].coordinate_format if request.robots else "matrix"
            visualizer = QuantumRoboticsVisualizer(
                grid_size=(env.M, env.N), title=f"{request.map_id} — solved",
                convention="robotics" if render_format == "cartesian" else "matrix",
            )
            fig = visualizer.create_animated_plot(
                obstacles=env.obstacles,
                problem=problem,
                robot_paths={
                    num_to_id.get(num, str(num)): coords
                    for num, coords in robot_paths.items()
                },
            )
            response.figure = json.loads(fig.to_json())

        return response
    except Exception as e:
        raise HTTPException(500, f"Planning failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
