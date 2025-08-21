from fastapi import FastAPI, UploadFile, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import quantum.config.hdf5parser as h5parser
import quantum.config.parser as config_parser
from quantum import map, QUBOBuilder
import quantum.pathFormulation as pathfinding
from profiles.robot import Robot, RegisterRobotRequest
from profiles.models import MapInfo, RobotMapsResponse, PlanRequest, PlanResponse
from config_api import load_solver_configs, global_solver_configs, global_aliases, global_penalties_params
from typing import Dict, Optional
import datetime

# Global registry: robot_id → Robot instance
robots: Dict[str, Robot] = {}

# This should be retrieved from config/robot_templates.yaml
# Atlhough I still don't have structure for this, need more data
TEMPLATES = {
    "default": {},
    "mobile-robot": {
        "active_solver": "simulated_annealing"
    },
    "quantum-agent": {
        "active_solver": "dwave.3x3"
    },
    "research-qaoa": {
        "active_solver": "pennylane.qaoa_QNG"
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager to handle startup and shutdown events.
    """
    try:
        solvers_config = config_parser.load_config("../quantum/config/solvers.yaml", sections=["solvers", "aliases"])
        solvers = solvers_config.get("solvers", {})
        aliases = solvers_config.get("aliases", {})

        load_solver_configs(solvers, aliases)
        print("Solver configurations loaded.")

        penalties_conf = config_parser.load_config("../quantum/config/config.yaml", sections=["penalty_sets"])
        global_penalties_params.update(penalties_conf["penalty_sets"])
        
        yield  # Application is ready to handle requests
    finally:
        # Cleanup if needed
        global_solver_configs.clear()
        global_aliases.clear()
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/solvers")
def list_solvers() -> Dict[str, dict]:
    """
    List all available solvers and their configurations.
    
    Returns:
        Dict mapping solver_id → configuration
    """
    return global_solver_configs


@app.post("/robots/{robot_id}/maps/{map_id}")
async def upload_map(robot_id: str, map_id: str,
                     file: UploadFile,
                     materials_file: Optional[UploadFile] = None):
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


@app.get("/robots/{robot_id}/maps",  response_model=RobotMapsResponse)
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
                is_active=(map_id == robot.active_map)
            )
        except Exception as e:
            result[map_id] = {
                "error": f"Failed to read metadata: {str(e)}"
            }
    
    return {
        "robot_id": robot_id,
        "map_count": len(result),
        "maps": result
    }


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
        "metadata": "Map from HDF5 with custom layers"
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
    
    return {
        "status": "deleted",
        "robot_id": robot_id,
        "map_id": map_id
    }

# ROBOTS


@app.post("/robots")
def register_robot(request: RegisterRobotRequest):
    if not request.robot_id:
        raise HTTPException(400, "robot_id is required")

    if request.robot_id in robots:
        # Optional: return existing
        return {"status": "already_registered",
                "robot": robots[request.robot_id].to_dict()}

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

    return {
        "status": "registered",
        "robot": robot.to_dict()
    }


@app.get("/robots/{robot_id}")
def get_robot(robot_id: str):
    if robot_id not in robots:
        raise HTTPException(404, "Robot not found")
    return {"robot": robots[robot_id].to_dict()}


@app.get("/robots")
def list_robots():
    return {
        "robots": [robot.to_dict() for robot in robots.values()]
    }

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
        problem = pathfinding.PathfindingProblem(
            map_obj,
            start=tuple(request.start),
            end=tuple(request.goal)
        )
        builder = QUBOBuilder.QUBOBuilder(problem, penalties=global_penalties_params["alt_later"], name="standard")
        builder.build()
        path = solver.solve_qubo(builder)
        decoded_path = solver.decode_path(path["solution"], problem)
        energy = solver.total_energy(path)
        print("Energy", energy)
        response = PlanResponse(
            path=decoded_path,
            cost=energy,
            # success=path["success"],
            map_id=request.map_id,
            # solve_time_ms=path["solve_time_ms"],
            solver_used=request.solver or robot.active_solver,
            metrics={
                "start": request.start,
                "goal": request.goal,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
            }
        )
        if request.details:
            response.solver_details = solver.to_dict()

        return response
    except Exception as e:
        raise HTTPException(500, f"Planning failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
