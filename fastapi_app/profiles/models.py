from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any

# Maps
class MapInfo(BaseModel):
    name: str
    grid_size: str
    resolution: float                # meters per grid cell (1.0 for a synthetic map)
    # world pose (x, y, yaw) of grid cell [M-1, 0]; null when the map carries none
    origin: Optional[List[float]] = None
    materials: List[str]
    loaded: bool
    is_active: bool


class RobotMapsResponse(BaseModel):
    robot_id: str
    map_count: int
    maps: Dict[str, MapInfo]


class RobotMapDetail(BaseModel):
    """GET /robots/{robot_id}/maps/{map_id} — one map in a robot's namespace."""
    robot_id: str
    map_id: str
    name: str
    grid_size: List[Optional[int]]   # [M, N]
    resolution: Optional[float] = None
    origin: Optional[List[float]] = None  # world pose (x, y, yaw) of grid cell [M-1, 0]
    materials: List[str]
    is_active: bool
    has_terrain: bool
    has_elevation: bool
    metadata: str


class RegisteredMapInfo(BaseModel):
    description: str
    loaded: bool                    # whether the HDF5 file has been parsed into memory yet
    grid_size: Optional[str] = None
    has_grid: bool                  # only meaningful once loaded=true; unloaded maps show False either way
    has_graph: bool
    source: str                     # map path (registry entries) or "uploaded"
    # Robotics geo-metadata — grid-only, and only populated once loaded=true
    # (null before first load, and null for graph-only maps). resolution defaults
    # to 1.0 m/cell and origin to [0, 0, 0] for synthetic maps with no real frame.
    resolution: Optional[float] = None
    origin: Optional[List[float]] = None  # world pose (x, y, yaw) of grid cell [M-1, 0]


class MapRegistryResponse(BaseModel):
    map_count: int
    maps: Dict[str, RegisteredMapInfo]


class RegisteredMapDetail(BaseModel):
    """Single-map introspection for GET /v1/maps/{map_id}. Forces a lazy load so
    resolution/origin/materials reflect the actual HDF5 contents."""
    map_id: str
    description: str
    loaded: bool
    source: str
    grid_size: Optional[str] = None
    has_grid: bool
    has_graph: bool
    resolution: Optional[float] = None
    origin: Optional[List[float]] = None  # world pose (x, y, yaw) of grid cell [M-1, 0]
    materials: List[str] = []


class MapUploadResponse(BaseModel):
    status: str
    map_id: str
    grid_size: Optional[str] = None
    has_graph: bool
    # Geo-metadata read from the uploaded HDF5. Null when the upload carried no grid.
    resolution: Optional[float] = None
    origin: Optional[List[float]] = None


# Stateless planning (multi-robot capable)
class RobotSpec(BaseModel):
    id: Optional[str] = None        # auto-assigned ("robot_0", ...) if omitted
    start: list[float]              # int cells for "matrix"/"cartesian"; real meters for "world"
    goal: list[float]
    start_time: int = 0
    priority: float = 1.0
    safety_radius: float = 0.5
    coordinate_format: str = "matrix"  # "matrix" (row, col), "cartesian" (x, y robotics/Y-up),
    # or "world" (real-world x, y meters in the map's frame — see quantum/maps/pgm2HDF5.py);
    # applies to this robot's start/goal, and its returned path is formatted the same way


class StatelessPlanRequest(BaseModel):
    map_id: str
    solver: str                     # required — stateless, no "active solver" to fall back to
    format: str = "grid"            # "grid" or "graph" — which representation of map_id to plan on
    robots: List[RobotSpec]         # one entry for a single robot, more for multi-robot
    penalty_set: str = "crash"
    T: Optional[int] = None         # omit/null to auto-compute; a window of 0 steps is never valid
    details: bool = False
    render: bool = False            # also return an animated Plotly figure (data+layout+frames) of the solved paths; grid only
    clip_at_goal: bool = False      # trim each robot's returned path once parked at goal, keeping only the first arrival

    @field_validator("robots")
    @classmethod
    def _non_empty_robots(cls, robots: List[RobotSpec]) -> List[RobotSpec]:
        if not robots:
            raise ValueError("'robots' must contain at least one entry.")
        return robots

    @field_validator("T")
    @classmethod
    def _positive_T(cls, T: Optional[int]) -> Optional[int]:
        if T is not None and T < 1:
            raise ValueError(
                "'T' must be a positive number of timesteps, or omitted/null to "
                "auto-compute from robot start/goal distances."
            )
        return T


class RobotPathResult(BaseModel):
    robot_id: str
    path: List[List[float]]         # ordered by timestep, in coordinate_format below
    coordinate_format: str = "matrix"  # convention this robot's start/goal/path used


class StatelessPlanResponse(BaseModel):
    paths: List[RobotPathResult]
    cost: float                     # total energy across all robots/windows
    map_id: str
    solver_used: str
    solver_details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    figure: Optional[Dict[str, Any]] = None  # {"data": [...], "layout": {...}, "frames": [...]}; set only if request.render was true


# Stateful (per-robot) planning
class PlanRequest(BaseModel):
    map_id: str
    start: list[float]              # int cells for "matrix"/"cartesian"; real meters for "world"
    goal:  list[float]
    solver: Optional[str] = None  # if None → use robot's active_solver
    penalty_set: str = "crash"     # QUBO penalty landscape; see quantum/config/config.yaml
    details: bool = False
    coordinate_format: str = "matrix"  # "matrix" (row, col), "cartesian" (x, y robotics/Y-up),
    # or "world" (real-world x, y meters in the map's frame — see quantum/maps/pgm2HDF5.py)
    clip_at_goal: bool = False  # trim the returned path once parked at goal, keeping only the first arrival


class PlanResponse(BaseModel):
    # Always present
    path: List[List[float]]         # decoded path, in coordinate_format below
    coordinate_format: str = "matrix"
    cost: float                     # best energy/cost
    map_id: str                     # which map was used
    solver_used: str                # e.g., "dwave.general", "pennylane.train.qaoa_QNG"

    # Optional: solver-specific details (only if requested)
    solver_details: Optional[Dict[str, Any]] = None

    # Optional: metrics
    metrics: Optional[Dict[str, Any]] = None
