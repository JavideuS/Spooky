from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any

# Maps
class MapInfo(BaseModel):
    name: str
    grid_size: str
    resolution: float | str
    materials: List[str]
    loaded: bool
    is_active: bool


class RobotMapsResponse(BaseModel):
    robot_id: str
    map_count: int
    maps: Dict[str, MapInfo]


class RegisteredMapInfo(BaseModel):
    description: str
    loaded: bool                    # whether the HDF5 file has been parsed into memory yet
    grid_size: Optional[str] = None
    has_grid: bool                  # only meaningful once loaded=true; unloaded maps show False either way
    has_graph: bool
    source: str                     # map path (registry entries) or "uploaded"


class MapRegistryResponse(BaseModel):
    map_count: int
    maps: Dict[str, RegisteredMapInfo]


class MapUploadResponse(BaseModel):
    status: str
    map_id: str
    grid_size: Optional[str] = None
    has_graph: bool


# Stateless planning (multi-robot capable)
class RobotSpec(BaseModel):
    id: Optional[str] = None        # auto-assigned ("robot_0", ...) if omitted
    start: list[int]
    goal: list[int]
    start_time: int = 0
    priority: float = 1.0
    safety_radius: float = 0.5
    coordinate_format: str = "matrix"  # "matrix" (row, col) or "cartesian" (x, y robotics/Y-up);
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
    path: List[List[int]]           # ordered by timestep, in coordinate_format below
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
    start: list[int]
    goal:  list[int]
    solver: Optional[str] = None  # if None → use robot's active_solver
    details: bool = False
    coordinate_format: str = "matrix"  # "matrix" (row, col) or "cartesian" (x, y robotics/Y-up)

class PlanResponse(BaseModel):
    # ✅ Always present
    path: List[List[int]]           # decoded path, in coordinate_format below
    coordinate_format: str = "matrix"
    cost: float                     # best energy/cost
    # success: bool                   # did the solver succeed?
    map_id: str                     # which map was used
    # solve_time_ms: float            # wall-clock time
    solver_used: str                # e.g., "dwave.general", "pennylane.qaoa_QNG"
    
    # Optional: solver-specific details (only if requested)
    solver_details: Optional[Dict[str, Any]] = None
    
    # Optional: metrics
    metrics: Optional[Dict[str, Any]] = None