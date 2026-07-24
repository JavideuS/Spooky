from pydantic import BaseModel, model_validator
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


class StatelessPlanRequest(BaseModel):
    map_id: str
    solver: str                     # required — stateless, no "active solver" to fall back to
    format: str = "grid"            # "grid" or "graph" — which representation of map_id to plan on
    start: Optional[list[int]] = None   # single-robot shorthand; always [row, col], even in graph mode
    goal: Optional[list[int]] = None
    robots: Optional[List[RobotSpec]] = None  # multi-robot form
    penalty_set: str = "crash"
    T: Optional[int] = None
    details: bool = False

    @model_validator(mode="after")
    def _exactly_one_robot_shape(self):
        single = self.start is not None and self.goal is not None
        multi = bool(self.robots)
        if single == multi:  # both provided, or neither
            raise ValueError(
                "Provide either 'start' and 'goal' (single robot) or a non-empty 'robots' list "
                "(multi-robot) — not both, not neither."
            )
        return self


class RobotPathResult(BaseModel):
    robot_id: str
    path: List[List[int]]           # [[row, col], ...] ordered by timestep


class StatelessPlanResponse(BaseModel):
    paths: List[RobotPathResult]
    cost: float                     # total energy across all robots/windows
    map_id: str
    solver_used: str
    solver_details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


# Stateful (per-robot) planning
class PlanRequest(BaseModel):
    map_id: str
    start: list[int]
    goal:  list[int]
    solver: Optional[str] = None  # if None → use robot's active_solver
    details: bool = False

class PlanResponse(BaseModel):
    # ✅ Always present
    path: List[List[int]]           # decoded path in grid coordinates
    cost: float                     # best energy/cost
    # success: bool                   # did the solver succeed?
    map_id: str                     # which map was used
    # solve_time_ms: float            # wall-clock time
    solver_used: str                # e.g., "dwave.3x3", "pennylane.qaoa_QNG"
    
    # Optional: solver-specific details (only if requested)
    solver_details: Optional[Dict[str, Any]] = None
    
    # Optional: metrics
    metrics: Optional[Dict[str, Any]] = None