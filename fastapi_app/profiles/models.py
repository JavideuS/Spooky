from pydantic import BaseModel
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

# Solver
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