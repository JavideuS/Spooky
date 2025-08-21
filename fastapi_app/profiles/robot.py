from typing import Dict
from quantum.solvers import SolverFactory
from quantum.map import Grid
from quantum.pathFormulation import PathfindingProblem as Problem
from pydantic import BaseModel
from typing import Optional
from config_api import global_solver_configs, global_aliases


class RegisterRobotRequest(BaseModel):
    robot_id: str
    template: Optional[str] = "default"


class Robot:
    def __init__(self, robot_id: str, template: str = "default"):
        self.robot_id = robot_id
        self.template = template
        
        # Isolated storage
        self.maps: Dict[str, Grid] = {}
        self.active_map = None  # e.g., "3x3_no_obs"
        self.problem = None
        self.solvers: Dict[str, SolverFactory] = {}
        self.active_solver = None  # e.g., "dwave.3x3"
        self.metadata = {
            "created_at": "2025-04-05T12:00:00Z",
            "last_seen": None,
            "status": "registered"  # Running, idle, etc.
        }

    def load_map(self, map_id: str, map_data: Dict):
        """Load map from parsed HDF5 data"""
        try:
            grid = Grid.from_hdf5_data(map_data)
            self.maps[map_id] = grid
            return {"status": "loaded", "map_id": map_id}
        except Exception as e:
            raise ValueError(f"Failed to load map: {str(e)}")
        
    def get_solver(self, solver_key: str = None):
        key = solver_key or self.active_solver  # ← uses active by default

        if key not in global_solver_configs:
            raise ValueError(f"Unknown solver config: {key}")

        if key not in self.solvers:
            config = global_solver_configs[key]
            try:
                solver = SolverFactory.create_solver_from_config(config)
                self.solvers[key] = solver
            except Exception as e:
                raise RuntimeError(f"Failed to create solver '{key}': {str(e)}")

        return self.solvers[key]

    def to_dict(self):
        """Export current state/context"""
        return {
            "robot_id": self.robot_id,
            "template": self.template,
            "map_count": len(self.maps),
            "maps": list(self.maps.keys()),
            "active_map": self.active_map,
            "active_solver": self.active_solver,
            "status": "active"
        }
