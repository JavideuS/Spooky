"""
In-memory registries for the FastAPI planner service.

Two things are cached here for the lifetime of the process:
  - Maps: named map_id -> quantum.map.Grid and/or quantum.map.Graph, lazily
    parsed from HDF5 on first use (never eagerly loaded at startup — some
    maps are 1000x1000). Synthetic maps generally carry both a grid and a
    graph representation in the same HDF5 file; both are parsed together
    from a single file read the first time either is requested.
  - Solvers: named solver key (e.g. "dwave.general") -> solver instance, built
    once from the config loaded by config_api.py and reused across requests.

Both registries are module-level dicts, mutated in place by design (mirrors
config_api.py's pattern) so other modules can `from registry import ...` and
see live state without re-importing.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import quantum
from quantum.config.hdf5parser import load_both_from_hdf5
from quantum.map import Grid, Graph
from quantum.maps.yaml2HDF5 import generate_map_from_yaml
from quantum.solvers.solver_factory import SolverFactory

from config_api import global_solver_configs

# Map paths in maps.yaml are relative to the quantum package. Anchor on the
# package itself rather than on the cwd ("../quantum" only resolves when the
# app is launched from fastapi_app/): this points at the repo checkout under an
# editable install and at site-packages otherwise, so the app runs from any
# directory and against an installed copy of the library.
QUANTUM_ROOT = Path(quantum.__file__).resolve().parent

# .h5 map files aren't bundled in the installed package (they're generated
# from the .yaml sources, which are — see pyproject.toml's package-data
# comment). When QUANTUM_ROOT has no .h5 for a map, one is generated here
# instead of in QUANTUM_ROOT: an installed package dir may be read-only, and
# even when it isn't, anything written there is wiped on the next reinstall.
MAP_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spooky" / "maps"


class MapEntry:
    def __init__(self, path: Optional[str] = None, description: str = "",
                 grid: Optional[Grid] = None, graph: Optional[Graph] = None):
        self.path = path  # relative to quantum/, no extension; None for uploaded maps
        self.description = description
        self.grid = grid    # populated lazily unless pre-loaded (uploads)
        self.graph = graph  # populated lazily unless pre-loaded (uploads)

    @property
    def loaded(self) -> bool:
        return self.grid is not None or self.graph is not None


_map_registry: Dict[str, MapEntry] = {}
_solver_instances: Dict[str, Any] = {}


def load_map_registry(maps_config: Dict[str, dict]) -> None:
    """Populate the registry from maps.yaml's parsed 'maps' section. Does not load any map data."""
    _map_registry.clear()
    for map_id, entry in maps_config.items():
        _map_registry[map_id] = MapEntry(path=entry["path"], description=entry.get("description", ""))


def register_uploaded_map(map_id: str, grid: Optional[Grid] = None, graph: Optional[Graph] = None,
                           description: str = "uploaded") -> None:
    """Add a runtime-uploaded map to the registry, already parsed. Not persisted to maps.yaml."""
    _map_registry[map_id] = MapEntry(path=None, description=description, grid=grid, graph=graph)


def _resolve_h5_path(relative_path: str) -> Path:
    """
    Locate the .h5 for a map, generating it into MAP_CACHE_ROOT from the
    bundled .yaml if no .h5 exists yet (installed copy or previously cached).
    """
    installed = QUANTUM_ROOT / f"{relative_path}.h5"
    if installed.exists():
        return installed

    cached = MAP_CACHE_ROOT / f"{relative_path}.h5"
    if not cached.exists():
        yaml_path = QUANTUM_ROOT / f"{relative_path}.yaml"
        cached.parent.mkdir(parents=True, exist_ok=True)
        generate_map_from_yaml(
            str(yaml_path),
            output_dir=str(cached.parent),
            materials_path=str(QUANTUM_ROOT / "config" / "materials.yaml"),
        )
    return cached


def _ensure_loaded(entry: MapEntry) -> None:
    """Parse both representations from HDF5 in one read, if a registry entry hasn't been loaded yet."""
    if entry.loaded or entry.path is None:
        return
    h5_path = _resolve_h5_path(entry.path)
    data = load_both_from_hdf5(str(h5_path))
    if data["has_map"] and data["map_data"]:
        entry.grid = Grid.from_hdf5_data(data["map_data"])
    if data["has_graph"] and data["graph_data"]:
        entry.graph = Graph.from_hdf5_data(data["graph_data"])


def get_map(map_id: str, format: str = "grid"):
    """Return the Grid or Graph for map_id (format: "grid" or "graph"), loading + caching on first access."""
    if map_id not in _map_registry:
        raise KeyError(map_id)

    entry = _map_registry[map_id]
    _ensure_loaded(entry)

    representation = entry.grid if format == "grid" else entry.graph
    if representation is None:
        raise ValueError(f"Map '{map_id}' has no {format} representation")

    return representation


def list_maps() -> Dict[str, dict]:
    return {
        map_id: {
            "description": entry.description,
            "loaded": entry.loaded,
            "grid_size": f"{entry.grid.M}x{entry.grid.N}" if entry.grid else None,
            "has_grid": entry.grid is not None,
            "has_graph": entry.graph is not None,
            "source": "uploaded" if entry.path is None else entry.path,
        }
        for map_id, entry in _map_registry.items()
    }


def get_solver(solver_key: str):
    """Return a cached solver instance for solver_key, building it on first use."""
    if solver_key not in global_solver_configs:
        raise KeyError(solver_key)

    if solver_key not in _solver_instances:
        config = global_solver_configs[solver_key]
        _solver_instances[solver_key] = SolverFactory.create_solver_from_config(config)

    return _solver_instances[solver_key]
