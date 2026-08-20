#!/usr/bin/env python3
"""
Convert a ROS map_server / nav2 map_saver map (.pgm/.png + .yaml) into
Spooky's native HDF5 map format, so maps produced by an external ROS mapping
stack (SLAM, a saved static map, ...) can be fed straight into a solver.

Usage:
    python pgm2HDF5.py /path/to/my_map.yaml
    python pgm2HDF5.py /path/to/my_map.yaml --output-dir quantum/maps/synthetic/imported
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

from quantum.maps.yaml2HDF5 import grid_to_graph_edges
from quantum.utils.coordinates import world_to_grid_cell, grid_cell_to_world

# Standard map_server/map_saver defaults (see ROS wiki: map_server).
DEFAULT_OCCUPIED_THRESH = 0.65
DEFAULT_FREE_THRESH = 0.196


def load_ros_map_yaml(yaml_path):
    """Load and validate a map_server-style map .yaml (image/resolution/origin)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    required = {"image", "resolution", "origin"}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"ROS map yaml {yaml_path} is missing required key(s): {sorted(missing)}")

    return meta


def pgm_to_occupancy(pgm_path, meta, unknown_as_obstacle=True):
    """
    Convert a ROS map_server/nav2 .pgm raster into a Spooky occupancy grid.

    Follows the standard map_server thresholding convention: the pixel value
    is normalized to an occupancy probability in [0, 1] (darker = more
    occupied, unless `negate` flips that), then split into occupied/free/
    unknown via `occupied_thresh`/`free_thresh`. Spooky's grid has no
    "unknown" state, so unknown (gray, unmapped) cells are folded into
    obstacles by default -- the conservative choice for planning -- or into
    free space if `unknown_as_obstacle=False`.

    PGM row 0 is the top of the raster image, which is already Spooky's
    matrix convention (row 0 = top), so no vertical flip happens here. This
    is a different array than ROS's OccupancyGrid.data: map_server builds
    that by flipping the raster so index 0 lands at the map's world-frame
    origin (bottom-left) -- that flip is a ROS-side concern, not part of this
    raw-pixel conversion.
    """
    img = np.array(Image.open(pgm_path).convert("L"), dtype=np.float64)

    negate = bool(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", DEFAULT_OCCUPIED_THRESH))
    free_thresh = float(meta.get("free_thresh", DEFAULT_FREE_THRESH))

    occ_prob = (img / 255.0) if negate else ((255.0 - img) / 255.0)

    occupancy = np.zeros(img.shape, dtype=np.uint8)
    occupancy[occ_prob > occupied_thresh] = 1
    if unknown_as_obstacle:
        unknown_mask = (occ_prob >= free_thresh) & (occ_prob <= occupied_thresh)
        occupancy[unknown_mask] = 1

    return occupancy


def pool_occupancy(occupancy, factor, mode="max", occupancy_threshold=0.2):
    """
    Coarsen an occupancy grid by merging `factor x factor` blocks of
    neighboring cells into one cell -- e.g. factor=2 merges each 2x2 group of
    4 cells into a single cell. QUBO variable count is `robots * window_T *
    M * N`, and windowing (`base_qubo.py`) only splits along T, so a
    ROS-imported map at its native pixel resolution (M*N in the hundreds of
    thousands) is orders of magnitude past `var_limit` (~1650) before a
    solver ever sees it -- this is the fix for that.

    mode="max" (default): a block is obstacle if ANY cell in it is obstacle.
    Conservative (matches costmap inflation), but can over-block narrow
    corridors/doorways when `factor` is aggressive.
    mode="threshold": a block is obstacle only if more than
    `occupancy_threshold` of its cells are obstacle. Less conservative, but
    a thin wall can be averaged away entirely.

    If M or N isn't evenly divisible by `factor`, the grid is padded up to
    the next multiple by replicating the edge row/column (not a constant
    obstacle fill): with mode="max", padding with a constant 1 would force
    the *entire* last block-row/column to obstacle regardless of what the
    real cells underneath contain, since a single fabricated obstacle cell
    in a block is enough to poison it under max-pooling.
    """
    if factor <= 1:
        return occupancy

    M, N = occupancy.shape
    pad_M = (-M) % factor
    pad_N = (-N) % factor
    if pad_M or pad_N:
        occupancy = np.pad(occupancy, ((0, pad_M), (0, pad_N)), mode="edge")

    Mp, Np = occupancy.shape
    blocks = occupancy.reshape(Mp // factor, factor, Np // factor, factor)

    if mode == "max":
        pooled = blocks.max(axis=(1, 3))
    elif mode == "threshold":
        pooled = (blocks.mean(axis=(1, 3)) > occupancy_threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown pool mode: {mode!r} (expected 'max' or 'threshold')")

    return pooled.astype(np.uint8)


def upsample_occupancy(occupancy, factor):
    """
    Inverse of `pool_occupancy`: replicate each cell into a `factor x factor`
    block, restoring the coarse grid's shape (scaled by `factor`) via nearest-
    neighbor. This does not recover detail lost during pooling -- it only
    re-expands a coarse decision back to the finer grid, e.g. to re-align a
    coarse-grid path against the original map for visualization or handoff to
    a stack that expects the map's native resolution.
    """
    if factor <= 1:
        return occupancy
    return np.repeat(np.repeat(occupancy, factor, axis=0), factor, axis=1)


def _load_geo_meta(h5_source):
    """Accept either an .h5 path (opened here) or a pre-loaded (M, origin, resolution) tuple."""
    if isinstance(h5_source, tuple):
        return h5_source
    with h5py.File(h5_source, "r") as f:
        M = f["map_structure"].shape[0]
        origin = tuple(f.attrs["origin"])
        resolution = float(f.attrs["resolution"])
    return M, origin, resolution


def world_to_grid(x, y, h5_source):
    """
    Convert a world-frame point (meters, ROS "map" frame) into the (row, col)
    grid cell of an .h5 map produced by `generate_map_from_pgm`. Thin
    file-I/O wrapper around `quantum.utils.coordinates.world_to_grid_cell` --
    see that function for the actual math and conventions.
    """
    M, origin, resolution = _load_geo_meta(h5_source)
    return world_to_grid_cell(x, y, M, origin, resolution)


def grid_to_world(row, col, h5_source):
    """
    Inverse of `world_to_grid`. Thin file-I/O wrapper around
    `quantum.utils.coordinates.grid_cell_to_world`.
    """
    M, origin, resolution = _load_geo_meta(h5_source)
    return grid_cell_to_world(row, col, M, origin, resolution)


def load_waypoints_yaml(path):
    """
    Load named world-frame points of interest from a companion YAML:

        waypoints:
          origin: [0.0, 0.0, 0.0]     # x, y, yaw(optional, default 0)
          dock_1: [3.2, -1.5]
          inspection_zone_a: [-4.0, 2.0]

    These are map-level landmarks (docks, charging stations, inspection
    zones, ...) -- not a robot's start/goal, which is per-problem data that
    stays in the sibling problems YAML instead.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("waypoints", {})


def _normalize_waypoints(waypoints):
    """dict name -> [x, y] or [x, y, yaw] --> name -> (x, y, yaw) with yaw defaulted to 0.0."""
    normalized = {}
    for name, point in waypoints.items():
        point = list(point)
        if len(point) == 2:
            point.append(0.0)
        elif len(point) != 3:
            raise ValueError(f"Waypoint {name!r} must be [x, y] or [x, y, yaw], got {point}")
        normalized[name] = (float(point[0]), float(point[1]), float(point[2]))
    return normalized


def read_waypoints(h5_source):
    """
    Read the named waypoints stored by `generate_map_from_pgm`/`add_waypoints`
    back out of an .h5, as {name: {"world": (x, y, yaw), "grid": (row, col)}}.
    """
    with h5py.File(h5_source, "r") as f:
        if "waypoints" not in f:
            return {}
        world_grp = f["waypoints/world"]
        grid_grp = f["waypoints/grid"]
        return {
            name: {
                "world": tuple(world_grp[name][:]),
                "grid": tuple(int(v) for v in grid_grp[name][:]),
            }
            for name in world_grp.keys()
        }


def _resolve_waypoints_input(waypoints, waypoints_path, default_waypoints_path):
    """
    Shared merge logic for `generate_map_from_pgm`/`add_waypoints`: an
    explicit `waypoints_path` file, else `default_waypoints_path` if it
    exists, merged with an explicit `waypoints` dict (which wins on a name
    collision). Returns name -> (x, y, yaw).
    """
    resolved = {}
    if waypoints_path is not None:
        resolved.update(load_waypoints_yaml(waypoints_path))
    elif default_waypoints_path.exists():
        resolved.update(load_waypoints_yaml(default_waypoints_path))
    if waypoints is not None:
        resolved.update(waypoints)
    return _normalize_waypoints(resolved)


def _compute_waypoint_cells(resolved_waypoints, M, origin, resolution):
    """name -> (x, y, yaw) --> name -> (row, col), with the offending name in any ValueError."""
    cells = {}
    for wp_name, (wx, wy, wyaw) in resolved_waypoints.items():
        try:
            cells[wp_name] = world_to_grid_cell(wx, wy, M, origin, resolution)
        except ValueError as e:
            raise ValueError(f"Waypoint {wp_name!r}: {e}") from e
    return cells


def add_waypoints(h5_path, waypoints=None, waypoints_path=None, overwrite=False):
    """
    Add named waypoints to an existing .h5 written by `generate_map_from_pgm`,
    without reconstructing the map. A waypoint's grid cell only depends on
    the map's `origin`/`resolution`/row count, which are already stored in
    the file, so this just opens it in append mode -- `map_structure` and
    `graph` are never touched.

    `overwrite=False` (default) raises if any name in `waypoints`/
    `waypoints_path` already exists in the file; pass `overwrite=True` to
    replace those entries instead.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "a") as f:
        if "map_structure" not in f:
            raise ValueError(f"{h5_path} has no map_structure -- is this a Spooky map .h5?")

        M = f["map_structure"].shape[0]
        origin = tuple(f.attrs["origin"])
        resolution = float(f.attrs["resolution"])

        default_waypoints_path = h5_path.with_name(f"{h5_path.stem}_waypoints.yaml")
        resolved = _resolve_waypoints_input(waypoints, waypoints_path, default_waypoints_path)
        if not resolved:
            return {}

        world_grp = f.require_group("waypoints/world")
        grid_grp = f.require_group("waypoints/grid")

        collisions = [name for name in resolved if name in world_grp]
        if collisions and not overwrite:
            raise ValueError(
                f"Waypoint name(s) already exist in {h5_path}: {collisions} "
                f"-- pass overwrite=True to replace them"
            )

        cells = _compute_waypoint_cells(resolved, M, origin, resolution)
        for wp_name, world_point in resolved.items():
            if wp_name in world_grp:
                del world_grp[wp_name]
                del grid_grp[wp_name]
            world_grp.create_dataset(wp_name, data=np.array(world_point, dtype=np.float64))
            grid_grp.create_dataset(wp_name, data=np.array(cells[wp_name], dtype=np.int32))

    print(f"Added {len(resolved)} waypoint(s) to {h5_path}: {sorted(resolved)}")
    return cells


def generate_map_from_pgm(yaml_path, output_dir=None, map_name=None,
                           unknown_as_obstacle=True, connectivity=4,
                           downsample_factor=None, target_resolution=None,
                           pool_mode="max", occupancy_threshold=0.2,
                           waypoints=None, waypoints_path=None):
    """
    Convert a ROS map_server/nav2 map (.yaml + .pgm/.png) into a Spooky .h5
    map, matching the layout `yaml2HDF5.generate_map_from_yaml` produces
    (map_structure + graph/nodes+edges), so it loads through the same
    `PathfindingProblem.from_unified_data` / `from_map_config` path.

    At most one of `downsample_factor` (explicit integer, e.g. 2 merges each
    2x2 block into 1 cell) or `target_resolution` (desired meters/cell --
    the factor is derived by rounding `target_resolution / source_resolution`)
    may be given, to coarsen the map down to a solver-feasible cell count.
    See `pool_occupancy` for why this is usually necessary for real maps.

    `waypoints`: optional dict of name -> [x, y] / [x, y, yaw] world-frame
    points (see `load_waypoints_yaml`). `waypoints_path`: optional companion
    YAML to load more from. If neither is given, a sibling
    `<yaml_stem>_waypoints.yaml` is loaded automatically if it exists. Both
    sources are merged (explicit `waypoints` wins on a name collision); each
    is stored in the output .h5 as both its original world-frame point and
    its resolved (row, col) grid cell, since -- unlike a robot's start/goal,
    which is per-problem and stays out of the map file -- these are static
    landmarks that belong to the map itself.
    """
    if downsample_factor is not None and target_resolution is not None:
        raise ValueError("Pass only one of downsample_factor or target_resolution, not both")

    yaml_path = Path(yaml_path)
    meta = load_ros_map_yaml(yaml_path)

    pgm_path = yaml_path.parent / meta["image"]
    if not pgm_path.exists():
        raise FileNotFoundError(f"Map image not found: {pgm_path}")

    default_waypoints_path = yaml_path.with_name(f"{yaml_path.stem}_waypoints.yaml")
    resolved_waypoints = _resolve_waypoints_input(waypoints, waypoints_path, default_waypoints_path)

    occupancy = pgm_to_occupancy(pgm_path, meta, unknown_as_obstacle=unknown_as_obstacle)
    source_resolution = float(meta.get("resolution", 1.0))

    factor = 1
    if target_resolution is not None:
        factor = max(1, round(target_resolution / source_resolution))
    elif downsample_factor is not None:
        factor = max(1, int(downsample_factor))

    if factor > 1:
        occupancy = pool_occupancy(occupancy, factor, mode=pool_mode,
                                    occupancy_threshold=occupancy_threshold)
    resolution = source_resolution * factor
    M, N = occupancy.shape
    map_origin = tuple(meta.get("origin", [0.0, 0.0, 0.0]))
    waypoint_cells = _compute_waypoint_cells(resolved_waypoints, M, map_origin, resolution)

    nodes, edges = grid_to_graph_edges(occupancy, connectivity=connectivity)

    name = map_name or yaml_path.stem
    output_dir = Path(output_dir) if output_dir else yaml_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{name}.h5"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("map_structure", data=occupancy, compression="gzip")

        grp = f.create_group("graph")
        grp.create_dataset("nodes", data=nodes, compression="gzip")
        grp.create_dataset("edges", data=edges, compression="gzip")

        f.attrs["map_name"] = name
        f.attrs["grid_size"] = f"{M}x{N}"
        f.attrs["resolution"] = resolution
        f.attrs["source_resolution"] = source_resolution
        f.attrs["downsample_factor"] = factor
        # World-frame pose (x, y, yaw) of grid cell [M-1, 0], meters/radians,
        # straight from the ROS yaml. No Spooky solver reads this today -- a
        # grid cell is otherwise unit-less -- it's kept so a caller can map a
        # decoded (row, col) path back to world coordinates for its own stack.
        # Note: if downsampling pads the grid (see pool_occupancy), this
        # origin is off by less than one coarse cell -- not worth correcting
        # given nothing currently consumes it downstream of this file.
        f.attrs["origin"] = list(meta.get("origin", [0.0, 0.0, 0.0]))
        f.attrs["source_format"] = "ros_pgm"
        f.attrs["generated_from"] = os.path.relpath(str(pgm_path))
        f.attrs["generated_at"] = np.bytes_(str(np.datetime64("now")))

        if resolved_waypoints:
            world_grp = f.create_group("waypoints/world")
            grid_grp = f.create_group("waypoints/grid")
            for wp_name, world_point in resolved_waypoints.items():
                world_grp.create_dataset(wp_name, data=np.array(world_point, dtype=np.float64))
                grid_grp.create_dataset(wp_name, data=np.array(waypoint_cells[wp_name], dtype=np.int32))

    print(
        f"Generated: {h5_path}  ({M}x{N}, {int(occupancy.sum())} occupied cells, "
        f"{resolution:.3g} m/cell"
        + (f", downsampled {factor}x from {source_resolution:.3g} m/px)" if factor > 1 else ")")
        + (f", {len(resolved_waypoints)} waypoint(s)" if resolved_waypoints else "")
    )
    return h5_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a ROS map_server/nav2 map (.pgm/.png + .yaml) into Spooky's HDF5 map format"
    )
    parser.add_argument("yaml_path", help="Path to the ROS map .yaml (map_server/map_saver format)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: alongside the yaml)")
    parser.add_argument("--name", default=None, help="Map name / output filename stem (default: yaml stem)")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument(
        "--unknown-as-free", action="store_true",
        help="Treat unknown (unmapped/gray) cells as free instead of the default obstacle",
    )
    downsample_group = parser.add_mutually_exclusive_group()
    downsample_group.add_argument(
        "--downsample-factor", type=int, default=None,
        help="Merge each NxN block of cells into one (e.g. 2 merges every 2x2=4 cells into 1)",
    )
    downsample_group.add_argument(
        "--target-resolution", type=float, default=None,
        help="Desired meters/cell; factor is derived from the source map's resolution",
    )
    parser.add_argument(
        "--pool-mode", choices=["max", "threshold"], default="max",
        help="max: any obstacle cell in a block blocks it (default, conservative). "
             "threshold: block is obstacle only if occupied fraction exceeds --occupancy-threshold",
    )
    parser.add_argument(
        "--occupancy-threshold", type=float, default=0.2,
        help="Occupied-fraction cutoff used by --pool-mode threshold (default: 0.2)",
    )
    parser.add_argument(
        "--waypoints-file", default=None,
        help="YAML of named world-frame points (see load_waypoints_yaml); "
             "default: <yaml_stem>_waypoints.yaml alongside the map yaml, if it exists",
    )
    args = parser.parse_args()

    generate_map_from_pgm(
        args.yaml_path,
        output_dir=args.output_dir,
        map_name=args.name,
        unknown_as_obstacle=not args.unknown_as_free,
        connectivity=args.connectivity,
        downsample_factor=args.downsample_factor,
        target_resolution=args.target_resolution,
        pool_mode=args.pool_mode,
        occupancy_threshold=args.occupancy_threshold,
        waypoints_path=args.waypoints_file,
    )


if __name__ == "__main__":
    main()
