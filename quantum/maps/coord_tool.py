#!/usr/bin/env python3
"""
Convert a single point between Spooky's three coordinate conventions for a
given map, using that map's own (M, origin, resolution) read straight from
its .h5. Handy for debugging things like a waypoint that pgm2HDF5.py reports
as falling outside the map -- pass the same world coordinates here to see
exactly which cell they resolve to (or how far out of bounds they are) and
what that same cell looks like in the other two conventions.

Conventions (see quantum/utils/coordinates.py for the full explanation):
    --world  X Y     ROS "map" frame, meters (AMCL pose, rviz goal, waypoint YAML)
    --matrix ROW COL Spooky's native indexing; row 0 = top
    --xy     X Y     robotics/Y-up grid cell; origin at bottom-left

Usage:
    python3 maps/coord_tool.py maps/synthetic/fraunhofer/cml.h5 --world 11.0 23.0
    python3 maps/coord_tool.py maps/synthetic/fraunhofer/cml.h5 --matrix 12 7
    python3 maps/coord_tool.py maps/synthetic/fraunhofer/cml.h5 --xy 7 40
"""
import argparse

from quantum.config.hdf5parser import load_map_from_hdf5
from quantum.utils.coordinates import (
    grid_cell_to_world,
    to_matrix_rc,
    to_robotics_xy,
    world_to_grid_cell,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", help="Path to the map .h5 file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--world", nargs=2, type=float, metavar=("X", "Y"), help="World-frame point, meters")
    group.add_argument("--matrix", nargs=2, type=int, metavar=("ROW", "COL"), help="Spooky matrix (row, col) cell")
    group.add_argument("--xy", nargs=2, type=int, metavar=("X", "Y"), help="Robotics/Y-up (x, y) cell")
    args = parser.parse_args()

    map_data = load_map_from_hdf5(args.map)
    M, N = map_data["grid"]["M"], map_data["grid"]["N"]
    origin = map_data["origin"]
    resolution = map_data["resolution"]

    print(f"Map: {args.map}  ({M} rows x {N} cols)  resolution={resolution}  origin={tuple(origin)}")

    if args.world:
        x, y = args.world
        try:
            row, col = world_to_grid_cell(x, y, M, origin, resolution)
        except ValueError as e:
            print(f"World ({x}, {y}) -> OUT OF BOUNDS: {e}")
            return
    elif args.matrix:
        row, col = args.matrix
        x, y = grid_cell_to_world(row, col, M, origin, resolution)
    else:
        rx, ry = args.xy
        row, col = to_matrix_rc(rx, ry, M)
        x, y = grid_cell_to_world(row, col, M, origin, resolution)

    rx, ry = to_robotics_xy(row, col, M)
    in_bounds = 0 <= row < M and 0 <= col < N
    print(f"world (x, y)   = ({x:.4f}, {y:.4f})")
    print(f"matrix (row, col) = ({row}, {col}){'' if in_bounds else '  [OUT OF BOUNDS for this map]'}")
    print(f"xy (x, y)      = ({rx}, {ry})")


if __name__ == "__main__":
    main()
