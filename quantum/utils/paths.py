import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# One place for all benchmark / sweep / calibration output, anchored to the
# repo root so it lands in the same directory no matter which working
# directory a script was launched from (a solver script run from quantum/
# used to drop a second results/ there). Override with SPOOKY_RESULTS_DIR for
# a non-editable install where quantum/ isn't inside the repo checkout.
#   paths.py -> quantum/utils/ -> quantum/ -> repo root
RESULTS_DIR = Path(
    os.environ.get(
        "SPOOKY_RESULTS_DIR", str(Path(__file__).resolve().parents[2] / "results")
    )
)


def merge_paths(old_path: List[Tuple[int, int, int]], new_path: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Merge two single-robot paths with global timesteps.
        
        Assumes both paths already use global timesteps (offset by window's current_T).

        Args:
            old_path: Existing global path [(i, j, t), ...]
            new_path: New global path [(i, j, t), ...]

        Returns:
            Merged path [(i, j, t), ...] with no position duplication
        """
        if not old_path:
            return new_path.copy()

        if not new_path:
            return old_path.copy()

        merged = old_path.copy()
        last_i, last_j, last_t = merged[-1]
        first_i, first_j, first_t = new_path[0]

        # Check if first position of new path duplicates last position of old path
        if (last_i, last_j) == (first_i, first_j):
            # Skip the duplicate position
            merged.extend(new_path[1:])
        else:
            # No duplicate, just concatenate
            merged.extend(new_path)
        
        return merged

def clip_path_at_goal(coords: List[Tuple[int, int, int]], goal: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """
        Trim the trailing steps where a single-robot path is already parked at
        goal, keeping only the first arrival. Pure/output-only: does not touch
        any solver or windowing state — callers decide whether/where to use it.

        Args:
            coords: Single robot's path [(i, j, t), ...], sorted by t
            goal: Robot's goal position (i, j)

        Returns:
            Path truncated right after the first timestep the robot reaches
            goal and never leaves again. Unchanged if the robot never parks
            at goal for the remainder of the path.
        """
        cut = len(coords)
        for idx in range(len(coords) - 1, -1, -1):
            i, j, _ = coords[idx]
            if (i, j) != goal:
                break
            cut = idx
        return coords[:cut + 1]

def decode_position(idx: int, problem) -> Tuple[int, int, int, int]:
        """
        Decode variable index to position, time, and robot number.

        Args:
            idx: Variable index
            problem: Problem instance

        Returns:
            Tuple of (i, j, t, robot_num) coordinates
        """
        if problem.get_format_type() == "graph":
            nodes_per_robot = len(problem.graph.nodes) * problem.T
            robot_num = idx // nodes_per_robot
            reduced_idx = idx % nodes_per_robot
            t = reduced_idx // len(problem.graph.nodes)
            graph_idx = reduced_idx % len(problem.graph.nodes)
            pos = problem.graph.get_node_position(graph_idx)
            return int(pos[0]), int(pos[1]), t, robot_num
        M = problem.grid.M
        N = problem.grid.N
        T = problem.T
        robot_num = idx // (M * N * T)
        reduced_idx = idx % (M * N * T)
        t = reduced_idx // (M * N)
        pos = reduced_idx % (M * N)
        i = pos // N
        j = pos % N
        return i, j, t, robot_num