from typing import Dict, Any, List, Tuple

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