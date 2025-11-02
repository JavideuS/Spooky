from typing import Dict, Any, List, Tuple

def merge_paths(old_path: List[Tuple[int, int, int]], new_path: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Merge two single-robot paths, fixing duplicate positions and time continuity.

        Args:
            old_path: Existing global path [(i, j, t), ...]
            new_path: New local path [(i, j, t), ...] (usually starts at t=0)

        Returns:
            Merged path [(i, j, t), ...] with consistent times and no position duplication
        """
        if not old_path:
            return new_path.copy()

        if not new_path:
            return old_path.copy()

        merged = old_path.copy()
        last_i, last_j, last_t = merged[-1]
        first_i, first_j, first_t = new_path[0]

        trimmed_new = new_path.copy()

        # Clip duplicate starting cell
        if (last_i, last_j) == (first_i, first_j):
            trimmed_new = trimmed_new[1:]

        # Offset new times so continuity holds
        adjusted_new = [(i, j, t + last_t) for (i, j, t) in trimmed_new]

        merged.extend(adjusted_new)
        return merged
