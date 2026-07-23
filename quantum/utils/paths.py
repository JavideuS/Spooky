import math
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

def bit_width(n: int) -> int:
        """
        Number of bits B = ceil(log2(n)) needed to binary-encode n distinct
        position states. n <= 1 needs no bits (only one possible state).
        """
        if n <= 1:
            return 0
        return math.ceil(math.log2(n))

def bits_to_code(bits: List[int]) -> int:
        """
        Convert a list of bit values into an integer code.

        Single source of truth for the binary-encoding bit convention used
        throughout the codebase: bits[b] is the coefficient of 2**b
        (LSB-first), matching BaseQUBO.binary_var_index(robot, t, b) directly
        — i.e. bits[b] is the value of the variable at binary_var_index(r, t, b).
        Any code that turns a bitstring into a position (or vice versa) must
        go through this convention to stay consistent.
        """
        code = 0
        for b, bit in enumerate(bits):
            code |= (int(bit) & 1) << b
        return code

def decode_position_binary(bits: List[int], problem) -> Tuple[int, int]:
        """
        Decode the B bits of a single (robot, timestep) block into a
        grid/graph (i, j) position. See bits_to_code for the bit convention.

        A binary code can exceed the number of valid positions (2^B > N);
        those codes are invalid/non-existent states with no corresponding
        position. Callers are responsible for ensuring the code is in range —
        this function does not guard against it.

        Args:
            bits: B bit values (0/1), bits[b] as returned by binary_var_index(robot, t, b)
            problem: Problem instance

        Returns:
            (i, j) position tuple
        """
        code = bits_to_code(bits)

        if problem.get_format_type() == "graph":
            pos = problem.graph.get_node_position(code)
            return int(pos[0]), int(pos[1])

        N = problem.grid.N
        return code // N, code % N