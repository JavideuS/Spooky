import pennylane as qml
import numpy as np
from abc import ABC, abstractmethod


class BaseQUBO(ABC):
    """
    Base class for QUBO builders that provides shared fields and common
    utilities. Subclasses should implement build() and may override any
    inherited methods as needed.

    Required attributes on the provided problem instance:
    - grid or graph structure as required by the subclass
    - start, end (optional depending on subclass)
    - T (total time horizon if using windowed approach)
    """

    def __init__(
        self,
        problem,
        penalties,
        name="unnamed",
        var_limit=601,
        window_max_steps=None,
        distance_scaling=None,
    ):
        self.problem = problem
        self.penalties = penalties
        self.name = name
        self.var_limit = var_limit
        # Windowing/time slicing support
        self.t_max = window_max_steps or self.max_window_size()
        self.total_t = getattr(self.problem, "T", 1)
        self.iter = 0
        self.T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        # Keep a copy of the initial start position if present
        self.initial_pos = getattr(problem, "start", None)
        # QUBO dict
        self.Q = {}
        # Optional knob used by grid subclass
        self.distance_scaling = distance_scaling

    # Subclasses must implement build to populate self.Q
    @abstractmethod
    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for the current window and return it."""
        raise NotImplementedError

    # Shared: QUBO -> Ising mapping (identical across formats)
    def qubo_to_ising(self):
        """
        Convert the QUBO (upper triangle) dictionary to an Ising Hamiltonian.
        Returns (qml.Hamiltonian, constant_offset).
        """
        linear_coeffs = {}
        quadratic_terms = {}
        constant = 0.0
        for (i, j), qij in self.Q.items():
            if i == j:
                constant += qij / 2
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 2
            else:
                constant += qij / 4
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 4
                linear_coeffs[j] = linear_coeffs.get(j, 0) - qij / 4
                quadratic_terms[i, j] = (
                    quadratic_terms.get((i, j), 0) + qij / 4
                )

        coeffs = []
        observables = []
        for i in sorted(linear_coeffs):
            if linear_coeffs[i] != 0:
                coeffs.append(linear_coeffs[i])
                observables.append(qml.PauliZ(i))
        for (i, j), val in quadratic_terms.items():
            if val != 0:
                coeffs.append(val)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        Hc = qml.Hamiltonian(coeffs, observables)
        return Hc, constant

    # Shared: count wires from Q
    def get_num_wires(self):
        """
        Return the number of unique variable indices in the current QUBO.
        """
        if not self.Q:
            raise ValueError("QUBO dictionary is empty. Build the QUBO first.")
        qubit_indices = set()
        for (i, j) in self.Q.keys():
            qubit_indices.update([i, j])
        return len(qubit_indices)

    # Shared: compute max window size based on var_limit
    def max_window_size(self):
        """
        Estimate max window size using problem dimensions if available.
        Subclasses relying on non-grid encodings can override this.
        """
        # Default grid-style estimation if attributes exist
        grid = getattr(self.problem, "grid", None)
        if grid is not None and hasattr(grid, "M") and hasattr(grid, "N"):
            M = grid.M
            N = grid.N
            if M and N:
                return max(1, self.var_limit // (M * N))
        # Fallback: 1 time slice
        return 1

    # Shared: window update/reset
    def update_problem(self, new_start):
        """Advance window and optionally update start state for next build."""
        self.iter += 1
        new_T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        if new_T > 0:
            self.problem.start = new_start
            self.T = new_T
            self.build()

    def reset_problem(self):
        """Reset windowing and restore initial start position if available."""
        self.problem.start = self.initial_pos
        self.iter = 0
        self.T = min(self.t_max, self.total_t - (self.iter * self.t_max))

    # Shared utilities for Q manipulation
    def dict_to_array(self, fill_value=0):
        if not self.Q:
            return np.array([[]])
        rows, cols = zip(*self.Q.keys())
        shape = (max(rows) + 1, max(cols) + 1)
        arr = np.full(shape, fill_value, dtype=float)
        for (r, c), val in self.Q.items():
            arr[r, c] = val
        return arr

    def reduce_qubo(self, fixed_vars):
        """
        Reduce a QUBO dictionary by fixing variables.

        fixed_vars can be:
        - dict {idx: value}
        - numpy array of length total_vars, with 0/1 for fixed vars and np.nan
          for free ones

        Returns: (reduced_Q, const_offset)
        """
        if isinstance(fixed_vars, np.ndarray):
            fixed_dict = {
                i: int(v) for i, v in enumerate(fixed_vars) if not np.isnan(v)
            }
        else:
            fixed_dict = fixed_vars
        Q = self.Q.copy()
        const_offset = 0
        for var, val in fixed_dict.items():
            if val == 0:
                keys_to_remove = [k for k in Q if var in k]
                for k in keys_to_remove:
                    Q.pop(k, None)
            elif val == 1:
                neighbors = [
                    j for (i, j) in Q if i == var and j != var
                ] + [
                    i for (i, j) in Q if j == var and i != var
                ]
                for nb in neighbors:
                    coeff = Q.get((var, nb), 0) + Q.get((nb, var), 0)
                    Q[(nb, nb)] = Q.get((nb, nb), 0) + coeff
                const_offset += Q.get((var, var), 0)
                keys_to_remove = [k for k in Q if var in k]
                for k in keys_to_remove:
                    Q.pop(k, None)
        return Q, const_offset

    def list_to_dict_solution(self, solution_list):
        if isinstance(solution_list, list) and len(solution_list) > 0:
            solution_dict = (
                solution_list[0] if isinstance(solution_list[0], dict) else {}
            )
        elif isinstance(solution_list, dict):
            solution_dict = solution_list
        else:
            solution_dict = {}
        return solution_dict

    def reconstruct_solution(self, reduced_sol, fixed_vars, total_vars):
        if isinstance(reduced_sol, list):
            reduced_sol = self.list_to_dict_solution(reduced_sol)
        full_sol_dict = {}
        for i in range(total_vars):
            if i in fixed_vars:
                full_sol_dict[i] = np.int8(fixed_vars[i])
            elif i in reduced_sol:
                full_sol_dict[i] = np.int8(reduced_sol[i])
            else:
                full_sol_dict[i] = np.int8(0)
        return full_sol_dict
