import pennylane as qml
import numpy as np
from abc import ABC, abstractmethod
from quantum.utils import paths


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
        self.current_T = 0  # This represents the current global time in the problem
        self.total_t = getattr(self.problem, "T", 1)
        self.t_max = window_max_steps or self.max_window_size()
        self.iter = 0
        # QUBO dict
        self.Q = {}
        self.initial_num_vars = 0  # To be set by subclass during build
        # Optional knob used by grid subclass
        self.distance_scaling = distance_scaling
        self.prev_solution = []

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

    def get_wires(self):
        """
        Return set of unique variable indices in the current QUBO.
        """
        if not self.Q:
            raise ValueError("QUBO dictionary is empty. Build the QUBO first.")
        qubit_indices = set()
        for (i, j) in self.Q.keys():
            qubit_indices.update([i, j])
        return qubit_indices

    # Shared: count wires from Q
    def get_num_wires(self):
        """
        Return the number of unique variable indices in the current QUBO.
        """
        return len(self.get_wires())

    # Shared: compute max window size based on var_limit
    def max_window_size(self):
        """
        Estimate max window size using problem dimensions if available.
        Subclasses relying on non-grid encodings can override this.
        """
        if self.problem.get_format_type() == "graph":
            vars_per_time = self.num_nodes
        elif self.problem.get_format_type() == "grid":
            M = self.problem.grid.M
            N = self.problem.grid.N
            vars_per_time = M * N
        
        robot_active_timeline = self.problem.get_robot_per_timestep()
        needed_vars = 0
        for t in range(self.current_T, self.total_t):
            num_active_robots = len(robot_active_timeline[t])
            needed_vars += num_active_robots * vars_per_time
            if needed_vars > self.var_limit:
                return t - self.current_T
        # We need to substract initial offset from current_T
        # The +1 is because range is exclusive at the end (else it would keep infinite loop)
        return self.total_t - self.current_T + 1

    def get_active_robot_in_window(self):
        """
        Get list of robot IDs that are active in the current window.
        """
        active_robots = set()
        robot_active_timeline = self.problem.get_robot_per_timestep()
        for t in range(self.current_T, self.current_T + self.t_max):
            for robot_id in robot_active_timeline.get(t, []):
                active_robots.add(robot_id)
        return list(active_robots)
    
    def get_active_robots_per_timestep_in_window(self):
        """
        Get a dict mapping each timestep in the current window
        to the list of active robot IDs at that timestep.
        """
        robot_active_timeline = self.problem.get_robot_per_timestep()
        active_per_timestep = {}

        for t in range(self.current_T, self.current_T + self.t_max):
            active_robots = robot_active_timeline.get(t, [])
            if active_robots:  # only include if something active
                active_per_timestep[t] = active_robots

        return active_per_timestep

    # Shared: window update/reset
    def update_problem(self, solution=[]):
        """Advance window and optionally update start state for next build."""
        self.iter += 1  # Now iter is only used for information
        # The new method to keep time is current_T which to keep track of overlapping first step with last step
        # It simply does t_max - 1 (meaning that if you only can render one time step you will always be in that time)
        # Because you need to render the step where you were and the following iteration (qubo renders the transition)
        # Note that this also helps debug since now the global time don't care about iterations (which clipped time)
        self.current_T += self.t_max - 1
        # We update current time and recalculate t_max for next window
        self.t_max = self.max_window_size()
        if self.t_max > 0:
            for robot_num, new_segment in solution.items():
                robot_id = list(self.problem.robots.keys())[robot_num]
                old_path = self.problem.robots[robot_id].path
                merged = paths.merge_paths(old_path, new_segment)

                self.problem.robots[robot_id].path = merged
                self.problem.robots[robot_id].current_position = merged[-1][:2]
            self.build()

    def reset_problem(self):
        """Reset windowing and restore initial start position if available."""
        for robot_id in self.problem.robots.keys():
            robot = self.problem.robots[robot_id]
            robot.path = []
            robot.current_position = robot.start
        self.iter = 0
        self.current_T = 0
        self.t_max = self.max_window_size()

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
        Assumes Q uses upper triangular form: (i,j) with i <= j only.
        
        fixed_vars can be:
        - dict {idx: value}
        - numpy array of length total_vars, with 0/1 for fixed vars and np.nan for free ones
        
        Returns: (reduced_Q, const_offset)
        """
        if isinstance(fixed_vars, np.ndarray):
            fixed_dict = {
                i: int(v) for i, v in enumerate(fixed_vars) if not np.isnan(v)
            }
        else:
            fixed_dict = fixed_vars
        
        Q = {}
        const_offset = 0
        
        for key, coeff in self.Q.items():
            i, j = key
            i_fixed = i in fixed_dict
            j_fixed = j in fixed_dict
            
            if not i_fixed and not j_fixed:
                # Both free: keep as is
                Q[key] = coeff
            elif i_fixed and j_fixed:
                # Both fixed: add to constant
                const_offset += coeff * fixed_dict[i] * fixed_dict[j]
            elif i_fixed:
                # Only i fixed
                if fixed_dict[i] == 1:
                    Q[(j, j)] = Q.get((j, j), 0) + coeff
            else:  # j_fixed
                # Only j fixed
                if fixed_dict[j] == 1:
                    Q[(i, i)] = Q.get((i, i), 0) + coeff
        
        return Q, const_offset

    def diag_fixed_vars(self):
        """
        Identify fixed variables based on outliers in diagonal coefficients
        per time step.

        For each time step:
        1. Collect all diagonal coefficients for variables in that time step
        2. If there are multiple variables with different coefficients:
           - Variables with smaller coefficients are fixed to 0 (outliers)
           - If one variable has a significantly larger coefficient,
             it's fixed to 1
        3. If there's only one variable in a time step, it's fixed to 1

        Note that this functions works better when there was already a
        pre-reduction. And be mind that it is iterative, when it clears at
        one iteration, it updates coefficient so it can find more fixed
        variables in the next iteration.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        from collections import Counter
        fixed = {}
        # We could get all dictionary items, but it simpler to iterate
        # over known qubit numbers
        n = self.initial_num_vars
        type = self.problem.get_format_type()
        if type == "grid":
            M, N = self.problem.grid.M, self.problem.grid.N
            vars_per_time = M * N
        elif type == "graph":
            vars_per_time = self.num_nodes
        time_step_vars = {}
        
        for i in range(n):
            if (i, i) in self.Q:
                diag_coeff = self.Q[(i, i)]
                if diag_coeff != 0:
                    t = i // vars_per_time
                    if t not in time_step_vars:
                        time_step_vars[t] = []
                    time_step_vars[t].append((i, diag_coeff))
                
        for t in time_step_vars:
            # print(f"Time step {t}: vars and diag coeffs: {time_step_vars[t]}")
            # If only one variable, fix to 1
            if len(time_step_vars[t]) == 1:
                var_idx = time_step_vars[t][0][0]
                fixed[var_idx] = 1
            
            counts = Counter([v[1] for v in time_step_vars[t]])
            # If all coefficients are the same, skip
            if len(counts) <= 1:
                continue
            else:
                # Since we are minimizing, the minimum is the best (set to 1)
                min_coeff = min(counts)
                if counts[min_coeff] == 1:
                    for var_idx, coeff in time_step_vars[t]:
                        if coeff == min_coeff:
                            fixed[var_idx] = 1
                        else:
                            fixed[var_idx] = 0
                else:
                    # These are the worst (largest) coeffs, set to 0
                    max_coeff = max(counts)
                    # print("ddx")
                    if counts[max_coeff] >= 1:
                        for var_idx, coeff in time_step_vars[t]:
                            if coeff == max_coeff:
                                fixed[var_idx] = 0

            # print(counts)
        return fixed

    def reduce_diag_fixed_vars_iterative(self):
        """
        Iteratively apply diag_fixed_vars until no new fixed variables are found.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        total_fixed = {}
        while True:
            new_fixed = self.diag_fixed_vars()
            if not new_fixed:
                break
            total_fixed.update(new_fixed)
            self.Q, _ = self.reduce_qubo(new_fixed)
        return total_fixed

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
